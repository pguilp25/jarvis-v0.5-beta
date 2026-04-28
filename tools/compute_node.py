"""
Computational Execution Node — JARVIS v5

Secure Python REPL: subprocess execution with hard timeout,
process-tree kill, redundant retry, alternate code fallback.
"""

import os
import sys
import signal
import asyncio
import tempfile
import time
from pathlib import Path
from core.cli import step, status, warn, error, success


COMPUTE_TIMEOUT = 60
COMPUTE_MAX_OUTPUT = 50_000
COMPUTE_TEMP_DIR = Path(tempfile.gettempdir()) / "jarvis_compute"
BANNED_IMPORTS = {"shutil.rmtree", "os.system", "subprocess.call", "eval(", "exec("}
BANNED_COMMANDS = {"rm -rf", "dd if=", "mkfs", ":(){", "fork"}


def _check_code_safety(code: str) -> tuple[bool, str]:
    for banned in BANNED_IMPORTS:
        if banned in code:
            return False, f"Blocked: contains '{banned}'"
    for banned in BANNED_COMMANDS:
        if banned in code:
            return False, f"Blocked: contains '{banned}'"
    if "requests." in code or "urllib" in code or "http.client" in code:
        return False, "Blocked: network calls not allowed in compute scripts"
    return True, "OK"


async def execute_python(code: str, timeout: float = COMPUTE_TIMEOUT, label: str = "compute") -> dict:
    """Execute a Python script in a subprocess with hard timeout. Kills on hang."""
    is_safe, reason = _check_code_safety(code)
    if not is_safe:
        return {"success": False, "stdout": "", "stderr": f"SAFETY BLOCK: {reason}",
                "timeout": False, "exit_code": None, "duration": 0.0}

    COMPUTE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    script_path = COMPUTE_TEMP_DIR / f"jarvis_{label}_{int(time.time())}.py"
    try:
        script_path.write_text(code, encoding="utf-8")
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": f"Failed to write script: {e}",
                "timeout": False, "exit_code": None, "duration": 0.0}

    step(f"Executing {label} ({timeout}s timeout)...")
    t0 = time.monotonic()
    timed_out = False

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            warn(f"{label}: TIMEOUT after {timeout}s — killing process tree")
            try:
                if hasattr(os, "killpg") and proc.pid:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, OSError):
                pass
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            except Exception:
                stdout_bytes, stderr_bytes = b"", b"Process killed after timeout"

        duration = time.monotonic() - t0
        stdout = stdout_bytes.decode("utf-8", errors="replace")[:COMPUTE_MAX_OUTPUT]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:COMPUTE_MAX_OUTPUT]

        if timed_out:
            stderr += (f"\n\n--- JARVIS TIMEOUT ---\nProcess killed after {timeout}s.\n"
                       "Constraint space likely too large. Refine heuristics:\n"
                       "  - Add symmetry-breaking constraints\n"
                       "  - Reduce variable count\n"
                       "  - Use incremental solving (solver.push/pop)\n")

        result = {"success": not timed_out and proc.returncode == 0,
                  "stdout": stdout, "stderr": stderr, "timeout": timed_out,
                  "exit_code": proc.returncode, "duration": duration}

        if result["success"]: success(f"{label}: completed in {duration:.1f}s")
        elif timed_out: warn(f"{label}: timed out after {timeout}s")
        else: warn(f"{label}: failed (exit {proc.returncode}) in {duration:.1f}s")
        return result

    except Exception as e:
        duration = time.monotonic() - t0
        error(f"{label}: execution error: {e}")
        return {"success": False, "stdout": "", "stderr": f"Execution error: {e}",
                "timeout": False, "exit_code": None, "duration": duration}
    finally:
        try: script_path.unlink(missing_ok=True)
        except Exception: pass


async def execute_with_redundancy(code: str, timeout: float = COMPUTE_TIMEOUT,
                                   label: str = "compute", alternate_code: str | None = None) -> dict:
    """
    Redundant execution:
      1. Try primary code
      2. If timeout/fail → retry primary once
      3. If still fails → try alternate code twice
      4. If all fail → return skip signal
    """
    # Attempt 1: primary
    status("Compute: attempt 1/4 (primary)")
    result = await execute_python(code, timeout, f"{label}_try1")
    if result["success"]:
        return result

    # Attempt 2: retry primary (Z3 nondeterminism)
    if result["timeout"]:
        status("Compute: attempt 2/4 (primary retry)")
        result = await execute_python(code, timeout, f"{label}_try2")
        if result["success"]:
            return result

    # Attempt 3-4: alternate code
    if alternate_code:
        for alt_try in range(2):
            status(f"Compute: attempt {3 + alt_try}/4 (alternate)")
            result = await execute_python(alternate_code, timeout, f"{label}_alt{alt_try + 1}")
            if result["success"]:
                return result

    # All failed
    warn(f"Compute: all attempts failed for {label}")
    result["skip_model"] = True
    result["stderr"] += "\n\n--- ALL RETRIES EXHAUSTED ---\nSkipping this model for this compute cycle.\n"
    return result


def extract_python_code(ai_response: str) -> str | None:
    """Extract Python code block from AI response."""
    for marker in ["```python", "```Python", "```PYTHON"]:
        idx = ai_response.find(marker)
        if idx != -1:
            start = idx + len(marker)
            end = ai_response.find("```", start)
            if end != -1:
                return ai_response[start:end].strip()
    # Fallback: any fenced block
    idx = ai_response.find("```")
    if idx != -1:
        start = idx + 3
        newline = ai_response.find("\n", start)
        if newline != -1 and newline - start < 20:
            start = newline + 1
        end = ai_response.find("```", start)
        if end != -1:
            code = ai_response[start:end].strip()
            if any(kw in code for kw in ["import ", "def ", "print(", "for ", "from "]):
                return code
    return None


def format_result_for_moa(result: dict, code: str) -> str:
    """Format execution result for the MoA to reason about."""
    parts = []
    if result.get("timeout"):
        parts.append(f"## EXECUTION RESULT: TIMEOUT\nDuration: {result['duration']:.1f}s (killed)")
    elif result.get("success"):
        parts.append(f"## EXECUTION RESULT: SUCCESS\nDuration: {result['duration']:.1f}s")
    else:
        parts.append(f"## EXECUTION RESULT: ERROR\nExit code: {result.get('exit_code', '?')}")

    if result.get("stdout"):
        stdout = result["stdout"]
        if len(stdout) > 8000:
            stdout = stdout[:4000] + "\n...[truncated]...\n" + stdout[-4000:]
        parts.append(f"\n### STDOUT:\n{stdout}")
    if result.get("stderr"):
        stderr = result["stderr"]
        if len(stderr) > 4000:
            stderr = stderr[:2000] + "\n...[truncated]...\n" + stderr[-2000:]
        parts.append(f"\n### STDERR:\n{stderr}")
    if result.get("skip_model"):
        parts.append("\n### NOTE: All retries exhausted. Model SKIPPED this cycle.")

    code_preview = code[:3000] + "..." if len(code) > 3000 else code
    parts.append(f"\n### CODE:\n```python\n{code_preview}\n```")
    return "\n".join(parts)


def cleanup_temp_files():
    if COMPUTE_TEMP_DIR.exists():
        for f in COMPUTE_TEMP_DIR.glob("jarvis_*.py"):
            try: f.unlink()
            except Exception: pass
