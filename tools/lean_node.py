"""
Lean 4 Compiler Node — JARVIS v5.1

Executes .lean files via `lean` subprocess, captures compiler errors,
returns structured results for the MoA to iterate on.

Unlike the Python compute node, Lean compilation is deterministic:
  - No timeout nondeterminism (same code = same result every time)
  - Errors are structured (line:col: error: ...)
  - "No errors" = proof is CORRECT. Period. No LLM judge needed.

The Lean compiler IS the evaluator. If it accepts, the proof is valid.
"""

import asyncio
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from core.cli import step, status, warn, error, success


# ─── Configuration ───────────────────────────────────────────────────────────

LEAN_TIMEOUT = 120              # seconds — Lean type-checking can be slow
LEAN_MAX_OUTPUT = 100_000       # chars
LEAN_WORK_DIR = Path.home() / ".jarvis_lean"
LEAN_SAVE_DIR = Path.home() / "Desktop"  # Successful proofs saved here


# ─── Find Lean Binary ───────────────────────────────────────────────────────

def find_lean() -> str | None:
    """Find the lean binary. Checks elan default + PATH."""
    candidates = [
        Path.home() / ".elan" / "bin" / "lean",
        Path.home() / ".elan" / "toolchains" / "leanprover-lean4-v4.15.0" / "bin" / "lean",  # adjust version
    ]
    # Also check PATH
    lean_path = shutil.which("lean")
    if lean_path:
        candidates.insert(0, Path(lean_path))

    for p in candidates:
        if p.exists() and os.access(p, os.X_OK):
            return str(p)

    # Last resort: just try "lean" and let subprocess fail with a clear error
    return "lean"


LEAN_BIN = find_lean()


# ─── Execute Lean File ──────────────────────────────────────────────────────

async def execute_lean(
    code: str,
    timeout: float = LEAN_TIMEOUT,
    label: str = "proof",
) -> dict:
    """
    Write a .lean file and compile it. Returns:
    {
        "success": bool,      # True = no errors = PROOF VALID
        "stdout": str,
        "stderr": str,
        "timeout": bool,
        "exit_code": int|None,
        "duration": float,
        "lean_file": str,     # path to the .lean file (kept on success)
        "errors": list[str],  # parsed error messages
    }
    """
    LEAN_WORK_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    lean_file = LEAN_WORK_DIR / f"jarvis_{label}_{timestamp}.lean"

    try:
        lean_file.write_text(code, encoding="utf-8")
    except Exception as e:
        return {
            "success": False, "stdout": "", "stderr": f"Failed to write .lean file: {e}",
            "timeout": False, "exit_code": None, "duration": 0.0,
            "lean_file": str(lean_file), "errors": [str(e)],
        }

    step(f"Compiling {lean_file.name} ({timeout}s timeout)...")
    t0 = time.monotonic()
    timed_out = False

    try:
        proc = await asyncio.create_subprocess_exec(
            LEAN_BIN, str(lean_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            warn(f"Lean: TIMEOUT after {timeout}s — killing")
            try:
                if hasattr(os, "killpg") and proc.pid:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, OSError):
                pass
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=5.0
                )
            except Exception:
                stdout_bytes, stderr_bytes = b"", b"Lean killed after timeout"

        duration = time.monotonic() - t0
        stdout = stdout_bytes.decode("utf-8", errors="replace")[:LEAN_MAX_OUTPUT]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:LEAN_MAX_OUTPUT]

        # Parse errors from stderr
        errors_list = _parse_lean_errors(stderr)

        # Success = exit code 0, no error lines, not timed out
        is_success = (
            not timed_out
            and proc.returncode == 0
            and not errors_list
        )

        result = {
            "success": is_success,
            "stdout": stdout,
            "stderr": stderr,
            "timeout": timed_out,
            "exit_code": proc.returncode,
            "duration": duration,
            "lean_file": str(lean_file),
            "errors": errors_list,
        }

        if is_success:
            success(f"LEAN PROOF ACCEPTED — {lean_file.name} ({duration:.1f}s)")
        elif timed_out:
            warn(f"Lean: timed out after {timeout}s")
        else:
            n = len(errors_list)
            warn(f"Lean: {n} error{'s' if n != 1 else ''} ({duration:.1f}s)")

        return result

    except FileNotFoundError:
        error(f"Lean binary not found at '{LEAN_BIN}'. Install: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh")
        return {
            "success": False, "stdout": "",
            "stderr": f"Lean not found at '{LEAN_BIN}'. Install elan first.",
            "timeout": False, "exit_code": None, "duration": 0.0,
            "lean_file": str(lean_file), "errors": ["lean binary not found"],
        }
    except Exception as e:
        duration = time.monotonic() - t0
        error(f"Lean execution error: {e}")
        return {
            "success": False, "stdout": "", "stderr": str(e),
            "timeout": False, "exit_code": None, "duration": duration,
            "lean_file": str(lean_file), "errors": [str(e)],
        }


# ─── Parse Lean Errors ──────────────────────────────────────────────────────

def _parse_lean_errors(stderr: str) -> list[str]:
    """Extract structured error messages from Lean compiler output."""
    errors = []
    for line in stderr.split("\n"):
        stripped = line.strip()
        # Lean 4 errors look like: filename.lean:10:4: error: ...
        if ": error:" in stripped or ": error " in stripped:
            errors.append(stripped)
        elif stripped.startswith("error:"):
            errors.append(stripped)
        # Also catch "unknown identifier", "type mismatch" etc
        elif any(kw in stripped.lower() for kw in [
            "unknown identifier", "type mismatch", "unsolved goals",
            "function expected", "failed to synthesize", "declaration uses",
            "unknown namespace", "ambiguous",
        ]):
            errors.append(stripped)
    return errors


# ─── Save Proof to Desktop ──────────────────────────────────────────────────

def save_proof_to_desktop(lean_file: str, code: str, label: str = "proof") -> str:
    """
    Copy a verified .lean proof to ~/Desktop with a meaningful name.
    Returns the saved path.
    """
    LEAN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
    dest = LEAN_SAVE_DIR / f"JARVIS_PROOF_{safe_label}_{timestamp}.lean"
    dest.write_text(code, encoding="utf-8")
    success(f"Proof saved to {dest}")
    return str(dest)


# ─── Format Lean Errors for MoA ─────────────────────────────────────────────

def format_lean_errors_for_moa(result: dict, code: str) -> str:
    """Format Lean compiler output for the MoA to debug."""
    parts = []

    if result["success"]:
        parts.append("## LEAN COMPILER: PROOF ACCEPTED ✓")
        parts.append("The proof type-checks. All goals accomplished.")
    elif result["timeout"]:
        parts.append(f"## LEAN COMPILER: TIMEOUT after {result['duration']:.1f}s")
        parts.append("The proof is too expensive to type-check. Simplify the proof term.")
    else:
        parts.append(f"## LEAN COMPILER: {len(result['errors'])} ERROR(S)")
        parts.append(f"Exit code: {result['exit_code']}")
        if result["errors"]:
            parts.append("\n### ERRORS:")
            for e in result["errors"][:20]:  # Cap at 20 errors
                parts.append(f"  {e}")
        if result["stderr"] and result["stderr"] not in "\n".join(result["errors"]):
            stderr_extra = result["stderr"][:4000]
            parts.append(f"\n### FULL STDERR:\n{stderr_extra}")

    # Include the code
    code_preview = code[:5000] + "\n-- ... (truncated)" if len(code) > 5000 else code
    parts.append(f"\n### LEAN CODE THAT WAS COMPILED:\n```lean\n{code_preview}\n```")

    return "\n".join(parts)


# ─── Extract Lean Code from AI Response ──────────────────────────────────────

def extract_lean_code(ai_response: str) -> str | None:
    """Extract Lean 4 code block from AI response."""
    for marker in ["```lean", "```Lean", "```lean4", "```Lean4"]:
        idx = ai_response.find(marker)
        if idx != -1:
            start = idx + len(marker)
            end = ai_response.find("```", start)
            if end != -1:
                return ai_response[start:end].strip()

    # Fallback: any fenced block that looks like Lean
    idx = ai_response.find("```")
    if idx != -1:
        start = idx + 3
        newline = ai_response.find("\n", start)
        if newline != -1 and newline - start < 20:
            start = newline + 1
        end = ai_response.find("```", start)
        if end != -1:
            code = ai_response[start:end].strip()
            if any(kw in code for kw in ["theorem ", "lemma ", "def ", "import Mathlib", "#check", "example "]):
                return code

    return None


# ─── Cleanup ────────────────────────────────────────────────────────────────

def cleanup_lean_files():
    """Remove temp .lean files (NOT desktop saves)."""
    if LEAN_WORK_DIR.exists():
        for f in LEAN_WORK_DIR.glob("jarvis_*.lean"):
            try:
                f.unlink()
            except Exception:
                pass
