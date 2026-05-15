#!/usr/bin/env python3
"""
JARVIS v0.5.1 — Multi-brain AI agent.

Context flow:
  1. full_history grows forever (raw, never compressed in place)
  2. compressed_context (~30K) built in background when full_history > 50K
  3. AI context = compressed_context (if exists) or raw recent
  4. If AI context > 10K → input compression to 5-10K before feeding to model
  5. Fast detector runs BEFORE any compression (uses raw recent, fits 8K)
  6. After response: raw exchange added to full_history, background compress if needed
  7. Self-eval: lenient, 1 retry with feedback if it fails
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.state import AgentState, new_state
from core.abort import is_abort, check_override
from core.input_handler import process_input
from core.tokens import count_tokens
from core.fast_detector import fast_detect
from core.decorticator import decorticate, route
from core.self_eval import self_eval
from core.formatter import format_output
from core.memory import ConversationMemory
from core.persistence import save_session, load_session, clear_session
from core.confidence import confidence
from core.costs import cost_tracker
from core.compressor import (
    compress_background, compress_for_input,
    needs_background_compression, needs_input_compression,
)
from core.cli import status, step, success, warn, error, budget_warn
from core import thought_logger

from workflows.chat import CHAT_HANDLERS
from workflows.stubs import AGENT_MAP


# ─── Pending Sandbox (for coding agent confirmation) ─────────────────────────

_pending_sandbox = None
_pending_maps = None

_project_root: str | None = None


def _load_project_root() -> str | None:
    """Load saved project root from .jarvis/project_root."""
    p = Path.home() / ".jarvis" / "project_root"
    if p.exists():
        root = p.read_text().strip()
        if Path(root).is_dir():
            return root
    return None


def _save_project_root(root: str | None):
    """Save project root to .jarvis/project_root."""
    d = Path.home() / ".jarvis"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "project_root"
    if root:
        p.write_text(root)
    elif p.exists():
        p.unlink()


# ─── Slash Commands ──────────────────────────────────────────────────────────

def handle_slash_command(cmd_raw: str, memory: ConversationMemory) -> str | None:
    global _project_root

    # Handle /project BEFORE lowercase (paths are case-sensitive)
    stripped = cmd_raw.strip()
    if stripped.lower().startswith("/project"):
        arg = stripped[len("/project"):].strip()
        if not arg:
            # Show current project
            if _project_root:
                return f"Current project: {_project_root}"
            return "No project set. Use: /project /path/to/your/project"

        # Set project root
        p = Path(arg).expanduser().resolve()
        if not p.is_dir():
            return f"Not a directory: {p}"
        _project_root = str(p)
        _save_project_root(_project_root)
        return f"Project set: {_project_root}"

    cmd = stripped.lower()

    if cmd == "/help":
        return """Available commands:
  /costs    — Show budget usage
  /memory   — Show conversation memory stats
  /project  — Show current project directory
  /project <path> — Set project directory for coding agent
  /index    — Index project code (Nemotron Super builds code maps)
  /index force — Force re-index even if cache is fresh
  /clear    — Clear conversation history
  /history  — Show recent conversation
  /help     — This message

Overrides:
  !!simple      — Force complexity 1-2 (fast)
  !!medium      — Force complexity 5-6 (intelligent)
  !!hard        — Force complexity 9-10 (full ensemble + debate)
  !!deep        — Deep thinking mode (100+ cycles, for hard conjectures)
  !!conjecture  — Same as !!deep
  !!compute     — Compute mode (code-execution + Z3 solver research loop)
  !!prove       — Formal proof mode (Lean 4 compiler — self-eval DISABLED)
  !!image       — Image generation (MiniMax expands prompt → SD3 Medium)
  !!deepcode    — Deep code mode (3-layer debate plan + parallel coders + review)

Web UI:
  python ui_main.py          — Launch web interface at localhost:3000
  python ui_main.py --port N — Use custom port"""

    if cmd == "/costs":
        return cost_tracker.summary()

    if cmd == "/memory":
        n = len(memory.full_history)
        full_tok = memory.full_token_count()
        comp_tok = memory.compressed_token_count()
        has_comp = "yes" if memory.compressed_context else "no"
        return (
            f"Messages: {n}\n"
            f"Full history: {full_tok:,} tokens\n"
            f"Background compressed: {has_comp} ({comp_tok:,} tokens)\n"
            f"Background trigger: 50K → 30K\n"
            f"Input trigger: 10K → 5-10K"
        )

    if cmd == "/clear":
        memory.clear()
        clear_session()
        return "Conversation cleared."

    if cmd == "/history":
        if not memory.full_history:
            return "No conversation history."
        recent = memory.get_recent_raw(10)
        lines = []
        for m in recent:
            role = m["role"].upper()
            content = m["content"][:200]
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    return None


# ─── Background Compression ─────────────────────────────────────────────────

async def maybe_compress_background(memory: ConversationMemory):
    """Run background compression if full history > 50K. Runs after response."""
    if memory.compression_running:
        return
    if not needs_background_compression(memory.full_history):
        return

    memory.compression_running = True
    try:
        compressed = await compress_background(memory.full_history)
        memory.compressed_context = compressed
        save_session(memory)
    except Exception as e:
        warn(f"Background compression failed: {e}")
    finally:
        memory.compression_running = False


# ─── Context Notes ───────────────────────────────────────────────────────────

def _extract_notes(answer: str) -> tuple[str, str]:
    """
    Extract [CONTEXT_NOTES] from the AI's answer.
    Returns (clean_answer, notes). Notes are stripped before showing to user.
    """
    marker = "[CONTEXT_NOTES]"
    if marker in answer:
        parts = answer.split(marker, 1)
        clean = parts[0].strip()
        notes = parts[1].strip()
        return clean, notes
    # Try lowercase/variations
    for m in ["[Context_Notes]", "[context_notes]", "[CONTEXT NOTES]"]:
        if m in answer:
            parts = answer.split(m, 1)
            return parts[0].strip(), parts[1].strip()
    return answer, ""


def _extract_thinking(text: str) -> tuple[str, str]:
    """Extract <think>...</think> blocks from response text.
    Returns (clean_text, thinking_trace).
    """
    import re
    thinking_parts = []
    # Match <think>...</think> blocks (the streaming clients use these markers)
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        thinking_parts = [m.strip() for m in matches if m.strip()]
        clean = re.sub(pattern, '', text, flags=re.DOTALL).strip()
        return clean, "\n\n".join(thinking_parts)
    return text, ""

# ─── Main Pipeline ───────────────────────────────────────────────────────────

async def process_turn(user_input: str, memory: ConversationMemory) -> str:
    """
    Process one user turn. Full pipeline: detect → classify → dispatch → eval → format.
    """
    # Budget check
    tier = cost_tracker.get_tier()
    if tier in ("emergency", "cautious", "restricted"):
        budget_warn(cost_tracker.spend_pct)

    # Start thought logging session
    thought_logger.new_session(user_input)

    # Strip mode prefixes for display/memory (keep raw for routing)
    display_input = user_input
    for _prefix in ["!!deepcode ", "!!simple ", "!!medium ", "!!hard ", "!!deep ",
                     "!!conjecture ", "!!compute ", "!!prove ", "!!image "]:
        if display_input.lower().startswith(_prefix):
            display_input = display_input[len(_prefix):].strip()
            break
    # Use display_input for memory.add("user", ...) everywhere below
    user_input_display = display_input

    # State
    state = new_state(user_input)
    handler_name = None  # Set by override or decorticator

    # ── Abort ────────────────────────────────────────────────────────────
    if is_abort(user_input):
        memory.add("user", user_input_display)
        memory.add("assistant", "OK, starting fresh. What would you like instead?")
        return "OK, starting fresh. What would you like instead?"

    # ── Override ─────────────────────────────────────────────────────────
    # Special: !!conjecture, !!compute, !!prove route to deep iterative mode
    if state["raw_input"].strip().lower().startswith("!!conjecture"):
        state["raw_input"] = state["raw_input"].strip()[len("!!conjecture"):].strip()
        state["processed_input"] = state["raw_input"]
        state["classification"] = {
            "expanded_prompt": "", "domain": "science",
            "complexity": 10, "agent": "conjecture", "intent": "deep research",
        }
        status("Override: conjecture mode (deep iterative)")
        handler_name = "conjecture"
    elif state["raw_input"].strip().lower().startswith("!!compute"):
        # Keep !!compute in the query so deep_thinking_v5 detects it
        state["processed_input"] = state["raw_input"].strip()
        state["classification"] = {
            "expanded_prompt": "", "domain": "science",
            "complexity": 10, "agent": "conjecture", "intent": "compute research",
        }
        status("Override: compute mode (code-execution research loop)")
        handler_name = "conjecture"
    elif state["raw_input"].strip().lower().startswith("!!prove"):
        # Keep !!prove in the query so deep_thinking_v5 detects it
        state["processed_input"] = state["raw_input"].strip()
        state["classification"] = {
            "expanded_prompt": "", "domain": "math",
            "complexity": 10, "agent": "conjecture", "intent": "formal proof",
        }
        status("Override: PROVE mode (Lean 4 formal proof — self-eval DISABLED)")
        handler_name = "conjecture"
    elif state["raw_input"].strip().lower().startswith("!!image"):
        state["raw_input"] = state["raw_input"].strip()[len("!!image"):].strip()
        state["processed_input"] = state["raw_input"]
        state["classification"] = {
            "expanded_prompt": "", "domain": "general",
            "complexity": 5, "agent": "image", "intent": "generate image",
        }
        status("Override: image generation mode")
        handler_name = "image"
    elif state["raw_input"].strip().lower().startswith("!!deepcode"):
        state["raw_input"] = state["raw_input"].strip()[len("!!deepcode"):].strip()
        state["processed_input"] = state["raw_input"]
        state["classification"] = {
            "expanded_prompt": "", "domain": "code",
            "complexity": 9, "agent": "code", "intent": "deep code changes",
        }
        status("Override: DEEP CODE mode (3-layer debate plan + parallel coders + review)")
        handler_name = "code"
    else:
        cleaned, forced = check_override(state["raw_input"])
        if forced is not None:
            state["raw_input"] = cleaned
            state["forced_complexity"] = forced
            status(f"Override: forced complexity {forced}")

    # ── Input processing ─────────────────────────────────────────────────
    state = await process_input(state)
    query = state.get("processed_input", state["raw_input"])

    # ── Fast detector / Decorticator (skip if already routed by override) ────
    if state.get("classification", {}).get("agent") in ("conjecture", "image", "code"):
        pass  # Already routed by !!conjecture/!!image/!!deepcode/etc

    elif state.get("forced_complexity") is not None:
        fc = state["forced_complexity"]
        if fc == 99:
            # !!deep → deep thinking mode
            state["classification"] = {
                "expanded_prompt": "", "domain": "science",
                "complexity": 10, "agent": "conjecture", "intent": "deep iterative",
            }
            status("Override: deep thinking mode (iterative)")
            handler_name = "conjecture"
        else:
            state["classification"] = {
                "expanded_prompt": "", "domain": "general",
                "complexity": fc, "agent": "chat", "intent": "",
            }
            handler_name = "chat_fast" if fc <= 2 else ("chat_intelligent" if fc <= 6 else "chat_very_intelligent")

    else:
        fast_result = await fast_detect(query)

        if fast_result["is_fast"] and fast_result["quick_answer"]:
            # Hardcoded responses (thanks, hi, bye) → instant, no self-eval
            if fast_result.get("hardcoded"):
                answer = fast_result["quick_answer"]
                memory.add("user", user_input_display)
                memory.add("assistant", answer, notes="- Acknowledgment/greeting")
                save_session(memory)
                return answer

            state["classification"] = {
                "expanded_prompt": "", "domain": "general",
                "complexity": 1, "agent": "chat", "intent": "",
            }
            # Self-eval the quick answer
            eval_result = await self_eval(query, fast_result["quick_answer"], complexity=1)
            if eval_result["passed"]:
                raw_quick = fast_result["quick_answer"]
                raw_clean_ft, thinking_trace_ft = _extract_thinking(raw_quick)
                clean, notes = _extract_notes(raw_clean_ft)
                answer = await format_output(clean)
                confidence.record("1/1", 1)
                conf = confidence.get_statement("1/1", 1)
                if conf:
                    answer += f"\n\n{conf}"
                memory.add("user", user_input_display)
                memory.add("assistant", raw_clean_ft, notes=notes, thinking_trace=thinking_trace_ft)
                save_session(memory)
                return answer

            handler_name = "chat_fast"
        else:
            # ── Build AI context (compressed_context or raw recent) ──────
            ai_context = memory.get_ai_context()
            ai_context_tokens = count_tokens(ai_context)

            # ── Input compression if AI context > 10K ────────────────────
            if needs_input_compression(ai_context_tokens):
                ai_context = await compress_for_input(ai_context)
                ai_context_tokens = count_tokens(ai_context)

            state["context_tokens"] = ai_context_tokens
            state["conversation_history"] = [{"role": "system", "content": ai_context}] if ai_context else []

            # ── Decorticator (skip if override already set handler) ────
            if not handler_name:
                state = await decorticate(state)
                handler_name = route(state["classification"])

    # ── Dispatch ─────────────────────────────────────────────────────────
    step(f"Dispatch → {handler_name}")

    # Ensure handler has AI context
    if not state.get("conversation_history"):
        ai_context = memory.get_ai_context()
        if needs_input_compression(count_tokens(ai_context)):
            ai_context = await compress_for_input(ai_context)
        state["conversation_history"] = [{"role": "system", "content": ai_context}] if ai_context else []
        state["context_tokens"] = count_tokens(ai_context)

    handler = CHAT_HANDLERS.get(handler_name) or AGENT_MAP.get(handler_name)
    if not handler:
        error(f"Unknown handler: {handler_name}")
        memory.add("user", user_input_display)
        return f"Internal error: unknown handler '{handler_name}'"

    # ── Inject project root for coding agent ─────────────────────────────
    if handler_name in ("code", "code_fluidx3d", "code_arduino"):
        if _project_root:
            state["project_root"] = _project_root
        else:
            memory.add("user", user_input_display)
            return (
                "No project directory set. The coding agent needs to know which "
                "folder to work on.\n\n"
                "Set it with:\n  /project ~/my-project-folder\n\n"
                "Then try your request again."
            )

    state = await handler(state)

    # ── Capture pending sandbox for coding agent confirmation ─────────
    global _pending_sandbox, _pending_maps
    if state.get("pending_sandbox"):
        _pending_sandbox = state["pending_sandbox"]
    if state.get("updated_maps"):
        _pending_maps = state["updated_maps"]

    # ══════════════════════════════════════════════════════════════════════
    #  HARD OUTPUT BYPASS — for !!prove mode AND image generation
    #  If the handler set bypass flags, the output should NOT be touched
    #  by self-eval or the formatter. Print raw. Save. Done.
    # ══════════════════════════════════════════════════════════════════════
    if state.get("bypass_self_eval") and state.get("bypass_formatter"):
        answer = state.get("final_answer", "No answer generated.")
        conf = state.get("confidence", "")
        proof_file = state.get("proof_file", "")

        # ── Image generation ──
        img_ctx = state.get("image_context")
        if img_ctx:
            raw_output = answer  # Just the filepath

            # Save with rich context notes so follow-ups work
            notes = (
                f"- IMAGE GENERATED: {img_ctx['query']}\n"
                f"- Expanded prompt: {img_ctx['expanded_prompt'][:200]}\n"
                f"- File: {img_ctx['filepath']}\n"
                f"- Aspect: {img_ctx['aspect']}"
            )
            try:
                memory.add("user", user_input_display)
                _, img_thinking = _extract_thinking(answer)
                memory.add("assistant", f"[IMAGE: {img_ctx['filepath']}]", notes=notes, thinking_trace=img_thinking)
                save_session(memory)
            except Exception as e:
                warn(f"Memory save failed: {e}")

            success("Image generated — bypassed self-eval and formatter")
            return raw_output

        # ── Lean proof ──
        raw_output = ""
        if proof_file:
            raw_output += f"══ COMPILER-VERIFIED PROOF ══\n"
            raw_output += f"Saved to: {proof_file}\n\n"

        raw_output += answer

        if conf:
            raw_output += f"\n\n{conf}"

        # Save to memory (raw, no formatting)
        try:
            memory.add("user", user_input_display)
            clean_proof, proof_thinking = _extract_thinking(raw_output)
            memory.add("assistant", clean_proof, notes="- Lean proof (compiler-verified, bypassed self-eval)", thinking_trace=proof_thinking)
            save_session(memory)
        except Exception as e:
            warn(f"Memory save failed: {e}")

        success("Hard Output Bypass — raw proof delivered (no self-eval, no formatter)")
        return raw_output

    # ── Normal pipeline: Self-eval + retry ────────────────────────────────
    answer = state.get("final_answer", "No answer generated.")
    complexity = state.get("classification", {}).get("complexity", 5)
    search_results = state.get("search_results", "")

    # Skip self-eval for coding agent — the code's output is a diff/summary,
    # not an answer to be judged against the user's question. Self-eval
    # evaluates the text content (did we answer the question?) which is wrong
    # for code outputs (the answer is the actual code change, not prose).
    if handler_name in ("code", "code_fluidx3d", "code_arduino"):
        status("Skipping self-eval for coding agent")
    else:
        eval_result = await self_eval(query, answer, complexity=complexity, search_results=search_results)

        if not eval_result["passed"]:
            # Retry: re-run ONLY the handler with feedback, NOT the full pipeline
            warn("Self-eval failed — retrying handler with feedback...")
            feedback = eval_result["feedback"]
            state["raw_input"] = f"{user_input}\n[SELF-EVAL FEEDBACK: {feedback}. Please fix this issue.]"
            state["processed_input"] = state["raw_input"]
            # expanded_prompt no longer used — handlers read raw_input directly
            state = await handler(state)
            answer = state.get("final_answer", answer)
            # Second eval — accept regardless
            eval2 = await self_eval(query, answer, complexity=complexity, search_results=search_results)
            if not eval2["passed"]:
                warn("Self-eval failed again — accepting answer")

    # ── Extract thinking trace from RAW answer (before formatting strips ◫...◎ markers)
    raw_clean, thinking_trace = _extract_thinking(answer)
    # ── Format + output (crash-proof — NEVER lose the answer) ─────────
    notes = ""
    try:
        clean_answer, notes = _extract_notes(raw_clean)
        formatted = await format_output(clean_answer)
        conf = state.get("confidence", "")
        if conf:
            formatted += f"\n\n{conf}"
    except Exception as e:
        warn(f"Post-processing failed ({e}) — returning raw answer")
        formatted = raw_clean
    # ── Add RAW to full_history ──────────────────────────────────────────
    try:
        memory.add("user", user_input_display)
        memory.add("assistant", raw_clean, notes=notes, thinking_trace=thinking_trace)
        save_session(memory)
    except Exception as e:
        warn(f"Memory save failed: {e}")
    return formatted


# ─── CLI Loop ────────────────────────────────────────────────────────────────

async def main():
    global _project_root, _pending_sandbox, _pending_maps

    print("\033[1m\033[96m")
    print("   ╔═══════════════════════════════════╗")
    print("   ║         J.A.R.V.I.S. v0.5.1       ║")
    print("   ║   Multi-Brain AI Agent System      ║")
    print("   ╚═══════════════════════════════════╝")
    print("\033[0m")
    print("  Type /help for commands. Type 'quit' to exit.")

    # Load saved project root
    _project_root = _load_project_root()
    if _project_root:
        print(f"  Project: {_project_root}")
    else:
        print("  No project set. Use /project <path> for coding tasks.")

    # Load saved API keys from settings.json (shared with web UI)
    try:
        import json
        settings_file = Path.home() / ".jarvis" / "settings.json"
        if settings_file.exists():
            saved = json.loads(settings_file.read_text(encoding="utf-8"))
            loaded = []
            for key in ["NVIDIA_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "OPENROUTER_API_KEY"]:
                val = saved.get(key, "")
                if val and not os.environ.get(key):
                    os.environ[key] = val
                    loaded.append(key.replace("_API_KEY", "").replace("_API_KEYS", ""))
            if loaded:
                print(f"  Loaded API keys: {', '.join(loaded)}")
    except Exception:
        pass

    print()

    memory = load_session() or ConversationMemory()

    while True:
        try:
            user_input = input("\033[1m\033[97mYou: \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            save_session(memory)
            print("Session saved. Goodbye!")
            break

        slash_response = handle_slash_command(user_input, memory)
        if slash_response is not None:
            print(f"\n\033[93m{slash_response}\033[0m\n")
            continue

        # /index — async command (needs await)
        if user_input.strip().lower().startswith("/index"):
            if not _project_root:
                print(f"\n\033[93mNo project set. Use /project <path> first.\033[0m\n")
            else:
                try:
                    from tools.code_index import generate_maps, list_sections
                    force = "force" in user_input.lower()
                    maps = await generate_maps(_project_root, force=force)
                    sections = list_sections(maps["detailed"])
                    print(f"\n\033[92m✅ Project indexed: {len(sections)} sections mapped\033[0m")
                    print(f"\033[93mSections: {', '.join(sections[:15])}")
                    if len(sections) > 15:
                        print(f"  ... and {len(sections) - 15} more")
                    print(f"\033[0m\n")
                except Exception as e:
                    print(f"\n\033[91mIndex failed: {e}\033[0m\n")
            continue

        try:
            answer = await process_turn(user_input, memory)
            print(f"\n\033[1m\033[92mJARVIS:\033[0m {answer}\n")

            # ── Coding agent confirmation ─────────────────────────────
            if _pending_sandbox is not None:
                try:
                    confirm = input("\033[1m\033[97mApply changes? (y/n): \033[0m").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    confirm = "n"

                if confirm in ("y", "yes"):
                    applied = _pending_sandbox.apply()
                    print(f"\n\033[92m✅ Changes applied:\033[0m")
                    for a in applied:
                        print(f"  {a}")
                    print()

                    # Save AI-updated maps (no re-read needed)
                    if _project_root and _pending_maps:
                        try:
                            from tools.code_index import patch_maps
                            print(f"\033[96m  Saving updated code maps...\033[0m")
                            new_hash = patch_maps(
                                _project_root,
                                _pending_maps["general"],
                                _pending_maps["detailed"],
                            )
                            print(f"\033[92m  ✓ Maps patched (hash: {new_hash[:8]})\033[0m\n")
                        except Exception as e:
                            print(f"\033[93m  ⚠️ Map patch failed: {e}\033[0m\n")
                else:
                    sb_dir = _pending_sandbox.sandbox_dir
                    print(f"\n\033[93m⚠️ Changes NOT applied. Files still in sandbox: {sb_dir}\033[0m\n")

                _pending_sandbox = None
                _pending_maps = None

            # Background compression (runs after response shown)
            asyncio.ensure_future(maybe_compress_background(memory))

        except Exception as e:
            error(f"Pipeline error: {e}")
            print(f"\n\033[91mError: {e}\033[0m\n")


if __name__ == "__main__":
    asyncio.run(main())
