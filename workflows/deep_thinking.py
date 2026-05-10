"""
Deep Thinking — 100+ debate cycles for hard conjectures.

Same debate as very intelligent, but loops. Step B AIs self-declare SOLVED/UNSOLVED.
If any AI claims SOLVED, all 5 verify that claim. If majority confirms → resolved.

No Flash Lite — all Gemini calls use paid Flash (gemini-3.1-flash).
Trigger: !!deep or !!conjecture
"""

import asyncio
from core.retry import call_with_retry
from core.debate import STEP_A_PROMPT, STEP_B_PROMPT
from core.ensemble import run_ensemble, _build_prompt
from core.synthesizer import verify_agreement, merge_answers, synthesize
from core.state import AgentState
from core.cli import step, status, success, warn
from domains.prompts import get_assumption_prompt
from core.system_knowledge import SYSTEM_KNOWLEDGE


ATTEMPT_LOG_LIMIT = 8000

ALL_NVIDIA = [
    "nvidia/deepseek-v4-pro",
    "nvidia/glm-5",
    "nvidia/minimax-m2.5",
    "nvidia/qwen-3.5",
    "nvidia/nemotron-super",
]

# ─── Verification prompt — all 5 check if a claimed solution actually works ──

VERIFY_SOLUTION_PROMPT = SYSTEM_KNOWLEDGE + """

One AI claims to have SOLVED this problem. Your job: rigorously verify their solution.

══════════════════════════════════════════════════════════
ORIGINAL PROBLEM:
══════════════════════════════════════════════════════════
{problem}
══════════════════════════════════════════════════════════

CLAIMED SOLUTION (by {solver_model}):
{claimed_solution}

YOUR TASK:
1. Read the claimed solution carefully
2. Check EVERY step of the reasoning — is each step logically valid?
3. Look for: hidden assumptions, circular reasoning, gaps in logic, arithmetic errors
4. Try to find a COUNTEREXAMPLE that breaks the solution

Reply with ONLY:
CONFIRMED — the solution is correct. [one sentence: why it's valid]
or
REJECTED — the solution has a flaw. [explain the specific flaw]"""


async def _call(model: str, prompt: str) -> dict:
    result = await call_with_retry(model, prompt, max_tokens=16384)
    return {"model": model, "answer": result}


def _extract_attempt_log(answer: str) -> str:
    """Extract [ATTEMPT_LOG] section from an AI's answer."""
    marker = "[ATTEMPT_LOG]"
    idx = answer.find(marker)
    if idx == -1:
        idx = answer.lower().find("[attempt_log]")
    if idx == -1:
        return ""
    log_text = answer[idx + len(marker):].strip()
    for stop in ["[CONTEXT_NOTES]", "[context_notes]", "[STATUS]", "[status]", "═══"]:
        stop_idx = log_text.find(stop)
        if stop_idx != -1:
            log_text = log_text[:stop_idx].strip()
    return log_text


def _extract_status(answer: str) -> str:
    """Extract [STATUS] declaration — SOLVED or UNSOLVED."""
    marker = "[STATUS]"
    idx = answer.find(marker)
    if idx == -1:
        idx = answer.lower().find("[status]")
    if idx == -1:
        return "UNSOLVED"
    status_text = answer[idx + len(marker):].strip()
    # Take first line only
    first_line = status_text.split("\n")[0].strip().upper()
    if first_line.startswith("SOLVED"):
        return "SOLVED"
    return "UNSOLVED"


def _trim_log(log: str) -> str:
    """Hard trim — drop oldest cycle entries to stay under limit."""
    parts = log.split("--- CYCLE")
    if len(parts) <= 2:
        return log[-ATTEMPT_LOG_LIMIT:]
    while len("--- CYCLE".join(parts)) > ATTEMPT_LOG_LIMIT and len(parts) > 2:
        parts.pop(1)
    trimmed = "--- CYCLE".join(parts)
    if len(trimmed) > ATTEMPT_LOG_LIMIT:
        trimmed = trimmed[-ATTEMPT_LOG_LIMIT:]
    return trimmed


def _get_context(state: AgentState) -> str:
    history = state.get("conversation_history", [])
    if not history:
        return ""
    return "\n".join(m.get("content", "")[:2000] for m in history[-3:])


def _get_last_exchange(state: AgentState) -> str:
    history = state.get("conversation_history", [])
    if not history:
        return ""
    content = history[-1].get("content", "") if history else ""
    lines = content.split("\n") if content else []
    last_msgs = []
    for line in reversed(lines):
        if "] USER:" in line or "] ASSISTANT:" in line:
            last_msgs.insert(0, line)
            if len(last_msgs) >= 2:
                break
    return "\n".join(last_msgs) if last_msgs else ""


# ─── Main Loop ──────────────────────────────────────────────────────────────

async def chat_deep_thinking(state: AgentState) -> AgentState:
    """
    Deep iterative solver — loops debate until an AI claims SOLVED and peers confirm.
    """
    step("Chat — deep thinking")
    classification = state["classification"]
    query = state.get("processed_input", state["raw_input"])
    domain = classification.get("domain", "general")
    complexity = 10
    ctx_tokens = state.get("context_tokens", 0)
    context = _get_context(state)

    max_cycles = 100
    attempt_log = ""

    # ── Initial round: 2-step ensemble ───────────────────────────────────
    step("Initial round: 2-step ensemble")
    assumption = get_assumption_prompt(domain, complexity)
    last_exchange = _get_last_exchange(state)

    answers = await run_ensemble(
        query=query, context=context, domain=domain,
        complexity=complexity, context_tokens=ctx_tokens,
        assumption_prompt=assumption, last_exchange=last_exchange,
    )

    if not answers:
        state["final_answer"] = "Failed to get initial responses."
        return state

    original_prompt = _build_prompt(query, context, assumption, last_exchange, complexity)

    # Get initial best
    current_best = await _get_best(query, answers)
    status(f"Initial answer: {len(current_best)} chars from {len(answers)} models")

    # ── Debate cycles ────────────────────────────────────────────────────
    for cycle in range(1, max_cycles + 1):
        step(f"═══ Deep Thinking Cycle {cycle}/{max_cycles} ═══")

        all_answers_text = "\n\n".join(
            f"═══ AI-{i+1} ({a['model'].split('/')[-1]}) ═══\n{a['answer']}"
            for i, a in enumerate(answers)
        )

        # ── Step A: Hypothesize flaws (all 5 parallel) ───────────────────
        step(f"Cycle {cycle} Step A: hypothesize flaws (all 5)")
        step_a_prompt = STEP_A_PROMPT.format(
            original_prompt=original_prompt,
            other_answers=all_answers_text,
        )
        if attempt_log:
            step_a_prompt += (
                f"\n\nATTEMPT LOG (what was already tried — DO NOT repeat failed approaches):\n"
                f"{attempt_log}"
            )

        step_a_results = list(await asyncio.gather(*[
            _call(m, step_a_prompt) for m in ALL_NVIDIA
        ]))
        status(f"Step A: {len(step_a_results)} hypothesis sets")

        combined_hypotheses = "\n\n".join(
            f"═══ REVIEWER {i+1} ({r['model'].split('/')[-1]}) ═══\n{r['answer']}"
            for i, r in enumerate(step_a_results)
        )

        # ── Step B: Verify + improve (all 5 parallel) ───────────────────
        step(f"Cycle {cycle} Step B: verify + improve (all 5)")
        step_b_prompt = STEP_B_PROMPT.format(
            original_prompt=original_prompt,
            hypotheses=combined_hypotheses,
        )

        answers = list(await asyncio.gather(*[
            _call(m, step_b_prompt) for m in ALL_NVIDIA
        ]))
        status(f"Step B: {len(answers)} improved answers")

        # ── Output 1: Best merged answer ─────────────────────────────────
        current_best = await _get_best(query, answers)

        # ── Output 2: Extract attempt logs from each AI ──────────────────
        cycle_entries = []
        for a in answers:
            ai_name = a['model'].split('/')[-1]
            log_section = _extract_attempt_log(a['answer'])
            if log_section:
                cycle_entries.append(f"  {ai_name}:\n{log_section}")

        if cycle_entries:
            attempt_log += f"\n--- CYCLE {cycle} ---\n" + "\n".join(cycle_entries)
        else:
            fallback = "\n".join(
                f"  {a['model'].split('/')[-1]}: {a['answer'][-200:]}"
                for a in answers
            )
            attempt_log += f"\n--- CYCLE {cycle} ---\n{fallback}"

        if len(attempt_log) > ATTEMPT_LOG_LIMIT:
            attempt_log = _trim_log(attempt_log)

        status(f"Cycle {cycle}: best={len(current_best)} chars, log={len(attempt_log)} chars")

        # ── Check if any AI claims SOLVED ────────────────────────────────
        solvers = []
        for a in answers:
            if _extract_status(a['answer']) == "SOLVED":
                solvers.append(a)

        if solvers:
            # At least one AI claims solved — ALL 5 verify the claim
            solver = solvers[0]  # Take first solver's claim
            solver_name = solver['model'].split('/')[-1]
            step(f"🔍 {solver_name} claims SOLVED — all 5 verifying...")

            verify_prompt = VERIFY_SOLUTION_PROMPT.format(
                problem=query,
                solver_model=solver_name,
                claimed_solution=solver['answer'][:12000],
            )

            verifications = list(await asyncio.gather(*[
                _call(m, verify_prompt) for m in ALL_NVIDIA
            ]))

            confirmed = 0
            rejected = 0
            for v in verifications:
                v_name = v['model'].split('/')[-1]
                v_text = v['answer'].strip().upper()
                if v_text.startswith("CONFIRMED"):
                    confirmed += 1
                    status(f"  ✓ {v_name}: CONFIRMED")
                else:
                    rejected += 1
                    status(f"  ✗ {v_name}: REJECTED")

            if confirmed >= 3:  # Majority confirms
                success(f"VERIFIED SOLVED — {confirmed}/5 confirmed after {cycle} cycles!")
                state["final_answer"] = solver['answer']
                state["confidence"] = f"[SOLVED — verified {confirmed}/5 after {cycle} deep thinking cycles]"
                state["attempt_log"] = attempt_log
                return state
            else:
                warn(f"Claim rejected ({confirmed}/5 confirmed) — continuing...")
                attempt_log += f"\n  ⚠️ {solver_name} claimed SOLVED but rejected ({confirmed}/5)"

        status(f"Cycle {cycle}: not yet solved — continuing...")

    # Max cycles
    warn(f"Max {max_cycles} cycles reached")
    state["final_answer"] = current_best
    state["confidence"] = f"[{max_cycles} cycles completed, unresolved]"
    state["attempt_log"] = attempt_log
    return state


# ─── Helpers ─────────────────────────────────────────────────────────────────

async def _get_best(query: str, answers: list[dict]) -> str:
    verdict = await verify_agreement(query, answers)
    if verdict == "agree":
        return answers[-1]["answer"]
    elif verdict == "complementary":
        result = await merge_answers(query, answers)
        return result["answer"]
    else:
        result = await synthesize(query, answers)
        return result["answer"]
