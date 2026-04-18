"""
Conjecture mode — deep iterative thinking for extremely hard problems.

Same as very_intelligent but loops 100+ cycles. Each cycle:
  Step A: Hypothesize flaws + new approaches (5 AIs parallel)
  Step B: Prove/disprove + improve (5 AIs parallel)
  Summary: Compact log of what was tried + worked/didn't
  Check: Is the conjecture fully resolved?

The cycle history (summaries only) prevents infinite loops without context explosion.
Each cycle sees: original question + ALL past summaries + latest full answers.

Trigger: !!conjecture prefix
"""

import asyncio
from core.retry import call_with_retry
from core.ensemble import run_ensemble, _build_step_a, _build_step_b
from core.synthesizer import verify_agreement, merge_answers, synthesize
from core.state import AgentState
from core.confidence import confidence
from core.costs import cost_tracker
from core.cli import step, status, success, warn, error
from domains.prompts import get_assumption_prompt
from core.system_knowledge import SYSTEM_KNOWLEDGE
from clients.gemini import call_flash


# ─── Cycle Prompts ──────────────────────────────────────────────────────────

CYCLE_STEP_A_PROMPT = SYSTEM_KNOWLEDGE + """

You are cycle {cycle_num} of a deep iterative research process on a VERY HARD problem.
Previous cycles have tried various approaches. Your job: find NEW angles to try.

══════════════════════════════════════════════════════════
ORIGINAL PROBLEM:
══════════════════════════════════════════════════════════
{question}
══════════════════════════════════════════════════════════

HISTORY OF ALL PREVIOUS ATTEMPTS (what was tried and whether it worked):
{cycle_history}

CURRENT BEST ANSWERS FROM LAST CYCLE:
{current_answers}

YOUR TASK — HYPOTHESES ONLY (do NOT write a full answer):

1. What approaches have ALREADY been tried? (from history above — do NOT repeat them)
2. What NEW angles, methods, or frameworks could be applied that haven't been tried?
3. FLAW HYPOTHESES in current answers:
   F1: [specific flaw — say WHY you think it's wrong]
   F2: [etc.]
4. NEW APPROACH HYPOTHESES:
   N1: [a completely different way to tackle this problem]
   N2: [another novel approach]
5. KEY INSIGHTS that should be PRESERVED from previous cycles.

CRITICAL: You MUST propose at least one approach NOT in the history above.
If you can't find a new approach, explain why the problem may be unsolvable or why
the current answer is the best possible."""


CYCLE_STEP_B_PROMPT = SYSTEM_KNOWLEDGE + """

You are in the VERIFICATION phase of cycle {cycle_num}.
Reviewers proposed new hypotheses and approaches. Prove or disprove them.

══════════════════════════════════════════════════════════
ORIGINAL PROBLEM:
══════════════════════════════════════════════════════════
{question}
══════════════════════════════════════════════════════════

REVIEWERS' HYPOTHESES AND NEW APPROACHES:
{hypotheses}

YOUR TASK:

1. For each FLAW HYPOTHESIS:
   ✓ CONFIRMED — this is wrong. Correct version: [...]
   ✗ REJECTED — the original was correct because: [...]

2. For each NEW APPROACH:
   Test it. Does it lead somewhere? Show your work.
   ✓ PROMISING — here's what it yields: [...]
   ✗ DEAD END — because: [...]

3. WRITE YOUR COMPLETE UPDATED ANSWER incorporating all improvements.
   Include ALL preserved insights from previous cycles."""


SUMMARIZE_CYCLE_PROMPT = """Summarize this research cycle concisely for the history log.

Cycle {cycle_num} produced these outputs:
{cycle_output}

Write a CONCISE summary (200-400 words) covering:
1. What NEW approaches were tried in this cycle?
2. What FLAWS were found and fixed?
3. What DEAD ENDS were identified? (so future cycles don't repeat them)
4. What is the CURRENT STATE of the answer?
5. What OPEN QUESTIONS remain?

Format:
CYCLE {cycle_num} SUMMARY:
TRIED: [what was attempted]
WORKED: [what improved the answer]
FAILED: [what didn't work and why]
OPEN: [remaining questions/gaps]
STATUS: [brief current state]"""


RESOLUTION_CHECK_PROMPT = SYSTEM_KNOWLEDGE + """

Has this problem been FULLY RESOLVED?

ORIGINAL PROBLEM:
{question}

CURRENT BEST ANSWER:
{best_answer}

CYCLE HISTORY:
{cycle_history}

Think carefully:
1. Does the answer fully address the original problem?
2. Are there remaining flaws or gaps?
3. Have the last 2-3 cycles made meaningful progress, or are we going in circles?
4. Is further iteration likely to improve the answer significantly?

Reply with ONLY one of:
RESOLVED — the answer is as complete and correct as possible
PROGRESSING — meaningful progress is being made, continue
STALLED — no significant progress in recent cycles, accept current answer
IMPOSSIBLE — the problem cannot be fully solved, accept partial answer"""


# ─── Conjecture Runner ──────────────────────────────────────────────────────

NVIDIA_MODELS = [
    "nvidia/deepseek-v3.2",
    "nvidia/glm-5",
    "nvidia/minimax-m2.5",
    "nvidia/qwen-3.5",
    "nvidia/nemotron-super",
]


async def conjecture_agent(state: AgentState) -> AgentState:
    """
    Deep iterative thinking — loops cycles until resolved or max reached.
    """
    step("Conjecture mode — deep iterative thinking")
    classification = state.get("classification", {})
    query = state.get("processed_input", state["raw_input"])
    domain = classification.get("domain", "general")
    complexity = max(classification.get("complexity", 10), 9)  # Always high
    ctx_tokens = state.get("context_tokens", 0)
    context = _get_context(state)

    max_cycles = 100
    cycle_history = []  # List of compact summaries
    current_answers = []  # Latest full answers from last cycle

    # ── Initial round: same as very intelligent 2-step ensemble ──────────
    step("Cycle 0: Initial 2-step ensemble")
    assumption = get_assumption_prompt(domain, complexity)
    last_exchange = _get_last_exchange(state)

    current_answers_raw = await run_ensemble(
        query=query, context=context, domain=domain,
        complexity=complexity, context_tokens=ctx_tokens,
        assumption_prompt=assumption, last_exchange=last_exchange,
    )

    if not current_answers_raw:
        state["final_answer"] = "Failed to get initial answers."
        return state

    current_answers = current_answers_raw
    status(f"Cycle 0: {len(current_answers)} initial answers")

    # ── Cycle loop ───────────────────────────────────────────────────────
    for cycle_num in range(1, max_cycles + 1):
        step(f"═══ Cycle {cycle_num}/{max_cycles} ═══")

        # Format current answers for the cycle prompt
        answers_text = "\n\n".join(
            f"═══ AI-{i+1} ({a['model'].split('/')[-1]}) ═══\n{a['answer']}"
            for i, a in enumerate(current_answers)
        )

        history_text = "\n\n".join(cycle_history) if cycle_history else "(First cycle — no history yet)"

        # ── Step A: Hypothesize flaws + NEW approaches ───────────────
        step(f"Cycle {cycle_num} Step A: hypothesize (all 5)")
        step_a_prompt = CYCLE_STEP_A_PROMPT.format(
            cycle_num=cycle_num,
            question=query,
            cycle_history=history_text,
            current_answers=answers_text,
        )

        step_a_results = list(await asyncio.gather(*[
            _call_model(m, step_a_prompt) for m in NVIDIA_MODELS
        ]))
        status(f"Step A: {len(step_a_results)} hypothesis sets")

        # Combine all hypotheses
        combined_hypotheses = "\n\n".join(
            f"═══ REVIEWER {i+1} ({r['model'].split('/')[-1]}) ═══\n{r['answer']}"
            for i, r in enumerate(step_a_results)
        )

        # ── Step B: Prove/disprove + improve ─────────────────────────
        step(f"Cycle {cycle_num} Step B: verify + improve (all 5)")
        step_b_prompt = CYCLE_STEP_B_PROMPT.format(
            cycle_num=cycle_num,
            question=query,
            hypotheses=combined_hypotheses,
        )

        new_answers = list(await asyncio.gather(*[
            _call_model(m, step_b_prompt) for m in NVIDIA_MODELS
        ]))
        status(f"Step B: {len(new_answers)} improved answers")

        # ── Output 1: Full answers (for next cycle) ─────────────────
        current_answers = new_answers

        # ── Output 2: Cycle summary (compact, for history) ──────────
        step(f"Cycle {cycle_num}: Summarizing")
        cycle_output = "\n\n".join(
            f"AI-{i+1}: {a['answer'][:2000]}" for i, a in enumerate(new_answers)
        )
        summary = await call_flash(
            SUMMARIZE_CYCLE_PROMPT.format(
                cycle_num=cycle_num,
                cycle_output=cycle_output,
            ),
            max_tokens=1024,
        )
        cycle_history.append(summary)
        status(f"History: {len(cycle_history)} cycles, ~{sum(len(s) for s in cycle_history)} chars")

        # ── Resolution check ─────────────────────────────────────────
        step(f"Cycle {cycle_num}: Checking resolution")

        # Pick best answer for check (last AI typically strongest)
        best = current_answers[-1]["answer"]

        resolution = await call_flash(
            RESOLUTION_CHECK_PROMPT.format(
                question=query,
                best_answer=best[:8000],
                cycle_history="\n\n".join(cycle_history[-5:]),  # Last 5 cycles
            ),
            max_tokens=64,
        )

        res_upper = resolution.strip().upper()
        if res_upper.startswith("RESOLVED"):
            success(f"Conjecture RESOLVED after {cycle_num} cycles")
            break
        elif res_upper.startswith("STALLED"):
            warn(f"Conjecture STALLED after {cycle_num} cycles — accepting best answer")
            break
        elif res_upper.startswith("IMPOSSIBLE"):
            warn(f"Conjecture deemed IMPOSSIBLE after {cycle_num} cycles")
            break
        else:
            status(f"Cycle {cycle_num}: PROGRESSING — continuing")

    # ── Final synthesis ──────────────────────────────────────────────────
    step("Final synthesis")
    verdict = await verify_agreement(query, current_answers)

    if verdict == "agree":
        best = current_answers[-1]["answer"]
        vote_split = f"{len(current_answers)}/{len(current_answers)}"
    elif verdict == "complementary":
        result = await merge_answers(query, current_answers)
        best = result["answer"]
        vote_split = result["vote_split"]
    else:
        result = await synthesize(query, current_answers)
        best = result["answer"]
        vote_split = result["vote_split"]

    confidence.record(vote_split, len(current_answers))
    state["final_answer"] = best
    state["confidence"] = confidence.get_statement(vote_split, len(current_answers))
    state["cycle_count"] = cycle_num if 'cycle_num' in dir() else 0
    state["cycle_history"] = cycle_history
    success(f"Conjecture complete — {len(cycle_history)} cycles, {vote_split}")
    return state


# ─── Helpers ─────────────────────────────────────────────────────────────────

async def _call_model(model: str, prompt: str) -> dict:
    result = await call_with_retry(model, prompt, max_tokens=16384)
    return {"model": model, "answer": result}


def _get_context(state: AgentState) -> str:
    history = state.get("conversation_history", [])
    if not history:
        return ""
    return "\n".join(m.get("content", "")[:500] for m in history[-3:])


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
