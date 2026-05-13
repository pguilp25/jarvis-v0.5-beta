"""
Debate — 2-step critique protocol to avoid 504 timeouts.

Step A: Find what's wrong + what could be added (hypotheses only — don't prove yet)
Step B: Prove/disprove each hypothesis + write improved answer

Each step is a lighter call. Total quality is higher because each step is focused.

Mini-debate (intelligent 5-6): race 3 AIs per step, first 2 win
Full debate (very intelligent 9-10): all 5 parallel per step
"""

import asyncio
from core.retry import call_with_retry
from core.cli import step, debating, status
from core.system_knowledge import SYSTEM_KNOWLEDGE


async def _race_first_n_debate(calls: list[tuple[str, str]], n: int = 2) -> list[dict]:
    """Race models, return first N to finish, cancel the rest."""
    results = []
    pending = set()

    async def _run(model, prompt):
        from core.tool_call import call_with_tools
        return await call_with_tools(
            model, prompt, enable_code_search=False, enable_web_search=True, max_tokens=16384,
        )

    for model, prompt in calls:
        task = asyncio.create_task(_run(model, prompt))
        task.model_id = model
        pending.add(task)

    while len(results) < n and pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                results.append(task.result())
            except Exception as e:
                status(f"⚠️ {task.model_id} failed: {e}")

    for task in pending:
        task.cancel()
        status(f"Dropped slowest: {task.model_id}")

    return results[:n]


# ─── Step A: Find flaws + additions (hypotheses only) ───────────────────────

STEP_A_PROMPT = SYSTEM_KNOWLEDGE + """

You are a critical reviewer. Read the original prompt and the AI answers below.

DO NOT write a full answer. Only produce HYPOTHESES about what's wrong and what's missing.

══════════════════════════════════════════════════════════
ORIGINAL PROMPT:
══════════════════════════════════════════════════════════
{original_prompt}
══════════════════════════════════════════════════════════

PREVIOUS AIs' ANSWERS:
{other_answers}

YOUR TASK — HYPOTHESES ONLY (do NOT write a full answer):

0. INFER INTENT:
   Before reviewing, restate what the user ACTUALLY wants in your own words.
   What is their underlying goal? What would a perfect answer look like?
   If the prompt is vague or unclear, what is the most likely interpretation?

1. CONSTRAINTS CHECK:
   List every constraint from the original prompt.
   For each, note which AIs addressed it and which missed it.

2. FLAW HYPOTHESES (things that might be WRONG):
   F1: [specific claim that might be incorrect — say WHY you suspect it]
   F2: [another potential flaw]
   F3: [etc.]

3. ADDITION HYPOTHESES (things that could be ADDED):
   A1: [missing topic/calculation/detail that would improve the answer]
   A2: [another addition]
   A3: [etc.]

4. BEST PARTS (what the AIs got RIGHT — keep these):
   B1: [specific good claim/calculation from AI-X — quote it]
   B2: [another good part]
   B3: [etc.]

IMPORTANT: Include EVERYTHING the next step needs to prove/disprove your hypotheses.
Include specific numbers, references, or reasoning that supports your suspicions."""


# ─── Step B: Prove/disprove + write improved answer ─────────────────────────

STEP_B_PROMPT = SYSTEM_KNOWLEDGE + """

You are in the VERIFICATION phase. A review AI produced hypotheses about what's wrong 
and what's missing in previous answers. Your job:

1. PROVE or DISPROVE each flaw hypothesis
2. DECIDE which additions are worth including
3. WRITE the complete improved answer

══════════════════════════════════════════════════════════
ORIGINAL PROMPT:
══════════════════════════════════════════════════════════
{original_prompt}
══════════════════════════════════════════════════════════

REVIEWER's HYPOTHESES:
{hypotheses}

YOUR TASK:

0. INFER INTENT:
   Before verifying, restate what the user ACTUALLY wants in your own words.
   This ensures your improved answer addresses their real goal, not just the literal words.

1. For each FLAW HYPOTHESIS:
   ✓ CONFIRMED — this is wrong. Here's the correct version: [...]
   ✗ REJECTED — the original was actually correct because: [...]

2. For each ADDITION HYPOTHESIS:
   ✓ INCLUDE — this improves the answer: [brief version to include]
   ✗ SKIP — not needed because: [reason]

3. Keep all BEST PARTS identified by the reviewer.

4. WRITE YOUR COMPLETE ANSWER:
   - Built from the verified best parts
   - With confirmed flaws FIXED
   - With approved additions INCLUDED
   - Addressing ALL constraints from the original prompt

AFTER your complete answer, you MUST write BOTH of these sections:

[ATTEMPT_LOG]
List everything you tried in this cycle — each approach, calculation, or idea:
- TRIED: [what you attempted]. RESULT: [worked/failed — explain WHY in one sentence]
- TRIED: [another approach]. RESULT: [worked/failed — why]
(list ALL attempts, even failed ones — this prevents future cycles from repeating them)

[STATUS]
SOLVED — if you believe the problem is FULLY resolved with a complete proof/solution
UNSOLVED — if gaps remain (briefly say what's still missing)"""


# ─── Mini-Debate (2-step, race 3) ───────────────────────────────────────────

async def mini_debate(
    question: str,
    answers: list[dict],
    context_tokens: int = 0,
    context: str = "",
    original_prompt: str = "",
) -> list[dict]:
    """
    Mini-debate for complexity 5-6. 2-step, race 3 per step.
    Step A: hypothesize flaws (race 3, first 2 win)
    Step B: prove/disprove + improve (race 3, first 2 win)
    """
    debating()
    prompt_to_show = original_prompt or question
    models = ["groq/llama-4-scout", "nvidia/minimax-m2.7", "nvidia/qwen-3.5"]

    all_answers_text = "\n\n".join(
        f"═══ AI-{i+1} ({a['model'].split('/')[-1]}) ═══\n{a['answer']}"
        for i, a in enumerate(answers)
    )

    # Step A: Find flaws + additions
    step("Debate Step A: hypothesize flaws (race 3)")
    step_a_prompt = STEP_A_PROMPT.format(
        original_prompt=prompt_to_show,
        other_answers=all_answers_text,
    )
    step_a_results = await _race_first_n_debate(
        [(m, step_a_prompt) for m in models], n=2,
    )
    status(f"Step A: {len(step_a_results)} hypothesis sets")

    # Combine hypotheses from both winners
    combined_hypotheses = "\n\n".join(
        f"═══ REVIEWER {i+1} ({r['model'].split('/')[-1]}) ═══\n{r['answer']}"
        for i, r in enumerate(step_a_results)
    )

    # Step B: Prove/disprove + write improved answer
    step("Debate Step B: verify + improve (race 3)")
    step_b_prompt = STEP_B_PROMPT.format(
        original_prompt=prompt_to_show,
        hypotheses=combined_hypotheses,
    )
    results = await _race_first_n_debate(
        [(m, step_b_prompt) for m in models], n=2,
    )

    status(f"Debate complete — {len(results)} improved answers")
    return results


# ─── Full Debate (2-step, all 5) ────────────────────────────────────────────

async def full_debate(
    question: str,
    answers: list[dict],
    context_tokens: int = 0,
    context: str = "",
    original_prompt: str = "",
) -> list[dict]:
    """
    Full debate for complexity 9-10. 2-step, all 5 parallel per step.
    Step A: 5 AIs hypothesize flaws in parallel
    Step B: 5 AIs prove/disprove + improve in parallel
    """
    debating()
    prompt_to_show = original_prompt or question
    models = [
        "nvidia/deepseek-v4-pro",
        "nvidia/glm-5",
        "nvidia/minimax-m2.7",
        "nvidia/qwen-3.5",
        "nvidia/kimi-k2.6",
    ]

    all_answers_text = "\n\n".join(
        f"═══ AI-{i+1} ({a['model'].split('/')[-1]}) ═══\n{a['answer']}"
        for i, a in enumerate(answers)
    )

    # Step A: All 5 hypothesize flaws in parallel
    step("Debate Step A: hypothesize flaws (all 5)")
    step_a_prompt = STEP_A_PROMPT.format(
        original_prompt=prompt_to_show,
        other_answers=all_answers_text,
    )

    async def _call(model, prompt):
        from core.tool_call import call_with_tools
        return await call_with_tools(
            model, prompt, enable_code_search=False, enable_web_search=True, max_tokens=16384,
        )

    step_a_results = list(await asyncio.gather(*[
        _call(m, step_a_prompt) for m in models
    ]))
    status(f"Step A: {len(step_a_results)} hypothesis sets")

    # Combine all hypotheses
    combined_hypotheses = "\n\n".join(
        f"═══ REVIEWER {i+1} ({r['model'].split('/')[-1]}) ═══\n{r['answer']}"
        for i, r in enumerate(step_a_results)
    )

    # Step B: All 5 prove/disprove + write improved answer
    step("Debate Step B: verify + improve (all 5)")
    step_b_prompt = STEP_B_PROMPT.format(
        original_prompt=prompt_to_show,
        hypotheses=combined_hypotheses,
    )

    results = list(await asyncio.gather(*[
        _call(m, step_b_prompt) for m in models
    ]))

    status(f"Full debate complete — {len(results)} improved answers")
    return results
