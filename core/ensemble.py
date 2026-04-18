"""
Ensemble — runs 2-5 AIs ALL IN PARALLEL. Groq + NVIDIA fire simultaneously.
At ~5 RPM real usage, we're nowhere near the 40 RPM limit.
"""

import asyncio
from core.retry import call_with_retry
from core.model_selector import select_domain_pair, select_for_context
from core.cli import step, status


async def run_ensemble(
    query: str,
    context: str,
    domain: str,
    complexity: int,
    context_tokens: int,
    assumption_prompt: str = "",
    last_exchange: str = "",
) -> list[dict]:
    """
    Run multiple AIs on the same query IN PARALLEL.
    Complexity >= 7: 2-step thinking (plan → verify) to avoid 504 timeouts.
    Returns list of {model, answer}.
    """
    step(f"Ensemble — domain={domain}, complexity={complexity}, ctx={context_tokens}")

    if complexity >= 7:
        # 2-STEP THINKING — each step is lighter, avoids 504 timeouts
        return await _two_step_ensemble(query, context, domain, complexity, context_tokens, assumption_prompt, last_exchange)

    # Simple single-prompt for lower complexity
    full_prompt = _build_prompt(query, context, assumption_prompt, last_exchange, complexity=complexity)

    if context_tokens < 8000:
        results = await _mixed_ensemble_small(full_prompt, domain)
    elif context_tokens < 25000:
        results = await _mixed_ensemble_medium(full_prompt, domain)
    else:
        results = await _nvidia_ensemble_3(full_prompt, domain, context_tokens)

    status(f"Got {len(results)} answers")
    return results


async def _two_step_ensemble(
    query: str, context: str, domain: str, complexity: int,
    context_tokens: int, assumption_prompt: str, last_exchange: str,
) -> list[dict]:
    """
    2-step thinking for complexity >= 7.
    Step A: Plan + hypotheses (all parallel, lighter calls)
    Step B: Verify + write final answer (all parallel, uses Step A output)
    """
    # Build Step A prompt
    step_a_prompt = _build_step_a(query, context, assumption_prompt, last_exchange)

    # Select models
    if complexity >= 9:
        models = [
            "nvidia/deepseek-v3.2",
            "nvidia/glm-5",
            "nvidia/minimax-m2.5",
            "nvidia/qwen-3.5",
            "nvidia/nemotron-super",
        ]
    else:
        pair = select_domain_pair(domain)
        models = list(pair)

    # Step A: All models plan in parallel
    step("Step A: Planning + hypotheses")
    step_a_results = list(await asyncio.gather(*[
        _call_one(m, step_a_prompt) for m in models
    ]))
    status(f"Step A done: {len(step_a_results)} plans")

    # Step B: Each model verifies ITS OWN plan and writes final answer
    step("Step B: Verify + write answer")
    step_b_tasks = []
    for plan in step_a_results:
        step_b_prompt = _build_step_b(query, plan["answer"])
        step_b_tasks.append(_call_one(plan["model"], step_b_prompt))

    results = list(await asyncio.gather(*step_b_tasks))
    status(f"Step B done: {len(results)} verified answers")
    return results


def _build_prompt(query: str, context: str, assumption_prompt: str, last_exchange: str = "", complexity: int = 5) -> str:
    """Build prompt. For complexity < 7, single prompt. For >= 7, use _build_step_a/b instead."""
    from core.system_knowledge import SYSTEM_KNOWLEDGE
    from core.agent_context import get_agent_context
    agent_name = "chat_very_intelligent" if complexity >= 7 else "chat_intelligent"
    parts = [get_agent_context(agent_name), "", SYSTEM_KNOWLEDGE, ""]
    if context:
        parts.append(f"CONVERSATION CONTEXT:\n{context}\n")
    if last_exchange:
        parts.append(f"LAST EXCHANGE:\n{last_exchange}\n")
    if assumption_prompt:
        parts.append(f"ANALYSIS REQUIREMENTS:\n{assumption_prompt}\n")

    parts.append("""INSTRUCTIONS:

CONTEXT IS CRITICAL:
- BEFORE answering, check: does the user's message make sense ON ITS OWN?
- If it's ambiguous or uses words like "it", "that", "this", "more", "also",
  "another", "same", "again", "continue", "what about" — it REQUIRES the
  conversation context above.
- Re-read the LAST EXCHANGE. Figure out exactly what the user is referring to.
  State it explicitly in your answer.
- Do NOT drift to a new topic. Stay on the thread of the conversation.
- Do NOT answer a different question than what they're asking about.

ANSWERING:
- FIRST: Restate what the user is actually asking for in your own words.
  If the prompt is vague, infer using the conversation context.
- Read the USER QUERY below carefully — address EVERY point.
- Provide a thorough, well-reasoned answer.
- TOOL: If you need current info, write [WEBSEARCH: your query] on its own line.

AFTER your answer, on a NEW line write exactly:
[CONTEXT_NOTES]
Then 1-3 short bullet points noting: what topic was discussed, any subject changes, key terms.""")

    parts.append(f"""
════════════════════════════════════════
USER QUERY (read this completely — including ALL numbered items, constraints, and rules):
════════════════════════════════════════
{query}
════════════════════════════════════════""")

    return "\n".join(parts)


# ─── 2-Step Thinking (complexity >= 7) ───────────────────────────────────────

def _build_step_a(query: str, context: str, assumption_prompt: str, last_exchange: str = "") -> str:
    """Step A: Plan + Define Hypotheses. Lighter call — no full answer needed."""
    from core.system_knowledge import SYSTEM_KNOWLEDGE
    from core.agent_context import get_agent_context
    parts = [get_agent_context("chat_very_intelligent"), "", SYSTEM_KNOWLEDGE, ""]
    if context:
        parts.append(f"CONVERSATION CONTEXT:\n{context}\n")
    if last_exchange:
        parts.append(f"LAST EXCHANGE:\n{last_exchange}\n")
    if assumption_prompt:
        parts.append(f"ANALYSIS REQUIREMENTS:\n{assumption_prompt}\n")

    parts.append(f"""
════════════════════════════════════════
USER QUERY (read completely — ALL constraints and rules):
════════════════════════════════════════
{query}
════════════════════════════════════════

YOUR TASK — PLANNING PHASE ONLY (do NOT write the full answer yet):

0. INFER INTENT: Restate what the user ACTUALLY wants in your own words.
   If the query is ambiguous or uses words like "it", "that", "this", "more",
   "also" — check the CONVERSATION CONTEXT and LAST EXCHANGE above.
   The user is continuing a conversation. Their question builds on what came before.
   Do NOT interpret the query in isolation if context exists.

1. CONSTRAINTS: List EVERY constraint, requirement, and condition from the query above.
   Number them. Only what the user ACTUALLY asked for.

2. APPROACH: For each constraint, briefly explain how you would address it.

3. HYPOTHESES: Based on your analysis, state your key conclusions/recommendations as hypotheses:
   H1: [your first key claim]
   H2: [your second key claim]
   H3: [etc.]
   
   For each hypothesis, note what EVIDENCE would support or refute it.

4. RISKS: What could go wrong? What failure modes exist? What tradeoffs are you making?

5. KEY DATA: Include any calculations, numbers, specific values, or references that the 
   verification step will need. Do NOT leave anything out — the next step only sees YOUR output.

IMPORTANT: Include EVERYTHING the next step needs. Be thorough in your planning.""")

    return "\n".join(parts)


def _build_step_b(query: str, step_a_output: str) -> str:
    """Step B: Verify hypotheses + write final answer. Uses Step A output."""
    from core.system_knowledge import SYSTEM_KNOWLEDGE
    return f"""{SYSTEM_KNOWLEDGE}

You are in the VERIFICATION & WRITING phase. A planning AI already analyzed the query 
and produced hypotheses. Your job:

1. READ the original query and the planning output below
2. VERIFY each hypothesis — is it correct? Does it address the constraint?
3. WRITE the complete, thorough final answer

════════════════════════════════════════
ORIGINAL QUERY:
════════════════════════════════════════
{query}
════════════════════════════════════════

PLANNING AI's OUTPUT (hypotheses + approach):
{step_a_output}

YOUR TASK:
0. INFER INTENT: Before verifying, restate what the user ACTUALLY wants in your own words.
   This ensures your answer addresses their real goal, not just the literal words.

1. VERIFY each hypothesis:
   ✓ = correct, supported by evidence/logic
   ✗ = wrong or incomplete → FIX IT with correct information
   
2. CHECK: Does the plan address ALL constraints from the query? If any are missing, add them.

3. WRITE YOUR COMPLETE ANSWER — incorporate all verified hypotheses, corrected errors, 
   and missing constraints. Be thorough.

AFTER your answer, on a NEW line write exactly:
[CONTEXT_NOTES]
Then 1-3 short bullet points noting what was discussed."""


async def _call_one(model: str, prompt: str, max_tokens: int = 16384) -> dict:
    """Call one model with mid-thought web search. Used by gather()."""
    from core.tool_call import call_with_tools
    return await call_with_tools(
        model, prompt, project_root=None, max_tokens=max_tokens,
        enable_code_search=False, enable_web_search=True,
    )


async def _mixed_ensemble_small(prompt: str, domain: str) -> list[dict]:
    """< 8K context: 2 Groq + 1 NVIDIA — ALL parallel."""
    pair = select_domain_pair(domain)
    tasks = [
        _call_one("groq/kimi-k2", prompt),
        _call_one("groq/gpt-oss-120b", prompt),
        _call_one(pair[0], prompt),
    ]
    return list(await asyncio.gather(*tasks))


async def _mixed_ensemble_medium(prompt: str, domain: str) -> list[dict]:
    """8-25K context: 1 Groq + 2 NVIDIA — ALL parallel."""
    pair = select_domain_pair(domain)
    tasks = [
        _call_one("groq/llama-4-scout", prompt),
        _call_one(pair[0], prompt),
        _call_one(pair[1], prompt),
    ]
    return list(await asyncio.gather(*tasks))


async def _nvidia_ensemble_3(prompt: str, domain: str, ctx_tokens: int) -> list[dict]:
    """25-72K context: 3 NVIDIA — ALL parallel."""
    available = select_for_context(ctx_tokens, "medium")[:3]
    if len(available) < 3:
        available = select_for_context(ctx_tokens, "simple")[:3]

    tasks = [_call_one(m, prompt) for m in available]
    return list(await asyncio.gather(*tasks))


async def _nvidia_pair(prompt: str, pair: tuple) -> list[dict]:
    """2 NVIDIA domain-matched — parallel."""
    tasks = [_call_one(m, prompt, max_tokens=16384) for m in pair]
    return list(await asyncio.gather(*tasks))


async def _nvidia_ensemble_5(prompt: str, ctx_tokens: int) -> list[dict]:
    """Full 5-model ensemble — ALL parallel."""
    all_nvidia = [
        "nvidia/deepseek-v3.2",
        "nvidia/glm-5",
        "nvidia/minimax-m2.5",
        "nvidia/qwen-3.5",
        "nvidia/nemotron-super",
    ]
    available = [m for m in all_nvidia if m in select_for_context(ctx_tokens, "extreme")]
    if len(available) < 3:
        available = [m for m in all_nvidia if m in select_for_context(ctx_tokens, "hard")]

    tasks = [_call_one(m, prompt, max_tokens=16384) for m in available]
    return list(await asyncio.gather(*tasks))
