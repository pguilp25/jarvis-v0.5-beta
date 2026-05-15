"""
Chat workflow — three tiers:
  - Fast (complexity 1-2): llama-4-scout answer → self-check
  - Intelligent (complexity 3-6): 3 AIs parallel, verifier, mini-debate at 5-6
  - Very intelligent (complexity 7-10): 2-5 AIs parallel, full debate, majority vote

Self-eval + retry loop is handled by main.py, not here.
Handlers set state["final_answer"] with the raw best answer.
"""

import asyncio
import re
from core.state import AgentState
from core.retry import call_with_retry
from core.confidence import confidence
from core.ensemble import run_ensemble
from core.synthesizer import verify_agreement, synthesize, merge_answers
from core.debate import mini_debate, full_debate
from core.costs import cost_tracker
from core.cli import step, status, success, warn
from domains.prompts import get_assumption_prompt
from clients.gemini import call_gemini


# ─── Search Tool for Intelligent AIs ────────────────────────────────────────

SEARCH_TOOL_INSTRUCTION = """

IMPORTANT — YOUR TOOLS:
Your training data is MONTHS old. You do NOT know current facts.
If you need current information, write on its own line:
[WEBSEARCH: your search query here]
Results will be fetched and you can continue your answer with real data.
You can request up to 3 searches. Only use when your training is likely outdated.
CRITICAL: Use EXACT names from the system knowledge above when searching. Do NOT invent model names."""


def _extract_search_requests(text: str) -> list[str]:
    """Extract search requests from AI response. Handles multiple formats."""
    queries = []
    # Old format: SEARCH: "query"
    for match in re.finditer(r'SEARCH:\s*["\']([^"\']+)["\']', text):
        queries.append(match.group(1))
    # New format: [WEBSEARCH: query]
    for match in re.finditer(r'\[WEBSEARCH:\s*(.+?)\]', text, re.IGNORECASE):
        queries.append(match.group(1))
    return queries


def _clean_search_tags(text: str) -> str:
    """Remove search tags from final answer."""
    lines = text.split("\n")
    cleaned = [l for l in lines
               if not l.strip().startswith("SEARCH:")
               and not re.match(r'\s*\[WEBSEARCH:', l, re.IGNORECASE)]
    return "\n".join(cleaned).strip()


def _get_context(state: AgentState) -> str:
    """Get AI context from state (compressed or raw)."""
    history = state.get("conversation_history", [])
    if not history:
        return ""
    return "\n".join(m.get("content", "")[:500] for m in history[-6:])


def _get_last_exchange(state: AgentState) -> str:
    """Get the last user+assistant exchange. Critical for resolving 'it', 'that', etc."""
    history = state.get("conversation_history", [])
    if not history:
        return ""
    # Find last 2 messages from raw content
    content = history[-1].get("content", "") if history else ""
    # Parse the timeline to find last user+assistant pair
    lines = content.split("\n") if content else []
    last_msgs = []
    for line in reversed(lines):
        if "] USER:" in line or "] ASSISTANT:" in line:
            last_msgs.insert(0, line)
            if len(last_msgs) >= 2:
                break
    return "\n".join(last_msgs) if last_msgs else ""


def _build_ensemble_prompt(query: str, context: str, assumption: str, last_exchange: str = "", extra_instructions: str = "", agent_name: str = "chat_intelligent") -> str:
    """Build prompt with user query at the END — clear delimiters prevent lost-in-the-middle."""
    from core.system_knowledge import SYSTEM_KNOWLEDGE
    from core.agent_context import get_agent_context

    parts = [get_agent_context(agent_name), "", SYSTEM_KNOWLEDGE, ""]

    if context:
        parts.append(
            f"CONVERSATION CONTEXT (recent turns for tracking the thread):\n"
            f"{context}\n"
        )
    if last_exchange:
        parts.append(
            f"LAST EXCHANGE (the immediate prior message — the user is most\n"
            f"likely referring to this):\n{last_exchange}\n"
        )
    if assumption:
        parts.append(f"ANALYSIS REQUIREMENTS:\n{assumption}\n")

    # Instructions BEFORE the query
    instructions = """INSTRUCTIONS:

CONTEXT IS CRITICAL — READ THIS CAREFULLY:
- BEFORE answering, check: does the user's message make sense ON ITS OWN?
- If it's ambiguous, vague, or uses words like "it", "that", "this", "the", "more",
  "also", "another", "same", "again", "continue", "keep going", "what about" —
  then it REQUIRES the conversation context above to understand.
- When context is needed: re-read the LAST EXCHANGE and the FULL CONTEXT above.
  Figure out what the user is referring to. State it explicitly in your answer
  so there's no ambiguity.
- Do NOT answer a different question than what the user is asking about.
  If they were discussing Python sorting and say "what about performance?",
  they mean sorting performance, not general performance.
- Do NOT drift to a new topic unless the user explicitly changes it.
- Stay on the thread of the conversation. Build on previous exchanges.

ANSWERING:
- Read the user's query below carefully — it may contain numbered constraints, rules, or lists.
- Address EVERY constraint in the query. Do not skip any.
- Provide a thorough, well-reasoned answer.

AFTER your answer, on a NEW line write exactly:
[CONTEXT_NOTES]
Then 1-3 short bullet points noting: what topic was discussed, any subject changes, key terms."""

    if extra_instructions:
        instructions += "\n" + extra_instructions

    parts.append(instructions)

    # User query goes LAST — clear delimiters so models can't miss it.
    # Uses the exact `USER REQUEST]` / `[END USER REQUEST]` block style
    # SYSTEM_KNOWLEDGE tells the model to look for.
    parts.append(f"""
══════════════════════════════════════════════════════════════════════
[USER REQUEST] — the human's actual task (this is what you must serve)
══════════════════════════════════════════════════════════════════════
{query}
══════════════════════════════════════════════════════════════════════
[END USER REQUEST] — read it completely, including ALL numbered items,
constraints, and rules.
══════════════════════════════════════════════════════════════════════""")

    return "\n".join(parts)


async def _call_one(model: str, prompt: str, max_tokens: int = 16384) -> dict:
    """Call one model with mid-thought web search. Used by race + gather."""
    from core.tool_call import call_with_tools
    return await call_with_tools(
        model, prompt, project_root=None, max_tokens=max_tokens,
        enable_code_search=False, enable_web_search=True,
    )


async def _race_first_n(calls: list[tuple[str, str]], n: int = 2) -> list[dict]:
    """
    Fire all models in parallel, return the first N to finish.
    Cancel the rest.
    """
    from core.cli import status

    results = []
    pending = set()

    async def _run(model, prompt):
        return await _call_one(model, prompt)

    for model, prompt in calls:
        task = asyncio.create_task(_run(model, prompt))
        task.model_id = model
        pending.add(task)

    while len(results) < n and pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                result = task.result()
                results.append(result)
            except Exception as e:
                status(f"⚠️ {task.model_id} failed: {e}", "yellow")

    for task in pending:
        task.cancel()
        status(f"Dropped slowest: {task.model_id}")

    return results[:n]


# ─── Fast Chat (Complexity 1-2) ─────────────────────────────────────────────

async def chat_fast(state: AgentState) -> AgentState:
    """
    Fast bypass. All Groq, < 1 second.
    llama-4-scout → answer → self-check
    """
    step("Chat — fast")
    query = state.get("processed_input", state["raw_input"])
    context = _get_context(state)
    last_exchange = _get_last_exchange(state)

    ctx_parts = []
    if context:
        ctx_parts.append(f"Context:\n{context}\n")
    if last_exchange:
        ctx_parts.append(f"LAST EXCHANGE (user likely refers to this):\n{last_exchange}\n")
    ctx_str = "\n".join(ctx_parts)

    prompt = f"""You are JARVIS, a multi-brain AI assistant. Fast chat mode — be concise and direct.

BEFORE answering:
1. Does the user's message below make sense ON ITS OWN, or does it need the
   conversation context above?
   - "thanks", "ok", "cool" → standalone acknowledgment. Reply warmly.
   - Clear standalone questions → answer directly.
   - Words like "it", "that", "this", "more", "also", "what about", "explain",
     "do it in", "the same", "another" → NEEDS CONTEXT. Re-read the LAST EXCHANGE.
     Figure out what they're referring to. Answer about THAT topic.
2. STAY ON TOPIC. If the conversation was about X and they ask a follow-up,
   their question is about X, not something else. Do NOT drift.
3. Answer based on your understanding.

AFTER your answer, on a NEW line write exactly:
[CONTEXT_NOTES]
Then 1-3 short bullet points noting: topic discussed, subject changes, key terms.

{ctx_str}══════════════════════════════════════════════════════════════════════
[USER REQUEST] — the human's actual message (this is what you must serve)
══════════════════════════════════════════════════════════════════════
{query}
══════════════════════════════════════════════════════════════════════
[END USER REQUEST]
══════════════════════════════════════════════════════════════════════"""

    # Step 1: answer
    answer = await call_with_retry("groq/llama-4-scout", prompt, max_tokens=16384)

    # Step 2: self-check
    check = await call_with_retry(
        "groq/llama-4-scout",
        f"Question: {query}\nAnswer: {answer}\n\n"
        f"Does this answer make sense? If yes, reply GOOD. If not, reply with a corrected answer.",
        max_tokens=16384,
    )
    if not check.strip().upper().startswith("GOOD"):
        answer = check

    confidence.record("1/1", 1)
    state["final_answer"] = answer
    state["confidence"] = confidence.get_statement("1/1", 1)
    success("Fast chat complete")
    return state


# ─── Intelligent Chat (Complexity 3-6) ──────────────────────────────────────

async def chat_intelligent(state: AgentState) -> AgentState:
    """
    3 AIs race (with search tool) → first 2 continue → verifier → mini-debate at 5-6
    AIs can request SEARCH: "query" — we fetch results and re-prompt them.
    """
    step("Chat — intelligent")
    classification = state["classification"]
    query = state.get("processed_input", state["raw_input"])
    domain = classification.get("domain", "general")
    complexity = classification.get("complexity", 5)
    ctx_tokens = state.get("context_tokens", 0)
    context = _get_context(state)

    assumption = get_assumption_prompt(domain, complexity)
    last_exchange = _get_last_exchange(state)

    # Auto-inject relevant knowledge
    from knowledge import get_auto_inject
    knowledge = get_auto_inject(query)
    extra = SEARCH_TOOL_INSTRUCTION
    if knowledge:
        extra += f"\n\nGUIDELINES (follow these):\n{knowledge}"

    full_prompt = _build_ensemble_prompt(query, context, assumption, last_exchange,
                                         extra_instructions=extra)
    # Track which prompt the AIs actually received (may change after search)
    final_prompt = full_prompt

    # Round 1: Race 3 AIs — take first 2
    answers = await _race_first_n(
        [
            ("groq/llama-4-scout", full_prompt),
            ("nvidia/glm-5.1", full_prompt),
            ("nvidia/deepseek-v4-flash", full_prompt),
        ],
        n=2,
    )

    if not answers:
        state["final_answer"] = "Failed to get responses from AI ensemble."
        return state

    status(f"First 2 finished: {[a['model'].split('/')[-1] for a in answers]}")

    # Check if any AI requested web search
    all_search_reqs = []
    for ans in answers:
        all_search_reqs.extend(_extract_search_requests(ans["answer"]))

    if all_search_reqs:
        # Use Gemini grounded search — execute ALL requests in parallel
        from tools.search import grounded_ask
        step(f"AIs requested {len(all_search_reqs)} searches — using Gemini grounded search")

        # Run ALL search requests in parallel (no cap)
        search_tasks = [grounded_ask(sq) for sq in all_search_reqs]
        search_results = list(await asyncio.gather(*search_tasks, return_exceptions=True))

        research_parts = []
        for sq, result in zip(all_search_reqs, search_results):
            if isinstance(result, Exception):
                warn(f"Search '{sq[:40]}' failed: {result}")
                continue
            sources_str = ", ".join(s.get("title", "")[:40] for s in result.get("sources", [])[:3])
            research_parts.append(
                f"SEARCH: {sq}\nFINDINGS: {result['answer'][:4000]}\nSOURCES: {sources_str}"
            )
            status(f"Search '{sq[:40]}' → grounded answer + {len(result.get('sources', []))} sources")

        if research_parts:
            research_text = "\n\n".join(research_parts)
            state["search_results"] = research_text
            step("Round 2: Racing 3 AIs with search results")
            base_prompt = _build_ensemble_prompt(query, context, assumption, last_exchange)
            augmented_prompt = (
                f"WEB SEARCH RESULTS (these are current, from Google — trust them over your training):\n"
                f"{research_text}\n\n{base_prompt}"
            )
            # Race 3 fresh AIs with search data — first 2 win
            answers = await _race_first_n(
                [
                    ("groq/llama-4-scout", augmented_prompt),
                    ("nvidia/glm-5.1", augmented_prompt),
                    ("nvidia/deepseek-v4-flash", augmented_prompt),
                ],
                n=2,
            )
            status(f"Round 2 winners: {[a['model'].split('/')[-1] for a in answers]}")
            final_prompt = augmented_prompt  # Debate will see this version

    # Clean search tags from final answers
    for ans in answers:
        ans["answer"] = _clean_search_tags(ans["answer"])

    # Verifier
    verdict = await verify_agreement(query, answers)

    if verdict == "agree":
        best_answer = answers[-1]["answer"]
        vote_split = f"{len(answers)}/{len(answers)}"
    elif verdict == "complementary":
        result = await merge_answers(query, answers, fast=True)
        best_answer = result["answer"]
        vote_split = result["vote_split"]
    elif complexity <= 4:
        result = await synthesize(query, answers)
        if result["tied"] and cost_tracker.can_use_paid():
            best_answer = await _gemini_tiebreak(query, answers, "gemini/2.5-pro")
            vote_split = f"tiebreak/{len(answers)}"
        else:
            best_answer = result["answer"]
            vote_split = result["vote_split"]
    else:
        # Conflict at complexity 5-6 → mini-debate with the 2 winners only
        step("Mini-debate (complexity 5-6)")
        revised = await mini_debate(query, answers, ctx_tokens, context=context, original_prompt=final_prompt)
        re_verdict = await verify_agreement(query, revised)
        if re_verdict == "agree":
            best_answer = revised[-1]["answer"]
            vote_split = f"{len(revised)}/{len(revised)}"
        elif re_verdict == "complementary":
            result = await merge_answers(query, revised, fast=True)
            best_answer = result["answer"]
            vote_split = result["vote_split"]
        else:
            result = await synthesize(query, revised)
            if result["tied"] and cost_tracker.can_use_paid():
                best_answer = await _gemini_tiebreak(query, revised, "gemini/2.5-pro")
                vote_split = f"tiebreak/{len(revised)}"
            else:
                best_answer = result["answer"]
                vote_split = result["vote_split"]

    confidence.record(vote_split, len(answers))
    state["final_answer"] = best_answer
    state["confidence"] = confidence.get_statement(vote_split, len(answers))
    success("Intelligent chat complete")
    return state


# ─── Very Intelligent Chat (Complexity 7-10) ────────────────────────────────

async def chat_very_intelligent(state: AgentState) -> AgentState:
    """
    Full ensemble parallel → debate (9-10) → majority vote
    """
    step("Chat — very intelligent")
    classification = state["classification"]
    query = state.get("processed_input", state["raw_input"])
    domain = classification.get("domain", "general")
    complexity = classification.get("complexity", 9)
    ctx_tokens = state.get("context_tokens", 0)
    context = _get_context(state)

    # Ensemble with deep analysis (ALL parallel)
    assumption = get_assumption_prompt(domain, complexity)
    last_exchange = _get_last_exchange(state)

    # Auto-inject relevant knowledge
    from knowledge import get_auto_inject
    knowledge = get_auto_inject(query)
    if knowledge:
        assumption += f"\n\nGUIDELINES (follow these):\n{knowledge}"

    answers = await run_ensemble(
        query=query, context=context, domain=domain,
        complexity=complexity, context_tokens=ctx_tokens,
        assumption_prompt=assumption, last_exchange=last_exchange,
    )

    if not answers:
        state["final_answer"] = "Failed to get responses from AI ensemble."
        return state

    # Build the same prompt the ensemble AIs saw — for debate to reference
    from core.ensemble import _build_prompt
    original_prompt = _build_prompt(query, context, assumption, last_exchange, complexity)

    # Debate ALWAYS at 9-10 — models must engage with each other's findings
    if complexity >= 9 and len(answers) >= 3:
        step("Full debate — forced engagement (complexity 9-10)")
        answers = await full_debate(query, answers, ctx_tokens, context=context, original_prompt=original_prompt)

    # Final synthesis
    verdict = await verify_agreement(query, answers)

    if verdict == "agree":
        best_answer = answers[-1]["answer"]
        vote_split = f"{len(answers)}/{len(answers)}"
    elif verdict == "complementary":
        result = await merge_answers(query, answers)
        best_answer = result["answer"]
        vote_split = result["vote_split"]
    else:
        # Conflict — majority vote, Gemini only if tied
        result = await synthesize(query, answers)
        if result["tied"] and cost_tracker.can_use_paid():
            step("Gemini 3.1 Pro tiebreaker")
            best_answer = await _gemini_tiebreak(query, answers, "gemini/3.1-pro")
            vote_split = f"tiebreak/{result['total']}"
        else:
            best_answer = result["answer"]
            vote_split = result["vote_split"]

    confidence.record(vote_split, len(answers))
    state["final_answer"] = best_answer
    state["confidence"] = confidence.get_statement(vote_split, len(answers))
    success("Very intelligent chat complete")
    return state


# ─── Gemini Tiebreaker ──────────────────────────────────────────────────────

async def _gemini_tiebreak(question: str, answers: list[dict], model: str) -> str:
    block = "\n\n".join(
        f"--- AI-{i+1} ({a['model'].split('/')[-1]}) ---\n{a['answer'][:3000]}"
        for i, a in enumerate(answers)
    )
    return await call_gemini(
        model,
        f"Multiple AIs disagree. Give the best synthesis.\n\n"
        f"Question: {question}\n\n{block}\n\nBest answer:",
        max_tokens=16384,
    )


# ─── Dispatch ────────────────────────────────────────────────────────────────

CHAT_HANDLERS = {
    "chat_fast": chat_fast,
    "chat_intelligent": chat_intelligent,
    "chat_very_intelligent": chat_very_intelligent,
}
