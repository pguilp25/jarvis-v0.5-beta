"""
Research workflow — 9 steps:
  1. Plan search queries
  2. Search loop (max 5 iterations)
  3. Three independent researchers (each sees ALL results, each loops to fill gaps)
  4. Deliberation — each sees others' findings, cross-verifies
  5. Merge into final research report
  6. Draw conclusions (3 AIs parallel)
  7. Compare conclusions — agree/complementary/conflict
  8. Verify sources + freshness
  9. Format + confidence

Circuit breakers: 5 search iterations, 2 re-research, 3 briefing loops per researcher
"""

import asyncio
from core.state import AgentState
from core.retry import call_with_retry
from core.synthesizer import verify_agreement, merge_answers, synthesize
from core.confidence import confidence
from core.formatter import format_output
from core.model_selector import select_domain_pair
from core.tokens import count_tokens
from core.costs import cost_tracker
from core.cli import step, status, success, warn, error
from tools.search import web_search, web_extract
from domains.prompts import get_assumption_prompt


# ─── Prompts ─────────────────────────────────────────────────────────────────

from core.system_knowledge import SYSTEM_KNOWLEDGE
from core.agent_context import get_agent_context as _get_ac
_SK = _get_ac("research") + "\n\n" + SYSTEM_KNOWLEDGE + "\n\n"

PLAN_PROMPT = _SK + """You are a research planner. The user wants information on a topic.

CONTEXT IS CRITICAL:
- If the user's question is ambiguous (uses "it", "that", "this", "more about"),
  use the CONVERSATION CONTEXT below to understand what they're asking about.
- Do NOT research a different topic than what the conversation is about.
- If the user said "search for more about that", figure out what "that" is from context.

CONVERSATION CONTEXT:
{context}

FIRST: Restate what the user is actually asking for in your own words.
What is their underlying goal? If the question is vague, use the conversation
context above to infer the most likely interpretation.
What would a perfect answer include?

Then generate 3-5 specific search queries to find comprehensive, reliable information.

User question: {question}
Domain: {domain}

For each query, explain what you're looking for:
1. Query: "..." — Goal: find ...
2. Query: "..." — Goal: find ...
(etc.)

Also specify:
- ENOUGH_CRITERIA: What constitutes a complete answer? (e.g. "3+ sources agreeing on key points")
- SOURCE_PRIORITY: What sources to prioritize (e.g. "peer-reviewed papers > official docs > tutorials")

Keep queries short and specific (3-6 words each)."""


ENOUGH_PROMPT = """You are evaluating whether we have enough information to answer a question.

Question: {question}

Information gathered so far:
{briefing}

Do we have enough information for a thorough, well-sourced answer?
Reply with ONLY:
YES — we have enough to answer comprehensively
NO: [what specific information is still missing, suggest 1-2 new search queries]"""


BRIEFING_PROMPT = _SK + """You are an independent researcher. Compile a thorough factual briefing
from these search results about the question below.

CONTEXT IS CRITICAL:
- The question may be ambiguous. Use the conversation context to understand what
  the user is actually asking about. Do NOT research a different topic.

FIRST: Restate what the user actually wants to know in your own words.
What would a perfect answer include? Keep this in mind while compiling.

Include all relevant facts, data, and claims. Note which source each fact comes from.
Organize by subtopic. Be thorough — another researcher will cross-check your work.

Question: {question}

Search results:
{results}

Your factual briefing (cite sources):

AFTER your briefing, if you notice GAPS — important aspects the results DON'T cover:
SEARCH_MORE: "query here"
(one per line, max 2). Only for genuinely missing information."""


DELIBERATION_PROMPT = """You are a researcher in a peer review round.
You already wrote your own briefing (below). Now you can see what two other
independent researchers found about the same question.

Your job:
1. What did the OTHERS find that YOU missed? Add it to your briefing.
2. What did YOU find that the OTHERS missed? Keep it — it's valuable.
3. Do any findings CONTRADICT each other? Flag them explicitly.
4. Cross-verify: if a claim appears in 2+ briefings, it's likely correct.
   If it appears in only 1, mark it as [UNVERIFIED].

Question: {question}

YOUR BRIEFING:
{my_briefing}

RESEARCHER B's BRIEFING:
{briefing_b}

RESEARCHER C's BRIEFING:
{briefing_c}

Write your REVISED, COMPLETE briefing incorporating the best of all three:"""


MERGE_RESEARCH_PROMPT = """Three researchers independently investigated a question,
then cross-verified each other's findings. Merge their final briefings into ONE
comprehensive, well-organized research report.

Question: {question}

RESEARCHER A (final):
{briefing_a}

RESEARCHER B (final):
{briefing_b}

RESEARCHER C (final):
{briefing_c}

Rules:
- Include ALL unique findings from each researcher
- Claims verified by 2+ researchers = high confidence
- Claims from only 1 researcher = note as less certain
- Resolve contradictions by noting both positions
- Organize logically, remove duplicates

Merged research report:"""


CONCLUDE_PROMPT = _SK + """You are a research analyst. Draw conclusions from this briefing.

CONTEXT IS CRITICAL:
- The user's original question may have been ambiguous. The conversation context
  tells you what they ACTUALLY want to know.
- Your conclusion must answer THEIR question, not a related but different question.
- Stay on the exact topic they asked about. Do NOT drift to adjacent topics.

Question: {question}

{assumption_prompt}

Research briefing:
{briefing}

Instructions:
- Answer the question thoroughly using the evidence in the briefing
- Cite specific sources for key claims
- Note areas of uncertainty or conflicting information
- If the briefing is insufficient, say what's missing
- Make sure your answer directly addresses what the USER asked, not what the
  research happened to find about a broader topic

AFTER your answer, on a NEW line write exactly:
[CONTEXT_NOTES]
Then 1-3 bullet points about what was researched and key findings."""


VERIFY_PROMPT = """Verify this research answer against its sources.

Question: {question}

Answer to verify:
{answer}

Original sources used:
{sources}

Check:
1. Are all claims supported by the cited sources?
2. Are sources recent enough? (Flag anything older than 2 years for fast-moving topics)
3. Any obvious contradictions between sources?
4. Any claims that seem unsupported?

Reply with:
VERIFIED — all claims supported, sources adequate
UNSUPPORTED: [which claims lack source support]
OUTDATED: [which sources are too old, suggest new search queries]"""


# ─── Research Agent ──────────────────────────────────────────────────────────

async def research_agent(state: AgentState) -> AgentState:
    """
    Full 7-step research workflow.
    """
    step("Research agent")
    classification = state.get("classification", {})
    query = state.get("processed_input", state["raw_input"])
    domain = classification.get("domain", "general")
    complexity = classification.get("complexity", 5)

    all_results = []       # Accumulated search results
    max_search_loops = 5
    tavily_budget = 15     # Max credits for this task

    # ── Step 1: Plan search ──────────────────────────────────────────────
    step("Step 1: Plan search")

    # Get conversation context for ambiguous queries
    conv_history = state.get("conversation_history", [])
    conv_context = ""
    if conv_history:
        conv_context = "\n".join(m.get("content", "")[:500] for m in conv_history[-6:])

    plan = await call_with_retry(
        "groq/llama-4-scout",
        PLAN_PROMPT.format(question=query, domain=domain, context=conv_context or "(no previous conversation)"),
        max_tokens=1024,
    )
    queries = _parse_queries(plan)
    if not queries:
        queries = [query]  # Fallback: search the raw question
    status(f"Planned {len(queries)} queries")

    # ── Step 2: Search loop ──────────────────────────────────────────────
    step("Step 2: Search loop")
    for iteration in range(max_search_loops):
        for q in queries:
            if cost_tracker.tavily_credits >= 950 or len(all_results) > 30:
                break
            try:
                results = await web_search(q, max_results=5)
                all_results.extend(results)
                status(f"Search '{q[:40]}' → {len(results)} results (total: {len(all_results)})")
            except Exception as e:
                warn(f"Search failed: {e}")

        # Check if we have enough
        if len(all_results) >= 5:
            briefing_text = _format_search_results(all_results)
            enough = await call_with_retry(
                "groq/llama-4-scout",
                ENOUGH_PROMPT.format(question=query, briefing=briefing_text[:6000]),
                max_tokens=256,
            )

            if enough.strip().upper().startswith("YES"):
                status(f"Enough info after {iteration + 1} iteration(s)")
                break
            elif iteration < max_search_loops - 1:
                # Parse new queries from the NO response
                new_queries = _parse_inline_queries(enough)
                if new_queries:
                    queries = new_queries
                    status(f"Refining: {len(new_queries)} new queries")
                else:
                    break
        elif iteration == 0:
            continue  # Try again with same queries
        else:
            break

    if not all_results:
        state["final_answer"] = "Could not find any search results. Try rephrasing your question."
        return state

    # ══ RESEARCH PIPELINE ════════════════════════════════════════════════
    # Loop A: Steps 3→4 (briefing convergence, max 2)
    # Loop B: Steps 6→7 (conclusion convergence, max 2)
    briefing_models = ["groq/llama-4-scout", "nvidia/deepseek-v4-pro", "nvidia/deepseek-v4-flash"]
    disagreement_context = ""

    # ── LOOP A: Steps 3-4 (briefing convergence) ─────────────────────
    max_briefing_loops = 2
    for briefing_loop in range(max_briefing_loops):
            if briefing_loop > 0:
                step(f"── Briefing re-research (round {briefing_loop + 1}) ──")

            # ── Step 3: Three independent researchers ────────────────────
            step("Step 3: Three independent researchers")

            async def _independent_research(model, results_pool, question, disagreement):
                local_results = list(results_pool)
                briefing = ""
                extra = ""
                if disagreement:
                    extra = (
                        f"\n\nIMPORTANT — Previous round had a DISAGREEMENT:\n"
                        f"{disagreement}\n"
                        f"Focus on RESOLVING this. Find evidence for/against each position.\n"
                    )
                for loop_i in range(3):
                    from core.tool_call import call_with_tools as _cwt
                    _r = await _cwt(
                        model,
                        BRIEFING_PROMPT.format(
                            question=question + extra,
                            results=_format_search_results(local_results)[:25000],
                        ),
                        enable_code_search=False, enable_web_search=True,
                        max_tokens=16384,
                    )
                    raw = _r["answer"]
                    briefing, searches = _extract_search_requests(raw)
                    if not searches or loop_i >= 2:
                        break
                    for sq in searches[:2]:
                        try:
                            new_results = await web_search(sq, max_results=3)
                            local_results.extend(new_results)
                        except Exception:
                            pass
                return briefing, local_results

            research_tasks = [
                _independent_research(m, all_results, query, disagreement_context)
                for m in briefing_models
            ]
            researcher_outputs = list(await asyncio.gather(*research_tasks))
            briefings = [out[0] for out in researcher_outputs]

            for _, local_results in researcher_outputs:
                for r in local_results:
                    if r not in all_results:
                        all_results.append(r)

            status(f"3 briefings: {[count_tokens(b) for b in briefings]} tokens")

            # ── Step 4: Deliberation — cross-verify ──────────────────────
            step("Step 4: Deliberation — cross-verification")

            async def _deliberate(model, prompt):
                from core.tool_call import call_with_tools as _cwt
                r = await _cwt(model, prompt, enable_code_search=False,
                               enable_web_search=True, max_tokens=16384)
                return r["answer"]

            revised_briefings = list(await asyncio.gather(
                _deliberate(
                    briefing_models[0],
                    DELIBERATION_PROMPT.format(
                        question=query, my_briefing=briefings[0][:8000],
                        briefing_b=briefings[1][:8000], briefing_c=briefings[2][:8000],
                    ),
                ),
                _deliberate(
                    briefing_models[1],
                    DELIBERATION_PROMPT.format(
                        question=query, my_briefing=briefings[1][:8000],
                        briefing_b=briefings[0][:8000], briefing_c=briefings[2][:8000],
                    ),
                ),
                _deliberate(
                    briefing_models[2],
                    DELIBERATION_PROMPT.format(
                        question=query, my_briefing=briefings[2][:8000],
                        briefing_b=briefings[0][:8000], briefing_c=briefings[1][:8000],
                    ),
                ),
            ))

            # Check if briefings converge after deliberation
            step("Checking briefing convergence")
            briefing_check = await verify_agreement(
                query,
                [{"model": m, "answer": b} for m, b in zip(briefing_models, revised_briefings)]
            )

            if briefing_check in ("agree", "complementary"):
                status(f"Briefings converged ({briefing_check})")
                break  # Exit Loop A
            else:
                if briefing_loop >= max_briefing_loops - 1:
                    warn("Briefings still diverge — continuing with best effort")
                    break
                warn("Briefings conflict — re-researching")
                disagreement_context = (
                    f"Briefings disagree:\n"
                    f"A: {revised_briefings[0][:500]}\n"
                    f"B: {revised_briefings[1][:500]}\n"
                    f"C: {revised_briefings[2][:500]}"
                )
                # Search for more info and loop back to step 3
                for sq in [query + " evidence", query + " data"]:
                    try:
                        new_results = await web_search(sq, max_results=3)
                        all_results.extend(new_results)
                    except Exception:
                        pass

    # ── Step 5: Merge into final report ──────────────────────────────
    step("Step 5: Merge research")
    briefing = await call_with_retry(
        "nvidia/deepseek-v4-flash",
        MERGE_RESEARCH_PROMPT.format(
            question=query,
            briefing_a=revised_briefings[0][:10000],
            briefing_b=revised_briefings[1][:10000],
            briefing_c=revised_briefings[2][:10000],
        ),
        max_tokens=16384,
    )
    status(f"Merged report: {count_tokens(briefing):,} tokens")

    # ── LOOP B: Steps 6-7 (conclusion convergence, max 2) ───────────
    max_conclusion_loops = 2
    conclusion_feedback = ""

    for conc_loop in range(max_conclusion_loops):
        if conc_loop > 0:
            step(f"── Conclusion retry (round {conc_loop + 1}) ──")

        # ── Step 6: Draw conclusions (3 AIs parallel) ────────────────
        step("Step 6: Draw conclusions")
        assumption = get_assumption_prompt(domain, complexity)

        extra_instruction = ""
        if conclusion_feedback:
            extra_instruction = (
                f"\n\nPREVIOUS ROUND DISAGREEMENT:\n{conclusion_feedback}\n"
                f"The briefing is solid. The issue is in how conclusions were drawn.\n"
                f"Read the briefing carefully and resolve the disagreement.\n"
            )

        conclude_prompt = CONCLUDE_PROMPT.format(
            question=query,
            assumption_prompt=assumption + extra_instruction,
            briefing=briefing[:30000],
        )

        conclusions = list(await asyncio.gather(
            _conclude_one("groq/llama-4-scout", conclude_prompt),
            _conclude_one("nvidia/deepseek-v4-pro", conclude_prompt),
            _conclude_one("nvidia/deepseek-v4-flash", conclude_prompt),
        ))
        status(f"Got {len(conclusions)} conclusions")

        # ── Step 7: Compare conclusions ──────────────────────────────
        step("Step 7: Compare conclusions")
        verdict = await verify_agreement(query, conclusions)

        if verdict == "agree":
            best = conclusions[-1]["answer"]
            vote_split = f"{len(conclusions)}/{len(conclusions)}"
            break
        elif verdict == "complementary":
            result = await merge_answers(query, conclusions)
            best = result["answer"]
            vote_split = result["vote_split"]
            break
        else:
            if conc_loop >= max_conclusion_loops - 1:
                warn("Conclusions still conflict — accepting majority vote")
                result = await synthesize(query, conclusions)
                best = result["answer"]
                vote_split = result["vote_split"]
                break

            warn("Conclusions conflict — retrying step 6 with feedback")
            conclusion_feedback = (
                f"A ({conclusions[0]['model']}): {conclusions[0]['answer'][:500]}\n"
                f"B ({conclusions[1]['model']}): {conclusions[1]['answer'][:500]}\n"
                f"C ({conclusions[2]['model']}): {conclusions[2]['answer'][:500]}"
            )

    # ── Step 8: Verify sources + freshness ───────────────────────────────
    step("Step 8: Verify sources")
    sources_text = "\n".join(f"- {r['title']}: {r['url']}" for r in all_results[:20])

    try:
        verification = await call_with_retry(
            "groq/llama-4-scout",
            VERIFY_PROMPT.format(
                question=query,
                answer=best[:4000],
                sources=sources_text,
            ),
            max_tokens=512,
        )

        if verification.strip().upper().startswith("VERIFIED"):
            success("Sources verified")
        elif "OUTDATED" in verification.upper():
            warn("Some sources may be outdated")
            # Could trigger one more search but keep it simple
        elif "UNSUPPORTED" in verification.upper():
            warn("Some claims may lack source support")
    except Exception as e:
        warn(f"Verification failed: {e}")

    # ── Step 9: Format + confidence ──────────────────────────────────────
    step("Step 9: Format + confidence")
    confidence.record(vote_split, len(conclusions))

    state["final_answer"] = best
    state["search_results"] = briefing[:4000]  # For self-eval to verify against
    state["confidence"] = confidence.get_statement(vote_split, len(conclusions))
    success(f"Research complete — {len(all_results)} sources, {len(conclusions)} conclusions")
    return state


# ─── Helpers ─────────────────────────────────────────────────────────────────

async def _conclude_one(model: str, prompt: str) -> dict:
    """Draw one conclusion, return {model, answer}."""
    result = await call_with_retry(model, prompt, max_tokens=16384)
    return {"model": model, "answer": result}


def _format_search_results(results: list[dict]) -> str:
    """Format search results for prompts.

    Uses the `══` block style that the rest of the codebase uses for
    injected content (cf. tool_call.py / synthesizer.py answer blocks)
    so models recognize "this is a system-injected block" at a glance.
    """
    parts = []
    for i, r in enumerate(results):
        title = r.get("title", "Untitled")
        url = r.get("url", "N/A")
        body = r.get("content", "")[:500]
        parts.append(
            f"══════════════════════════════════════════════════════════════════════\n"
            f"SOURCE {i+1} of {len(results)} — {title}\n"
            f"URL: {url}\n"
            f"══════════════════════════════════════════════════════════════════════\n"
            f"{body}\n"
        )
    return "\n".join(parts)


def _parse_queries(plan: str) -> list[str]:
    """Extract search queries from a plan."""
    queries = []
    for line in plan.split("\n"):
        line = line.strip()
        if "Query:" in line or "query:" in line:
            if '"' in line:
                parts = line.split('"')
                if len(parts) >= 2:
                    queries.append(parts[1])
            elif ":" in line:
                q = line.split("Query:")[-1].split("Goal:")[0].strip().strip('"\'')
                if q:
                    queries.append(q)
    return queries


def _parse_inline_queries(text: str) -> list[str]:
    """Extract search queries from a NO response."""
    queries = []
    for line in text.split("\n"):
        line = line.strip()
        if '"' in line:
            parts = line.split('"')
            for i in range(1, len(parts), 2):
                if len(parts[i]) > 3:
                    queries.append(parts[i])
    return queries[:3]


def _extract_search_requests(raw_briefing: str) -> tuple[str, list[str]]:
    """
    Extract SEARCH_MORE requests from a briefing.
    Returns (clean_briefing, list_of_search_queries).
    """
    lines = raw_briefing.split("\n")
    clean_lines = []
    searches = []
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("SEARCH_MORE:"):
            q = stripped.split(":", 1)[-1].strip().strip('"\'')
            if q and len(q) > 3:
                searches.append(q)
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines).strip(), searches
