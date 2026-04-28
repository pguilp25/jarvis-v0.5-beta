"""
Research sub-agent — lightweight version used inside chat agents.
Frontier AI writes search plan, llama-4-scout executes and compiles.
Uses real Tavily/DDG search.
"""

from core.retry import call_with_retry
from core.cli import step, status
from tools.search import web_search


PLAN_PROMPT = """The AI answering this question needs current information from the web.

Question: {question}
What the AI needs: {need}

Write 2-3 search queries. Keep them short and specific (3-6 words each):
1. Query: "..." Goal: find ...
2. Query: "..." Goal: find ..."""


COMPILE_PROMPT = """Compile a brief factual briefing from these search results.
Only include relevant, factual information. Note sources.

Question: {question}

Search results:
{results}

Brief factual briefing:"""


async def research_sub(question: str, need: str) -> str:
    """
    Quick research for chat agents. Returns a factual briefing string.
    """
    step("Research sub-agent")

    # Plan
    plan = await call_with_retry(
        "groq/llama-4-scout",
        PLAN_PROMPT.format(question=question, need=need[:500]),
        max_tokens=4096,
    )

    # Parse and execute
    queries = _parse_queries(plan)
    if not queries:
        queries = [question[:500]]

    all_results = []
    for q in queries[:3]:
        try:
            results = await web_search(q, max_results=3)
            all_results.extend(results)
        except Exception as e:
            status(f"Search failed: {e}")

    if not all_results:
        return "No search results found."

    # Compile
    results_text = "\n".join(
        f"[{r.get('title', 'N/A')}] ({r.get('url', '')})\n{r.get('content', '')[:300]}"
        for r in all_results[:10]
    )

    briefing = await call_with_retry(
        "groq/llama-4-scout",
        COMPILE_PROMPT.format(question=question, results=results_text),
        max_tokens=8192,
    )

    return briefing


def _parse_queries(plan: str) -> list[str]:
    queries = []
    for line in plan.split("\n"):
        if '"' in line:
            parts = line.split('"')
            if len(parts) >= 2:
                queries.append(parts[1])
    return queries
