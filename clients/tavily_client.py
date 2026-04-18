"""
Tavily search client — primary web search.
1K free credits/month. Each search = 1 credit. Extract = 1 credit.
"""

import os
from core.cli import status
from core.costs import cost_tracker


def _get_key() -> str:
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        raise RuntimeError("TAVILY_API_KEY not set")
    return key


async def tavily_search(query: str, max_results: int = 5, search_depth: str = "basic") -> list[dict]:
    """
    Search the web via Tavily. Returns list of {title, url, content, score}.
    search_depth: "basic" (1 credit) or "advanced" (2 credits).
    """
    from tavily import TavilyClient

    if cost_tracker.tavily_credits >= 950:
        raise RuntimeError("Tavily credits nearly exhausted (950+/1000)")

    status(f"Tavily search: {query}")
    client = TavilyClient(api_key=_get_key())

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )
    except Exception as e:
        raise RuntimeError(f"Tavily search failed: {e}")

    credits = 2 if search_depth == "advanced" else 1
    cost_tracker.log_tavily(credits)

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0),
        })

    status(f"Got {len(results)} results")
    return results


async def tavily_extract(urls: list[str]) -> list[dict]:
    """
    Extract full content from URLs via Tavily. 1 credit per URL.
    Returns list of {url, content}.
    """
    from tavily import TavilyClient

    if cost_tracker.tavily_credits >= 950:
        raise RuntimeError("Tavily credits nearly exhausted")

    status(f"Tavily extract: {len(urls)} URLs")
    client = TavilyClient(api_key=_get_key())

    try:
        response = client.extract(urls=urls[:5])  # Max 5 at once
    except Exception as e:
        raise RuntimeError(f"Tavily extract failed: {e}")

    cost_tracker.log_tavily(len(urls))

    results = []
    for r in response.get("results", []):
        results.append({
            "url": r.get("url", ""),
            "content": r.get("raw_content", r.get("content", "")),
        })

    return results
