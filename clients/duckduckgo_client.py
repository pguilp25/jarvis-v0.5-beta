"""
DuckDuckGo search — free unlimited fallback.
Uses the `ddgs` package (renamed from `duckduckgo_search`).
Install: pip install ddgs
"""

from core.cli import status


async def ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search via DuckDuckGo. Free, no API key needed.
    Returns list of {title, url, content}.
    """
    status(f"DDG search: {query}")

    try:
        # Try new package name first
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        ddgs = DDGS()
        raw = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        raise RuntimeError(f"DDG search failed: {e}")

    results = []
    for r in raw:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("href", r.get("link", "")),
            "content": r.get("body", r.get("snippet", "")),
            "score": 0,
        })

    status(f"Got {len(results)} results")
    return results
