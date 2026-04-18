"""
Unified search — Gemini Grounded Search primary, Tavily/DDG fallback.

Gemini 2.5 Flash + Google Search grounding: 1500 RPD per key (free).
It searches Google AND synthesizes — way better than raw search results.
"""

import os
from core.cli import status, warn


async def web_search(query: str, max_results: int = 5, depth: str = "basic") -> list[dict]:
    """
    Search the web. Returns list of {title, url, content, score}.
    Primary: Gemini grounded search (converts answer into results format).
    Fallback: Tavily → DDG.
    """
    # Try Gemini grounded search first
    try:
        from clients.gemini import grounded_search
        result = await grounded_search(query)
        
        # Convert grounded answer + sources into results format
        results = []
        
        # The answer itself is a high-quality result
        results.append({
            "title": f"Gemini Search: {query[:60]}",
            "url": "",
            "content": result["answer"][:2000],
            "score": 1.0,
        })
        
        # Add individual sources
        for src in result["sources"][:max_results - 1]:
            results.append({
                "title": src.get("title", ""),
                "url": src.get("url", ""),
                "content": "",  # Gemini already synthesized the content
                "score": 0.8,
            })
        
        status(f"Got {len(results)} results (grounded)")
        return results
        
    except Exception as e:
        warn(f"Gemini grounded search failed: {e} — trying fallback")

    # Fallback: Tavily
    if os.environ.get("TAVILY_API_KEY"):
        try:
            from clients.tavily_client import tavily_search
            return await tavily_search(query, max_results=max_results, search_depth=depth)
        except Exception as e:
            warn(f"Tavily failed: {e} — trying DDG")

    # Fallback: DuckDuckGo
    try:
        from clients.duckduckgo_client import ddg_search
        return await ddg_search(query, max_results=max_results)
    except Exception as e:
        raise RuntimeError(f"All search methods failed: {e}")


async def grounded_ask(question: str, system: str = "") -> dict:
    """
    Ask a question with Google Search grounding. Returns full answer + sources.
    Used by intelligent chat search tool — one call replaces search + compile.
    """
    from clients.gemini import grounded_search
    return await grounded_search(question, system=system)


async def web_extract(urls: list[str]) -> list[dict]:
    """Extract content from URLs. Tavily if available."""
    if os.environ.get("TAVILY_API_KEY"):
        try:
            from clients.tavily_client import tavily_extract
            return await tavily_extract(urls)
        except Exception as e:
            warn(f"Tavily extract failed: {e}")
    return [{"url": url, "content": ""} for url in urls]
