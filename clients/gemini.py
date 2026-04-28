"""
Gemini API client — multi-key round-robin.

Set multiple keys from different Google projects to multiply rate limits:
  GEMINI_API_KEYS=key1,key2,key3
  (or single key: GEMINI_API_KEY=key1)

Gemini 3.1 Flash Lite per project: 15 RPM, 250K TPM, 500 RPD
With 3 projects: 45 RPM, 750K TPM, 1500 RPD — plenty for all utility calls.
"""

import os
import aiohttp
from core.cli import thinking
from core.costs import cost_tracker
from core.tokens import count_tokens

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# ─── Multi-key round-robin ───────────────────────────────────────────────────

_keys: list[str] = []
_key_index: int = 0


def _load_keys():
    global _keys
    if _keys:
        return
    # Try comma-separated list first
    multi = os.environ.get("GEMINI_API_KEYS", "")
    if multi:
        _keys = [k.strip() for k in multi.split(",") if k.strip()]
    # Fallback to single key
    if not _keys:
        single = os.environ.get("GEMINI_API_KEY", "")
        if single:
            _keys = [single]
    if not _keys:
        raise RuntimeError("GEMINI_API_KEY or GEMINI_API_KEYS not set")


def _next_key() -> str:
    """Round-robin through available keys."""
    global _key_index
    _load_keys()
    key = _keys[_key_index % len(_keys)]
    _key_index += 1
    return key


# ─── API Call ────────────────────────────────────────────────────────────────

# Model ID mapping
GEMINI_MODEL_MAP = {
    "gemini/flash-lite":     "gemini-3.1-flash-lite-preview",
    "gemini/3.1-flash-lite": "gemini-3.1-flash-lite-preview",
    "gemini/3-flash":        "gemini-3-flash-preview",
    "gemini/flash":          "gemini-2.5-flash",
    "gemini/3.1-flash":      "gemini-2.5-flash",
    "gemini/2.5-pro":        "gemini-2.5-pro",
    "gemini/3.1-pro":        "gemini-3.1-pro-preview",
}


async def call_gemini(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 16384,
) -> str:
    """
    Call a Gemini model with round-robin key selection.
    """
    thinking(model_id)

    api_model = GEMINI_MODEL_MAP.get(model_id, model_id.split("/", 1)[-1])
    key = _next_key()
    url = f"{GEMINI_API_BASE}/{api_model}:generateContent?key={key}"

    body: dict = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "thinkingConfig": {
                "thinkingBudget": -1,
            },
        },
    }

    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Gemini {api_model} HTTP {resp.status}: {text[:200]}")
            data = await resp.json()

    try:
        candidate = data["candidates"][0]

        # Handle empty content (MAX_TOKENS with thinking eating all budget)
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        if not parts:
            finish = candidate.get("finishReason", "UNKNOWN")
            raise RuntimeError(f"Gemini {api_model} returned empty content (finishReason: {finish})")

        # Extract text, skip thinking parts
        result = ""
        for part in parts:
            if isinstance(part, dict):
                if part.get("thought"):
                    continue
                if "text" in part:
                    result += part["text"]
            elif isinstance(part, str):
                result += part
        if not result:
            raise RuntimeError(f"Gemini {api_model} no text in response parts")
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini {api_model} unexpected response: {str(data)[:200]}")

    # Log cost for paid models only
    input_tokens = count_tokens(prompt + system)
    output_tokens = count_tokens(result)
    cost = cost_tracker.estimate_cost(model_id, input_tokens, output_tokens)
    if cost > 0:
        cost_tracker.log_call(model_id, input_tokens, output_tokens, cost)

    return result


# ─── Convenience: Flash models ───────────────────────────────────────────────

async def call_flash_lite(prompt: str, system: str = "", max_tokens: int = 16384) -> str:
    """Gemini 3.1 Flash Lite (free preview) — kept for backward compat."""
    return await call_gemini("gemini/flash-lite", prompt, system, max_tokens=max_tokens)


async def call_flash(prompt: str, system: str = "", max_tokens: int = 16384) -> str:
    """
    Gemini Flash (paid) — falls back to groq/llama-4-scout on error.
    """
    try:
        return await call_gemini("gemini/flash", prompt, system, max_tokens=max_tokens)
    except Exception as e:
        from core.cli import warn
        warn(f"Gemini Flash failed ({str(e)[:120]}) — falling back to groq/llama-4-scout")
        from clients.groq import call_groq_stream
        return await call_groq_stream(
            "groq/llama-4-scout", prompt, system,
            max_tokens=min(max_tokens, 8192),
            log_label="Flash fallback",
        )


# ─── Grounded Search: Gemini 2.5 Flash + Google Search ──────────────────────

async def grounded_search(
    query: str,
    system: str = "",
    max_tokens: int = 16384,
) -> dict:
    """
    Ask Gemini 2.5 Flash with Google Search grounding.
    Returns {"answer": str, "sources": list[dict], "queries": list[str]}
    
    This replaces Tavily — Gemini searches Google, reads results, and synthesizes.
    1500 RPD per API key (free). With 3 keys = 4500 RPD.
    """
    from core.cli import thinking
    thinking("gemini/2.5-flash (grounded)")

    key = _next_key()
    url = f"{GEMINI_API_BASE}/gemini-2.5-flash:generateContent?key={key}"

    body = {
        "contents": [{"parts": [{"text": query}]}],
        "tools": [{"google_search": {}}],
    }

    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Gemini grounded search HTTP {resp.status}: {text[:200]}")
            data = await resp.json()

    # Extract text
    try:
        parts = data["candidates"][0]["content"]["parts"]
        answer = ""
        for part in parts:
            if isinstance(part, dict):
                if part.get("thought"):
                    continue
                if "text" in part:
                    answer += part["text"]
        if not answer:
            answer = "(no text in response)"
    except (KeyError, IndexError):
        raise RuntimeError(f"Grounded search failed: {str(data)[:200]}")

    # Extract grounding metadata (sources)
    sources = []
    search_queries = []
    try:
        metadata = data["candidates"][0].get("groundingMetadata", {})
        search_queries = metadata.get("webSearchQueries", [])
        for chunk in metadata.get("groundingChunks", []):
            web = chunk.get("web", {})
            if web:
                sources.append({
                    "title": web.get("title", ""),
                    "url": web.get("uri", ""),
                })
    except (KeyError, IndexError):
        pass

    return {
        "answer": answer,
        "sources": sources,
        "queries": search_queries,
    }
