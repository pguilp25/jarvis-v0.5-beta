"""
OpenRouter API client — OpenAI-compatible endpoint.
Used for free models like Nemotron Super.
"""

import json as _json
import os
import aiohttp
from core.cli import thinking
from core import thought_logger

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model ID mapping
OPENROUTER_MODELS = {
    "openrouter/qwen3.6-plus": "qwen/qwen3.6-plus-preview:free",
}


def _get_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return key


async def call_openrouter(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """Call an OpenRouter model. Returns response text."""
    thinking(model_id)

    api_model = OPENROUTER_MODELS.get(model_id, model_id.split("/", 1)[-1])

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {_get_key()}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=600)) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"OpenRouter {resp.status}: {body[:300]}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


async def call_openrouter_stream(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    log_label: str = "",
    stop_check=None,
) -> str:
    """Stream an OpenRouter model response via SSE."""
    thinking(model_id)
    thought_logger.write_header(model_id, log_label)

    api_model = OPENROUTER_MODELS.get(model_id, model_id.split("/", 1)[-1])

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {_get_key()}",
        "Content-Type": "application/json",
    }

    full = ""
    # stop_check must only see VISIBLE content (no <think> block).
    visible = ""
    in_thinking = False
    async with aiohttp.ClientSession() as session:
        async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=600)) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"OpenRouter {resp.status}: {body[:300]}")

            async for line in resp.content:
                line = line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = _json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if reasoning:
                        if not in_thinking:
                            full += "<think>"
                            thought_logger.write_chunk(model_id, "<think>")
                            in_thinking = True
                        full += reasoning
                        thought_logger.write_chunk(model_id, reasoning)
                    chunk = delta.get("content") or ""
                    if chunk:
                        if in_thinking:
                            full += "</think>\n\n"
                            thought_logger.write_chunk(model_id, "</think>\n\n")
                            in_thinking = False
                        full += chunk
                        visible += chunk
                        thought_logger.write_chunk(model_id, chunk)
                        if stop_check and ("]" in chunk or "\n" in chunk):
                            if stop_check(visible):
                                break
                except _json.JSONDecodeError:
                    continue
            if in_thinking:
                full += "</think>\n\n"
                thought_logger.write_chunk(model_id, "</think>\n\n")

    return full
