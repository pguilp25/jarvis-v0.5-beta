"""
NVIDIA NIM API client — 5 frontier models, rate-limited at 40 RPM shared.
Uses OpenAI-compatible /v1/chat/completions endpoint.
Supports both sequential and PARALLEL calls, and SSE streaming to thought_logger.
"""

import json as _json
import os
import asyncio
import aiohttp
from typing import Optional
from config import NVIDIA_MODEL_IDS, NVIDIA_SLEEP_BETWEEN
from core.cli import thinking
from core.rate_limiter import nvidia_limiter

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def _get_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    return key


async def call_nvidia(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """
    Call an NVIDIA model. model_id is our config name like 'nvidia/deepseek-v3.2'.
    Acquires rate limiter before calling. Returns response text.
    """
    await nvidia_limiter.acquire()
    thinking(model_id)

    api_model = NVIDIA_MODEL_IDS.get(model_id, model_id.split("/", 1)[-1])

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        # max_tokens removed — no output limit
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {_get_key()}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(NVIDIA_API_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"NVIDIA {api_model} HTTP {resp.status}: {body[:200]}")
            data = await resp.json()

    return data["choices"][0]["message"]["content"]


async def call_nvidia_stream(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    log_label: str = "",
    stop_check: object = None,
) -> str:
    """
    Call an NVIDIA model with SSE streaming.
    Streams each response chunk to thought_logger as it arrives.
    If stop_check(accumulated_text) returns True, stops early.
    Returns the complete response text.
    """
    from core import thought_logger

    await nvidia_limiter.acquire()
    thinking(model_id)
    thought_logger.write_header(model_id, log_label)

    api_model = NVIDIA_MODEL_IDS.get(model_id, model_id.split("/", 1)[-1])

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        # max_tokens removed — no output limit
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {_get_key()}",
        "Content-Type": "application/json",
    }

    chunks: list[str] = []
    async with aiohttp.ClientSession() as session:
        async with session.post(
            NVIDIA_API_URL, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=3600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"NVIDIA {api_model} HTTP {resp.status}: {body[:200]}")

            # Buffer raw bytes, split on newlines to produce complete SSE event lines.
            # Reading in arbitrary chunks from iter_any() can split a line mid-way,
            # which would break JSON parsing of the "data: {...}" payload.
            buf = b""
            done = False
            async for raw in resp.content.iter_any():
                buf += raw
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8").rstrip("\r")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        done = True
                        break
                    try:
                        obj = _json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            chunks.append(delta)
                            thought_logger.write_chunk(model_id, delta)
                            # Check stop on ] (tool tags) or \n (cycle detection)
                            if stop_check and ("]" in delta or "\n" in delta):
                                accumulated = "".join(chunks)
                                if stop_check(accumulated):
                                    done = True
                                    break
                    except (ValueError, KeyError, IndexError):
                        pass
                if done:
                    break

    return "".join(chunks)


async def call_nvidia_parallel(calls: list[dict]) -> list[str]:
    """
    Run multiple NVIDIA calls IN PARALLEL. All fire at once, rate limiter
    still enforces 40 RPM. At ~5 RPM real usage, this is totally safe.
    Returns list of response texts in same order as calls.
    """
    async def _one(c):
        return await call_nvidia(
            model_id=c["model_id"],
            prompt=c["prompt"],
            system=c.get("system", ""),
            temperature=c.get("temperature", 0.3),
            max_tokens=c.get("max_tokens", 4096),
            json_mode=c.get("json_mode", False),
        )

    return await asyncio.gather(*[_one(c) for c in calls])


async def call_nvidia_sequential(calls: list[dict], sleep: float = NVIDIA_SLEEP_BETWEEN) -> list[str]:
    """
    Run multiple NVIDIA calls sequentially (old method, kept as fallback).
    """
    results = []
    for i, c in enumerate(calls):
        result = await call_nvidia(
            model_id=c["model_id"],
            prompt=c["prompt"],
            system=c.get("system", ""),
            temperature=c.get("temperature", 0.3),
            max_tokens=c.get("max_tokens", 4096),
            json_mode=c.get("json_mode", False),
        )
        results.append(result)
        if i < len(calls) - 1:
            await asyncio.sleep(sleep)
    return results
