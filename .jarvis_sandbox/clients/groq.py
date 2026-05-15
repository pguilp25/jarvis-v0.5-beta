"""
Groq API client — async, supports all 6 free models.
Uses OpenAI-compatible /v1/chat/completions endpoint.
Supports SSE streaming to thought_logger.
"""

import json as _json
import os
import asyncio
import aiohttp
from typing import Optional
from config import GROQ_MODEL_IDS
from core.cli import thinking, warn
from core.stream_guard import DegenerationDetector

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _get_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    return key


async def call_groq(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """
    Call a Groq model. model_id is our config name like 'groq/kimi-k2'.
    Returns the response text.
    """
    thinking(model_id)

    api_model = GROQ_MODEL_IDS.get(model_id, model_id.split("/", 1)[-1])

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        # max_tokens removed — no output limit  # Groq caps at 8192
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {_get_key()}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(GROQ_API_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Groq {api_model} HTTP {resp.status}: {body[:200]}")
            data = await resp.json()

    return data["choices"][0]["message"]["content"]


async def call_groq_stream(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
    log_label: str = "",
    stop_check: object = None,
) -> str:
    """
    Call a Groq model with SSE streaming.
    Streams each response chunk to thought_logger as it arrives.
    If stop_check(accumulated_text) returns True, stops early.
    Returns the complete response text.
    """
    from core import thought_logger

    thinking(model_id)
    thought_logger.write_header(model_id, log_label)

    api_model = GROQ_MODEL_IDS.get(model_id, model_id.split("/", 1)[-1])

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

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {_get_key()}",
        "Content-Type": "application/json",
    }

    chunks: list[str] = []
    # stop_check must only see VISIBLE content. If the model writes "[STOP]"
    # inside its CoT, the early-stop must NOT fire.
    visible_chunks: list[str] = []
    # Same degeneration / prompt-leak guard as nvidia.py.
    degen_guard = DegenerationDetector()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            GROQ_API_URL, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=3600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Groq {api_model} HTTP {resp.status}: {body[:200]}")

            buf = b""
            done = False
            in_thinking_block = False
            # Per-chunk idle timeout (see nvidia.py for full rationale).
            # Groq is generally much faster than NVIDIA but a stuck stream
            # would still waste the parent's 1-hour aiohttp budget without
            # this guard. Matches the NVIDIA cap so behaviour is uniform.
            STREAM_IDLE_TIMEOUT = 600.0
            while True:
                try:
                    raw = await asyncio.wait_for(
                        resp.content.readany(), timeout=STREAM_IDLE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError(
                        f"Groq {api_model} stream idle "
                        f"{STREAM_IDLE_TIMEOUT:.0f}s — server stalled"
                    )
                if not raw:
                    break  # EOF
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
                        delta_obj = obj["choices"][0]["delta"]
                        reasoning = delta_obj.get("reasoning") or delta_obj.get("reasoning_content") or ""
                        if reasoning:
                            if not in_thinking_block:
                                chunks.append("<think>")
                                thought_logger.write_chunk(model_id, "<think>")
                                in_thinking_block = True
                            chunks.append(reasoning)
                            thought_logger.write_chunk(model_id, reasoning)
                        delta = delta_obj.get("content") or ""
                        if delta:
                            if in_thinking_block:
                                chunks.append("</think>\n\n")
                                thought_logger.write_chunk(model_id, "</think>\n\n")
                                in_thinking_block = False
                            chunks.append(delta)
                            visible_chunks.append(delta)
                            thought_logger.write_chunk(model_id, delta)
                            if stop_check and ("]" in delta or "\n" in delta):
                                if stop_check("".join(visible_chunks)):
                                    done = True
                                    break
                            # Degeneration / prompt-leak guard.
                            if "\n" in delta:
                                reason = degen_guard.check("".join(visible_chunks))
                                if reason:
                                    warn(
                                        f"  [{model_id.split('/')[-1]}] stream "
                                        f"aborted — {reason}"
                                    )
                                    done = True
                                    break
                    except (ValueError, KeyError, IndexError):
                        pass
                if done:
                    break
            if in_thinking_block:
                chunks.append("</think>\n\n")
                thought_logger.write_chunk(model_id, "</think>\n\n")

    return "".join(chunks)


async def call_groq_parallel(calls: list[dict]) -> list[str]:
    """
    Run multiple Groq calls in parallel (different models, no shared rate limit).
    Each call is a dict with keys: model_id, prompt, system (optional).
    Returns list of response texts in same order.
    """
    import asyncio

    async def _one(c):
        return await call_groq(
            model_id=c["model_id"],
            prompt=c["prompt"],
            system=c.get("system", ""),
            temperature=c.get("temperature", 0.3),
            max_tokens=c.get("max_tokens", 4096),
            json_mode=c.get("json_mode", False),
        )

    return await asyncio.gather(*[_one(c) for c in calls])
