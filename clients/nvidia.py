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
from core.cli import thinking, warn
from core.rate_limiter import nvidia_limiter
from core.stream_guard import DegenerationDetector

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
    Call an NVIDIA model. model_id is our config name like 'nvidia/deepseek-v4-pro'.
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
        # Output-token floor — see call_nvidia_stream for rationale.
        "max_tokens": max(int(max_tokens), 4096),
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

    # ── Pre-flight context-budget check ──────────────────────────────
    # Rough estimate: ~4 chars per token for English/code. If our prompt
    # is already over the model's typical input cap, fail loudly with a
    # clear message instead of letting the server return the cryptic
    # "requested 0 output tokens" HTTP 400 (which happens when the server
    # computes max_output = context_limit - input and gets 0 or negative).
    # The threshold is a soft hint — we'd rather warn early than truncate
    # silently and lose the model's work.
    _approx_input_chars = sum(len(m.get("content", "")) for m in messages)
    _approx_input_tokens = _approx_input_chars // 4
    # Most NVIDIA models we use have 200k-256k context. We reserve 8k
    # for output and warn at 90% of a conservative 200k input cap.
    _SOFT_INPUT_CAP = 190_000  # tokens
    if _approx_input_tokens > _SOFT_INPUT_CAP:
        from core.cli import warn as _warn
        _warn(
            f"  [{model_id.split('/')[-1]}] prompt is ~{_approx_input_tokens:,} "
            f"tokens — over the {_SOFT_INPUT_CAP:,} soft cap. The model may "
            f"refuse with HTTP 400 'requested 0 output tokens'. Consider "
            f"narrowing [KEEP:] ranges or splitting the step."
        )

    payload = {
        "model": api_model,
        "messages": messages,
        "temperature": temperature,
        # Reserve a floor for output. Without this, when input nearly fills
        # the context the server computes max_output = 0 and returns the
        # opaque "requested 0 output tokens" error. With an explicit floor,
        # an overflowing request fails with a clear "context exceeded"
        # message we can surface and handle.
        "max_tokens": max(int(max_tokens), 4096),
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {_get_key()}",
        "Content-Type": "application/json",
    }

    chunks: list[str] = []
    # Visible-only buffer: stop_check must NEVER see reasoning_content. A
    # reasoning model that mentions "[STOP]" in its CoT would otherwise trigger
    # an early-stop while still thinking. Track visible content separately.
    visible_chunks: list[str] = []
    # Degeneration / prompt-leak guard. Aborts the stream as soon as the
    # model starts repeating or emits prompt-only scaffolding. Saves both
    # tokens and the next round's context (degenerate output here gets
    # echoed into YOUR WORK SO FAR otherwise).
    degen_guard = DegenerationDetector()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            NVIDIA_API_URL, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=3600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"NVIDIA {api_model} HTTP {resp.status}: {body[:200]}")

            buf = b""
            done = False
            in_thinking_block = False
            # Per-chunk idle timeout. NVIDIA NIM is genuinely slow — the
            # first token can take up to ~10 minutes on a complex prompt,
            # reasoning or not. The aiohttp `total` timeout is 1 hour so
            # it can't catch a dead connection; we need a between-chunk
            # cap that's still longer than NIM's worst legitimate wait.
            # 10 min: tighter than the 1-hour total, slack enough that a
            # slow first token (thinking or pre-token queue) doesn't trip
            # it. Earlier 90s/300s caps killed legitimate slow starts.
            STREAM_IDLE_TIMEOUT = 600.0
            while True:
                try:
                    raw = await asyncio.wait_for(
                        resp.content.readany(), timeout=STREAM_IDLE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    # Raise as a regular RuntimeError so retry.py treats it
                    # as a normal recoverable error (bounded retries +
                    # fallback). asyncio.TimeoutError would trigger the
                    # infinite-retry timeout path in retry.py — wrong for
                    # an idle stream that's likely a dead connection.
                    raise RuntimeError(
                        f"NVIDIA {api_model} stream idle "
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
                        # ── Reasoning content (hidden CoT) ──
                        # Some servers use `reasoning`, others `reasoning_content`.
                        reasoning = (
                            delta_obj.get("reasoning_content")
                            or delta_obj.get("reasoning")
                            or ""
                        )
                        if reasoning:
                            if not in_thinking_block:
                                opener = "<think>"
                                chunks.append(opener)
                                thought_logger.write_chunk(model_id, opener)
                                in_thinking_block = True
                            chunks.append(reasoning)
                            thought_logger.write_chunk(model_id, reasoning)
                        # ── Visible content ──
                        delta = delta_obj.get("content") or ""
                        if delta:
                            if in_thinking_block:
                                closer = "</think>\n\n"
                                chunks.append(closer)
                                thought_logger.write_chunk(model_id, closer)
                                in_thinking_block = False
                            chunks.append(delta)
                            visible_chunks.append(delta)
                            thought_logger.write_chunk(model_id, delta)
                            if stop_check and ("]" in delta or "\n" in delta):
                                if stop_check("".join(visible_chunks)):
                                    done = True
                                    break
                            # Degeneration / prompt-leak guard — check on
                            # newline-bearing deltas so we re-scan when a
                            # line completes. Cheap when not tripped.
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
