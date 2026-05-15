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
LIGHTNING_API_URL = "https://lightning.ai/api/v1/chat/completions"
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# Models we deliberately route to DeepInfra. Pro is intentionally NOT here:
# DeepInfra serves Pro FP4-quantized at only 66k context (vs 200k+ on NVIDIA
# and 1M native), so we keep Pro on NVIDIA/Lightning. Flash on DeepInfra
# keeps the full 1M context, which is what we want for huge code repos.
DEEPINFRA_MODELS = {
    "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
}


def _route(model_id: str) -> tuple[str, str, str]:
    """Pick endpoint, auth key, and provider-specific model slug.

    Priority per model:
      1. DeepInfra — only for models in DEEPINFRA_MODELS (currently Flash,
         for its full 1M context). Skipped for Pro because DeepInfra
         FP4-quantizes Pro down to 66k.
      2. Lightning AI — if LIGHTNING_API_KEY is set, much faster than
         NVIDIA's free DGX Cloud which returns sporadic 504s under load.
      3. NVIDIA — integrate.api.nvidia.com (free, occasionally flaky).
    """
    base = model_id.split("/", 1)[-1]

    dkey = os.environ.get("DEEPINFRA_API_KEY", "")
    if dkey and base in DEEPINFRA_MODELS:
        return DEEPINFRA_API_URL, dkey, DEEPINFRA_MODELS[base]

    lkey = os.environ.get("LIGHTNING_API_KEY", "")
    if lkey:
        return LIGHTNING_API_URL, lkey, f"lightning-ai/{base}"

    nkey = os.environ.get("NVIDIA_API_KEY", "")
    if not nkey:
        raise RuntimeError("None of DEEPINFRA_API_KEY / LIGHTNING_API_KEY / NVIDIA_API_KEY is set")
    return NVIDIA_API_URL, nkey, NVIDIA_MODEL_IDS.get(model_id, base)


def _get_key() -> str:
    # Kept for callers (clients/imagen.py) that still need the NVIDIA key directly.
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    return key


def _max_thinking_payload(model_id: str) -> dict:
    """Per-model parameters that force the strongest available reasoning mode.

    Defaults vary by provider/family:
      • DeepSeek V4 Pro/Flash → `reasoning_effort: "high"` by default; "xhigh"
        is the documented map for the "max" budget. We also set the explicit
        `thinking: {type: enabled}` so hosts that key off it (rather than
        reasoning_effort) still surface reasoning_content.
      • Kimi K2.6 → thinking is ON by default; we still send the explicit
        enable so a host that flipped the default doesn't silently disable it.
      • GLM-5.1 → thinking is ON by default; same belt-and-suspenders
        approach. We send the canonical `thinking` plus the vLLM-style
        `chat_template_kwargs` so it works against either parser.

    All values are OpenAI-compatible JSON fields. A host that does not
    recognize a field generally ignores it; if a provider returns HTTP 400
    on one of these, narrow this map for that model.
    """
    base = model_id.split("/", 1)[-1].lower()
    if base.startswith("deepseek-v4"):
        return {
            "reasoning_effort": "xhigh",
            "thinking": {"type": "enabled"},
        }
    if base.startswith("kimi-"):
        return {
            "thinking": {"type": "enabled"},
        }
    if base.startswith("glm-"):
        return {
            "thinking": {"type": "enabled"},
            "chat_template_kwargs": {"enable_thinking": True},
        }
    return {}


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

    url, key, api_model = _route(model_id)

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
        **_max_thinking_payload(model_id),
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
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

    url, key, api_model = _route(model_id)

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
        **_max_thinking_payload(model_id),
    }

    headers = {
        "Authorization": f"Bearer {key}",
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
            url, json=payload, headers=headers,
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
