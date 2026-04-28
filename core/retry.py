"""
Retry wrapper — timeout, retry with backoff, automatic fallback.
v5: connectivity check before API calls.
"""

import asyncio
from clients.api import call_api, call_api_stream
from config import NVIDIA_FALLBACKS, GROQ_FALLBACKS, MODELS
from core.cli import status, warn, error

try:
    from tools.connectivity import is_online, wait_for_connection
    _HAS_CONNECTIVITY = True
except ImportError:
    _HAS_CONNECTIVITY = False


def _default_timeout(model_id: str) -> float:
    """No practical time limit — let models finish thinking."""
    return 3600.0  # 1 hour — effectively unlimited


async def call_with_retry(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
    max_retries: int = 3,
    timeout: float = 0,  # 0 = auto-detect from provider
    log_label: str = "",
    stop_check: object = None,
) -> str:
    """
    Call a model with retries + exponential backoff + automatic fallback.
    Streams thinking to terminal via thought_logger.
    If stop_check(accumulated_text) returns True, stops the stream early.
    If all retries fail, tries the fallback model once.
    """
    if timeout <= 0:
        timeout = _default_timeout(model_id)

    # v5: pause if WiFi dropped
    if _HAS_CONNECTIVITY and not is_online():
        ok = await wait_for_connection(f"API call to {model_id}")
        if not ok:
            raise ConnectionError(f"Internet lost >10min during call to {model_id}")

    last_error = None

    for attempt in range(max_retries):
        try:
            result = await asyncio.wait_for(
                call_api_stream(model_id, prompt, system, temperature, max_tokens, json_mode, log_label,
                                stop_check=stop_check),
                timeout=timeout,
            )
            return result

        except asyncio.TimeoutError:
            last_error = f"Timeout after {timeout}s"
            warn(f"{model_id}: {last_error}. Retry {attempt+1}/{max_retries}...")
        except Exception as e:
            last_error = str(e)[:120]
            # Don't retry on 400 (bad request) — prompt/params are wrong, retrying won't help
            if "HTTP 400" in str(e):
                warn(f"{model_id}: Bad request — {last_error}")
                break  # Skip straight to fallback
            warn(f"{model_id}: {last_error}. Retry {attempt+1}/{max_retries}...")

        await asyncio.sleep(2 * (attempt + 1))

    # All retries exhausted — try fallback
    fallback = NVIDIA_FALLBACKS.get(model_id) or GROQ_FALLBACKS.get(model_id)
    if fallback:
        error(f"{model_id} unreachable. Falling back to {fallback}...")
        fb_timeout = _default_timeout(fallback)
        try:
            return await asyncio.wait_for(
                call_api_stream(fallback, prompt, system, temperature, max_tokens, json_mode, log_label,
                                stop_check=stop_check),
                timeout=fb_timeout,
            )
        except Exception as e2:
            raise RuntimeError(
                f"All retries failed for {model_id} ({last_error}) "
                f"AND fallback {fallback} failed ({e2})"
            )

    raise RuntimeError(f"All retries failed for {model_id}: {last_error}")


async def call_with_retry_stream(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
    max_retries: int = 3,
    timeout: float = 0,
    log_label: str = "",
    stop_check: object = None,
) -> str:
    """
    Stream a model call with retry + backoff + fallback.
    Writes chunks to thought_logger as they arrive.
    If stop_check(accumulated_text) returns True, stops the stream early.
    """
    if timeout <= 0:
        timeout = _default_timeout(model_id)

    if _HAS_CONNECTIVITY and not is_online():
        ok = await wait_for_connection(f"stream call to {model_id}")
        if not ok:
            raise ConnectionError(f"Internet lost >10min during stream call to {model_id}")

    last_error = None

    for attempt in range(max_retries):
        try:
            result = await asyncio.wait_for(
                call_api_stream(model_id, prompt, system, temperature, max_tokens, json_mode, log_label,
                                stop_check=stop_check),
                timeout=timeout,
            )
            return result

        except asyncio.TimeoutError:
            last_error = f"Timeout after {timeout}s"
            warn(f"{model_id}: {last_error}. Retry {attempt+1}/{max_retries}...")
        except Exception as e:
            last_error = str(e)[:120]
            if "HTTP 400" in str(e):
                warn(f"{model_id}: Bad request — {last_error}")
                break
            warn(f"{model_id}: {last_error}. Retry {attempt+1}/{max_retries}...")

        await asyncio.sleep(2 * (attempt + 1))

    fallback = NVIDIA_FALLBACKS.get(model_id) or GROQ_FALLBACKS.get(model_id)
    if fallback:
        error(f"{model_id} unreachable. Falling back to {fallback}...")
        fb_timeout = _default_timeout(fallback)
        try:
            return await asyncio.wait_for(
                call_api_stream(fallback, prompt, system, temperature, max_tokens, json_mode, log_label,
                                stop_check=stop_check),
                timeout=fb_timeout,
            )
        except Exception as e2:
            raise RuntimeError(
                f"All retries failed for {model_id} ({last_error}) "
                f"AND fallback {fallback} failed ({e2})"
            )

    raise RuntimeError(f"All retries failed for {model_id}: {last_error}")
