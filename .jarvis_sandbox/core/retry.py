"""
Retry wrapper — timeout, retry with backoff, automatic fallback.
v5: connectivity check before API calls.

Timeout policy:
  - Timeout errors: infinite retries with exponential backoff (10s, 20s, 40s … capped at 5min)
  - Other errors: up to max_retries, then fallback model
  - PERMANENT errors (HTTP 410 Gone, 404 Not Found, 401/403 auth): skip
    retries entirely, go straight to fallback. Retrying a deprecated
    model just wastes 6+ seconds of backoff per call — observed in
    practice when MiniMax 2.5 was sunset on NVIDIA NIM.
"""

import re
import asyncio
from clients.api import call_api, call_api_stream
from config import NVIDIA_FALLBACKS, GROQ_FALLBACKS, MODELS
from core.cli import status, warn, error

# Errors that CANNOT recover with a retry. Detected from the exception
# message. We match on the literal HTTP status prefix the API clients
# include, e.g. "HTTP 410:".
_PERMANENT_STATUS = re.compile(r'HTTP\s*(?:410|404|401|403)\b', re.IGNORECASE)


def _is_permanent_error(exc: BaseException) -> bool:
    """True if `exc` represents a permanent failure that retries can't fix.

    HTTP 410 (Gone) = model was deprecated/removed.
    HTTP 404 = model name not recognised by the provider.
    HTTP 401/403 = auth wrong — won't get better with a sleep.
    """
    return bool(_PERMANENT_STATUS.search(str(exc)))

try:
    from tools.connectivity import is_online, wait_for_connection
    _HAS_CONNECTIVITY = True
except ImportError:
    _HAS_CONNECTIVITY = False

# Timeout backoff: starts at 10s, doubles each retry, capped at 300s (5 min)
_TIMEOUT_BACKOFF_START = 10
_TIMEOUT_BACKOFF_CAP   = 300


def _default_timeout(model_id: str) -> float:
    """No practical time limit — let models finish thinking."""
    return 3600.0  # 1 hour — effectively unlimited


def _timeout_wait(timeout_attempt: int) -> float:
    """Exponential backoff for timeout retries: 10s → 20s → 40s … capped at 5min."""
    return min(_TIMEOUT_BACKOFF_START * (2 ** timeout_attempt), _TIMEOUT_BACKOFF_CAP)


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

    Timeout errors: infinite retries with increasing wait (10s → 20s → 40s … 5min max).
    Other errors:   up to max_retries, then tries the fallback model once.
    """
    if timeout <= 0:
        timeout = _default_timeout(model_id)

    # v5: pause if WiFi dropped
    if _HAS_CONNECTIVITY and not is_online():
        ok = await wait_for_connection(f"API call to {model_id}")
        if not ok:
            raise ConnectionError(f"Internet lost >10min during call to {model_id}")

    last_error = None
    error_attempt = 0   # counts non-timeout failures (has a limit)
    timeout_attempt = 0  # counts timeout failures (no limit)

    while True:
        try:
            result = await asyncio.wait_for(
                call_api_stream(model_id, prompt, system, temperature, max_tokens,
                                json_mode, log_label, stop_check=stop_check),
                timeout=timeout,
            )
            return result

        except asyncio.TimeoutError:
            wait = _timeout_wait(timeout_attempt)
            last_error = f"Timeout after {timeout}s"
            warn(f"  ⚠️  {model_id}: {last_error} — waiting {wait:.0f}s then retrying (no limit)...")
            timeout_attempt += 1
            await asyncio.sleep(wait)
            continue  # infinite retry on timeout

        except Exception as e:
            last_error = str(e)[:120]
            # Don't retry on 400 (bad request) — prompt/params are wrong
            if "HTTP 400" in str(e):
                warn(f"  {model_id}: Bad request — {last_error}")
                break
            # Don't retry on permanent failures (model deprecated, 410/404/
            # 401/403). Retrying a sunset model just burns backoff seconds.
            # Skip straight to fallback.
            if _is_permanent_error(e):
                warn(
                    f"  {model_id}: permanent error ({last_error}) — "
                    f"skipping retries, going to fallback"
                )
                break
            error_attempt += 1
            if error_attempt >= max_retries:
                break
            wait = 2 * error_attempt
            warn(f"  ⚠️  {model_id}: {last_error}. Retry {error_attempt}/{max_retries} in {wait}s...")
            await asyncio.sleep(wait)

    # Non-timeout retries exhausted — try fallback
    fallback = NVIDIA_FALLBACKS.get(model_id) or GROQ_FALLBACKS.get(model_id)
    if fallback:
        error(f"{model_id} unreachable. Falling back to {fallback}...")
        fb_timeout = _default_timeout(fallback)
        try:
            return await asyncio.wait_for(
                call_api_stream(fallback, prompt, system, temperature, max_tokens,
                                json_mode, log_label, stop_check=stop_check),
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
    Timeout errors: infinite retries with increasing wait.
    Other errors:   up to max_retries, then fallback model.
    """
    if timeout <= 0:
        timeout = _default_timeout(model_id)

    if _HAS_CONNECTIVITY and not is_online():
        ok = await wait_for_connection(f"stream call to {model_id}")
        if not ok:
            raise ConnectionError(f"Internet lost >10min during stream call to {model_id}")

    last_error = None
    error_attempt = 0
    timeout_attempt = 0

    while True:
        try:
            result = await asyncio.wait_for(
                call_api_stream(model_id, prompt, system, temperature, max_tokens,
                                json_mode, log_label, stop_check=stop_check),
                timeout=timeout,
            )
            return result

        except asyncio.TimeoutError:
            wait = _timeout_wait(timeout_attempt)
            last_error = f"Timeout after {timeout}s"
            warn(f"  ⚠️  {model_id}: {last_error} — waiting {wait:.0f}s then retrying (no limit)...")
            timeout_attempt += 1
            await asyncio.sleep(wait)
            continue

        except Exception as e:
            last_error = str(e)[:120]
            if "HTTP 400" in str(e):
                warn(f"  {model_id}: Bad request — {last_error}")
                break
            if _is_permanent_error(e):
                warn(
                    f"  {model_id}: permanent error ({last_error}) — "
                    f"skipping retries, going to fallback"
                )
                break
            error_attempt += 1
            if error_attempt >= max_retries:
                break
            wait = 2 * error_attempt
            warn(f"  ⚠️  {model_id}: {last_error}. Retry {error_attempt}/{max_retries} in {wait}s...")
            await asyncio.sleep(wait)

    fallback = NVIDIA_FALLBACKS.get(model_id) or GROQ_FALLBACKS.get(model_id)
    if fallback:
        error(f"{model_id} unreachable. Falling back to {fallback}...")
        fb_timeout = _default_timeout(fallback)
        try:
            return await asyncio.wait_for(
                call_api_stream(fallback, prompt, system, temperature, max_tokens,
                                json_mode, log_label, stop_check=stop_check),
                timeout=fb_timeout,
            )
        except Exception as e2:
            raise RuntimeError(
                f"All retries failed for {model_id} ({last_error}) "
                f"AND fallback {fallback} failed ({e2})"
            )

    raise RuntimeError(f"All retries failed for {model_id}: {last_error}")
