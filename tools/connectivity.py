"""
Connectivity Monitor — JARVIS v5

Pauses loop if internet drops, resumes when back. Designed for Chromebook WiFi.
"""

import asyncio
import socket
import time
from core.cli import status, warn, error, success

CHECK_HOSTS = [("8.8.8.8", 53), ("1.1.1.1", 53), ("208.67.222.222", 53)]
CHECK_TIMEOUT = 3.0
POLL_INTERVAL = 5.0
MAX_OFFLINE_WAIT = 600  # 10 min

_last_check_time = 0.0
_last_check_result = True
_CACHE_TTL = 10.0


def _check_once() -> bool:
    for host, port in CHECK_HOSTS:
        try:
            sock = socket.create_connection((host, port), timeout=CHECK_TIMEOUT)
            sock.close()
            return True
        except (OSError, socket.timeout):
            continue
    return False


def is_online() -> bool:
    global _last_check_time, _last_check_result
    now = time.monotonic()
    if now - _last_check_time < _CACHE_TTL:
        return _last_check_result
    _last_check_result = _check_once()
    _last_check_time = now
    return _last_check_result


async def wait_for_connection(context: str = "") -> bool:
    """Block until internet is back. Returns False if offline > MAX_OFFLINE_WAIT."""
    if is_online():
        return True

    ctx = f" ({context})" if context else ""
    warn(f"Internet connection lost{ctx}. Pausing...")
    warn(f"Checking every {POLL_INTERVAL}s for up to {MAX_OFFLINE_WAIT}s")

    t0 = time.monotonic()
    checks = 0
    while True:
        elapsed = time.monotonic() - t0
        if elapsed > MAX_OFFLINE_WAIT:
            error(f"Offline for {elapsed:.0f}s — exceeds limit")
            return False
        await asyncio.sleep(POLL_INTERVAL)
        checks += 1
        global _last_check_time
        _last_check_time = 0.0  # Force fresh check
        if is_online():
            success(f"Internet reconnected after {elapsed:.0f}s{ctx}. Resuming...")
            return True
        if checks % 6 == 0:
            status(f"Still offline... ({elapsed:.0f}s){ctx}")
