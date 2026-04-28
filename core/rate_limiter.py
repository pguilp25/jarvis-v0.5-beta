"""
NVIDIA rate limiter — 40 RPM shared across all 5 frontier models.
"""

import asyncio
import time
from collections import deque
from config import NVIDIA_MAX_RPM
from core.cli import rate_limit_wait


class NvidiaRateLimiter:
    def __init__(self, max_rpm: int = NVIDIA_MAX_RPM):
        self.max_rpm = max_rpm
        self.timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until we can make an NVIDIA call without exceeding RPM."""
        async with self._lock:
            now = time.time()
            # Remove timestamps older than 60s
            while self.timestamps and self.timestamps[0] < now - 60:
                self.timestamps.popleft()

            if len(self.timestamps) >= self.max_rpm:
                wait = 60 - (now - self.timestamps[0]) + 0.1
                rate_limit_wait(wait)
                await asyncio.sleep(wait)

            self.timestamps.append(time.time())

    def remaining(self) -> int:
        now = time.time()
        recent = sum(1 for t in self.timestamps if t > now - 60)
        return self.max_rpm - recent


# Global instance
nvidia_limiter = NvidiaRateLimiter()
