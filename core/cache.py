"""
TTL cache — time-based expiry for tool results.
"""

import time
from typing import Any, Optional


class AgentCache:
    def __init__(self):
        self.store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str, max_age: float = 300.0) -> Optional[Any]:
        if key in self.store:
            val, ts = self.store[key]
            if time.time() - ts < max_age:
                return val
            del self.store[key]
        return None

    def set(self, key: str, val: Any):
        self.store[key] = (val, time.time())

    def invalidate(self, prefix: str):
        for k in [k for k in self.store if k.startswith(prefix)]:
            del self.store[k]

    def clear(self):
        self.store.clear()


# Default TTLs (seconds)
TTL_FILE_READ = 300       # 5 min
TTL_CODEBASE_SEARCH = 300 # 5 min
TTL_PROJECT_MAP = 1800    # 30 min
TTL_WEB_SEARCH = 900      # 15 min

# Global instance
cache = AgentCache()
