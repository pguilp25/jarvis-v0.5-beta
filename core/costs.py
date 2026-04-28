"""
Cost tracking — logs every paid API call, auto-throttles when budget runs low.
"""

import time
import json
import os
from typing import Optional
from config import MONTHLY_BUDGET


class CostTracker:
    def __init__(self, budget: float = MONTHLY_BUDGET):
        self.budget = budget
        self.calls: list[dict] = []
        self.tavily_credits = 0
        self._month_file = os.path.expanduser("~/.jarvis_costs.json")
        self._load()

    # ─── Persistence ─────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self._month_file):
            try:
                data = json.load(open(self._month_file))
                self.calls = data.get("calls", [])
                self.tavily_credits = data.get("tavily_credits", 0)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        with open(self._month_file, "w") as f:
            json.dump({
                "calls": self.calls,
                "tavily_credits": self.tavily_credits,
            }, f)

    # ─── Logging ─────────────────────────────────────────────────────────

    def log_call(self, model: str, input_tokens: int, output_tokens: int, cost: float):
        self.calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "time": time.time(),
        })
        self._save()

    def log_tavily(self, credits: int = 1):
        self.tavily_credits += credits
        self._save()

    # ─── Budget Tier ─────────────────────────────────────────────────────

    @property
    def total_spent(self) -> float:
        return sum(c["cost"] for c in self.calls)

    @property
    def spend_pct(self) -> float:
        return self.total_spent / self.budget if self.budget > 0 else 0.0

    def get_tier(self) -> str:
        pct = self.spend_pct
        if pct < 0.70:
            return "normal"
        if pct < 0.85:
            return "cautious"       # avoid Gemini 3.1 Pro
        if pct < 0.95:
            return "restricted"     # free models only
        return "emergency"          # warn user

    def can_use_paid(self) -> bool:
        return self.get_tier() in ("normal", "cautious")

    # ─── Cost Estimation ─────────────────────────────────────────────────

    @staticmethod
    def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a model call. Free models return 0."""
        # Only Gemini models cost money in our setup
        costs = {
            "gemini/2.5-pro":   (0.00125, 0.010),   # per 1K tokens
            "gemini/3.1-pro":   (0.002,   0.012),
        }
        if model not in costs:
            return 0.0
        in_rate, out_rate = costs[model]
        return (input_tokens / 1000 * in_rate) + (output_tokens / 1000 * out_rate)

    # ─── Summary ─────────────────────────────────────────────────────────

    def summary(self) -> str:
        tier = self.get_tier()
        return (
            f"Budget: ${self.total_spent:.2f} / ${self.budget:.2f} "
            f"({self.spend_pct:.0%}) — tier: {tier} | "
            f"Tavily: {self.tavily_credits}/1000 credits"
        )

    def reset_month(self):
        self.calls.clear()
        self.tavily_credits = 0
        self._save()


# Global instance
cost_tracker = CostTracker()
