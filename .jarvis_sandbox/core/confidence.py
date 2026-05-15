"""
Confidence calibration — tracks model agreement vs correctness over time.
Appended to every output as confidence statement.
"""

import json
import os
from pathlib import Path
from collections import defaultdict


class ConfidenceTracker:
    def __init__(self, project_path: str = "."):
        self.path = Path(project_path) / ".jarvis" / "confidence.json"
        self.records: list[dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.records = json.load(open(self.path))
            except (json.JSONDecodeError, TypeError):
                self.records = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)

    def record(self, vote_split: str, total_models: int, correct: bool | None = None):
        """
        Record a task outcome.
        vote_split: e.g. "4/5", "3/3", "2/2"
        correct: True/False/None (None = not yet evaluated)
        """
        self.records.append({
            "vote_split": vote_split,
            "total_models": total_models,
            "correct": correct,
        })
        self._save()

    def mark_last(self, correct: bool):
        """Mark the most recent record as correct or incorrect (user feedback)."""
        if self.records:
            self.records[-1]["correct"] = correct
            self._save()

    def get_statement(self, vote_split: str = "", total_models: int = 0) -> str:
        """
        Generate a confidence statement based on historical accuracy.
        e.g. "[High confidence (4/5 agreement, 92% accuracy over 18 tasks)]"
        """
        if not self.records:
            if vote_split:
                return f"[{vote_split} agreement, no historical data yet]"
            return ""

        # Calculate accuracy for tasks with known outcomes
        evaluated = [r for r in self.records if r["correct"] is not None]
        if not evaluated:
            total = len(self.records)
            if vote_split:
                return f"[{vote_split} agreement, {total} tasks tracked]"
            return f"[{total} tasks tracked, no feedback yet]"

        correct = sum(1 for r in evaluated if r["correct"])
        accuracy = correct / len(evaluated) * 100

        if vote_split:
            return (
                f"[{vote_split} agreement, "
                f"{accuracy:.0f}% accuracy over {len(evaluated)} evaluated tasks]"
            )

        return f"[{accuracy:.0f}% accuracy over {len(evaluated)} evaluated tasks]"


# Global instance (will be re-initialized with project path in main)
confidence = ConfidenceTracker()
