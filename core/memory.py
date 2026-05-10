"""
Conversation memory — two layers with timestamps.

Every message gets a timestamp + message number.
AI context is formatted as a timeline so the model knows:
  - What was said when
  - What was the LAST exchange
  - What "earlier" and "just now" refer to
"""

from datetime import datetime
from core.tokens import count_tokens


class ConversationMemory:
    def __init__(self):
        self.full_history: list[dict] = []      # Raw with timestamps
        self.compressed_context: str = ""         # 5-10K for AI input
        self.compression_running: bool = False
        self._msg_counter: int = 0

    def add(self, role: str, content: str, notes: str = "", thinking_trace: str = ""):
        """Add a raw message with timestamp, number, and optional context notes."""
        self._msg_counter += 1
        entry = {
            "role": role,
            "content": content,
            "time": datetime.now().strftime("%H:%M"),
            "n": self._msg_counter,
        }
        if notes:
            entry["notes"] = notes
        if thinking_trace:
            entry["thinking_trace"] = thinking_trace
        self.full_history.append(entry)

    def get_ai_context(self) -> str:
        """
        Get timestamped context for AIs.
        If compressed exists, append last few raw messages so AI always
        knows the most recent exchange (even if compression is stale).
        """
        parts = []

        if self.compressed_context:
            parts.append("[CONVERSATION SUMMARY]\n" + self.compressed_context)
            # Always append last 6 raw messages so AI sees the latest
            recent = self.full_history[-6:]
            if recent:
                parts.append("\n[RECENT MESSAGES]")
                parts.append(self._format_timeline(recent))
        else:
            # No compression — raw recent with timeline
            recent = self.full_history[-20:]
            if recent:
                parts.append("[CONVERSATION TIMELINE]")
                parts.append(self._format_timeline(recent))

        return "\n".join(parts)

    def get_recent_raw(self, n: int = 6) -> list[dict]:
        """Get last n raw messages."""
        return self.full_history[-n:]

    def _format_timeline(self, messages: list[dict]) -> str:
        """Format messages as a numbered timeline with timestamps and notes."""
        lines = []
        for m in messages:
            n = m.get("n", "?")
            t = m.get("time", "??:??")
            role = m["role"].upper()
            content = m["content"]
            lines.append(f"[#{n} {t}] {role}: {content}")
            if m.get("notes"):
                lines.append(f"  [NOTES: {m['notes']}]")
        return "\n".join(lines)

    def full_token_count(self) -> int:
        return sum(count_tokens(m.get("content", "")) for m in self.full_history)

    def compressed_token_count(self) -> int:
        return count_tokens(self.compressed_context) if self.compressed_context else 0

    def clear(self):
        self.full_history.clear()
        self.compressed_context = ""
        self.compression_running = False
        self._msg_counter = 0

    def to_dict(self) -> dict:
        return {
            "full_history": self.full_history,
            "compressed_context": self.compressed_context,
            "msg_counter": self._msg_counter,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        mem = cls()
        mem.full_history = data.get("full_history", [])
        mem.compressed_context = data.get("compressed_context", "")
        mem._msg_counter = data.get("msg_counter", len(mem.full_history))
        return mem
