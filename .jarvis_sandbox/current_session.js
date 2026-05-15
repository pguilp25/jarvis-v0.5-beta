def add(self, role: str, content: str, notes: str = "", thinking_trace: str = ""):
        """Add a raw message with timestamp, number, and optional context notes.
        
        Args:
            role: "user" or "assistant"
            content: The message content
            notes: Optional metadata/context notes
            thinking_trace: Optional reasoning trace from AI (for assistant messages)
        """
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