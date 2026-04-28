"""
Context check — counts tokens, flags if compression needed.
"""

from core.tokens import count_tokens
from core.state import AgentState
from config import COMPRESS_THRESHOLD


def check_context(state: AgentState) -> AgentState:
    """
    Count tokens in full context (prompt + conversation history).
    Set context_tokens on state. Does NOT compress — that's compressor.py.
    """
    parts = []

    # Living summary (if we have one and it replaced raw history)
    if state.get("living_summary"):
        parts.append(state["living_summary"])

    # Full conversation history
    for msg in state.get("conversation_history", []):
        parts.append(msg.get("content", ""))

    # Current input
    parts.append(state.get("processed_input", state.get("raw_input", "")))

    full_text = "\n".join(parts)
    state["context_tokens"] = count_tokens(full_text)
    return state


def needs_compression(state: AgentState) -> bool:
    """Check if context exceeds 72K threshold."""
    return state.get("context_tokens", 0) > COMPRESS_THRESHOLD
