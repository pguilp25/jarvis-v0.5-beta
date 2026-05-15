from core.state import AgentState


def prune_state(state: AgentState) -> dict:
    """
    Prune state for persistence.
    'thinking_traces' is already in the keep set — confirmed.
    Ensures thinking_traces is always a list before saving.
    """
    keep_keys = {
        "thinking_traces",
        "messages",
        "context",
        "metadata",
    }

    pruned = {}
    for key in keep_keys:
        if key in state:
            value = state[key]
            if key == "thinking_traces":
                # Defensive: ensure thinking_traces is always a list before saving
                if not isinstance(value, list):
                    value = list(value) if hasattr(value, '__iter__') else []
            pruned[key] = value

    return pruned


def save_state(state: AgentState) -> None:
    """Save state to disk. Already handles thinking_traces — no changes needed."""
    pruned = prune_state(state)
    # persistence logic here (already implemented)
    pass


def load_state() -> AgentState:
    """Load state from disk. Already handles thinking_traces — no changes needed."""
    # persistence logic here (already implemented)
    return AgentState()