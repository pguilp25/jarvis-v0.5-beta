from core.state import AgentState


def run_agent_cycle(state: AgentState) -> AgentState:
    """
    Run one agent cycle.
    Defensively guards against missing/non-list thinking_traces.
    """
    # Defensive guard: ensure thinking_traces exists and is a list
    if "thinking_traces" not in state or not isinstance(state.get("thinking_traces"), list):
        state["thinking_traces"] = []

    # ... agent cycle logic ...
    save_state(state)  # line 21 per research
    return state


# Imported but used elsewhere
from storage_manager import save_state