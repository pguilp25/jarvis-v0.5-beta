"""
Stub agents — placeholders for Phase 5 implementation.
Research agent is live (Phase 3).
Deep thinking / conjecture mode is live.
Coding agent is live (Phase 4).
"""

from core.state import AgentState
from core.cli import step, warn
from workflows.research import research_agent
from workflows.deep_thinking_v5 import chat_deep_thinking
from workflows.code import code_agent
from workflows.image import image_agent


async def fluidx3d_agent(state: AgentState) -> AgentState:
    step("FluidX3D agent [Phase 5 stub]")
    state["final_answer"] = "FluidX3D code agent not yet implemented. Coming in Phase 5."
    return state


async def arduino_agent(state: AgentState) -> AgentState:
    step("Arduino agent [Phase 5 stub]")
    state["final_answer"] = "Arduino code agent not yet implemented. Coming in Phase 5."
    return state


async def shell_agent(state: AgentState) -> AgentState:
    step("Shell agent [Phase 5 stub]")
    state["final_answer"] = "Shell agent not yet implemented. Coming in Phase 5."
    return state


AGENT_MAP = {
    "research":       research_agent,
    "conjecture":     chat_deep_thinking,
    "code":           code_agent,
    "code_fluidx3d":  fluidx3d_agent,
    "code_arduino":   arduino_agent,
    "image":          image_agent,
    "shell":          shell_agent,
}
