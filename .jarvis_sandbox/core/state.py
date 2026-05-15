"""
AgentState — shared state TypedDict passed through the pipeline.
"""

from __future__ import annotations
from typing import TypedDict, Optional, Any


class Classification(TypedDict, total=False):
    expanded_prompt: str
    domain: str          # general | math | code | cfd | arduino | science | web
    complexity: int      # 1-10
    agent: str           # chat | research | code | code_fluidx3d | code_arduino | image | shell
    intent: str


class AgentState(TypedDict, total=False):
    # Input
    raw_input: str
    processed_input: str       # after image/audio processing
    has_image: bool
    has_audio: bool
    image_data: Any
    audio_data: Any

    # Overrides
    forced_complexity: Optional[int]
    aborted: bool

    # Classification
    classification: Classification
    context_tokens: int

    # Pipeline tracking
    pipeline_calls: list       # log of every API call made
    current_step: str

    # Output
    final_answer: str
    confidence: str

    # Conversation
    conversation_history: list
    living_summary: str


def new_state(raw_input: str = "") -> AgentState:
    """Create a fresh state for a new user turn."""
    return AgentState(
        raw_input=raw_input,
        processed_input="",
        has_image=False,
        has_audio=False,
        image_data=None,
        audio_data=None,
        forced_complexity=None,
        aborted=False,
        classification={},
        context_tokens=0,
        pipeline_calls=[],
        current_step="input",
        final_answer="",
        confidence="",
        conversation_history=[],
        living_summary="",
    )
