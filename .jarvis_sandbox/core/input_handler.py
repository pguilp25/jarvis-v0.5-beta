"""
Input handler — routes image → Qwen 3.5 VL, audio → Whisper, text → passthrough.
"""

from core.state import AgentState
from core.cli import status


async def process_input(state: AgentState) -> AgentState:
    """Process multi-modal input into text."""

    text = state["raw_input"]

    # Image → text via Qwen 3.5 VL (NVIDIA)
    if state.get("has_image") and state.get("image_data"):
        status("Reading image with Qwen 3.5...")
        # TODO Phase 5: implement vision call
        # For now, skip image processing
        text += "\n[Image processing not yet implemented]"

    # Audio → text via Whisper (Groq)
    if state.get("has_audio") and state.get("audio_data"):
        status("Transcribing speech...")
        # TODO Phase 5: implement Whisper call
        text += "\n[Audio processing not yet implemented]"

    state["processed_input"] = text
    return state
