"""
Unified API — routes call to correct provider client.
"""

from clients.groq import call_groq, call_groq_stream
from clients.nvidia import call_nvidia, call_nvidia_stream
from clients.gemini import call_gemini
from clients.openrouter import call_openrouter, call_openrouter_stream
from config import MODELS


async def call_api(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """
    Call any model by its config name (e.g. 'groq/kimi-k2', 'nvidia/deepseek-v3.2').
    Routes to the correct provider client automatically.
    """
    provider = MODELS[model_id]["provider"]

    if provider == "groq":
        return await call_groq(model_id, prompt, system, temperature, max_tokens, json_mode)
    elif provider == "nvidia":
        return await call_nvidia(model_id, prompt, system, temperature, max_tokens, json_mode)
    elif provider == "gemini":
        return await call_gemini(model_id, prompt, system, temperature, max_tokens)
    elif provider == "openrouter":
        return await call_openrouter(model_id, prompt, system, temperature, max_tokens, json_mode)
    else:
        raise ValueError(f"Unknown provider '{provider}' for model '{model_id}'")


async def call_api_stream(
    model_id: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
    log_label: str = "",
    stop_check: object = None,
) -> str:
    """
    Stream any model's response to the thought_logger.
    NVIDIA and Groq: true SSE streaming (chunks written as they arrive).
    Gemini: full response written to log after completion (no SSE endpoint).
    If stop_check(accumulated_text) returns True, stops the stream early.
    Returns the complete response text.
    """
    from core import thought_logger

    provider = MODELS[model_id]["provider"]

    if provider == "groq":
        return await call_groq_stream(
            model_id, prompt, system, temperature, max_tokens, json_mode, log_label,
            stop_check=stop_check,
        )
    elif provider == "nvidia":
        return await call_nvidia_stream(
            model_id, prompt, system, temperature, max_tokens, log_label,
            stop_check=stop_check,
        )
    elif provider == "gemini":
        # Gemini uses a non-SSE REST API — write the full response to the log
        result = await call_gemini(model_id, prompt, system, temperature, max_tokens)
        thought_logger.write_header(model_id, log_label)
        thought_logger.write_chunk(model_id, result)
        return result
    elif provider == "openrouter":
        return await call_openrouter_stream(
            model_id, prompt, system, temperature, max_tokens, log_label,
            stop_check=stop_check,
        )
    else:
        raise ValueError(f"Unknown provider '{provider}' for model '{model_id}'")
