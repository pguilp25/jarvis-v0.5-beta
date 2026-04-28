import anthropic
from config import get_api_key, DEFAULT_MODEL


def generate_response(messages: list, model: str = DEFAULT_MODEL) -> dict:
    client = anthropic.Anthropic(api_key=get_api_key())

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=messages,
    )

    thinking_blocks = []
    response_text = ""

    for block in response.content:
        if block.type == "thinking":
            thinking_blocks.append(block.thinking)
        elif block.type == "text":
            response_text += block.text

    return {"response_text": response_text, "thinking_blocks": thinking_blocks}