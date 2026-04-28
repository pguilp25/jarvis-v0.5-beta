"""
Image generation via Stable Diffusion 3 Medium on NVIDIA NIM.

Uses the same NVIDIA_API_KEY as the LLM client.
Endpoint: ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium
"""

import os
import base64
import aiohttp
from datetime import datetime
from pathlib import Path
from core.cli import thinking, warn

SD3_ENDPOINT = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium"

# Aspect ratio → supported output size mapping
ASPECT_RATIOS = {
    "1:1":  "1:1",
    "16:9": "16:9",
    "9:16": "9:16",
    "4:3":  "4:3",
    "3:4":  "3:4",
}


def _get_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    return key


async def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    negative_prompt: str = "",
    cfg_scale: float = 5.0,
    steps: int = 50,
    seed: int = 0,
    output_dir: str | None = None,
) -> dict:
    """
    Generate an image with Stable Diffusion 3 Medium via NVIDIA NIM.

    Args:
        prompt: Detailed image description prompt.
        aspect_ratio: One of "1:1", "3:4", "4:3", "9:16", "16:9".
        negative_prompt: What to avoid in the image.
        cfg_scale: Classifier-free guidance scale (1-10).
        steps: Number of diffusion steps (10-50).
        seed: Random seed (0 for random).
        output_dir: Directory to save the image. Defaults to ~/jarvis_images/.

    Returns:
        {"path": str, "prompt_used": str, "model": str} on success.
        Raises RuntimeError on failure.
    """
    thinking("sd3-medium (NIM)")

    key = _get_key()
    ar = ASPECT_RATIOS.get(aspect_ratio, "1:1")

    payload = {
        "prompt": prompt,
        "cfg_scale": cfg_scale,
        "aspect_ratio": ar,
        "seed": seed,
        "steps": steps,
        "negative_prompt": negative_prompt,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            SD3_ENDPOINT, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"SD3 Medium HTTP {resp.status}: {text[:300]}")
            data = await resp.json()

    # Extract image — NIM returns {"image": "<base64 JPEG>"} directly
    try:
        img_b64 = data.get("image", "")

        # Fallback: some versions use {"artifacts": [{"base64": "..."}]}
        if not img_b64:
            artifacts = data.get("artifacts", [])
            if artifacts:
                img_b64 = artifacts[0].get("base64", "")
                if not img_b64:
                    finish = artifacts[0].get("finishReason", "")
                    if finish == "CONTENT_FILTERED":
                        raise RuntimeError("Image was blocked by safety filter. Try a different prompt.")

        if not img_b64:
            raise RuntimeError(f"SD3 returned no image data: {str(data)[:200]}")

        img_bytes = base64.b64decode(img_b64)
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"SD3 unexpected response: {e}")

    # Save to file (SD3 outputs JPEG)
    if output_dir is None:
        output_dir = str(Path.home() / "jarvis_images")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt[:40]).strip().replace(" ", "_")
    if not safe_name:
        safe_name = "image"
    filename = f"{timestamp}_{safe_name}.jpg"
    filepath = str(Path(output_dir) / filename)

    with open(filepath, "wb") as f:
        f.write(img_bytes)

    return {
        "path": filepath,
        "prompt_used": prompt,
        "model": "stabilityai/stable-diffusion-3-medium (NIM)",
    }
