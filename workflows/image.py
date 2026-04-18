"""
Image workflow — two-step generation:
  1. NVIDIA MiniMax expands the user's short prompt into a rich, detailed description
     using conversation context to resolve references ("that", "it", etc.)
  2. Stable Diffusion 3 Medium (on NVIDIA NIM) generates the image
  3. Image is saved to ~/jarvis_images/ and path returned
"""

import re
from core.state import AgentState
from core.cli import step, status, success, warn, error
from core.retry import call_with_retry
from clients.imagen import generate_image


# ─── Context Extraction (same pattern as chat.py) ───────────────────────────

def _get_context(state: AgentState) -> str:
    """Get AI context from state (compressed or raw)."""
    history = state.get("conversation_history", [])
    if not history:
        return ""
    return "\n".join(m.get("content", "")[:500] for m in history[-6:])


def _get_last_exchange(state: AgentState) -> str:
    """Get the last user+assistant exchange. Critical for resolving 'it', 'that', etc."""
    history = state.get("conversation_history", [])
    if not history:
        return ""
    content = history[-1].get("content", "") if history else ""
    lines = content.split("\n") if content else []
    last_msgs = []
    for line in reversed(lines):
        if "] USER:" in line or "] ASSISTANT:" in line:
            last_msgs.insert(0, line)
            if len(last_msgs) >= 4:
                break
    return "\n".join(last_msgs) if last_msgs else ""


# ─── Prompt Expansion ───────────────────────────────────────────────────────

EXPAND_SYSTEM = """You are an expert image prompt engineer for Stable Diffusion 3 Medium.
Your job: take a short image request and expand it into a highly detailed, COHERENT
prompt that will produce a stunning image.

YOU MUST THINK THROUGH EVERY DETAIL SYSTEMATICALLY:

1. SUBJECT — What exactly is in the image? Be hyper-specific.
   - If it's a person: age, gender, ethnicity, clothing, pose, expression, body language
   - If it's an animal: species, breed, size, posture, fur/feather details
   - If it's an object: exact type, material, condition (new/worn/rusty), size relative to scene
   - If it's a scene: what elements are present, how are they arranged spatially

2. COMPOSITION — How is the image framed?
   - Camera angle: eye level, bird's eye, worm's eye, Dutch angle, over-the-shoulder
   - Framing: close-up, medium shot, wide shot, extreme close-up, panoramic
   - Rule of thirds, centered, asymmetric, leading lines
   - Foreground, midground, background — what's in each layer
   - Depth of field: shallow (blurry background) or deep (everything sharp)

3. LIGHTING — This makes or breaks the image.
   - Direction: front-lit, backlit, side-lit, rim light, under-lit
   - Quality: soft/diffused, hard/dramatic, golden hour, overcast, studio
   - Color temperature: warm (amber/orange), cool (blue), neutral
   - Shadows: harsh, soft, long, minimal
   - Special: volumetric light, god rays, neon glow, bioluminescence

4. COLOR PALETTE — Be deliberate.
   - Dominant color, secondary color, accent color
   - Warm vs cool palette, monochromatic, complementary, analogous
   - Saturation level: vibrant, muted, pastel, desaturated

5. STYLE & MEDIUM — What does it look like?
   - If not specified, default to photorealistic
   - Photography: DSLR, film grain, lens type (35mm, 85mm, fisheye), bokeh
   - Digital art: concept art, matte painting, 3D render, cel-shaded
   - Traditional: oil painting, watercolor, pencil sketch, charcoal, gouache
   - Specific aesthetics: cyberpunk, art nouveau, minimalist, brutalist, vaporwave

6. MOOD & ATMOSPHERE — What should the viewer FEEL?
   - Emotion: serene, dramatic, mysterious, joyful, melancholic, tense
   - Atmosphere: foggy, crisp, hazy, stormy, ethereal, gritty
   - Time of day: dawn, midday, golden hour, dusk, night, blue hour

7. TEXTURES & DETAILS — What makes it feel real?
   - Surface textures: rough stone, smooth glass, weathered wood, wet metal
   - Environmental: dust particles, water droplets, lens flare, motion blur
   - Fine details: fabric wrinkles, skin pores, leaf veins, rust patterns

8. COHERENCE CHECK — Before outputting, verify:
   - Does the lighting match the time of day?
   - Does the color palette match the mood?
   - Are the materials physically plausible together?
   - Does the scale of objects make sense relative to each other?
   - Is the style consistent throughout (not mixing photorealism with cartoon)?
   - Would a real photographer/artist compose the scene this way?

RULES:
- Output ONLY the expanded prompt. No preamble, no explanation, no markdown.
- Keep under 250 words — SD3 works best with focused, non-repetitive detail.
- Do NOT use negative prompts or technical parameters.
- Do NOT list categories (don't write "Lighting:", "Style:") — write it as one
  flowing, natural description.
- If the user says "it", "that", "this", use the CONVERSATION CONTEXT to
  understand what they refer to and describe it concretely.

MODIFICATION REQUESTS:
- If the conversation context shows a previous image was generated (look for
  "IMAGE GENERATED" or "Expanded prompt:" in the context), and the user is asking
  to modify it ("make it more realistic", "change the colors", "different style",
  "make it make more sense", "try again but...", "more detailed", "zoom out"),
  then:
  1. Find the previous expanded prompt from the context
  2. Use it as your BASE — keep everything that was good
  3. Modify ONLY the aspects the user asked to change
  4. Write the complete new prompt (not just the changes)"""

EXPAND_PROMPT_WITH_CONTEXT = """CONVERSATION CONTEXT (use this to understand references and find previous image prompts):
{last_exchange}

{context}

───────────────────────────────

User's request: {query}

If this is a MODIFICATION of a previous image (the context shows a previous image was
generated), use the previous expanded prompt as your base and apply the user's requested
changes. If this is a NEW image, create the prompt from scratch.

Write the expanded prompt as one flowing description:"""

EXPAND_PROMPT_NO_CONTEXT = """The user wants an image of: {query}

Think through subject, composition, lighting, colors, style, mood, textures, and
coherence. Then write the expanded prompt as one flowing description:"""


async def expand_prompt(user_query: str, context: str = "", last_exchange: str = "") -> str:
    """
    Use NVIDIA MiniMax to expand a short image request into a detailed prompt.
    Passes conversation context so it can resolve references.
    Falls back to nvidia/deepseek-v3.2 if MiniMax fails.
    """
    step("Prompt Expansion (MiniMax)")

    if context or last_exchange:
        prompt = EXPAND_PROMPT_WITH_CONTEXT.format(
            query=user_query,
            context=context,
            last_exchange=last_exchange or "(no previous exchange)",
        )
    else:
        prompt = EXPAND_PROMPT_NO_CONTEXT.format(query=user_query)

    try:
        expanded = await call_with_retry(
            "nvidia/minimax-m2.5",
            prompt=prompt,
            system=EXPAND_SYSTEM,
            temperature=0.7,
            max_tokens=1024,
            log_label="prompt expansion",
        )
    except Exception as e:
        warn(f"MiniMax failed ({e}) — falling back to DeepSeek v3.2")
        expanded = await call_with_retry(
            "nvidia/deepseek-v3.2",
            prompt=prompt,
            system=EXPAND_SYSTEM,
            temperature=0.7,
            max_tokens=1024,
            log_label="prompt expansion fallback",
        )

    # Clean up: strip any markdown fences or quotes the model might add
    expanded = expanded.strip()
    if expanded.startswith('"') and expanded.endswith('"'):
        expanded = expanded[1:-1]
    if expanded.startswith("```"):
        lines = expanded.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        expanded = "\n".join(lines).strip()

    return expanded


# ─── Aspect Ratio Detection ─────────────────────────────────────────────────

def detect_aspect_ratio(query: str) -> str:
    """
    Infer aspect ratio from the user's request.
    Default: 1:1 (square).
    """
    q = query.lower()

    # Landscape hints
    if any(kw in q for kw in ["landscape", "wide", "panorama", "panoramic", "banner", "wallpaper", "desktop", "16:9", "16x9"]):
        return "16:9"

    # Portrait hints
    if any(kw in q for kw in ["portrait", "tall", "vertical", "phone", "mobile", "story", "stories", "9:16", "9x16"]):
        return "9:16"

    # 4:3 / 3:4
    if any(kw in q for kw in ["4:3", "4x3"]):
        return "4:3"
    if any(kw in q for kw in ["3:4", "3x4"]):
        return "3:4"

    return "1:1"


# ─── Image Agent ─────────────────────────────────────────────────────────────

async def image_agent(state: AgentState) -> AgentState:
    """
    Full image generation pipeline:
      1. Extract conversation context from state
      2. Expand prompt via MiniMax (with context)
      3. Detect aspect ratio
      4. Generate image via SD3 Medium on NIM
      5. Return result with file path
    """
    step("Image Agent")

    query = state.get("processed_input", state.get("raw_input", ""))

    # Strip !!image prefix if present
    if query.strip().lower().startswith("!!image"):
        query = query.strip()[len("!!image"):].strip()

    if not query:
        state["final_answer"] = "Please describe the image you'd like me to generate."
        return state

    # Extract conversation context
    context = _get_context(state)
    last_exchange = _get_last_exchange(state)

    # Step 1: Expand prompt (with context so MiniMax can resolve references)
    try:
        expanded = await expand_prompt(query, context=context, last_exchange=last_exchange)
        status(f"Expanded prompt ({len(expanded)} chars)")
    except Exception as e:
        error(f"Prompt expansion failed: {e}")
        state["final_answer"] = f"Failed to expand prompt: {e}"
        return state

    # Step 2: Detect aspect ratio
    aspect = detect_aspect_ratio(query)
    if aspect != "1:1":
        status(f"Aspect ratio: {aspect}")

    # Step 3: Generate image
    step("Image Generation (SD3 Medium — NIM)")
    try:
        result = await generate_image(
            prompt=expanded,
            aspect_ratio=aspect,
        )
        filepath = result["path"]
        model_used = result.get("model", "unknown")
        success(f"Image saved: {filepath}")
    except Exception as e:
        error(f"Image generation failed: {e}")
        state["final_answer"] = (
            f"Image generation failed: {e}\n\n"
            f"**Expanded prompt was:**\n{expanded}"
        )
        return state

    # Step 4: Just the image path — UI will render it
    state["final_answer"] = filepath

    # Store context for memory so follow-ups work
    state["image_context"] = {
        "query": query,
        "expanded_prompt": expanded,
        "filepath": filepath,
        "aspect": aspect,
    }

    # Skip self-eval for image generation (nothing to evaluate textually)
    state["bypass_self_eval"] = True
    state["bypass_formatter"] = True

    return state
