"""
Fast detector — catches simple queries before they enter the full pipeline.

Three layers:
  1. HARDCODED: acknowledgments, greetings → instant response, NO LLM call
  2. Step A: llama-4-scout sniff ("obviously simple?")
  3. Step B: gpt-oss-120b tries to answer
"""

import re
from core.retry import call_with_retry
from core.cli import step, status


# ─── Hardcoded patterns — NO LLM needed ─────────────────────────────────────

# Normalize: lowercase, strip punctuation
def _normalize(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.strip().lower())

# Acknowledgments — user is just saying thanks, not asking anything
ACKNOWLEDGMENTS = {
    "ok", "okay", "k", "thanks", "thank you", "thx", "ty", "tanks",
    "cool", "nice", "great", "awesome", "perfect", "got it", "understood",
    "alright", "sure", "yep", "yup", "yes", "no", "nope", "nah",
    "ok thanks", "ok thank you", "ok cool", "ok great", "ok perfect",
    "thanks a lot", "thank you so much", "merci", "ok merci",
    "good", "fine", "noted", "sounds good", "makes sense",
}

ACKNOWLEDGMENT_RESPONSES = [
    "You're welcome! Let me know if you need anything else.",
    "No problem! What would you like to work on next?",
    "Happy to help! Anything else?",
]

# Greetings
GREETINGS = {
    "hi", "hello", "hey", "yo", "sup", "whats up", "hows it going",
    "good morning", "good afternoon", "good evening", "bonjour", "salut",
}

GREETING_RESPONSES = [
    "Hey! What can I help you with?",
    "Hello! What are you working on?",
    "Hi there! What do you need?",
]

# Farewells
FAREWELLS = {
    "bye", "goodbye", "see you", "later", "cya", "au revoir",
    "good night", "gn", "bonne nuit",
}

FAREWELL_RESPONSES = [
    "See you later! Good luck with your project.",
    "Bye! Feel free to come back anytime.",
]


def _check_hardcoded(query: str) -> str | None:
    """Check if query matches a hardcoded pattern. Returns response or None."""
    normalized = _normalize(query)

    if normalized in ACKNOWLEDGMENTS:
        import random
        return random.choice(ACKNOWLEDGMENT_RESPONSES)

    if normalized in GREETINGS:
        import random
        return random.choice(GREETING_RESPONSES)

    if normalized in FAREWELLS:
        import random
        return random.choice(FAREWELL_RESPONSES)

    return None


# ─── LLM-based detection ────────────────────────────────────────────────────

STEP_A_PROMPT = """Is this query OBVIOUSLY SIMPLE? (trivial fact, yes/no, one-liner)
Do NOT consider greetings or acknowledgments — those are handled separately.

ALWAYS answer NO for:
- Image/photo generation requests ("generate", "create an image", "make a picture")
- Code writing or debugging requests
- Research or web search requests
- Math proofs or complex calculations
- Anything requiring a specialized agent

Query: {query}

Reply with ONLY one word: YES or NO"""


STEP_B_PROMPT = """Answer this query in 1-3 sentences. Be concise.
Use the conversation context if the query references something from it.
If it's a standalone question, answer it on its own.

CURRENT FACTS (your training may be outdated):
- Claude Opus 4.6 is Anthropic's latest, most advanced model (March 2026)
- Claude Sonnet 4.6 and Haiku 4.5 also exist
- If unsure about something recent, say NEEDS_SPECIAL: yes

YOU ARE JARVIS — you have these capabilities (say NEEDS_SPECIAL: yes for all of these):
- Image generation (you CAN generate images via Imagen 4 Ultra)
- Code writing and debugging
- Web research
- Math and formal proofs
Do NOT say "I can't" for things JARVIS can do. Route them to the right agent instead.

Query: {query}

After your answer, on a new line write:
CONFIDENCE: [1-10]
NEEDS_SPECIAL: [yes/no] (does this need image generation, math, code, domain expertise, or web search?)"""


async def fast_detect(query: str) -> dict:
    """
    Three-layer fast detection.
    Returns: {
        "is_fast": bool,
        "quick_answer": str or None,
        "confidence": int,
        "hardcoded": bool,
    }
    """
    step("Fast detect")

    # Layer 1: Hardcoded check — no LLM call at all
    hardcoded = _check_hardcoded(query)
    if hardcoded is not None:
        status(f"Hardcoded match → instant response")
        return {
            "is_fast": True,
            "quick_answer": hardcoded,
            "confidence": 10,
            "hardcoded": True,
        }

    # Layer 2: LLM sniff
    step_a = await call_with_retry(
        "groq/llama-4-scout",
        STEP_A_PROMPT.format(query=query),
        max_tokens=10,
    )

    obviously_simple = step_a.strip().upper().startswith("YES")

    if not obviously_simple:
        return {"is_fast": False, "quick_answer": None, "confidence": 0, "hardcoded": False}

    # Layer 3: Try to answer
    step_b = await call_with_retry(
        "groq/gpt-oss-120b",
        STEP_B_PROMPT.format(query=query),
        max_tokens=512,
    )

    # Parse
    lines = step_b.strip().split("\n")
    conf = 0
    needs_special = False
    answer_lines = []

    for line in lines:
        upper = line.strip().upper()
        if upper.startswith("CONFIDENCE:"):
            try:
                conf = int(line.split(":")[-1].strip().split("/")[0].split()[0])
            except (ValueError, IndexError):
                conf = 5
        elif upper.startswith("NEEDS_SPECIAL:"):
            needs_special = "YES" in upper
        else:
            answer_lines.append(line)

    quick_answer = "\n".join(answer_lines).strip()
    is_fast = conf >= 8 and not needs_special

    if is_fast:
        status(f"Fast bypass — confidence {conf}/10")
    else:
        status(f"Not simple enough — confidence {conf}, special={needs_special}")

    return {
        "is_fast": is_fast,
        "quick_answer": quick_answer if is_fast else None,
        "confidence": conf,
        "hardcoded": False,
    }
