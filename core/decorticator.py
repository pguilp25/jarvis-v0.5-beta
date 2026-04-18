"""
Decorticator - classifies user query and picks the right agent.
Router is Python if/else, not an LLM call.
"""

import json
import asyncio
from core.retry import call_with_retry
from core.model_selector import select_for_context
from core.state import AgentState, Classification
from core.cli import step, agree, disagree, status
from config import NVIDIA_SLEEP_BETWEEN


DECORTICATOR_PROMPT = """You are the ROUTER for a multi-agent AI system called JARVIS.
Your job: understand what the user wants and pick the right agent to handle it.

Think through this step by step:

STEP 1 — WHAT HAPPENED BEFORE?
Read the LAST EXCHANGE and CONTEXT below. Understand the conversation flow.
What was the user doing? What did JARVIS produce? Is the user continuing that
activity or starting something new?

STEP 2 — WHAT DOES THE USER WANT NOW?
Interpret their message in context. If their message is vague or ambiguous,
the conversation history tells you what they mean. A user who just got an image
and says "make it better" wants a new image. A user who just got code changes
and says "also handle errors" wants more code changes. Think about their goal.

STEP 3 — WHICH AGENT SERVES THAT GOAL?
Pick the agent whose capabilities match what the user is trying to accomplish.
Don't match keywords — understand the intent.

Query: {query}

LAST EXCHANGE:
{last_exchange}

Full context:
{context}

═══════════════════════════════════════════════════════════════
AGENTS — what each one DOES (pick the one that fits the user's goal):
═══════════════════════════════════════════════════════════════

"chat" — Thinks and answers questions. Explains concepts, gives advice,
   analyzes problems, writes text, does math, compares options, brainstorms.
   This is the DEFAULT. If the user wants to UNDERSTAND something, this handles it.
   Does NOT write code to files, does NOT search the web for live data,
   does NOT generate images.

"research" — Searches the web for CURRENT information. Use when the user
   needs facts that change over time: news, prices, events, releases, scores,
   weather, who holds a position now. Has access to Google Search.
   Does NOT answer from general knowledge — that's chat.

"code" — Reads, plans, and modifies code in the user's project. Use when
   the user wants files changed: new features, bug fixes, refactors.
   Has access to the codebase via maps, search, and file reading.
   Edits are sandboxed and require user approval.
   Does NOT answer questions ABOUT code — that's chat.

"code_fluidx3d" — Same as code, specialized for FluidX3D/OpenFOAM.
"code_arduino" — Same as code, specialized for Arduino/ESP32.

"image" — Generates images from text descriptions. Use when the user wants
   a visual output: a picture, illustration, render, logo, artwork.
   Also use when the user wants to REDO or MODIFY a previously generated image.
   The image agent takes the user's request, expands it into a detailed prompt,
   and feeds it to Stable Diffusion 3. It can use conversation context to
   understand what "it" refers to and what to change.

"shell" — Runs terminal commands directly. Use when the user wants to
   execute a command on their system.

═══════════════════════════════════════════════════════════════
KEY ROUTING PRINCIPLE:
═══════════════════════════════════════════════════════════════
If the user is continuing a previous activity (follow-up to image generation,
follow-up to code changes, follow-up to research), route to the SAME agent
that handled it before — unless the user clearly changed topics.

A user who received an image and says anything about changing, improving,
or redoing it → image. Not chat.

A user who received code changes and asks for more changes → code. Not chat.

A user who got research results and wants more depth → research. Not chat.

═══════════════════════════════════════════════════════════════
DOMAINS: "general", "math", "code", "cfd", "arduino", "science", "web"
COMPLEXITY: 1-2 trivial, 3-4 simple, 5-6 moderate, 7-8 complex, 9-10 deep
═══════════════════════════════════════════════════════════════

Respond with ONLY valid JSON, no markdown:
{{
    "domain": "<domain>",
    "complexity": <1-10>,
    "agent": "<agent>",
    "intent": "<what the user wants in one sentence>"
}}"""


def _parse_classification(text: str) -> Classification | None:
    """Try to parse JSON from model output, tolerating markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip markdown code fences
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
        return Classification(
            expanded_prompt="",  # Not used — handlers use raw query
            domain=data.get("domain", "general"),
            complexity=int(data.get("complexity", 5)),
            agent=data.get("agent", "chat"),
            intent=data.get("intent", ""),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _classifications_agree(a: Classification, b: Classification) -> bool:
    """Two classifications agree if agent and complexity tier match."""
    # Same agent
    if a.get("agent") != b.get("agent"):
        return False
    # Complexity within same tier (1-2, 3-6, 7-10)
    ca, cb = a.get("complexity", 5), b.get("complexity", 5)
    tier_a = 0 if ca <= 2 else (1 if ca <= 6 else 2)
    tier_b = 0 if cb <= 2 else (1 if cb <= 6 else 2)
    return tier_a == tier_b


async def decorticate(state: AgentState) -> AgentState:
    """
    Run decorticator. Single Gemini Flash call classifies the query.
    """
    step("Decorticator")

    query = state.get("processed_input", state.get("raw_input", ""))
    context_tokens = state.get("context_tokens", 0)

    # Build context string from conversation history
    history = state.get("conversation_history", [])
    context_str = ""
    last_exchange_str = "(no previous exchange)"
    if history:
        recent = history[-6:]
        context_str = "\n".join(
            f"{m['role']}: {m['content'][:500]}" for m in recent
        )
        # Extract last user+assistant pair from the timeline content
        content = history[-1].get("content", "") if history else ""
        lines = content.split("\n") if content else []
        last_msgs = []
        for line in reversed(lines):
            if "] USER:" in line or "] ASSISTANT:" in line or "[NOTES:" in line:
                last_msgs.insert(0, line)
                if len(last_msgs) >= 6:
                    break
        if last_msgs:
            last_exchange_str = "\n".join(last_msgs)

    prompt = DECORTICATOR_PROMPT.format(query=query, context=context_str or "(none)", last_exchange=last_exchange_str)

    # Single Gemini Flash Lite call — 250K context, fast, replaces 2-3 Groq calls
    from clients.gemini import call_flash
    result = await call_flash(prompt, max_tokens=1024)
    classification = _parse_classification(result)

    if classification:
        state["classification"] = classification
    else:
        # Parse failed — default
        state["classification"] = Classification(
            expanded_prompt="",
            domain="general",
            complexity=5,
            agent="chat",
            intent="unknown",
        )

    return state


def route(classification: Classification) -> str:
    """
    Pure Python routing - no LLM call. Returns the handler name.
    """
    agent = classification.get("agent", "chat")
    complexity = classification.get("complexity", 5)

    if agent == "chat":
        if complexity <= 2:
            return "chat_fast"
        elif complexity <= 6:
            return "chat_intelligent"
        else:
            return "chat_very_intelligent"

    return agent  # research, code, code_fluidx3d, code_arduino, image, shell
