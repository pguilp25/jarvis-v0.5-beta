"""
Knowledge System — domain expertise that AIs can consult.

Knowledge files live in the knowledge/ directory as markdown files.
Each file covers one topic with guidelines the AI should follow.

Two ways to access:
  1. AUTO-INJECT: decorticator detects relevant topics from the task
     and injects them into the AI's context automatically.
  2. ON-DEMAND: AI writes [KNOWLEDGE: topic] tag to consult a topic.
"""

import os
from pathlib import Path
from core.cli import status


# ─── Knowledge Base ───────────────────────────────────────────────────────

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# topic_id → {name, keywords, content}
_knowledge: dict[str, dict] = {}

# Keywords that trigger each knowledge topic
TOPIC_KEYWORDS = {
    "ui_design": {
        "name": "UI & Design Guidelines",
        "keywords": [
            "ui", "color", "css", "style", "design", "theme", "layout",
            "button", "font", "visual", "dark mode", "light mode",
            "background", "border", "shadow", "gradient", "palette",
            "brighter", "darker", "icon", "sidebar", "header", "footer",
            "responsive", "animation", "hover", "aesthetic", "pretty",
            "beautiful", "ugly", "look", "appearance",
        ],
    },
    "game_design": {
        "name": "Game Design Guidelines",
        "keywords": [
            "game", "score", "player", "level", "difficulty", "gameplay",
            "snake", "tetris", "pong", "platformer", "puzzle", "arcade",
            "spawn", "collision", "power-up", "enemy", "health", "lives",
            "game over", "high score", "leaderboard", "fun", "play",
        ],
    },
    "planning": {
        "name": "Project Planning Guidelines",
        "keywords": [
            "plan", "planning", "architect", "design doc", "spec",
            "requirements", "feature list", "scope", "roadmap",
            "build me", "create a", "make me a", "i want to build",
            "project idea", "startup", "app idea",
        ],
    },
}


def _load_knowledge():
    """Load all knowledge files from disk."""
    global _knowledge
    if _knowledge:
        return  # Already loaded

    if not KNOWLEDGE_DIR.exists():
        return

    for topic_id, meta in TOPIC_KEYWORDS.items():
        fpath = KNOWLEDGE_DIR / f"{topic_id}.md"
        if fpath.exists():
            try:
                content = fpath.read_text(encoding="utf-8")
                _knowledge[topic_id] = {
                    "name": meta["name"],
                    "content": content,
                }
            except Exception:
                pass


def detect_relevant_knowledge(task: str) -> list[str]:
    """
    Given a task description, return list of relevant topic IDs.
    Checks for keyword matches in the task text.
    """
    _load_knowledge()
    task_lower = task.lower()
    relevant = []

    for topic_id, meta in TOPIC_KEYWORDS.items():
        if topic_id not in _knowledge:
            continue
        # Check if any keyword appears in the task
        for kw in meta["keywords"]:
            if kw in task_lower:
                relevant.append(topic_id)
                break

    return relevant


def get_knowledge(topic: str) -> str:
    """
    Get knowledge content by topic name or ID.
    Fuzzy matches topic names.
    """
    _load_knowledge()
    topic_lower = topic.lower().strip()

    # Try exact ID match
    if topic_lower.replace(" ", "_") in _knowledge:
        k = _knowledge[topic_lower.replace(" ", "_")]
        return f"=== KNOWLEDGE: {k['name']} ===\n{k['content']}"

    # Try name match
    for topic_id, data in _knowledge.items():
        if topic_lower in data["name"].lower() or data["name"].lower() in topic_lower:
            return f"=== KNOWLEDGE: {data['name']} ===\n{data['content']}"

    # Try keyword match
    for topic_id, meta in TOPIC_KEYWORDS.items():
        if topic_id not in _knowledge:
            continue
        for kw in meta["keywords"]:
            if topic_lower in kw or kw in topic_lower:
                data = _knowledge[topic_id]
                return f"=== KNOWLEDGE: {data['name']} ===\n{data['content']}"

    available = list_knowledge()
    return f"(no knowledge matching '{topic}')\nAvailable: {', '.join(available)}"


def get_auto_inject(task: str) -> str:
    """
    Detect relevant knowledge from task and return combined text
    for auto-injection into AI context.
    """
    topics = detect_relevant_knowledge(task)
    if not topics:
        return ""

    parts = []
    for topic_id in topics:
        data = _knowledge[topic_id]
        parts.append(f"=== KNOWLEDGE: {data['name']} ===\n{data['content']}")
        status(f"Auto-injected knowledge: {data['name']}")

    return "\n\n".join(parts)


def list_knowledge() -> list[str]:
    """List all available knowledge topics."""
    _load_knowledge()
    return [data["name"] for data in _knowledge.values()]
