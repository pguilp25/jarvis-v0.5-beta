"""
Session persistence — save/restore conversation state.
Saves after EVERY response to .jarvis/current_session.json.
"""

import json
import os
from pathlib import Path
from core.memory import ConversationMemory
from core.cli import status


def _jarvis_dir(project_path: str = ".") -> Path:
    p = Path(project_path) / ".jarvis"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_session(memory: ConversationMemory, project_path: str = "."):
    """Save current session to disk."""
    path = _jarvis_dir(project_path) / "current_session.json"
    try:
        with open(path, "w") as f:
            json.dump(memory.to_dict(), f, indent=2)
    except Exception as e:
        status(f"Failed to save session: {e}", "yellow")


def load_session(project_path: str = ".") -> ConversationMemory | None:
    """Load last session from disk. Returns None if no session exists."""
    path = _jarvis_dir(project_path) / "current_session.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        mem = ConversationMemory.from_dict(data)
        if mem.full_history:
            status(f"Restored session ({len(mem.full_history)} messages)")
        return mem
    except Exception as e:
        status(f"Failed to load session: {e}", "yellow")
        return None


def clear_session(project_path: str = "."):
    """Delete saved session."""
    path = _jarvis_dir(project_path) / "current_session.json"
    if path.exists():
        path.unlink()
