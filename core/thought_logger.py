"""
Thought Logger — streams each model's reasoning to per-model markdown files
AND prints it live to the terminal (stderr) as it arrives.

One file per model per session, saved to ~/jarvis_thinking_logs/{timestamp}/.
Call new_session(prompt) once per user turn to create a fresh directory.
Each model's full reasoning is appended to its own file as chunks arrive,
and simultaneously echoed to stderr so you can watch models think in real time.

Files are named after the model short-name, e.g.:
  deepseek-v4-pro.md, glm-5.md, minimax-m2.5.md, qwen-3.5.md, nemotron-super.md
  kimi-k2.md, llama-4-scout.md, ...
"""

import atexit
import datetime
import sys
from pathlib import Path
from typing import IO

# ─── Session state ────────────────────────────────────────────────────────────

_session_dir: Path | None = None
_file_handles: dict[str, IO[str]] = {}

# When True, every chunk is also written live to stderr
_live: bool = True

# ─── Per-model ANSI colours (consistent across calls) ─────────────────────────
# Colours cycle through a palette keyed on the model's short name.

_PALETTE = [
    "\033[96m",   # bright cyan
    "\033[95m",   # bright magenta
    "\033[93m",   # bright yellow
    "\033[94m",   # bright blue
    "\033[92m",   # bright green
    "\033[91m",   # bright red
    "\033[97m",   # bright white
]
_RESET = "\033[0m"
_DIM   = "\033[2m"

# Sorted known model names → stable colour index
_KNOWN_MODELS = [
    "deepseek-v4-pro",
    "gemini",
    "glm-5",
    "gpt-oss-120b",
    "kimi-k2",
    "llama-3.1-8b",
    "llama-3.3-70b",
    "llama-4-scout",
    "minimax-m2.5",
    "nemotron-super",
    "qwen-3.5",
    "qwen3-32b",
]

_colour_cache: dict[str, str] = {}


def _model_colour(model_id: str) -> str:
    """Return a consistent ANSI colour string for this model."""
    name = model_id.split("/")[-1]
    if name not in _colour_cache:
        try:
            idx = _KNOWN_MODELS.index(name)
        except ValueError:
            idx = abs(hash(name))
        _colour_cache[name] = _PALETTE[idx % len(_PALETTE)]
    return _colour_cache[name]


# ─── Session management ───────────────────────────────────────────────────────

def new_session(prompt: str = "") -> Path:
    """Create a timestamped session directory. Call once per user prompt."""
    global _session_dir, _file_handles
    close_session()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path.home() / "jarvis_thinking_logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    if prompt:
        (log_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    _session_dir = log_dir
    _file_handles = {}
    return log_dir


def _model_filename(model_id: str) -> str:
    """Convert 'nvidia/deepseek-v4-pro' → 'deepseek-v4-pro.md'"""
    return model_id.split("/")[-1] + ".md"


def _get_file(model_id: str) -> IO[str]:
    """Return (creating if needed) the open write handle for this model."""
    global _file_handles, _session_dir
    if _session_dir is None:
        new_session()
    if model_id not in _file_handles:
        path = _session_dir / _model_filename(model_id)
        # line-buffered (buffering=1) so every write is immediately visible on disk
        _file_handles[model_id] = open(path, "a", encoding="utf-8", buffering=1)
    return _file_handles[model_id]


# ─── Public write API ─────────────────────────────────────────────────────────

def write_header(model_id: str, label: str = "") -> None:
    """
    Write a labelled section header before a new call.
    Writes to the model's log file AND, if live mode is on, prints a
    coloured banner to stderr so the user sees which model is starting.
    """
    f = _get_file(model_id)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    sep = "═" * 60
    name = model_id.split("/")[-1]
    title = f"{name} — {label}  [{ts}]" if label else f"{name}  [{ts}]"

    # ── file ──────────────────────────────────────────────────────────────
    f.write(f"\n\n{sep}\n## {title}\n{sep}\n\n")
    f.flush()

    # ── terminal ──────────────────────────────────────────────────────────
    if _live:
        colour = _model_colour(model_id)
        # Lead with \n so we always start the banner on a fresh line even if
        # a previous stream ended mid-line.
        banner = f"\n{colour}┌─ {title} ─┐{_RESET}"
        print(banner, file=sys.stderr, flush=True)


def write_chunk(model_id: str, chunk: str) -> None:
    """
    Append a streamed text chunk to the model's log file.
    If live mode is on, also echo it to stderr immediately (no newline added —
    the chunk is printed exactly as received so the stream reads naturally).
    """
    if not chunk:
        return

    # ── file ──────────────────────────────────────────────────────────────
    _get_file(model_id).write(chunk)

    # ── terminal ──────────────────────────────────────────────────────────
    if _live:
        colour = _model_colour(model_id)
        sys.stderr.write(f"{_DIM}{colour}{chunk}{_RESET}")
        sys.stderr.flush()


# ─── Live-mode control ────────────────────────────────────────────────────────

def enable_live() -> None:
    """Turn on live terminal echo (default)."""
    global _live
    _live = True


def disable_live() -> None:
    """Turn off live terminal echo (log to file only)."""
    global _live
    _live = False


# ─── Cleanup ──────────────────────────────────────────────────────────────────

def close_session() -> None:
    """Close all open file handles (call before creating a new session)."""
    global _file_handles
    for f in _file_handles.values():
        try:
            f.close()
        except Exception:
            pass
    _file_handles = {}


def session_dir() -> Path | None:
    """Return the active session directory path, or None if not started."""
    return _session_dir


# Ensure file handles are closed cleanly even if the program exits unexpectedly
atexit.register(close_session)
