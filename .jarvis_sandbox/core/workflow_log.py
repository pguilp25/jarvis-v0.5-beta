"""
Workflow-level debug log.

Per-model streams (in `<model>.md`) are great for watching ONE AI think,
but they make it hard to see the workflow as a whole — was the merger
called? Which plans went in? Which step failed?

This module writes:
  • `workflow.log`           chronological event log of every phase
  • `plans/layer1_<i>_<m>.md` final plan text from each Layer 1 model
  • `plans/layer2_<i>_<m>.md` improved plan from each Layer 2 model
  • `plans/final.md`          merged final plan
  • `steps/step_<n>.md`       per-implementation-step outcome
  • `review.md`               reviewer output + edits applied

All paths are relative to the active thought_logger session dir. If no
session has been opened (e.g. tests, ad-hoc scripts) the functions
no-op so they never crash a workflow that just wasn't instrumented.

The functions are thread/async safe via a single re-entrant lock — the
4 parallel planners often log simultaneously.
"""

from __future__ import annotations

import datetime
import threading
from pathlib import Path
from typing import Any

from core import thought_logger

_lock = threading.RLock()


def _session_dir() -> "Path | None":
    return thought_logger.session_dir()


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _append_log(line: str) -> None:
    sd = _session_dir()
    if sd is None:
        return
    with _lock:
        with open(sd / "workflow.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ─── Phase / event primitives ───────────────────────────────────────────

def phase_start(phase: str, sub: str = "", **meta: Any) -> None:
    """Mark the start of a workflow phase.

    Example:
        phase_start("plan", "Layer 1", n_models=4)
    """
    meta_str = " ".join(f"{k}={v}" for k, v in meta.items())
    sub_str = f" — {sub}" if sub else ""
    _append_log(f"[{_ts()}] ▶ START  {phase}{sub_str}  {meta_str}".rstrip())


def phase_end(phase: str, sub: str = "", **meta: Any) -> None:
    meta_str = " ".join(f"{k}={v}" for k, v in meta.items())
    sub_str = f" — {sub}" if sub else ""
    _append_log(f"[{_ts()}] ✓ END    {phase}{sub_str}  {meta_str}".rstrip())


def phase_event(msg: str, **meta: Any) -> None:
    """One-line event inside the current phase (no boundary marker)."""
    meta_str = " ".join(f"{k}={v}" for k, v in meta.items())
    _append_log(f"[{_ts()}]   • {msg}  {meta_str}".rstrip())


def phase_warn(msg: str, **meta: Any) -> None:
    meta_str = " ".join(f"{k}={v}" for k, v in meta.items())
    _append_log(f"[{_ts()}]   ⚠ {msg}  {meta_str}".rstrip())


def phase_error(msg: str, **meta: Any) -> None:
    meta_str = " ".join(f"{k}={v}" for k, v in meta.items())
    _append_log(f"[{_ts()}]   ✗ {msg}  {meta_str}".rstrip())


# ─── Artifact snapshots ─────────────────────────────────────────────────

def save_artifact(relpath: str, content: str, header: str = "") -> "Path | None":
    """Write `content` to `<session_dir>/<relpath>`.

    Returns the absolute path written, or None if no session is active.
    Creates intermediate directories. If `header` is non-empty, writes it
    as a fenced markdown comment at the top so the file is self-describing.
    """
    sd = _session_dir()
    if sd is None:
        return None
    target = sd / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    body = content if not header else f"<!-- {header} -->\n\n{content}"
    with _lock:
        target.write_text(body, encoding="utf-8")
    return target


def save_plan(layer: int, index: int, model_id: str, content: str) -> "Path | None":
    """Save one parallel-planner's final plan text.

    layer:  1 (initial plans) or 2 (improved plans) or 3 (merger output).
    index:  0-based position in the parallel pool.
    """
    short_model = model_id.split("/")[-1]
    name = f"plans/layer{layer}_{index:02d}_{short_model}.md"
    header = (
        f"layer={layer}  index={index}  model={model_id}  "
        f"chars={len(content)}  saved={datetime.datetime.now().isoformat(timespec='seconds')}"
    )
    return save_artifact(name, content, header=header)


def save_final_plan(content: str, merger_model: str = "") -> "Path | None":
    header = (
        f"role=final_merged_plan  merger={merger_model}  "
        f"chars={len(content)}  saved={datetime.datetime.now().isoformat(timespec='seconds')}"
    )
    return save_artifact("plans/final.md", content, header=header)


def save_step(step_num: int, step_name: str, model_id: str, content: str,
              extra: "dict[str, Any] | None" = None) -> "Path | None":
    """Save an implementation step's final coder output + summary."""
    short_model = model_id.split("/")[-1]
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in step_name)[:40]
    name = f"steps/step_{step_num:02d}_{safe_name}.md"
    meta = {
        "step": step_num, "name": step_name, "model": model_id,
        "chars": len(content),
    }
    if extra:
        meta.update(extra)
    header = "  ".join(f"{k}={v}" for k, v in meta.items())
    return save_artifact(name, content, header=header)


def save_review(content: str, model_id: str = "") -> "Path | None":
    header = (
        f"role=review  model={model_id}  chars={len(content)}  "
        f"saved={datetime.datetime.now().isoformat(timespec='seconds')}"
    )
    return save_artifact("review.md", content, header=header)


def save_test(test_command: str, output: str, passed: bool) -> "Path | None":
    header = f"role=test  passed={passed}  command={test_command!r}"
    return save_artifact("test.md", output, header=header)
