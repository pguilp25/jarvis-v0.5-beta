"""
Two-level compression:

1. BACKGROUND (after response shown to user):
   - Triggers when full_history > 50K tokens
   - Compresses full history to ~30K

2. INPUT (before feeding to model):
   - Triggers when AI context > 10K tokens
   - Compresses to 5-10K for model input only

Both use Gemini Flash Lite — 250K context window, fast, free.
"""

import asyncio
from core.tokens import count_tokens, truncate_to_tokens
from clients.gemini import call_flash
from core.cli import step, status, success

# ─── Thresholds ──────────────────────────────────────────────────────────────

BACKGROUND_TRIGGER = 50_000    # Full history > this → background compress
BACKGROUND_TARGET = 30_000     # Target after background compression

INPUT_TRIGGER = 10_000         # AI context > this → compress for model input
INPUT_TARGET_MIN = 5_000
INPUT_TARGET_MAX = 10_000

# ─── Prompts ─────────────────────────────────────────────────────────────────

SUMMARIZE_OLD_PROMPT = """Summarize this OLD conversation history as a chronological timeline.
Format each topic/phase as a bullet with approximate time:
  [HH:MM] Topic — what happened, what was decided

Keep ONLY: key decisions, outcomes, lessons, preferences.
Drop details, examples, back-and-forth, greetings.
Target: ~{target} tokens maximum.

Old history:
{context}

Chronological summary (oldest first):"""

SUMMARIZE_RECENT_PROMPT = """Summarize this RECENT conversation as a chronological timeline.
Format each topic/exchange as:
  [HH:MM] Topic — what was discussed, key details, decisions

Preserve more detail than a normal summary:
- Key decisions and WHY
- Technical details, numbers, code snippets
- Current task state, unresolved questions
- Failed approaches and lessons
Target: ~{target} tokens maximum.

Recent history:
{context}

Chronological summary (oldest first, more detail on latest):"""

MERGE_PROMPT = """Merge these two chronological conversation summaries into ONE timeline.
OLD covers earlier. RECENT covers latest.

Rules:
- Output as ONE continuous chronological timeline
- OLD entries: keep very brief, one line each
- RECENT entries: keep more detail
- Format: [HH:MM] Topic — summary
- Total output: {target_min}-{target_max} tokens
- Remove duplicates, keep chronological order

OLD (earlier):
{old_summary}

RECENT (latest):
{recent_summary}

Merged timeline:"""


# ─── Background Compression (50K → 30K) ─────────────────────────────────────

async def compress_background(full_history: list[dict]) -> str:
    """
    Compress full history to ~30K. Runs in background after response.
    Old info gets more summarized than recent.
    """
    step("Background compression (50K → 30K)")

    split_point = int(len(full_history) * 0.7)
    old_msgs = full_history[:split_point]
    recent_msgs = full_history[split_point:]

    old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_msgs)
    recent_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent_msgs)

    old_tokens = count_tokens(old_text)
    recent_tokens = count_tokens(recent_text)
    status(f"Old: {old_tokens:,} tok | Recent: {recent_tokens:,} tok")

    if old_tokens > 120_000:
        old_text = truncate_to_tokens(old_text, 120_000)
    if recent_tokens > 120_000:
        recent_text = truncate_to_tokens(recent_text, 120_000)

    # Two Flash Lite in parallel
    status("Summarizing old + recent in parallel...")
    old_summary, recent_summary = await asyncio.gather(
        call_flash(
            SUMMARIZE_OLD_PROMPT.format(context=old_text, target=10000),
            max_tokens=8192,
        ),
        call_flash(
            SUMMARIZE_RECENT_PROMPT.format(context=recent_text, target=18000),
            max_tokens=16384,
        ),
    )

    # Merge
    status("Merging...")
    merged = await call_flash(
        MERGE_PROMPT.format(
            old_summary=old_summary,
            recent_summary=recent_summary,
            target_min=25000,
            target_max=30000,
        ),
        max_tokens=16384,
    )

    final_tokens = count_tokens(merged)
    total_original = old_tokens + recent_tokens
    success(f"Background: {total_original:,} → {final_tokens:,} tokens ({total_original / max(final_tokens, 1):.1f}x)")
    return merged


# ─── Input Compression (10K → 5-10K) ────────────────────────────────────────

async def compress_for_input(context: str) -> str:
    """
    Compress AI context to 5-10K for model input. Does NOT modify memory.
    Returns a smaller context string.
    """
    token_count = count_tokens(context)

    if token_count <= INPUT_TRIGGER:
        return context  # No compression needed

    step(f"Input compression ({token_count:,} → 5-10K)")

    # Split into old/recent halves
    lines = context.split("\n")
    split = int(len(lines) * 0.6)
    old_text = "\n".join(lines[:split])
    recent_text = "\n".join(lines[split:])

    # Parallel summarize
    old_summary, recent_summary = await asyncio.gather(
        call_flash(
            SUMMARIZE_OLD_PROMPT.format(context=old_text, target=2000),
            max_tokens=2048,
        ),
        call_flash(
            SUMMARIZE_RECENT_PROMPT.format(context=recent_text, target=4000),
            max_tokens=4096,
        ),
    )

    # Merge to 5-10K
    merged = await call_flash(
        MERGE_PROMPT.format(
            old_summary=old_summary,
            recent_summary=recent_summary,
            target_min=INPUT_TARGET_MIN,
            target_max=INPUT_TARGET_MAX,
        ),
        max_tokens=8192,
    )

    final = count_tokens(merged)
    success(f"Input: {token_count:,} → {final:,} tokens")
    return merged


# ─── Trigger Checks ─────────────────────────────────────────────────────────

def needs_background_compression(full_history: list[dict]) -> bool:
    total = sum(count_tokens(m.get("content", "")) for m in full_history)
    return total > BACKGROUND_TRIGGER


def needs_input_compression(context_tokens: int) -> bool:
    return context_tokens > INPUT_TRIGGER
