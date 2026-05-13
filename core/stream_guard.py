"""
Stream-level safety detector.

Detects two failure modes that the JARVIS prompt format alone cannot prevent:

1. DEGENERATION — model gets stuck repeating the same line. Observed in
   practice: kimi-k2.6 generated `i4|status(f"Step A done...")` 464 times
   before max_tokens cut it off, burning ~10k wasted tokens AND filling
   the model's context with garbage that propagated to the next round.

2. PROMPT HALLUCINATION — model emits literal tokens that ONLY ever
   appear in JARVIS-side prompt scaffolding ("TOOL RESULTS (cumulative",
   "[← CODE: ...]", "══ CONTEXT MANIFEST"). When the model starts
   writing these, it has lost track of what's input vs output and the
   rest of its response is unreliable.

Both fire from inside the streaming loop: the client passes each new
visible delta to `DegenerationDetector.check(...)` and aborts the stream
when it returns a non-None reason. The partial response collected so
far is still usable — degenerate text only appears at the END.
"""

from __future__ import annotations

# ── Phrases that ONLY appear in JARVIS prompt scaffolding ───────────────
# If a model writes any of these in its visible response, it's regurgitating
# the prompt format back at us — its next tokens cannot be trusted.
_PROMPT_LEAK_PHRASES = (
    # New labelled section headers from the unified prompt template.
    # Models occasionally hallucinate these back into their response
    # when they've stopped writing real content and started parroting
    # the prompt scaffold. Catching them aborts the stream cleanly.
    "[YOUR TOOL INDEX]",
    "[YOUR PAST THINKING]",
    "[YOUR PLAN]",
    "[YOUR TOOL RESULTS]",   # legacy header still possible to leak
    "[YOUR PRIOR TURNS]",    # legacy header still possible to leak
    "[WRITE YOUR NEXT TURN BELOW]",
    "[PROJECT CONTEXT]",
    "[INPUT PLANS]",         # used by improve / merge phases
    "[END INPUT PLANS]",
    "[INPUT PLAN #",         # per-plan labels in improve / merge
    "────── ROUND",  # the round-section divider used in past_thinking
    # Legacy headers kept until all paths are migrated:
    "TOOL RESULTS (cumulative across all tool calls",
    "══ CONTEXT MANIFEST — what you have actually read",
    "TOOL CALLS THAT DID NOT FIRE",  # broad — catches "(check your syntax)"
                                     # hallucinations and the literal banner
    "BUDGET OVERFLOW — these results were DROPPED",
    "EDIT APPLICATION RESULTS — what the runtime did",
    "UNTERMINATED EDIT BLOCK(S)",
    "YOUR THINKING SO FAR — continuous reasoning",
    "YOUR PREVIOUS TURNS — DONE, not unfinished",
    # Continuation-cue arrows (any variant the prompt has used):
    "↓ Continue your thinking",
    "↓ Continue writing from where you stopped",
    "↓ NOW WRITE YOUR NEXT TURN",
    # Cumulative-result entry prefixes:
    "\n[← CODE:",
    "\n[← KEEP:",
    "\n[← VIEW:",
    "\n[← REFS:",
    "\n[← SEARCH:",
)


class DegenerationDetector:
    """Per-stream detector for repetition and prompt hallucination.

    Usage:
        det = DegenerationDetector()
        async for delta in stream:
            visible_chunks.append(delta)
            reason = det.check("".join(visible_chunks))
            if reason:
                # abort the stream — partial response is still valid
                break
    """

    # Tunables — chosen to be CONSERVATIVE (false-positive-resistant) so we
    # never abort a legitimate response that happens to be long-and-similar.
    LINE_MIN_LEN     = 20       # ignore short lines (`}`, blank, etc.)
    LINE_REPEAT_THRESHOLD = 8   # same line 8+ times in a row → degenerate
    BLOCK_LOOKBACK   = 4_000    # scan last 4k chars for periodic repetition
    BLOCK_MIN_LEN    = 40       # ignore tiny periodic patterns
    BLOCK_REPEAT_THRESHOLD = 6  # same N-char block ≥ 6× in lookback → degen

    def __init__(self) -> None:
        self._last_check_pos = 0   # only re-scan new content
        self._tripped = False
        self._reason: "str | None" = None

    def check(self, accumulated: str) -> "str | None":
        """Return a reason string if degenerate, else None.

        Once tripped, returns the same reason forever — the caller is
        expected to break the stream loop on the first non-None result.
        """
        if self._tripped:
            return self._reason

        # ── 1. Prompt-leak detection (fast, exact-substring) ──────────
        # Only check newly-appended text to keep this O(new) not O(total).
        new_text = accumulated[self._last_check_pos:]
        # Re-include a small overlap so a phrase split across check calls
        # still matches. Longest phrase is ~70 chars; 100 is safe.
        overlap_start = max(0, self._last_check_pos - 100)
        scan_text = accumulated[overlap_start:]
        for phrase in _PROMPT_LEAK_PHRASES:
            if phrase in scan_text:
                self._tripped = True
                self._reason = f"prompt-leak: model emitted '{phrase[:40]}...'"
                return self._reason
        self._last_check_pos = len(accumulated)

        # ── 2. Adjacent-line repetition (the kimi failure mode) ───────
        # Cheap fast-path: look only at the tail. If the last 8 non-trivial
        # lines are identical and long enough, the model is stuck.
        # Tail-size: 200 chars per expected line × threshold so realistic
        # 60-100 char code lines all fit. Earlier sizing of 20×10 = 200
        # chars total only held ~3 lines and missed the actual failure.
        tail_size = 250 * (self.LINE_REPEAT_THRESHOLD + 2)
        tail = accumulated[-tail_size:]
        lines = tail.splitlines()
        if len(lines) >= self.LINE_REPEAT_THRESHOLD:
            recent = lines[-self.LINE_REPEAT_THRESHOLD:]
            first = recent[0].strip()
            if (
                len(first) >= self.LINE_MIN_LEN
                and all(ln.strip() == first for ln in recent)
            ):
                self._tripped = True
                self._reason = (
                    f"line-repetition: '{first[:50]}...' "
                    f"x{self.LINE_REPEAT_THRESHOLD}"
                )
                return self._reason

        # ── 3. Low-line-diversity (general periodic pattern) ──────────
        # Catches loops where the model alternates 2-3 lines forever. We
        # don't need to know the period — just count how many DISTINCT
        # non-trivial lines exist in the last 20. If only 1-3 unique
        # values fill 15+ slots, it's a loop.
        all_lines = accumulated.splitlines()
        if len(all_lines) >= 15:
            recent = [
                ln.strip() for ln in all_lines[-20:]
                if len(ln.strip()) >= self.LINE_MIN_LEN
            ]
            if len(recent) >= 12:
                unique_count = len(set(recent))
                if unique_count <= 3:
                    self._tripped = True
                    self._reason = (
                        f"low-diversity: {len(recent)} recent lines, "
                        f"only {unique_count} unique"
                    )
                    return self._reason

        return None

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def reason(self) -> "str | None":
        return self._reason
