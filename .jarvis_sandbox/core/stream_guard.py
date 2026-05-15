"""
Stream-level degeneration detector.

Catches model failure modes that the JARVIS prompt format alone cannot
prevent — specifically, "stuck-line" repetition. Observed in practice:
kimi-k2.6 generated `i4|status(f"Step A done...")` 464 times before
max_tokens cut it off, burning ~10k wasted tokens AND filling the
model's context with garbage that propagated to the next round.

Fires from inside the streaming loop: the client passes each new
visible delta to `DegenerationDetector.check(...)` and aborts the stream
when it returns a non-None reason. The partial response collected so
far is still usable — degenerate text only appears at the END.

NOTE: this used to also detect "prompt leak" — when the model started
emitting tokens that ONLY appeared in JARVIS-side prompt scaffolding.
Removed: JARVIS is open source, so the prompts are not secret. The
genuine model-failure cases (stuck-line repetition, low-line diversity)
are still caught here; the prompt-template-echoing case was conflating
"model lost track" with "model said something I considered private,"
and the latter framing was wrong for an OSS project.
"""

from __future__ import annotations

import re

# Empty [tool use]…[/tool use] spam — observed in minimax-m2.7 R6 of
# 20260513_131849, which emitted ~250 of these in a row until max_tokens
# killed the stream. The existing line-length filter (LINE_MIN_LEN=20)
# misses it: `[tool use]` is 10 chars, `[/tool use]` is 11, the body is
# empty, so every line is filtered out as too short to count.
#
# Pattern: an opening tag immediately followed by whitespace/newlines and
# the closing tag, with no tool tags inside. Tolerates a few "almost
# empty" cases (a trailing comment, a stray space) by stopping at the
# first occurrence of `[CODE:`/`[REFS:`/etc. inside.
_EMPTY_TOOL_USE = re.compile(
    r'\[tool use\]\s*\[/tool use\]',
    re.IGNORECASE,
)

# Scaffold-hallucination — the runtime emits `────── ROUND N — your
# thinking ──────` / `────── ROUND N — your tool result ──────` headers
# in [YOUR PAST THINKING]. When the model writes these in its OWN
# response, it's pretending to be the runtime continuing the conversation
# — and the content right after is almost always fabricated (e.g.,
# imagined file contents, fake function signatures). Observed in
# deepseek-v4-flash R2 of 20260513_143353: model wrote a tool call, then
# instead of [STOP][CONFIRM_STOP] it wrote `────── ROUND 2 — your tool
# result ──────` and ~650 lines of invented `code_agent` source.
# Abort the stream the moment this scaffold appears.
_SCAFFOLD_MARKER = "────── ROUND"


class DegenerationDetector:
    """Per-stream detector for repetition-style degeneration.

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
        self._tripped = False
        self._reason: "str | None" = None

    def check(self, accumulated: str) -> "str | None":
        """Return a reason string if degenerate, else None.

        Once tripped, returns the same reason forever — the caller is
        expected to break the stream loop on the first non-None result.
        """
        if self._tripped:
            return self._reason

        # ── 0. Empty [tool use]…[/tool use] block spam ──────────────────
        # Model is generating tool-use shells with no tags inside. Real
        # tool calls always have at least one tag like [CODE:] / [REFS:]
        # / etc. inside the block. If we see 3+ empty blocks anywhere in
        # the accumulated text, the model has lost the plot — abort.
        _empty_blocks = _EMPTY_TOOL_USE.findall(accumulated)
        if len(_empty_blocks) >= 3:
            self._tripped = True
            self._reason = (
                f"empty-tool-use-spam: {len(_empty_blocks)} `[tool use]"
                f"…[/tool use]` blocks with no tags inside"
            )
            return self._reason

        # ── 0b. Scaffold-hallucination ──────────────────────────────────
        # Model is writing JARVIS's per-round header into its own response,
        # which means the content right after is fabricated. Abort.
        if _SCAFFOLD_MARKER in accumulated:
            self._tripped = True
            self._reason = (
                "scaffold-hallucination: model wrote `"
                f"{_SCAFFOLD_MARKER}` inside its response (this header is "
                "runtime-only; content after it is fabricated)"
            )
            return self._reason

        # ── 1. Adjacent-line repetition (the kimi failure mode) ───────
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

        # ── 2. Low-line-diversity (general periodic pattern) ──────────
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
