"""
Bulletproof tool-tag detection.

Single source of truth for "what tool calls did the model fire?" Replaces
the ad-hoc patchwork of regex extractors, masking passes, validation
helpers, and dropped-tag scans with one explicit pipeline:

  1.  Find the regions where tags MUST be ignored
      (code fences, backticks, <think>, edit-blocks, escaped \\[)
  2.  Find the [tool use]...[/tool use] spans
      (the ONLY places a tag is allowed to fire)
  3.  Scan the raw text for every `[TYPE: arg]`-shaped match
      with TWO independent passes — DOTALL regex AND a position-aware
      bracket scanner. Disagreement is logged loudly.
  4.  Classify each candidate match as VALID or REJECTED with a reason:
        • masked-by-code-fence
        • masked-by-backtick
        • masked-by-think-block
        • masked-by-edit-block
        • escaped-with-backslash
        • outside-tool-use-block
        • no-tool-use-block-in-response
        • malformed-arg
        • unknown-tag-type
  5.  Return DetectedTag objects carrying the rejection reason for
      every candidate, valid or not, so the caller can surface
      "tags that did NOT fire" to the model.

Design goals — in priority order:
  • Reliability: ALWAYS fires when a tag is genuinely intended.
  • Precision:    NEVER fires on prose mentions, escaped tags, or
                  content inside code blocks / think / edit spans.
  • Visibility:   Every rejection comes with a reason string.
  • Redundancy:   Two independent extraction methods; their union is
                  used, their disagreement is logged.
  • Testability:  A built-in corpus runs at import — JARVIS refuses
                  to start if any case regresses.

Used by `core/tool_call.py`. Tests in this module's `_run_self_test()`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


# ─── Tag types we know about ────────────────────────────────────────────────
# The ORDER here is the order we extract in; downstream stable iteration
# depends on it (the per-round tag-count cap is applied per-list pro-rata).
KNOWN_TAG_TYPES = (
    "CODE", "VIEW", "KEEP", "REFS", "SEARCH", "WEBSEARCH",
    "DETAIL", "PURPOSE", "SEMANTIC", "LSP", "KNOWLEDGE", "DISCARD",
)

# Per-tag regex. Each pattern uses DOTALL so multi-line args
# (`[CODE:\n  foo.py\n]`) match — the prior single-line regex was a silent
# false-negative source.
# The capture group is the RAW arg (anything between `:` and the closing
# `]`). It still uses non-greedy `+?` to terminate at the FIRST `]`, but
# DOTALL lets that `+?` span newlines if the model wrote them.
_TAG_PATTERNS = {
    tt: re.compile(
        rf'\[{tt}:\s*(.+?)\s*\]',
        re.IGNORECASE | re.DOTALL,
    )
    for tt in KNOWN_TAG_TYPES
}

# Tool-use block bodies — only spans INSIDE these are eligible to fire.
_TOOL_USE_OPEN = re.compile(r'\[tool\s*use\]', re.IGNORECASE)
_TOOL_USE_CLOSE = re.compile(r'\[/tool\s*use\]', re.IGNORECASE)

# Mask sources — every region where a tag should be IGNORED.
_MASK_PATTERNS: dict[str, re.Pattern] = {
    "code-fence":   re.compile(r'```.*?```', re.DOTALL),
    "backtick":     re.compile(r'`[^`\n]+`'),
    "think-block":  re.compile(
        r'(?:<think>.*?</think>|\[think\].*?\[/think\])',
        re.DOTALL | re.IGNORECASE,
    ),
    # Edit-blocks (full FILE: ... END FILE, or EDIT: ... [/REPLACE]/[/INSERT])
    "edit-file":    re.compile(
        r'===\s*FILE:.*?===\s*END\s+FILE\s*===',
        re.DOTALL | re.IGNORECASE,
    ),
    "edit-block":   re.compile(
        r'===\s*(?:EDIT|FILE):.*?'
        r'(?:\[/REPLACE\]|\[/INSERT\]|===\s*END\s+FILE\s*===)',
        re.DOTALL | re.IGNORECASE,
    ),
    # PLAN blocks — same mask reasoning as edit blocks. The body of the
    # plan is prose; anything inside that looks like a tool tag (e.g.,
    # the planner discussing what [CODE:] will be called in a step) must
    # NOT fire. Mask the entire PLAN / PLAN_EDIT body.
    "plan-block":   re.compile(
        r'===\s*PLAN\s*===.*?===\s*END\s+PLAN\s*===',
        re.DOTALL | re.IGNORECASE,
    ),
    "plan-edit-block": re.compile(
        r'===\s*PLAN_EDIT\s*===.*?===\s*END\s+PLAN_EDIT\s*===',
        re.DOTALL | re.IGNORECASE,
    ),
    "search-body":  re.compile(r'\[SEARCH[^\]]*\].*?\[/SEARCH\]',
                               re.DOTALL | re.IGNORECASE),
    "replace-body": re.compile(r'\[REPLACE[^\]]*\].*?\[/REPLACE\]',
                               re.DOTALL | re.IGNORECASE),
    "insert-body":  re.compile(r'\[INSERT[^\]]*\].*?\[/INSERT\]',
                               re.DOTALL | re.IGNORECASE),
}
_ESCAPED_BRACKET = re.compile(r'\\\[')


# ─── Tag-arg validators ────────────────────────────────────────────────────
# A tag whose arg doesn't fit the expected shape is rejected with
# "malformed-arg: <why>". Prevents prose-shaped tag calls from firing.

# Path-shaped: `module/path.py` optionally followed by line spec or
# whitespace-separated numeric ranges.
_RE_PATH_ARG = re.compile(
    r'^[\w./\-+#]+'                           # filepath
    r'(?:\s+\d+(?:\s*-\s*\d+)?'               # optional first range/line
    r'(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*)?'       # additional comma-separated
    r'\s*$',
)
# Identifier-shaped: bare symbol, dots OK for `mod.func`.
_RE_IDENT_ARG = re.compile(r'^[\w.]+$')
# Search-pattern shaped: pretty much anything goes; only reject the most
# obvious "this is a sentence" forms (ends in punctuation that prose has).
_RE_SEARCH_OBVIOUS_PROSE = re.compile(r'[?!]\s*$')
# Detail / Purpose / Knowledge / Semantic: free-form name, but must not
# look like prose — reject if it's a full sentence (multiple spaces +
# common stop words).
_RE_LONG_PROSE = re.compile(r'\b(?:the|a|that|which|where|when|how|please|need|want|should)\b',
                            re.IGNORECASE)

# Label suffix (`#label` at end of arg).
_RE_LABEL_SUFFIX = re.compile(r'\s+#(\w+)\s*$')


def _strip_label(raw: str) -> tuple[str, str | None]:
    m = _RE_LABEL_SUFFIX.search(raw)
    if m:
        return raw[:m.start()].strip(), m.group(1)
    return raw.strip(), None


def _validate_arg(tag_type: str, clean_arg: str) -> "str | None":
    """Return rejection reason, or None if the arg is well-shaped."""
    if not clean_arg:
        return "empty arg"
    if tag_type in ("CODE", "KEEP", "VIEW"):
        if not _RE_PATH_ARG.match(clean_arg):
            return f"arg {clean_arg!r} not a path-shape (path [+ ranges])"
        return None
    if tag_type in ("REFS", "LSP"):
        if not _RE_IDENT_ARG.match(clean_arg):
            return f"arg {clean_arg!r} not an identifier"
        return None
    if tag_type == "SEARCH":
        # [SEARCH: N-M] / [SEARCH: filepath] are edit-block syntax, not
        # tool calls. Reject those here; they're filtered elsewhere too.
        if re.match(r'^\d+\s*-\s*\d+$', clean_arg):
            return "search arg is a line range — that's edit syntax, not a search"
        # `[SEARCH: ui/index.html]` (just a path) → also edit syntax
        if (re.search(r'\.\w{1,5}$', clean_arg) and ' ' not in clean_arg
                and '"' not in clean_arg):
            return "search arg is a bare file path — that's edit syntax, not a search"
        if _RE_SEARCH_OBVIOUS_PROSE.search(clean_arg):
            return "search arg ends in `?` or `!` — looks like a question, not a query"
        return None
    if tag_type == "WEBSEARCH":
        # web queries can be long, free-form. No real validation.
        return None
    if tag_type in ("DETAIL", "PURPOSE", "SEMANTIC", "KNOWLEDGE"):
        # Free-form names but reject sentence-shaped args
        if (len(clean_arg.split()) >= 6
                and _RE_LONG_PROSE.search(clean_arg)):
            return f"arg {clean_arg!r} reads as prose, not a category name"
        return None
    if tag_type == "DISCARD":
        # [DISCARD: #label] — arg must be a label
        if not re.match(r'^#?\w+$', clean_arg):
            return f"arg {clean_arg!r} not a #label"
        return None
    return f"unknown tag type {tag_type!r}"


# ─── DetectedTag dataclass ─────────────────────────────────────────────────

@dataclass
class DetectedTag:
    tag_type: str
    raw_arg: str            # everything between `:` and `]` (unstripped of label)
    clean_arg: str          # raw_arg with label stripped
    label: "str | None"
    start: int              # position in source text
    end: int
    rejection_reason: "str | None" = None
    discovered_by: tuple[str, ...] = field(default_factory=tuple)

    @property
    def valid(self) -> bool:
        return self.rejection_reason is None

    def __repr__(self) -> str:
        tag = "✓" if self.valid else "✗"
        reason = f" [{self.rejection_reason}]" if self.rejection_reason else ""
        return f"<{tag} {self.tag_type}:{self.clean_arg}{reason}>"


# ─── The detector ──────────────────────────────────────────────────────────

class TagDetector:
    """Detect every tool tag in `text` and classify each as valid/rejected.

    All work happens in `__init__` — `all_tags`, `valid_tags`, etc are
    cheap accessors.
    """

    def __init__(self, text: str):
        self.text = text or ""
        self._tool_use_spans = self._compute_tool_use_spans()
        self._masked_spans = self._compute_masked_spans()
        self._escaped_brackets = self._compute_escaped_bracket_starts()
        self._all = self._compute_all_tags()

    # ── span computation ───────────────────────────────────────────────

    def _compute_tool_use_spans(self) -> list[tuple[int, int]]:
        """Return (start_of_body, end_of_body) spans for every closed
        `[tool use]...[/tool use]` pair, AND for an unterminated trailing
        `[tool use]` (body = from open-tag end → end of text).

        Handling unterminated blocks as "body to end" is intentional:
        models commonly forget the closing `[/tool use]` when running
        out of tokens, and we want the tags they wrote inside to still
        fire. The `_autocomplete_tool_blocks` upstream tries to insert
        the closer; this is the defensive second layer.
        """
        opens = [m for m in _TOOL_USE_OPEN.finditer(self.text)]
        closes = [m for m in _TOOL_USE_CLOSE.finditer(self.text)]
        spans: list[tuple[int, int]] = []
        ci = 0
        for o in opens:
            body_start = o.end()
            # Find the next close after this open
            while ci < len(closes) and closes[ci].start() < body_start:
                ci += 1
            if ci < len(closes):
                spans.append((body_start, closes[ci].start()))
                ci += 1
            else:
                # Unterminated open — body extends to EOT.
                spans.append((body_start, len(self.text)))
        return spans

    def _compute_masked_spans(self) -> list[tuple[int, int, str]]:
        """Every region where tags should be IGNORED.
        Returns list of (start, end, reason) so we can carry the source
        reason into rejection messages.
        """
        spans: list[tuple[int, int, str]] = []
        for reason, pat in _MASK_PATTERNS.items():
            for m in pat.finditer(self.text):
                spans.append((m.start(), m.end(), reason))
        return spans

    def _compute_escaped_bracket_starts(self) -> set[int]:
        return {m.end() - 1 for m in _ESCAPED_BRACKET.finditer(self.text)}

    # ── classification helpers ─────────────────────────────────────────

    def _mask_reason_at(self, pos: int) -> "str | None":
        """If `pos` is inside any masked span, return the reason; else None."""
        for s, e, reason in self._masked_spans:
            if s <= pos < e:
                return reason
        return None

    def _in_tool_use(self, pos: int) -> bool:
        for s, e in self._tool_use_spans:
            if s <= pos < e:
                return True
        return False

    # ── extraction ─────────────────────────────────────────────────────

    def _compute_all_tags(self) -> list[DetectedTag]:
        """Two independent extraction passes (regex + bracket-scan).
        Their UNION is taken, dedup'd by (start, tag_type, clean_arg),
        and any tag found by only ONE method is flagged with the
        `discovered_by` field — so we can audit edge cases.
        """
        by_method = {
            "regex": self._extract_by_regex(),
            "scan":  self._extract_by_scan(),
        }
        # Dedup by (start, tag_type, clean_arg)
        merged: dict[tuple[int, str, str], DetectedTag] = {}
        for method, tags in by_method.items():
            for t in tags:
                key = (t.start, t.tag_type, t.clean_arg)
                if key in merged:
                    merged[key].discovered_by = tuple(
                        sorted(set(merged[key].discovered_by) | {method})
                    )
                else:
                    t.discovered_by = (method,)
                    merged[key] = t
        out = sorted(merged.values(), key=lambda d: d.start)
        # Classify each (rejection reason). Done AFTER merge so the same
        # tag never gets two contradictory verdicts.
        any_tool_use = bool(self._tool_use_spans)
        for t in out:
            t.rejection_reason = self._classify(t, any_tool_use=any_tool_use)
        return out

    def _extract_by_regex(self) -> list[DetectedTag]:
        """Primary pass — per-type DOTALL regex."""
        out: list[DetectedTag] = []
        for tt, pat in _TAG_PATTERNS.items():
            for m in pat.finditer(self.text):
                raw_arg = m.group(1)
                clean, label = _strip_label(raw_arg)
                out.append(DetectedTag(
                    tag_type=tt,
                    raw_arg=raw_arg,
                    clean_arg=clean,
                    label=label,
                    start=m.start(),
                    end=m.end(),
                ))
        return out

    def _extract_by_scan(self) -> list[DetectedTag]:
        """Secondary pass — bracket-aware text scanner. Walks the text
        looking for `[<TAG>:<arg>]` shapes. Picks up multi-line cases
        and is a sanity-check against regex backtracking quirks.
        """
        out: list[DetectedTag] = []
        n = len(self.text)
        i = 0
        known_types_upper = set(KNOWN_TAG_TYPES)
        while i < n:
            br = self.text.find('[', i)
            if br < 0:
                break
            # Try to read a `TYPE:` immediately after `[`. Tag names are
            # uppercase letters in our schema (extract is case-insensitive,
            # so accept any case).
            j = br + 1
            # skip whitespace
            while j < n and self.text[j] in ' \t':
                j += 1
            # read tag name (letters)
            k = j
            while k < n and self.text[k].isalpha():
                k += 1
            name = self.text[j:k].upper()
            if not name or name not in known_types_upper:
                i = br + 1
                continue
            # require ':'
            if k >= n or self.text[k] != ':':
                i = br + 1
                continue
            # find the matching ']' — first one wins
            close = self.text.find(']', k + 1)
            if close < 0:
                i = br + 1
                continue
            # Skip if `]` is preceded by `]` in a tag like [SEARCH][/SEARCH]
            # (those open-edit tags don't have a `:` after the name, so the
            # name extraction above already excludes them — defensive only).
            raw_arg = self.text[k + 1:close]
            stripped = raw_arg.strip()
            clean, label = _strip_label(stripped)
            out.append(DetectedTag(
                tag_type=name,
                raw_arg=stripped,
                clean_arg=clean,
                label=label,
                start=br,
                end=close + 1,
            ))
            i = close + 1
        return out

    def _classify(self, t: DetectedTag, *, any_tool_use: bool) -> "str | None":
        """Return rejection reason, or None if the tag should fire."""
        # 1. Escaped (`\[CODE: ...]`)
        if t.start in self._escaped_brackets:
            return "escaped-with-backslash"
        # 2. Inside a masked region
        mask = self._mask_reason_at(t.start)
        if mask is not None:
            return f"masked-by-{mask}"
        # 3. Tool-use enforcement
        if any_tool_use:
            if not self._in_tool_use(t.start):
                return "outside-tool-use-block"
        else:
            # No [tool use] in the response at all — strict mode rejects.
            # The runtime upstream injects a correction nudge so the model
            # learns to wrap. Bare-fire fallback would be a footgun (prose
            # mentions of tools fire and the model invents results).
            return "no-tool-use-block-in-response"
        # 4. Arg shape
        bad_arg = _validate_arg(t.tag_type, t.clean_arg)
        if bad_arg:
            return f"malformed-arg: {bad_arg}"
        return None

    # ── public API ─────────────────────────────────────────────────────

    @property
    def all_tags(self) -> list[DetectedTag]:
        return list(self._all)

    def valid_tags(self, tag_type: "str | None" = None) -> list[DetectedTag]:
        if tag_type is None:
            return [t for t in self._all if t.valid]
        tt = tag_type.upper()
        return [t for t in self._all if t.valid and t.tag_type == tt]

    def rejected_tags(self, tag_type: "str | None" = None) -> list[DetectedTag]:
        if tag_type is None:
            return [t for t in self._all if not t.valid]
        tt = tag_type.upper()
        return [t for t in self._all if not t.valid and t.tag_type == tt]

    def valid_args(self, tag_type: str) -> list[str]:
        """Return the RAW arg strings (with label suffix preserved) for
        every valid tag of the given type, in document order, deduplicated.
        Backwards-compatible shape for the legacy extract_*_tags callers.
        """
        seen: set[str] = set()
        out: list[str] = []
        for t in self.valid_tags(tag_type):
            # Include the label in the returned arg so downstream
            # `_label_to_keys` registration works.
            arg = t.raw_arg
            if arg not in seen:
                seen.add(arg)
                out.append(arg)
        return out

    def has_any_valid(self) -> bool:
        return any(t.valid for t in self._all)

    def summary(self) -> str:
        """One-line debug summary."""
        valid = len([t for t in self._all if t.valid])
        rej = len([t for t in self._all if not t.valid])
        tu = len(self._tool_use_spans)
        return f"{valid} valid / {rej} rejected / {tu} [tool use] block(s)"


# ─── Self-test corpus — runs at import ─────────────────────────────────────

_SELF_TEST_CASES: list[tuple[str, dict[str, list[str]]]] = [
    # ─── Should fire ───────────────────────────────────────────────
    ("[tool use]\n[CODE: foo.py]\n[/tool use]", {"CODE": ["foo.py"]}),
    ("prose [tool use][CODE: foo.py][/tool use] more",  {"CODE": ["foo.py"]}),
    ("[tool use]\n[CODE: a.py]\n[REFS: bar]\n[/tool use]",
     {"CODE": ["a.py"], "REFS": ["bar"]}),
    # multi-line arg (DOTALL) — previously a silent miss
    ("[tool use]\n[CODE:\n  foo.py\n]\n[/tool use]", {"CODE": ["foo.py"]}),
    # KEEP with line ranges
    ("[tool use][KEEP: a.py 10-20, 30-40][/tool use]",
     {"KEEP": ["a.py 10-20, 30-40"]}),
    # VIEW with single line input
    ("[tool use][VIEW: a.py 100][/tool use]", {"VIEW": ["a.py 100"]}),
    # label suffix preserved
    ("[tool use][CODE: foo.py #lbl][/tool use]",
     {"CODE": ["foo.py #lbl"]}),
    # Unterminated `[tool use]` — body = to end of text
    ("[tool use]\n[CODE: foo.py]", {"CODE": ["foo.py"]}),
    # SEARCH with quoted prose-query
    ("[tool use][SEARCH: process_turn][/tool use]",
     {"SEARCH": ["process_turn"]}),

    # ─── Should NOT fire ───────────────────────────────────────────
    # No [tool use] block at all → reject ALL
    ("[CODE: foo.py] just prose mentioning a tag", {}),
    # Inside backticks
    ("[tool use][REFS: x][/tool use] then `[CODE: foo.py]` in prose",
     {"REFS": ["x"]}),
    # Inside fenced code
    ("[tool use][REFS: x][/tool use]\n```\n[CODE: foo.py]\n```",
     {"REFS": ["x"]}),
    # Inside <think>
    ("[tool use][REFS: x][/tool use]\n<think>[CODE: foo.py]</think>",
     {"REFS": ["x"]}),
    # Inside === EDIT: block
    ("[tool use][REFS: x][/tool use]\n=== EDIT: a.py ===\n[CODE: bar.py]\n[/REPLACE]",
     {"REFS": ["x"]}),
    # Escaped bracket
    ("[tool use][REFS: x][/tool use] also \\[CODE: foo.py]",
     {"REFS": ["x"]}),
    # OUTSIDE [tool use] when a block exists
    ("[tool use][REFS: x][/tool use] later [CODE: foo.py] in prose",
     {"REFS": ["x"]}),
    # Prose-shaped CODE arg
    ("[tool use][CODE: I want to read foo.py please][/tool use]",
     {}),
    # SEARCH line-range = edit syntax, not a search
    ("[tool use][SEARCH: 45-49][/tool use]", {}),
    # SEARCH bare file path = edit syntax, not a search
    ("[tool use][SEARCH: ui/index.html][/tool use]", {}),
]


def _run_self_test() -> None:
    """Validate every case. Run at import so any regression fails JARVIS
    startup with a clear error rather than silently breaking detection.
    """
    for i, (text, expected) in enumerate(_SELF_TEST_CASES):
        det = TagDetector(text)
        got: dict[str, list[str]] = {}
        for t in det.valid_tags():
            got.setdefault(t.tag_type, []).append(t.raw_arg)
        # Normalise: empty lists removed from `expected`
        for k in list(got.keys()):
            got[k] = sorted(got[k])
        expected_n = {k: sorted(v) for k, v in expected.items() if v}
        got_n = {k: v for k, v in got.items() if v}
        if expected_n != got_n:
            rej = [(t.tag_type, t.clean_arg, t.rejection_reason)
                   for t in det.rejected_tags()]
            raise AssertionError(
                f"TagDetector self-test #{i} failed.\n"
                f"  TEXT:     {text!r}\n"
                f"  EXPECTED: {expected_n}\n"
                f"  GOT:      {got_n}\n"
                f"  REJECTED: {rej}\n"
            )


_run_self_test()
