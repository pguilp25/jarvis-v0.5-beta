"""
Tool Call Loop — shared by all workflows.

Any AI can pause mid-thought to search:
  [SEARCH: pattern]    → ripgrep code search (coding agent)
  [WEBSEARCH: query]   → web search (research, chat)

JARVIS detects the tags, runs the searches, feeds results back,
and the AI continues from where it left off. Up to 5 rounds.
"""

import asyncio
import os
import re
from core.retry import call_with_retry
from core.cli import step, status, warn


# In-flight locks: prevent duplicate lookups across parallel AI calls.
# When two coders both request [REFS: foo] at the same time, only one
# actually runs the search — the other waits and gets the cached result.
_inflight_locks: dict[str, asyncio.Lock] = {}


# ─── Tag Patterns ────────────────────────────────────────────────────────────

SEARCH_TAG = re.compile(r'\[SEARCH:\s*(.+?)\]', re.IGNORECASE)
WEBSEARCH_TAG = re.compile(r'\[WEBSEARCH:\s*(.+?)\]', re.IGNORECASE)
DETAIL_TAG = re.compile(r'\[DETAIL:\s*(.+?)\]', re.IGNORECASE)
CODE_TAG = re.compile(r'\[CODE:\s*(.+?)\]', re.IGNORECASE)
REFS_TAG = re.compile(r'\[REFS:\s*(.+?)\]', re.IGNORECASE)
PURPOSE_TAG = re.compile(r'\[PURPOSE:\s*(.+?)\]', re.IGNORECASE)
SEMANTIC_TAG = re.compile(r'\[SEMANTIC:\s*(.+?)\]', re.IGNORECASE)
LSP_TAG = re.compile(r'\[LSP:\s*(.+?)\]', re.IGNORECASE)
KNOWLEDGE_TAG = re.compile(r'\[KNOWLEDGE:\s*(.+?)\]', re.IGNORECASE)
# KEEP strips a previously-loaded [CODE:] result to only the specified line
# ranges, removing the full file from context.  Format:
#   [KEEP: filepath 10-50, 80-120]
KEEP_TAG = re.compile(r'\[KEEP:\s*(.+?)\]', re.IGNORECASE)
# STOP signals "execute my tool calls now, then let me continue thinking."
# Robust two-tag combination — both halves must appear in order, separated
# by only whitespace/newlines. The CONFIRM_STOP half is a unique token
# that has no other reason to appear in any prose, code, or example: the
# model literally cannot produce it except by deliberate intent.
#
# This robustness matters when the model is editing its OWN codebase and
# constantly discusses signal tags in prose ("how does [STOP] vs [DONE]
# work?"). A single [STOP] alone — anywhere, in any context — does NOT
# fire. Only the full ordered combination does.
#
# Fires (case-insensitive, optional whitespace and one or more newlines
# between the two halves):
#   [STOP]\n[CONFIRM_STOP]        ← canonical: separate lines
#   [STOP][CONFIRM_STOP]          ← also fires: adjacent
#   [STOP]  [CONFIRM_STOP]        ← also fires: same line, spaces
#
# Does NOT fire:
#   [STOP]                              ← bare tag, anywhere
#   "discussion of [STOP] tag"          ← anywhere in prose
#   `[STOP]`                            ← in backticks
#   [STOP] then I'll [CONFIRM_STOP]     ← arbitrary text between halves
#   [CONFIRM_STOP] then [STOP]          ← wrong order
STOP_TAG = re.compile(r'\[STOP\]\s*\[CONFIRM_STOP\]', re.IGNORECASE)
# DONE signals the model is completely finished — apply edits and exit.
# Same two-tag-combination robustness as STOP.
DONE_TAG = re.compile(r'\[DONE\]\s*\[CONFIRM_DONE\]', re.IGNORECASE)
# CONTINUE signals "I'm not done writing my output but I have no tool
# calls — give me another round so I can finish." Used when a long plan,
# review, or analysis would overflow a single response. The runtime
# loops without firing any tool processing and feeds back a CONTINUATION
# banner so the model picks up where it stopped.
# Two-tag protocol identical to STOP/DONE.
CONTINUE_TAG = re.compile(r'\[CONTINUE\]\s*\[CONFIRM_CONTINUE\]', re.IGNORECASE)
# Bare-tag detectors — fire when the model wrote one half of the signal
# but not the other. Used to inject a correction so the model learns the
# combined form instead of looping silently.
_BARE_STOP = re.compile(r'(?<!\[)\[STOP\](?!\s*\[CONFIRM_STOP\])', re.IGNORECASE)
_BARE_DONE = re.compile(r'(?<!\[)\[DONE\](?!\s*\[CONFIRM_DONE\])', re.IGNORECASE)
_BARE_CONTINUE = re.compile(r'(?<!\[)\[CONTINUE\](?!\s*\[CONFIRM_CONTINUE\])', re.IGNORECASE)
# DISCARD removes a previously-loaded tool result by its #label.
# Format: [DISCARD: #label]
DISCARD_TAG = re.compile(r'\[DISCARD:\s*#(\w+)\]', re.IGNORECASE)
# Label suffix on tool calls — optional #label at the end of the argument.
# E.g. [REFS: process_turn #ref1] — the #ref1 is the label.
_LABEL_SUFFIX = re.compile(r'\s+#(\w+)\s*$')


def _strip_label(tag_arg: str) -> tuple[str, str | None]:
    """Strip optional #label from a tool argument. Returns (clean_arg, label_or_None)."""
    m = _LABEL_SUFFIX.search(tag_arg)
    if m:
        return tag_arg[:m.start()].strip(), m.group(1)
    return tag_arg.strip(), None


# Markers our deep-think preambles use, by canonical section name.
# Each entry: (display_name, regex_pattern). When round 1 completes,
# we scan _round_texts[0] against these and collect the ones present —
# the continuation prompt then lists them as "already done, don't redo."
_PREAMBLE_MARKERS = [
    ("DEEP THINK preamble",        re.compile(r'^\s*#{1,3}\s*DEEP\s+THINK\b', re.IGNORECASE | re.MULTILINE)),
    ("REAL GOAL / INTENT section", re.compile(r'^\s*#{1,4}\s*(?:[A-D]\.\s+)?(?:THE\s+)?REAL\s+(?:GOAL|INTENT)\b', re.IGNORECASE | re.MULTILINE)),
    ("HARDEST UNKNOWN section",    re.compile(r'^\s*#{1,4}\s*(?:[A-D]\.\s+)?(?:THE\s+)?HARDEST\s+UNKNOWN\b', re.IGNORECASE | re.MULTILINE)),
    ("PRE-MORTEM section",         re.compile(r'^\s*#{1,4}\s*(?:[A-D]\.\s+)?PRE-?MORTEM\b', re.IGNORECASE | re.MULTILINE)),
    ("APPROACHES / ARCHITECTURES", re.compile(r'^\s*#{1,4}\s*(?:[A-D]\.\s+)?(?:\d+-?\d*\s+)?(?:APPROACHES|ARCHITECTURES|SUBSTANTIVELY\s+DIFFERENT)\b', re.IGNORECASE | re.MULTILINE)),
    ("BLIND SPOT section",         re.compile(r'^\s*#{1,4}\s*(?:[A-D]\.\s+)?(?:THE\s+)?BLIND\s+SPOT\b', re.IGNORECASE | re.MULTILINE)),
    ("OPEN QUESTIONS list",        re.compile(r'^\s*#{1,3}\s*OPEN\s+QUESTIONS\b', re.IGNORECASE | re.MULTILINE)),
    ("INTEGRATION CHECKLIST",      re.compile(r'^\s*#{1,4}\s*\d?\.?\s*INTEGRATION\s+CHECKLIST\b', re.IGNORECASE | re.MULTILINE)),
    ("REQUIREMENT restatement",    re.compile(r'^\s*#{1,4}\s*\d?\.?\s*REQUIREMENT\b', re.IGNORECASE | re.MULTILINE)),
    ("PLAN OF EDITS",              re.compile(r'^\s*#{1,4}\s*\d?\.?\s*PLAN\s+OF\s+EDITS\b', re.IGNORECASE | re.MULTILINE)),
    ("WHAT COULD GO WRONG",        re.compile(r'^\s*#{1,4}\s*\d?\.?\s*WHAT\s+COULD\s+GO\s+WRONG\b', re.IGNORECASE | re.MULTILINE)),
    ("WHAT MUST BE TRUE",          re.compile(r'^\s*#{1,4}\s*\d?\.?\s*WHAT\s+MUST\s+BE\s+TRUE\b', re.IGNORECASE | re.MULTILINE)),
    ("EVIDENCE PLAN",              re.compile(r'^\s*#{1,4}\s*\d?\.?\s*EVIDENCE\s+PLAN\b', re.IGNORECASE | re.MULTILINE)),
]


def _detect_preamble_sections(text: str) -> list[str]:
    """Return the display names of every preamble section detected in `text`.

    Called once after round 1 to remember which deep-think sections the
    model has already completed. The continuation prompt quotes the list
    back as "✓ already done — do NOT redo these" so the model resumes
    instead of restarting its reasoning.
    """
    if not text:
        return []
    found = []
    for name, pat in _PREAMBLE_MARKERS:
        if pat.search(text):
            found.append(name)
    return found


def _build_continue_prompt(
    base_prompt: str,
    round_history_texts: list[str],
    round_num: int,
    max_rounds: int,
    preamble_done: list[str],
) -> str:
    """Construct the prompt for the round AFTER a [CONTINUE][CONFIRM_CONTINUE]
    signal. The model needs to keep writing its output (plan, review, etc.)
    without firing tools and without restarting its reasoning.

    The prompt design:
      1. A loud "CONTINUATION MODE" banner at the very top — this comes
         BEFORE the original task prompt so the model reads it first.
      2. The list of preamble sections already completed in round 1.
      3. The round-by-round history so the model sees where it left off.
      4. An explicit "resume from the last sentence" instruction.

    No CONTEXT MANIFEST, no RESULTS YOU REQUESTED — there were no tools.
    """
    rounds_left = max_rounds - round_num
    # Flow prior output as one continuous stream — no round labels.
    # The model reads its own previous response and continues writing
    # from the last sentence. The horizontal rule between rounds is
    # minimal: a signal that streaming paused/resumed, not a banner.
    history_block = "\n\n────────\n\n".join(round_history_texts)

    # The system prompt is kept intact so the model still has every
    # rule, role description, and signal definition it needs — only the
    # framing around its own prior output changes. The work-so-far is
    # streamed as one continuous narrative (no round labels) so it
    # feels like one ongoing response, not a series of restarts.
    return f"""{base_prompt}

══════════════════════════════════════════════════════════════════════
YOUR WORK SO FAR (continuous — you signaled [CONTINUE] for more space)
══════════════════════════════════════════════════════════════════════
{history_block}

──────────────────────────────────────────────────────────────────────
↓ Continue writing from where you stopped. Same response, same thought
  stream — just more space. No tools this round (you said you don't
  need any). When you finish, end with [DONE][CONFIRM_DONE], or signal
  another [CONTINUE][CONFIRM_CONTINUE] if you still need more space.
  Budget: {rounds_left} round(s) remain.
{("  (Already-written sections — don't restate, but revise if needed: "
  + ", ".join(preamble_done[:3])
  + ("…" if len(preamble_done) > 3 else "") + ")") if preamble_done else ""}
──────────────────────────────────────────────────────────────────────"""


# ─── Quote / Edit-block masking ─────────────────────────────────────────────
# Models often DISCUSS tool tags in prose ("I'll then [KEEP: file 50-80]")
# or copy them inside fenced code blocks while explaining a plan. The naive
# regex extractors used to fire on those, sending the model into a loop where
# every round it explained that it was about to call a tool, and the system
# went and called it again. The model never gets a chance to think.
#
# We mask out tool-tag-shaped substrings that appear inside:
#   1. Backtick-quoted spans (`...` and ```...```)
#   2. `=== EDIT: ... === ... [/REPLACE]` (or [/INSERT]) blocks — tags inside
#      an open edit block are file CONTENT being inserted, not tool calls.
#   3. Lines explicitly marked with the literal escape "\["  (model
#      convention: write `\[KEEP: ...]` to mention without invoking).
#
# Mask = replace every '[' with '\x00' so tag regexes don't match.  We never
# show the masked text to anyone — just feed it through extractors.

_FENCED_CODE_BLOCK = re.compile(r'```.*?```', re.DOTALL)
_INLINE_BACKTICK = re.compile(r'`[^`\n]+`')
_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
# Deliberate tool-use blocks: [tool use]...[/tool use]
# When ANY such block is present in the response, ONLY tags inside these
# blocks are executed — everything outside is treated as explanatory text.
# This prevents accidental/hallucinated tool calls.
_TOOL_USE_BLOCK = re.compile(r'\[tool use\](.*?)\[/tool use\]', re.DOTALL | re.IGNORECASE)
# Edit/code-writing blocks — content inside is CODE, not tool calls.
# Each pattern covers one form of code writing the coder can produce.
# All are masked so tool tags inside written code never fire accidentally.

# === EDIT: ... === end FILE === (full file creation).
# Body MUST stop at the next section header OR `=== END FILE ===`. The old
# pattern `.*?===\s*END\s+FILE\s*===` was a single lazy span — if a FILE
# block was missing its terminator, masking would consume the entire rest
# of the response, swallowing every tool tag after that point.
_EDIT_FILE_SPAN = re.compile(
    r'===\s*FILE:.*?'
    r'(?:===\s*END\s+FILE\s*===|(?====\s*(?:EDIT|FILE):))',
    re.DOTALL | re.IGNORECASE,
)
# [SEARCH]...[/SEARCH] — code the coder is searching for
_SEARCH_BLOCK = re.compile(r'\[SEARCH[^\]]*\](.*?)\[/SEARCH\]', re.DOTALL | re.IGNORECASE)
# [REPLACE]...[/REPLACE] — replacement code
_REPLACE_BLOCK = re.compile(r'\[REPLACE[^\]]*\](.*?)\[/REPLACE\]', re.DOTALL | re.IGNORECASE)
# [INSERT AFTER LINE N]...[/INSERT] — inserted code
_INSERT_BLOCK = re.compile(r'\[INSERT[^\]]*\](.*?)\[/INSERT\]', re.DOTALL | re.IGNORECASE)
# Legacy: === EDIT/FILE: ... <terminator>. Terminator is whichever comes
# first: a closing [/REPLACE] / [/INSERT], or `=== END FILE ===`, or the
# start of the NEXT section. Without the next-section fallback an EDIT
# block missing its closer ate the rest of the response and silently
# masked legitimate tool tags after it.
_EDIT_BLOCK_SPAN = re.compile(
    r'===\s*(?:EDIT|FILE):.*?'
    r'(?:'
        r'\[/REPLACE\]|\[/INSERT\]|===\s*END\s+FILE\s*==='
        r'|(?====\s*(?:EDIT|FILE):)'
    r')',
    re.DOTALL | re.IGNORECASE,
)
_BACKSLASH_BRACKET = re.compile(r'\\\[')


def _mask_quoted_tags_core(text: str, enforce_tool_use_blocks: bool) -> str:
    """Inner mask: applies backtick / fenced / think / edit-block / escape
    masking to `text`. Optionally also applies the [tool use] block
    enforcement (mask every `[` outside [tool use]...[/tool use] regions).

    The reason this is split into a parameter:
      • For TAG EXTRACTION ([CODE:], [REFS:], etc.), the [tool use] block
        enforcement is correct — only deliberately wrapped tags fire.
      • For SIGNAL DETECTION ([STOP][CONFIRM_STOP] et al.), the [tool use]
        enforcement is WRONG — the signal is supposed to go OUTSIDE the
        block (right after [/tool use]). Masking outside-of-block `[`
        chars hides the signal from the runtime, and the model thinks
        nothing fired. Use enforce_tool_use_blocks=False for signals.
    """
    if not text or '[' not in text:
        return text

    masked = list(text)

    def _blank(start: int, end: int) -> None:
        for i in range(start, min(end, len(masked))):
            if masked[i] == '[':
                masked[i] = '\x00'

    # 0. <think>...</think> blocks — model's internal reasoning.
    for m in _THINK_BLOCK.finditer(text):
        _blank(m.start(), m.end())

    # 1. Fenced code blocks (```...```)
    for m in _FENCED_CODE_BLOCK.finditer(text):
        _blank(m.start(), m.end())

    # 2. Inline backtick spans (`...`)
    for m in _INLINE_BACKTICK.finditer(text):
        _blank(m.start(), m.end())

    # 3. Code-writing blocks — mask all forms where the model writes actual code.
    for pattern in (_EDIT_FILE_SPAN, _SEARCH_BLOCK, _REPLACE_BLOCK,
                    _INSERT_BLOCK, _EDIT_BLOCK_SPAN):
        for m in pattern.finditer(text):
            _blank(m.start(), m.end())

    # 3b. Legacy edit block span
    for m in _EDIT_BLOCK_SPAN.finditer(text):
        _blank(m.start(), m.end())

    # 4. Explicit escape: `\[TAG: ...]` → mask just the leading `[`
    for m in _BACKSLASH_BRACKET.finditer(text):
        idx = m.end() - 1
        if 0 <= idx < len(masked) and masked[idx] == '[':
            masked[idx] = '\x00'

    # 5. [tool use]...[/tool use] enforcement (only when requested).
    if enforce_tool_use_blocks:
        tool_use_blocks = list(_TOOL_USE_BLOCK.finditer(text))
        if tool_use_blocks:
            inside = set()
            for m in tool_use_blocks:
                inside.update(range(m.start(1), m.end(1)))
            for i in range(len(masked)):
                if masked[i] == '[' and i not in inside:
                    masked[i] = '\x00'

    return ''.join(masked)


def _mask_quoted_tags(text: str) -> str:
    """FULL mask for tag extraction — applies all rules including
    [tool use] block enforcement. Only deliberately-wrapped tool tags
    survive this mask. Use this when extracting [CODE:], [REFS:], etc.
    """
    return _mask_quoted_tags_core(text, enforce_tool_use_blocks=True)


def _mask_for_signals(text: str) -> str:
    """Signal-detection mask — applies backtick / fenced / escape rules
    so the model can SAFELY discuss [STOP][CONFIRM_STOP] in prose, but
    does NOT apply [tool use] block enforcement. That second rule was
    causing two-tag signals written OUTSIDE [tool use] blocks (the
    canonical position — right after [/tool use]) to be masked out,
    which made the runtime miss the signal entirely and the model
    hallucinate tool results that never came.
    """
    return _mask_quoted_tags_core(text, enforce_tool_use_blocks=False)


# Module-level: these patterns never change — compile once, not per call.
# 1. Pure line-range patterns like "339-342" — [SEARCH: N-M] anchored edit
# 2. File paths like "ui/index.html" — [SEARCH: filepath] edit reference
# Routing these to ripgrep produces garbage and loops the model.
_SEARCH_LINE_RANGE = re.compile(r'^\d+\s*-\s*\d+$')
_SEARCH_FILE_PATH = re.compile(r'\.\w{1,5}$')


def extract_search_tags(text: str) -> list[str]:
    masked = _mask_quoted_tags(text)
    results = []
    for q in SEARCH_TAG.findall(masked):
        clean, _ = _strip_label(q)
        stripped = clean.strip()
        if _SEARCH_LINE_RANGE.match(stripped):
            continue  # anchored edit syntax [SEARCH: 45-49]
        if _SEARCH_FILE_PATH.search(stripped) and ' ' not in stripped:
            continue  # file path like "ui/index.html", not a search query
        results.append(q)
    return results

def extract_websearch_tags(text: str) -> list[str]:
    return WEBSEARCH_TAG.findall(_mask_quoted_tags(text))

def extract_detail_tags(text: str) -> list[str]:
    return DETAIL_TAG.findall(_mask_quoted_tags(text))

def extract_code_tags(text: str) -> list[str]:
    return CODE_TAG.findall(_mask_quoted_tags(text))

def extract_refs_tags(text: str) -> list[str]:
    return REFS_TAG.findall(_mask_quoted_tags(text))

def extract_purpose_tags(text: str) -> list[str]:
    return PURPOSE_TAG.findall(_mask_quoted_tags(text))

def extract_semantic_tags(text: str) -> list[str]:
    return SEMANTIC_TAG.findall(_mask_quoted_tags(text))

def extract_lsp_tags(text: str) -> list[str]:
    return LSP_TAG.findall(_mask_quoted_tags(text))

def extract_knowledge_tags(text: str) -> list[str]:
    return KNOWLEDGE_TAG.findall(_mask_quoted_tags(text))

def extract_keep_tags(text: str) -> list[str]:
    return KEEP_TAG.findall(_mask_quoted_tags(text))

def extract_discard_tags(text: str) -> list[str]:
    """Extract #labels from [DISCARD: #label] tags."""
    return DISCARD_TAG.findall(_mask_quoted_tags(text))

def has_tool_tags(text: str) -> bool:
    masked = _mask_quoted_tags(text)
    return bool(SEARCH_TAG.search(masked) or WEBSEARCH_TAG.search(masked)
                or DETAIL_TAG.search(masked) or CODE_TAG.search(masked)
                or REFS_TAG.search(masked) or PURPOSE_TAG.search(masked)
                or SEMANTIC_TAG.search(masked)
                or LSP_TAG.search(masked) or KNOWLEDGE_TAG.search(masked)
                or KEEP_TAG.search(masked) or DISCARD_TAG.search(masked))


# ─── Tool Runners ────────────────────────────────────────────────────────────

async def _run_code_searches(patterns: list[str], project_root: str) -> str:
    """Run ripgrep code searches. Returns formatted results."""
    from tools.codebase import search_code, format_search_results

    output_parts = []
    for pattern in patterns:
        status(f"    Code search: {pattern}")
        results = search_code(pattern, project_root)
        if results:
            output_parts.append(f"\n=== Code search: '{pattern}' ===")
            output_parts.append(format_search_results(results))
        else:
            output_parts.append(f"\n=== Code search '{pattern}': no matches ===")
    return "\n".join(output_parts)


async def _run_web_searches(queries: list[str]) -> str:
    """Run web searches. Returns formatted results."""
    output_parts = []
    for query in queries:
        status(f"    Web search: {query}")
        try:
            from tools.search import web_search
            results = await web_search(query, max_results=3)
            if results:
                output_parts.append(f"\n=== Web search: '{query}' ===")
                for r in results:
                    title = r.get("title", "")
                    content = r.get("content", "")[:500]
                    url = r.get("url", "")
                    output_parts.append(f"  {title}")
                    if url:
                        output_parts.append(f"  URL: {url}")
                    if content:
                        output_parts.append(f"  {content}")
                    output_parts.append("")
            else:
                output_parts.append(f"\n=== Web search '{query}': no results ===")
        except Exception as e:
            warn(f"Web search failed for '{query}': {e}")
            output_parts.append(f"\n=== Web search '{query}': error — {e} ===")
    return "\n".join(output_parts)


# ─── Detail Lookup ───────────────────────────────────────────────────────────

def _run_detail_lookups(section_names: list[str], detailed_map: str) -> str:
    """Look up sections from the detailed code map."""
    from tools.code_index import get_detail_section

    output_parts = []
    for name in section_names:
        status(f"    Detail lookup: {name}")
        section = get_detail_section(detailed_map, name)
        output_parts.append(f"\n=== Detail: '{name}' ===\n{section}")
    return "\n".join(output_parts)


# ─── Code File Reader ───────────────────────────────────────────────────────

def _parse_code_arg(raw: str) -> tuple[str, list[tuple[int, int]] | None]:
    """Parse a [CODE: ...] argument into (filepath, optional_line_ranges).

    Handles:
      [CODE: ui/server.py]           → ("ui/server.py", None)
      [CODE: ui/server.py 87-95]     → ("ui/server.py", [(87, 95)])
      [CODE: ui/server.py 87-95, 200-250]  → ("ui/server.py", [(87, 95), (200, 250)])
      [CODE: main.py 390-505]        → ("main.py", [(390, 505)])
    """
    raw = raw.strip()
    # Match a trailing sequence of "N-M" ranges (optionally comma-separated)
    # after the filepath.  The filepath itself never contains digits-dash-digits
    # as a trailing token.
    range_pat = re.compile(r'\s+((?:\d+\s*-\s*\d+)(?:\s*,\s*\d+\s*-\s*\d+)*)\s*$')
    m = range_pat.search(raw)
    if not m:
        return raw, None
    filepath = raw[:m.start()].strip()
    ranges = []
    for rng in re.findall(r'(\d+)\s*-\s*(\d+)', m.group(1)):
        ranges.append((int(rng[0]), int(rng[1])))
    return filepath, ranges if ranges else None


# Patterns for skeleton extraction — matches top-level structural lines
# across the languages we usually see. Each entry: (regex, label).
# The regex must capture an identifier or a useful signature fragment.
# We anchor on column 0 (top-level) plus 1 level of indent (4 spaces or
# a tab) to also pick up methods and class-level functions.
_SKELETON_PATTERNS = [
    # Python
    (re.compile(r'^(?:    |\t)?(?:async\s+)?def\s+(\w+)\s*\(', re.MULTILINE), 'def'),
    (re.compile(r'^(?:    |\t)?class\s+(\w+)', re.MULTILINE), 'class'),
    (re.compile(r'^([A-Z_][A-Z0-9_]{2,})\s*=', re.MULTILINE), 'CONST'),
    # JavaScript / TypeScript
    (re.compile(r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)', re.MULTILINE), 'function'),
    (re.compile(r'^(?:export\s+)?class\s+(\w+)', re.MULTILINE), 'class'),
    (re.compile(r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=', re.MULTILINE), 'const'),
    # Markdown / reST headers
    (re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE), 'header'),
]


def _build_file_skeleton(all_lines: list[str], max_items: int = 200) -> str:
    """Return a compact skeleton of a file: top-level / one-indent
    definitions with their line numbers. Used when [CODE:] is called on
    a file too large to return in full — the skeleton lets the model
    decide which line ranges to ask for via [KEEP:].

    Output format (one per line):
      LNNNN  def function_name
      LNNNN  class ClassName
      LNNNN  CONST_NAME
      LNNNN  ## Section Header

    Caps the number of items at `max_items` to keep the skeleton tiny —
    the goal is to fit in <2k tokens even for a 10k-line file.
    """
    content = "\n".join(all_lines)
    items: list[tuple[int, str]] = []  # (line_number, label_text)

    for pattern, kind in _SKELETON_PATTERNS:
        for m in pattern.finditer(content):
            # Compute line number from match position
            line_no = content.count('\n', 0, m.start()) + 1
            if kind == 'header':
                level, text = m.group(1), m.group(2).strip()
                items.append((line_no, f"{level} {text[:80]}"))
            elif kind == 'CONST':
                items.append((line_no, f"CONST {m.group(1)}"))
            else:
                items.append((line_no, f"{kind} {m.group(1)}"))

    # De-duplicate by (line_no, label) and sort by line number
    seen: set[tuple[int, str]] = set()
    unique = []
    for ln, lbl in sorted(items, key=lambda t: t[0]):
        key = (ln, lbl)
        if key in seen:
            continue
        seen.add(key)
        unique.append((ln, lbl))

    if len(unique) > max_items:
        # Sample evenly across the file so the user sees structure end-to-end
        step = len(unique) / max_items
        unique = [unique[int(i * step)] for i in range(max_items)]

    if not unique:
        return "(no top-level definitions detected — request a [KEEP:] range)"

    return "\n".join(f"  L{ln:<6} {lbl}" for ln, lbl in unique)


def _run_code_reads(
    filepaths: list[str], project_root: str,
    viewed_versions: "dict[str, str] | None" = None,
) -> str:
    """Read source code files from the sandbox.

    Always reads from .jarvis_sandbox/ — that's the working copy where
    all edits are applied. The real project is untouched.

    Supports optional line-range arguments:
      [CODE: path N-M]        → return only lines N through M
      [CODE: path N-M, A-B]   → return multiple ranges

    If `viewed_versions` is provided, the content of every successfully-read
    file is recorded there (keyed by filepath). This is what the model just
    saw, so any [REPLACE LINES X-Y] edits the model writes after this read
    have line numbers relative to THIS content. The on_stop callback uses
    that snapshot as the basis for line edits, instead of whatever the
    file looks like at apply time (which may have changed via earlier
    mid-stream edits in the same response).
    """
    import os
    from tools.codebase import read_file, norm_path, add_line_numbers

    KEEP_HINT_THRESHOLD = 400   # lines — recommend KEEP above this
    KEEP_FORCE_THRESHOLD = 1500 # lines — REQUIRE KEEP above this; full-file
                                # [CODE:] returns a skeleton view instead.
                                # Why: workflows/code.py (5773 lines, ~60k tokens)
                                # is sometimes already in the prompt as
                                # {file_content}, and reading it back via [CODE:]
                                # doubled the context to ~120k of file content +
                                # 3000-line prompt + history → blew past the model
                                # context limit (z-ai/glm5 ~200k) with HTTP 400
                                # "requested 0 output tokens" errors. Returning a
                                # skeleton keeps [CODE:] safe for any file size.
    sandbox_dir = os.path.join(project_root, ".jarvis_sandbox")

    output_parts = []
    for raw_fpath in filepaths:
        # Parse optional line ranges from the argument
        fpath, line_ranges = _parse_code_arg(raw_fpath)
        fpath = norm_path(fpath.strip())
        if line_ranges:
            range_str = ", ".join(f"{a}-{b}" for a, b in line_ranges)
            status(f"    Reading code: {fpath} (lines {range_str})")
        else:
            status(f"    Reading code: {fpath}")

        content = None
        source = None        # "sandbox" or "project" — tracked for the header
        sandbox_exists = False

        # The SANDBOX is the canonical post-edit state. We always read it
        # first. Silently falling back to the project root file when the
        # sandbox state looks "weird" (empty, starts with `[`) was the
        # bug behind the 19-round step-3 loop: bad edits truncated the
        # sandbox file → fallback served the original 84-line project
        # file → model thought "edit didn't apply" → retried forever.
        # Now we ONLY fall back when the sandbox file genuinely does
        # not exist, and we surface every other condition as an explicit
        # status the model can act on.
        sandbox_path = os.path.join(sandbox_dir, fpath)
        if os.path.isfile(sandbox_path):
            sandbox_exists = True
            try:
                with open(sandbox_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                source = "sandbox"
            except Exception as e:
                # Read failure on the sandbox file is unusual and worth
                # surfacing — don't mask it by serving the original.
                output_parts.append(
                    f"\n=== Code: {fpath} — SANDBOX READ ERROR: {e} ===\n"
                    f"The sandbox copy at {sandbox_path} exists but cannot be read.\n"
                    f"This usually means a prior edit corrupted the file's encoding.\n"
                    f"Recovery: write [REVERT FILE: {fpath}] to restore the pre-edit\n"
                    f"snapshot, then plan the correct edit from clean state.\n"
                )
                continue
        else:
            # Sandbox doesn't have the file at all. Fall back to project root
            # — this is legitimate (sandbox lazy-loads files on first reference,
            # so a file the workflow hasn't touched yet only lives at project_root).
            full_path = os.path.join(project_root, fpath)
            content = read_file(full_path)
            source = "project"

        # Empty-sandbox-file guard: a sandbox file that is exactly empty
        # almost always indicates a destructive edit (the model wrote a
        # `=== FILE: ... ===` that produced no output, or a SEARCH/REPLACE
        # that obliterated the body). Surface this LOUDLY instead of
        # falling back to the project file — the model needs to know the
        # damage so it can REVERT.
        if sandbox_exists and content is not None and content == "":
            output_parts.append(
                f"\n=== Code: {fpath} — SANDBOX FILE IS EMPTY (0 bytes) ===\n"
                f"⛔ The sandbox copy of {fpath} is now empty. This is almost\n"
                f"   certainly the result of a destructive edit (e.g. a `=== FILE:`\n"
                f"   rewrite that produced no body, or a SEARCH/REPLACE that\n"
                f"   matched the entire file and replaced it with nothing).\n"
                f"\n"
                f"RECOVERY OPTIONS:\n"
                f"  1. [REVERT FILE: {fpath}]   — pop the pre-edit snapshot off\n"
                f"     the undo stack and restore the file. Do this BEFORE your\n"
                f"     next [STOP][CONFIRM_STOP]. After revert, plan the correct\n"
                f"     edit from the restored state.\n"
                f"  2. If the emptying was intentional (rare), continue and write\n"
                f"     fresh content with a new === FILE: {fpath} === block.\n"
                f"\n"
                f"The runtime is NOT silently falling back to the project file.\n"
                f"The sandbox is canonical — what you see here is what the next\n"
                f"step's coder will see.\n"
            )
            continue

        # Binary / unreadable files return a [... — skipped] string — treat as missing.
        # Only the literal `[BINARY` / `[READ ERROR` / `[FILE NOT FOUND` prefixes from
        # the read_file helper count as failures; legitimate files whose content
        # happens to start with `[` (JSON arrays, TOML arrays of tables, Lua tables,
        # etc.) must NOT be rejected — that was the silent "FILE NOT FOUND" bug on
        # any list-shaped JSON.
        _READ_FAIL_PREFIXES = ("[BINARY", "[READ ERROR", "[FILE NOT FOUND", "[ERROR")
        if content and any(content.startswith(p) for p in _READ_FAIL_PREFIXES):
            output_parts.append(f"\n=== Code: {fpath} — {content.strip()} ===")
            continue

        if content is not None and content != "" and not any(
            content.startswith(p) for p in _READ_FAIL_PREFIXES
        ):
            # Record the FULL file version the model is about to see — line
            # numbers in any subsequent [REPLACE LINES] are relative to this.
            if viewed_versions is not None:
                viewed_versions[fpath] = content

            all_lines = content.split('\n')
            total_lines = len(all_lines)

            if line_ranges:
                # Return only the requested line ranges with correct numbering.
                # Format matches add_line_numbers: `iN|{code} {lineno}` (single
                # space). Stays consistent with KEEP output so the model sees
                # ONE format end-to-end.
                selected_parts = []
                for start, end in line_ranges:
                    start = max(1, start)
                    end = min(total_lines, end)
                    slice_lines = all_lines[start - 1:end]
                    renumbered = []
                    for i, line in enumerate(slice_lines):
                        expanded = line.expandtabs(4)
                        stripped = expanded.lstrip(' ')
                        n_indent = len(expanded) - len(stripped)
                        renumbered.append(f"i{n_indent}|{stripped} {start + i}")
                    selected_parts.append('\n'.join(renumbered))

                range_str = ", ".join(f"{a}-{b}" for a, b in line_ranges)
                combined = '\n'.join(selected_parts)
                source_tag = source or "sandbox"
                output_parts.append(
                    f"\n=== Code: {fpath} (lines {range_str} of {total_lines} — from {source_tag}) ===\n{combined}"
                )
            else:
                if total_lines > KEEP_FORCE_THRESHOLD:
                    # Huge file — return a skeleton instead of the full body.
                    # Loading the full file would blow the model's context.
                    skeleton = _build_file_skeleton(all_lines)
                    output_parts.append(
                        f"\n=== Code: {fpath} ({total_lines} lines — SKELETON ONLY) ===\n"
                        f"⛔ This file is too large to return in full "
                        f"({total_lines} lines > {KEEP_FORCE_THRESHOLD} threshold). "
                        f"Loading the entire file would overflow the model's "
                        f"context window. Below is the file's SKELETON — "
                        f"top-level definitions with their line numbers. To read "
                        f"the body of any item, follow up with a narrowed read:\n"
                        f"  [tool use] [KEEP: {fpath} START-END] [/tool use]\n"
                        f"  [STOP]\n  [CONFIRM_STOP]\n"
                        f"where START-END is the line range around the item you "
                        f"need. Pick ≤ 300 lines total across all ranges.\n"
                        f"\n{skeleton}\n"
                    )
                    # Don't record skeleton as viewed_versions — line-anchored
                    # edits against a skeleton view would be wrong. Force the
                    # model to KEEP first, which DOES record the real content.
                    if viewed_versions is not None:
                        viewed_versions.pop(fpath, None)
                else:
                    numbered = add_line_numbers(content)
                    large_note = ""
                    if total_lines > KEEP_HINT_THRESHOLD:
                        large_note = (
                            f"⚠ Big file ({total_lines} lines) — "
                            f"[KEEP: {fpath} X-Y, A-B] recommended to select only "
                            f"the lines you need.\n"
                        )
                    # Annotate the source so the model knows whether it's
                    # looking at the sandbox (post-edit state) or the
                    # project root (untouched original). When the sandbox
                    # hasn't seen this file yet, "project" is normal —
                    # but if a file is being EDITED and shows "project",
                    # that's a sign edits aren't being applied.
                    source_tag = source or "sandbox"
                    output_parts.append(
                        f"\n=== Code: {fpath} ({total_lines} lines — from {source_tag}) ===\n"
                        f"{large_note}"
                        f"{numbered}"
                    )
        else:
            output_parts.append(f"\n=== Code: {fpath} — FILE NOT FOUND ===")
    return "\n".join(output_parts)


# ─── KEEP Handler ────────────────────────────────────────────────────────────

async def _run_keep(
    keep_args: list[str], project_root: str,
    persistent_lookups: dict[str, str],
    research_cache: dict | None = None,
    viewed_versions: "dict[str, str] | None" = None,
    on_keep_seen: "Callable[[str, str], None] | None" = None,
) -> str:
    """Process [KEEP: filepath X-Y, A-B] tags.

    1. Parse filepath + line ranges from the tag argument
    2. Find the original file content (from persistent_lookups or disk)
    3. Build filtered view with preserved line numbers
    4. Run auto-RAG on kept lines
    5. REPLACE the CODE entry in persistent_lookups with the filtered view
    6. Return the filtered view + dependency summary

    on_keep_seen, if provided, is called with (canonical_key, raw_arg) for
    each KEEP that actually fires. The caller uses it to register the KEEP
    in its manifest / re-read counter so loop detection works for KEEP
    just like it does for CODE.
    """
    import os
    from tools.codebase import read_file, norm_path
    from workflows.code import _parse_keep_ranges, _filter_by_ranges, _auto_rag

    output_parts = []

    for arg in keep_args:
        arg = arg.strip()

        # Strip optional #label suffix BEFORE range parsing — without this,
        # `[KEEP: foo.py 10-20 #lbl]` would feed "10-20 #lbl" into the range
        # parser and the label survives into the filepath. The label is
        # purely for DISCARD identification; the canonical KEEP key never
        # includes it.
        arg_no_label, _kept_label = _strip_label(arg)

        # Parse: "filepath X-Y, A-B" or "filepath X-Y A-B"
        # The filepath is everything before the first digit-dash-digit pattern.
        # BUT — if the filepath itself contains a `N-M` segment (e.g.
        # `tools/v2-3/foo.py 50-80`), naive parsing splits in the wrong place.
        # Heuristic: also look for an explicit whitespace boundary between
        # filename and ranges; prefer that when present.
        ws_split_match = re.search(
            r'^(\S+\.(?:py|js|ts|jsx|tsx|html|css|json|lean|c|cpp|h|rs|java|go|rb|toml|yaml|yml|md|mjs|cjs|svelte|vue|lua|sh))\s+(.+)$',
            arg_no_label, re.IGNORECASE,
        )
        if ws_split_match:
            filepath = ws_split_match.group(1).strip()
            ranges_text = ws_split_match.group(2).strip()
        else:
            range_match = re.search(r'(\d+)\s*-\s*(\d+)', arg_no_label)
            if not range_match:
                output_parts.append(f"=== KEEP: invalid format '{arg}' — use [KEEP: filepath X-Y, A-B] ===")
                continue
            filepath = arg_no_label[:range_match.start()].strip()
            ranges_text = arg_no_label[range_match.start():]
        filepath = norm_path(filepath)

        status(f"    KEEP: {filepath}")

        # Find original content — check persistent_lookups first
        original_content = None
        norm_key = filepath.strip().lower()
        code_key = f"CODE:{norm_key}"

        # Search persistent_lookups for a matching CODE entry.
        # Match order, MOST-SPECIFIC FIRST:
        #   1. exact key match
        #   2. path is a proper suffix of the KEEP target (the KEEP wrote
        #      the full relative path, the CODE used the basename)
        #   3. path is a proper SUFFIX of the existing key (KEEP used the
        #      basename, CODE used the full path)
        # Bidirectional `endswith` without a SLASH guard used to pick the
        # wrong file when basenames collided (e.g. `foo/bar.py` and
        # `qux/bar.py` both end with `bar.py`). We now require the
        # boundary to fall on a path separator so partial-token matches
        # like `lib.py` ↔ `mylib.py` cannot collide.
        def _suffix_with_sep(longer: str, shorter: str) -> bool:
            if longer == shorter:
                return True
            if not longer.endswith(shorter):
                return False
            cut = len(longer) - len(shorter)
            return cut == 0 or longer[cut - 1] in '/\\'

        matched_key = None
        # Pass 1: exact match.
        if code_key in persistent_lookups:
            matched_key = code_key
        # Pass 2: existing key has the KEEP target as a path-bounded suffix.
        if matched_key is None:
            for key in persistent_lookups:
                if not key.startswith("CODE:"):
                    continue
                key_path = key[5:]
                if _suffix_with_sep(key_path, norm_key):
                    matched_key = key
                    break
        # Pass 3: KEEP target has the existing key as a path-bounded suffix.
        if matched_key is None:
            for key in persistent_lookups:
                if not key.startswith("CODE:"):
                    continue
                key_path = key[5:]
                if _suffix_with_sep(norm_key, key_path):
                    matched_key = key
                    break

        # Read from sandbox first, then fall back to project root
        sandbox_dir = os.path.join(project_root, ".jarvis_sandbox")
        sandbox_path = os.path.join(sandbox_dir, filepath)
        raw_content = None
        if os.path.isfile(sandbox_path):
            try:
                with open(sandbox_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_content = f.read()
            except Exception:
                raw_content = None

        # Only the read_file failure prefixes count as a miss. Legitimate
        # files whose content starts with `[` (JSON arrays, TOML arrays,
        # Lua tables …) MUST NOT trip the rejection — that produced the
        # silent "file not found" loop on list-shaped JSON.
        _READ_FAIL_PREFIXES = ("[BINARY", "[READ ERROR", "[FILE NOT FOUND", "[ERROR")
        def _looks_like_read_failure(s: "str | None") -> bool:
            return bool(s) and any(s.startswith(p) for p in _READ_FAIL_PREFIXES)

        if not raw_content:
            full_path = os.path.join(project_root, filepath)
            raw_content = read_file(full_path)
        if not raw_content or _looks_like_read_failure(raw_content):
            raw_content = read_file(filepath)

        if not raw_content or _looks_like_read_failure(raw_content):
            output_parts.append(f"=== KEEP: file not found '{filepath}' ===")
            continue

        # Record what the model is about to see — KEEP preserves real line
        # numbers, so any subsequent [REPLACE LINES X-Y] anchors to THIS
        # snapshot (not whatever the file looks like at apply time after
        # mid-stream edits).
        if viewed_versions is not None:
            viewed_versions[filepath] = raw_content

        # Parse KEEP ranges
        ranges = _parse_keep_ranges(ranges_text, filepath)
        if not ranges:
            # Try parsing the full arg
            ranges = _parse_keep_ranges(arg, filepath)

        if not ranges:
            output_parts.append(f"=== KEEP: no valid ranges in '{arg}' ===")
            continue

        # Build filtered view
        filtered = _filter_by_ranges(raw_content, ranges, filepath)
        kept_lines = sum(e - s + 1 for s, e in ranges)
        total_lines = raw_content.count('\n') + 1

        # Auto-RAG: find dependencies in kept code
        deps = await _auto_rag(filtered, filepath, project_root, research_cache)

        # Build the replacement result
        replacement = (
            f"\n=== Code: {filepath} (KEPT {kept_lines}/{total_lines} lines, "
            f"line numbers accurate for [REPLACE LINES]) ===\n"
            f"{filtered}\n"
        )
        if deps:
            replacement += f"\n{deps}\n"

        # REPLACE the CODE entry in persistent_lookups
        if matched_key:
            persistent_lookups[matched_key] = replacement
            status(f"    KEEP: {filepath}: {kept_lines}/{total_lines} lines kept, "
                   f"replaced full file in context")
        else:
            # No CODE entry to replace — just add it
            persistent_lookups[code_key] = replacement
            status(f"    KEEP: {filepath}: {kept_lines}/{total_lines} lines kept")

        # Notify the caller so it can register this KEEP in its manifest /
        # re-read counter. Without this, KEEP loop detection never fires
        # because _manifest only sees keys that flow through _store.
        if on_keep_seen is not None:
            keep_key = f"KEEP:{(filepath.strip() + ' ' + ranges_text.strip()).lower()}"
            try:
                on_keep_seen(keep_key, arg)
            except Exception:
                pass

        output_parts.append(replacement)

    return "\n".join(output_parts)


# ─── Reference Search ───────────────────────────────────────────────────────

async def _run_refs_searches(names: list[str], project_root: str) -> str:
    """Ripgrep word-boundary search for all references to a name."""
    from tools.codebase import search_refs

    output_parts = []
    for name in names:
        name = name.strip()
        status(f"    Refs search: {name}")
        result = search_refs(name, project_root)
        output_parts.append(result)
    return "\n".join(output_parts)


async def _run_lsp_searches(names: list[str], project_root: str) -> str:
    """LSP semantic search — finds dependencies, types, indirect references."""
    output_parts = []
    for name in names:
        name = name.strip()
        status(f"    LSP search: {name}")
        try:
            from tools.lsp import lsp_find_references
            result = await lsp_find_references(name, project_root)
            if result:
                output_parts.append(result)
            else:
                output_parts.append(f"=== LSP for '{name}': no LSP server available, use [REFS: {name}] instead ===")
        except Exception as e:
            output_parts.append(f"=== LSP for '{name}': failed ({str(e)[:80]}), use [REFS: {name}] instead ===")
    return "\n".join(output_parts)


def _run_purpose_lookups(categories: list[str], purpose_map: str, project_root: str) -> str:
    """Look up purpose categories and return actual code snippets with context."""
    from tools.code_index import get_purpose_snippets

    output_parts = []
    for cat in categories:
        status(f"    Purpose lookup: {cat}")
        result = get_purpose_snippets(purpose_map, cat, project_root)
        output_parts.append(result)
    return "\n".join(output_parts)


def _run_knowledge_lookups(topics: list[str]) -> str:
    """Look up knowledge topics."""
    from knowledge import get_knowledge

    output_parts = []
    for topic in topics:
        status(f"    Knowledge: {topic}")
        result = get_knowledge(topic.strip())
        output_parts.append(result)
    return "\n".join(output_parts)


# ─── Tool Tag Detection (for stream early-stop) ─────────────────────────────

_ALL_TAGS = re.compile(
    r'\[(SEARCH|WEBSEARCH|DETAIL|CODE|REFS|PURPOSE|LSP|KNOWLEDGE|KEEP|DISCARD):\s*.+?\]'
    r'|\[STOP\]\s*\[CONFIRM_STOP\]'
    r'|\[DONE\]\s*\[CONFIRM_DONE\]'
    r'|\[CONTINUE\]\s*\[CONFIRM_CONTINUE\]',
    re.IGNORECASE,
)


def _text_has_complete_tag(text: str) -> bool:
    """Return True if text contains at least one complete tool tag or
    fully-formed two-tag signal. Bare `[STOP]` alone NO LONGER counts —
    the two-tag protocol means bare halves are inert text."""
    return bool(_ALL_TAGS.search(text))


_TOOL_USE_OPEN  = re.compile(r'\[tool use\]',  re.IGNORECASE)
_TOOL_USE_CLOSE = re.compile(r'\[/tool use\]', re.IGNORECASE)


def _autocomplete_tool_blocks(text: str) -> tuple[str, int]:
    """Close any unclosed [tool use] blocks by inserting a [/tool use] tag
    BEFORE the NEXT [tool use] open (or before [STOP]/[DONE]/[CONTINUE], or
    at end of text) for each orphaned opener.

    Walks opens and closes in document order so each missing closer lands at
    the correct boundary instead of being collapsed onto a single position.

    Returns (fixed_text, number_of_blocks_fixed).
    A non-zero count means the model wrote [tool use] but forgot [/tool use].
    """
    opens  = list(_TOOL_USE_OPEN.finditer(text))
    closes = list(_TOOL_USE_CLOSE.finditer(text))
    if len(opens) <= len(closes):
        return text, 0

    # Build interleaved list of events: open / close / signal-or-stop boundary.
    # Each event has (pos, kind). Walk left→right pairing opens to closes;
    # any open without a close gets a synthetic [/tool use] inserted at the
    # first boundary AFTER the open (next open / next signal / end-of-text).
    BoundaryRe = re.compile(
        r'\[tool use\]|\[/tool use\]|\[STOP\]\s*\[CONFIRM_STOP\]'
        r'|\[DONE\]\s*\[CONFIRM_DONE\]|\[CONTINUE\]\s*\[CONFIRM_CONTINUE\]',
        re.IGNORECASE,
    )
    insertions: list[int] = []
    open_stack: list[int] = []
    for m in BoundaryRe.finditer(text):
        tok = m.group(0).lower()
        if tok == '[tool use]':
            if open_stack:
                # Previous open never closed — synthesize close right before
                # THIS open (i.e. at m.start()).
                insertions.append(m.start())
                open_stack.pop()
            open_stack.append(m.start())
        elif tok == '[/tool use]':
            if open_stack:
                open_stack.pop()
        else:
            # signal: any open before this should be closed before the signal.
            while open_stack:
                insertions.append(m.start())
                open_stack.pop()

    # Any remaining unclosed opens close at end of text.
    while open_stack:
        insertions.append(len(text))
        open_stack.pop()

    if not insertions:
        return text, 0

    # Insert from right to left so earlier offsets stay valid.
    insertions.sort()
    fixed = text
    closer = '[/tool use]'
    for pos in reversed(insertions):
        # Ensure we sit on its own boundary — surround with newlines if needed.
        prefix = '' if (pos == 0 or fixed[pos - 1] in '\n ') else '\n'
        suffix = '' if (pos >= len(fixed) or fixed[pos] in '\n ') else '\n'
        fixed = fixed[:pos] + f"{prefix}{closer}{suffix}" + fixed[pos:]
    return fixed, len(insertions)


def _describe_tool_mode(result: str) -> str:
    """Return a short string describing whether block or fallback mode is active."""
    blocks = list(_TOOL_USE_BLOCK.finditer(result))
    if blocks:
        return f"block mode ({len(blocks)} [tool use] block(s))"
    return "bare-tag fallback (model omitted [tool use] wrapper)"


def _tag_summary(
    code_tags, web_tags, detail_tags, file_tags, refs_tags,
    purpose_tags, semantic_tags, lsp_tags, knowledge_tags, keep_tags,
    research_cache: dict | None,
    persistent_lookups: dict,
) -> str:
    """Build a one-line summary of tags found this round with cache annotations."""
    parts = []
    def _note(label: str, tags: list[str], type_key: str) -> None:
        if not tags:
            return
        hits = 0
        if research_cache is not None and type_key not in ("CODE", "KEEP"):
            for t in tags:
                clean, _ = _strip_label(t)
                k = f"{type_key}:{clean.strip().lower()}"
                if k in research_cache or k in persistent_lookups:
                    hits += 1
        hit_str = f" ({hits} cached)" if hits else ""
        parts.append(f"{label}×{len(tags)}{hit_str}")

    _note("CODE",     file_tags,     "CODE")
    _note("REFS",     refs_tags,     "REFS")
    _note("SEARCH",   code_tags,     "SEARCH")
    _note("WEB",      web_tags,      "WEBSEARCH")
    _note("DETAIL",   detail_tags,   "DETAIL")
    _note("PURPOSE",  purpose_tags,  "PURPOSE")
    _note("SEMANTIC", semantic_tags, "SEMANTIC")
    _note("LSP",      lsp_tags,      "LSP")
    _note("KNOW",     knowledge_tags,"KNOWLEDGE")
    _note("KEEP",     keep_tags,     "KEEP")
    return ", ".join(parts) if parts else "(none)"


# ─── Main Tool Call Loop ────────────────────────────────────────────────────

async def call_with_tools(
    model: str,
    prompt: str,
    project_root: str | None = None,
    max_tokens: int = 16384,
    max_rounds: int = 20,
    enable_code_search: bool = True,
    enable_web_search: bool = True,
    detailed_map: str | None = None,
    purpose_map: str | None = None,
    research_cache: dict | None = None,
    log_label: str = "",
    on_stop: "Callable[[str], str | None] | None" = None,
    viewed_versions: "dict[str, str] | None" = None,
    stop_on_tool_block: bool = False,
) -> dict:
    """
    Call a model with mid-thought tool use.

    The AI wraps tool calls in [tool use]...[/tool use] then [STOP].
    Only tags INSIDE [tool use] blocks execute — tags outside are ignored.
    JARVIS runs ALL requested lookups at once and feeds results back.

    Signals:
      [STOP]   → execute tool calls + on_stop callback, continue thinking
      [DONE]   → apply final edits, model is completely finished

    on_stop: optional callback called with full_response when [STOP] fires.
             Used by coders to apply pending edit blocks before tool lookups,
             so [CODE:] reads return the post-edit state.
             The callback MAY return a feedback string describing what
             happened to the edits (which applied, which were skipped and
             why). When it does, the string is prepended to the next-round
             prompt as an "EDIT APPLICATION RESULTS" block, so the model
             gets explicit signal instead of having to infer success by
             re-reading the file. Returning None means "no feedback to add."

    viewed_versions: optional dict updated whenever the model reads a file via
             [CODE: path]. Maps filepath → content the model just saw. Used by
             on_stop to anchor [REPLACE LINES] edits to the version the model
             was actually looking at, instead of whatever the file currently
             is on disk. Without this, a model that views V0, writes [REPLACE
             LINES 22-24], then writes more edits after a mid-stream [STOP]
             would have its V0-relative line numbers applied to the post-STOP
             file (which has different line numbers).

    Tool tags:
      [SEARCH: pattern]       → code search
      [WEBSEARCH: query]      → web search
      [DETAIL: section name]  → detailed code map lookup
      [CODE: path/to/file]    → read actual source code file
      [REFS: name]            → find all definitions, imports, usages
      [PURPOSE: category]     → all code serving a purpose (exact/fuzzy category name)
      [SEMANTIC: description] → vector embedding search over purpose categories, returns top 10 matches
      [DISCARD: #label]       → remove a labeled result from context

    research_cache: shared dict that accumulates all lookup results across
    multiple AI calls. Same tag won't re-run if cached.

    Returns {"model": str, "answer": str, "done": bool, "research": {tag_key: result}}.
    "done" is True when the model explicitly wrote [DONE] — it is NOT present in
    "answer" (stripped before return), so callers must check this flag, not the text.
    """
    full_response = ""
    _done_signaled = False
    current_prompt = prompt

    # [SEARCH:] and [REFS:] use ripgrep on project_root, but ripgrep respects
    # .gitignore and .jarvis_sandbox is in .gitignore — so edits applied to the
    # sandbox are invisible to those tools. When the sandbox exists, search it
    # directly instead, since it contains the live (post-edit) file state.
    _sandbox_dir = os.path.join(project_root, ".jarvis_sandbox") if project_root else None
    _search_root = (
        _sandbox_dir
        if (_sandbox_dir and os.path.isdir(_sandbox_dir))
        else project_root
    )

    # Track this call's research (also writes to shared cache if provided)
    local_research: dict[str, str] = {}
    # Persistent lookup results — survives across rounds. Keyed by "TYPE:arg".
    # When [KEEP:] fires, it REPLACES the corresponding [CODE:] entry, removing
    # the full file from context and inserting only the kept ranges.
    persistent_lookups: dict[str, str] = {}
    # Maps #label → list of TYPE:arg keys, for [DISCARD: #label] support.
    _label_to_keys: dict[str, list[str]] = {}
    # Stall guard: if the model issues only ALREADY-CACHED tools for two
    # rounds in a row, it's spinning — break and let it commit. Tracks the
    # set of tag-keys requested per round.
    _last_round_keys: set[str] = set()
    _stall_rounds: int = 0

    # ── Context manifest — tracks what this model has actually received ──────
    # {key: {"round": int, "tag_type": str, "arg": str}}
    # Only contains tools this model ran or whose shared-cache results it got.
    _manifest: dict[str, dict] = {}

    # Re-read tracker — counts how many times the model has re-issued a CODE
    # or KEEP for the SAME argument across rounds. CODE/KEEP can legitimately
    # re-fire (the file may have changed), but if the model re-requests the
    # IDENTICAL ranges multiple rounds in a row it's stuck in a "let me verify
    # one more time" loop. We escalate warnings to break the loop.
    # Maps "CODE:path" or "KEEP:path 10-20" → count.
    _reread_count: dict[str, int] = {}

    # Per-round response text — used to build tagged round history in prompt.
    _round_texts: list[str] = []  # _round_texts[i] = text produced in round i+1

    # Track which DEEP THINK preamble sections the model has already
    # completed in round 1, so the continuation prompt in rounds 2+ can
    # tell the model "you already wrote these — don't redo them."
    # Populated after round 1 by scanning _round_texts[0] for known
    # section markers.
    _preamble_done: list[str] = []

    # Edit-apply feedback captured from the on_stop callback. Each entry
    # is the string returned by on_stop in a round, recording which edit
    # blocks applied and which were skipped. The most recent entry is
    # injected into the next-round prompt as "EDIT APPLICATION RESULTS"
    # so the model sees explicit success/failure instead of having to
    # infer it from re-reading the file. This was the dominant cause
    # of the step-3 19-round loop on domains/prompts.py: the model
    # wrote 19 nearly-identical edits with no feedback that one of them
    # had already landed, then accidentally landed several duplicates.
    _last_edit_feedback: str | None = None

    # Track edit attempts per file across rounds for stall detection.
    # Each entry: list[bool] of round-by-round "did any edit on this
    # file apply successfully?" When the recent N attempts on a file
    # are all failures, we surface a stop-flailing nudge.
    _edit_attempts_per_file: dict[str, list[bool]] = {}

    def _stop_check(accumulated: str) -> bool:
        # STOP, DONE, and CONTINUE are all two-tag signals.
        # Use _mask_for_signals (NOT _mask_quoted_tags) — the signal
        # canonical position is OUTSIDE [tool use] blocks (right after
        # [/tool use]). The full _mask_quoted_tags masks every '[' outside
        # [tool use] blocks when any block is present — which used to
        # eat the signal and make the runtime miss [STOP][CONFIRM_STOP]
        # entirely. The lighter mask still protects against backtick /
        # fenced-block discussion of the syntax.
        masked = _mask_for_signals(accumulated)
        if DONE_TAG.search(masked):
            return True
        if STOP_TAG.search(masked):
            return True
        if CONTINUE_TAG.search(masked):
            return True

        # `stop_on_tool_block` is accepted for back-compat with call sites
        # that still pass it, but it intentionally does NOTHING here.
        # Stopping on `[/tool use]` alone used to cut the stream BEFORE the
        # model finished writing `[CONFIRM_STOP]`, causing tool execution
        # without a real signal. The two-tag pair is now the ONLY trigger.
        return False

    _empty_streak = 0
    for round_num in range(1, max_rounds + 1):
        result = await call_with_retry(
            model, current_prompt, max_tokens=max_tokens,
            stop_check=_stop_check,
            log_label=f"{log_label} — R{round_num}" if log_label else f"R{round_num}",
        )

        # ── Empty-response guard ─────────────────────────────────────
        # Some models return an empty string under load (no reasoning, no
        # visible content). Without this, the loop kept calling the same
        # model 30 times for nothing — every call writes a header to the
        # log and burns rate-limit. Two empty responses in a row → break.
        if not result or not result.strip():
            _empty_streak += 1
            if _empty_streak >= 2:
                warn(f"  [{model.split('/')[-1]}] round {round_num}: empty response "
                     f"× {_empty_streak} — breaking tool loop")
                break
            warn(f"  [{model.split('/')[-1]}] round {round_num}: empty response "
                 f"(streak {_empty_streak}) — nudging model")
            current_prompt = (
                current_prompt
                + "\n\nNote: your previous response was empty. "
                  "Please write your answer now."
            )
            continue
        _empty_streak = 0

        # ── Cross-round signal detection ──────────────────────────────
        # When streaming gets cut mid-signal (e.g. max_tokens hits at
        # `[STOP]\n[CONFIRM_` and the rest arrives on the next call),
        # checking only the current round's text would miss the signal.
        # To handle this, we check signals against the BRIDGE between
        # the previous round's tail and the current round's head, then
        # fall back to the current round on its own.
        #
        # Algorithm:
        #   1. Compute `bridge` = last 64 chars of prev round + current
        #      result. 64 chars is enough to span any signal pair (the
        #      longest is [CONTINUE][CONFIRM_CONTINUE] at ~30 chars).
        #   2. Detect signals on the masked bridge.
        #   3. If a signal spans the boundary, mark that and strip the
        #      cross-boundary fragments from both round texts.
        #   4. Also detect signals fully inside current `result` (normal case).
        prev_idx = len(_round_texts) - 1 if _round_texts else -1

        def _current_bridge() -> tuple[str, int, str]:
            """Recompute (masked_bridge, bridge_offset, prev_tail) from the
            CURRENT state of `result` and `_round_texts`. Must be called
            every time `_signal_in_bridge` is consulted — `result` and the
            prev round's text both mutate while we strip signals."""
            tail = ""
            if prev_idx >= 0:
                tail = _round_texts[prev_idx][-64:]
            return _mask_for_signals(tail + result), len(tail), tail

        def _signal_in_bridge(pattern):
            """Return the FIRST bridge-relative match for `pattern`, or None.

            Always recomputes the bridge from the current `result` and
            `_round_texts[prev_idx]` so consecutive calls reflect prior
            consumes. The earlier implementation cached `_masked_bridge`
            at round start, which made `while _signal_in_bridge(STOP_TAG)`
            an infinite loop whenever two signals appeared in the bridge:
            the regex kept returning the same match against a stale mask.
            """
            masked, _offset, _tail = _current_bridge()
            return pattern.search(masked)

        def _consume_bridge_signal(pattern) -> bool:
            """Strip ONE signal (the leftmost match) that may span the
            prev-round / current-round boundary. Updates `_round_texts[-1]`
            and current `result` so the signal text is removed from both
            halves. Returns True if anything was stripped.

            Three cases:
              (1) signal entirely in prev_tail  (s, e <= bridge_offset)
              (2) signal entirely in current    (s >= bridge_offset)
              (3) signal straddles boundary     (s < bridge_offset < e)

            Each call recomputes the bridge first so repeated consumes
            converge — caller may loop on `while _signal_in_bridge(...)`.
            """
            nonlocal result
            masked, bridge_offset, prev_tail = _current_bridge()
            m = pattern.search(masked)
            if not m:
                return False
            s, e = m.start(), m.end()
            if s < bridge_offset and prev_idx >= 0:
                # Signal at least starts in prev_tail.
                prev_text = _round_texts[prev_idx]
                tail_start_abs = len(prev_text) - len(prev_tail)
                prev_start_abs = tail_start_abs + s
                # The signal extends in prev up to min(e, bridge_offset)
                prev_end_in_bridge = min(e, bridge_offset)
                prev_end_abs = tail_start_abs + prev_end_in_bridge
                prev_end_abs = min(prev_end_abs, len(prev_text))
                _round_texts[prev_idx] = (
                    prev_text[:prev_start_abs] + prev_text[prev_end_abs:]
                )
                # Strip the remainder from current result if any.
                if e > bridge_offset:
                    cur_end_in_bridge = e - bridge_offset
                    result = result[cur_end_in_bridge:]
            else:
                # Whole signal inside current result.
                cur_s = s - bridge_offset
                cur_e = e - bridge_offset
                result = result[:cur_s] + result[cur_e:]
            return True

        # ── DONE: finish the loop ─────────────────────────────────────
        if _signal_in_bridge(DONE_TAG):
            _consume_bridge_signal(DONE_TAG)
            # Also remove any trailing STOP/CONTINUE in the bridge.
            while _signal_in_bridge(STOP_TAG):
                _consume_bridge_signal(STOP_TAG)
            while _signal_in_bridge(CONTINUE_TAG):
                _consume_bridge_signal(CONTINUE_TAG)
            result = result.rstrip()
            _round_texts.append(result)
            full_response += result
            _done_signaled = True
            break

        # ── CONTINUE: keep writing without tools ──────────────────────
        if _signal_in_bridge(CONTINUE_TAG):
            _consume_bridge_signal(CONTINUE_TAG)
            while _signal_in_bridge(STOP_TAG):
                _consume_bridge_signal(STOP_TAG)
            result = result.rstrip()
            _round_texts.append(result)
            full_response += result
            current_prompt = _build_continue_prompt(
                base_prompt=prompt,
                round_history_texts=_round_texts,
                round_num=round_num,
                max_rounds=max_rounds,
                preamble_done=_preamble_done,
            )
            status(f"  [{model.split('/')[-1]}] round {round_num}: "
                   f"[CONTINUE] signal — resuming output without tool processing")
            continue

        # ── STOP: apply tools + continue thinking ─────────────────────
        has_stop = bool(_signal_in_bridge(STOP_TAG))
        if has_stop:
            while _signal_in_bridge(STOP_TAG):
                _consume_bridge_signal(STOP_TAG)
        _round_texts.append(result)

        # ── Capture which preamble sections were written in round 1 ──────
        # Once round 1 finishes, scan _round_texts[0] for the section
        # markers our prompts use. The continuation prompt builder will
        # quote this list back to the model in subsequent rounds as
        # "you already completed these — do not redo them."
        if round_num == 1 and not _preamble_done:
            _preamble_done = _detect_preamble_sections(_round_texts[0])

        # ── Bare-tag correction: model wrote [STOP] / [DONE] / [CONTINUE] alone ──
        # In the two-tag signal protocol, a bare half is just text. But the
        # model may have INTENDED it as a signal and missed the protocol.
        # Detect the situation and inject a one-shot correction below.
        # Only fires when NO real signal fired this round.
        _suspected_bare_signal = False
        if not has_stop and not _done_signaled:
            _mask_bare = _mask_quoted_tags(result)
            # Strip real two-tag signal matches first so we don't
            # double-count [STOP] inside [STOP][CONFIRM_STOP] as bare.
            _mask_bare = STOP_TAG.sub('', _mask_bare)
            _mask_bare = DONE_TAG.sub('', _mask_bare)
            _mask_bare = CONTINUE_TAG.sub('', _mask_bare)
            if (_BARE_STOP.search(_mask_bare)
                    or _BARE_DONE.search(_mask_bare)
                    or _BARE_CONTINUE.search(_mask_bare)):
                _suspected_bare_signal = True

        # ── Auto-complete unclosed [tool use] blocks ──────────────────
        # Models like minimax write [tool use] [CODE:...] but forget [/tool use].
        # Without the closing tag, _mask_quoted_tags never activates block
        # enforcement, so tags in explanatory text can fire accidentally.
        # We close them here before extraction so the masker works correctly.
        result, n_autoclosed = _autocomplete_tool_blocks(result)
        if n_autoclosed:
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"auto-closed {n_autoclosed} unclosed [tool use] block(s) "
                f"— model forgot [/tool use]"
            )

        # ── Detect tool tags anywhere in the response ────────────────
        # If the model wrote [CODE: ...] or [REFS: ...] anywhere in its
        # response, we process them.
        code_tags = list(dict.fromkeys(extract_search_tags(result))) if enable_code_search else []
        web_tags = list(dict.fromkeys(extract_websearch_tags(result))) if enable_web_search else []
        detail_tags = list(dict.fromkeys(extract_detail_tags(result))) if detailed_map else []
        file_tags = list(dict.fromkeys(extract_code_tags(result))) if project_root else []
        refs_tags = list(dict.fromkeys(extract_refs_tags(result))) if project_root else []
        purpose_tags = list(dict.fromkeys(extract_purpose_tags(result))) if purpose_map else []
        semantic_tags = list(dict.fromkeys(extract_semantic_tags(result))) if purpose_map else []
        lsp_tags = list(dict.fromkeys(extract_lsp_tags(result))) if project_root else []
        knowledge_tags = list(dict.fromkeys(extract_knowledge_tags(result)))
        keep_tags = list(dict.fromkeys(extract_keep_tags(result))) if project_root else []
        discard_tags = extract_discard_tags(result)

        has_tags = bool(code_tags or web_tags or detail_tags or file_tags
                        or refs_tags or purpose_tags or semantic_tags or lsp_tags
                        or knowledge_tags or keep_tags or discard_tags)

        # ── Dropped-tag detection (visibility for the model) ─────────
        # The full _mask_quoted_tags enforces [tool use] blocks: any tag
        # written OUTSIDE [tool use]...[/tool use] gets masked away and
        # silently ignored. That's correct enforcement, but invisible to
        # the model — which then thinks all its tool calls fired and
        # hallucinates results for the ones that didn't.
        #
        # Detect this case: re-run tag extraction with the LIGHTER mask
        # (no [tool use] enforcement) and compare. Tags that appear in
        # the light extraction but NOT in the strict extraction are
        # "dropped" — they were detected but never fired. Surface those
        # to the model so it knows what's missing and can re-wrap them.
        _DROP_PATTERNS = [
            ("CODE",      CODE_TAG),
            ("KEEP",      KEEP_TAG),
            ("REFS",      REFS_TAG),
            ("SEARCH",    SEARCH_TAG),
            ("WEBSEARCH", WEBSEARCH_TAG),
            ("DETAIL",    DETAIL_TAG),
            ("PURPOSE",   PURPOSE_TAG),
            ("SEMANTIC",  SEMANTIC_TAG),
            ("LSP",       LSP_TAG),
            ("KNOWLEDGE", KNOWLEDGE_TAG),
        ]
        _strict_seen: set[str] = set()
        for tag_type, items in [
            ("CODE", file_tags), ("KEEP", keep_tags), ("REFS", refs_tags),
            ("SEARCH", code_tags), ("WEBSEARCH", web_tags), ("DETAIL", detail_tags),
            ("PURPOSE", purpose_tags), ("SEMANTIC", semantic_tags),
            ("LSP", lsp_tags), ("KNOWLEDGE", knowledge_tags),
        ]:
            for x in items:
                _strict_seen.add(f"{tag_type}:{_strip_label(x)[0].strip().lower()}")

        _light_masked = _mask_for_signals(result)
        dropped_tags: list[tuple[str, str]] = []  # (tag_type, arg)
        for tag_type, pat in _DROP_PATTERNS:
            for m in pat.findall(_light_masked):
                clean, _lbl = _strip_label(m)
                key = f"{tag_type}:{clean.strip().lower()}"
                if key in _strict_seen:
                    continue
                dropped_tags.append((tag_type, clean.strip()))

        # ── Bare-signal correction injection ─────────────────────────
        # Model wrote a lone [STOP] or [DONE] (no CONFIRM half) AND no
        # real tool tags this round. Most likely they MEANT to signal but
        # used the old single-tag form. Inject a one-shot note teaching
        # the two-tag protocol so they can correct themselves next round.
        # If real tool tags ARE present, the round is doing work — skip
        # the nag, the next round's results will let them try again.
        if _suspected_bare_signal and not has_tags:
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"bare [STOP]/[DONE] without CONFIRM half — "
                f"injecting two-tag protocol reminder"
            )
            full_response += result
            correction = (
                "\n\n[SYSTEM NOTE: You wrote [STOP] or [DONE] without its "
                "CONFIRM half — that does NOT fire the signal. The runtime "
                "uses a TWO-TAG combination to prevent accidental signals "
                "from prose mentions.\n\n"
                "To execute pending tool calls and continue thinking, write:\n"
                "  [STOP]\n  [CONFIRM_STOP]\n\n"
                "To finalize edits and end the loop (coders/reviewers only), write:\n"
                "  [DONE]\n  [CONFIRM_DONE]\n\n"
                "Both halves must appear in order, separated only by "
                "whitespace/newlines. A bare [STOP] alone — anywhere, "
                "in any context, including the end of your response — is "
                "treated as plain text and the loop continues.]\n\n"
                "Continue your response with the correct two-tag signal "
                "if that's what you meant.\n"
            )
            current_prompt = (
                current_prompt + "\n\nASSISTANT: " + full_response + correction
            )
            full_response = ""
            continue

        # ── Verbatim-restatement detection ────────────────────────────
        # In round 2+, if the model re-writes preamble sections AND the
        # content is substantially the same as round 1 (no new conclusion
        # named), it's wasting tokens. We only intervene when the redo is
        # pure restatement — REVISION is welcome and shouldn't be flagged.
        # Heuristic: section header present AND no REINFORCE/REVISE/DEEPER
        # marker AND content overlap with round 1 is high (≥60% of lines
        # appear in round 1 verbatim).
        if round_num >= 2 and _preamble_done:
            this_round_sections = _detect_preamble_sections(result)
            redone = [s for s in this_round_sections if s in _preamble_done]
            if redone:
                # Check for revision markers — if present, allow it.
                has_revision_marker = bool(re.search(
                    r'\b(REINFORCE|REVISE|REVISING|GO DEEPER|on reflection'
                    r'|new evidence|update(?:s|d)?\s+(?:my|the)\s+(?:plan|approach))\b',
                    result, re.IGNORECASE,
                ))
                # Check overlap with round 1 — if the lines mostly match
                # what's already there, it's pure restatement.
                round1_text = _round_texts[0] if _round_texts else ""
                round1_lines = set(
                    ln.strip() for ln in round1_text.splitlines()
                    if len(ln.strip()) > 20  # ignore short / boilerplate lines
                )
                this_lines = [
                    ln.strip() for ln in result.splitlines()
                    if len(ln.strip()) > 20
                ]
                if this_lines:
                    overlap = sum(1 for ln in this_lines if ln in round1_lines)
                    overlap_ratio = overlap / len(this_lines)
                else:
                    overlap_ratio = 0.0

                is_pure_restatement = (
                    not has_revision_marker and overlap_ratio >= 0.6
                )

                if is_pure_restatement:
                    warn(
                        f"  [{model.split('/')[-1]}] round {round_num}: "
                        f"verbatim restatement of {len(redone)} preamble "
                        f"section(s) (overlap {overlap_ratio:.0%}) — nudging"
                    )
                    full_response += result
                    redo_list = "\n".join(f"    ✓ {s}" for s in redone)
                    correction = (
                        f"\n\n[SYSTEM NOTE: you just repeated these sections "
                        f"from earlier in your response with no new conclusion:\n"
                        f"{redo_list}\n\n"
                        f"That content stands as written — it's in YOUR WORK "
                        f"SO FAR above. You can REVISE any of it explicitly "
                        f"('REVISE: Approach B is now better because new "
                        f"evidence X shows Y'). If you have nothing new to "
                        f"add to those sections, move on — integrate the tool "
                        f"results with one of REINFORCE / REVISE / GO DEEPER, "
                        f"then take the next concrete action. This is the "
                        f"same response — pick up at the next sentence.]\n\n"
                    )
                    current_prompt = (
                        current_prompt + "\n\nASSISTANT: " + full_response + correction
                    )
                    full_response = ""
                    continue

        # ── "Partial view" hallucination guard ────────────────────────
        # Self-check coders sometimes look at a small file (e.g. 66 lines)
        # and hallucinate that "the output only showed 2 lines, appears
        # to be a partial view" — then re-read the same file 4-5 rounds
        # in a loop. The file is whole; the model is wrong. Detect those
        # phrases and inject a one-shot correction that quotes the real
        # line count from the manifest so the model can see its own error.
        _PARTIAL_VIEW_PHRASES = re.compile(
            r'(?:'
            r'appears?\s+to\s+be\s+a\s+partial(?:/filtered)?\s+view'
            r'|this\s+can.?t\s+be\s+the\s+whole\s+file'
            r'|output\s+(?:seems|looks|appears)\s+(?:filtered|truncated|incomplete)'
            r'|only\s+\d+\s+lines\s+were\s+returned'
            r'|the\s+view\s+is\s+incomplete'
            r'|only\s+showed\s+\d+\s+lines'
            r')',
            re.IGNORECASE,
        )
        _has_partial_view_claim = bool(_PARTIAL_VIEW_PHRASES.search(result))
        # Only nag if the model ALSO re-requested a CODE/KEEP this round
        # (it's not just musing — it's about to loop). And only if NO
        # truncation header is actually present in persistent_lookups.
        _has_legit_truncation = any(
            "SKELETON ONLY" in v or "KEPT " in v
            for v in persistent_lookups.values()
        )
        if (_has_partial_view_claim and (file_tags or keep_tags)
                and not _has_legit_truncation):
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"detected 'partial view' hallucination — injecting "
                f"line-count anchor"
            )
            # Pull the actual line counts from persistent_lookups so the
            # injected note quotes the truth back to the model.
            line_facts = []
            for k, v in persistent_lookups.items():
                if not k.startswith("CODE:"):
                    continue
                m = re.search(r'\((\d+) lines\)', v)
                if m:
                    line_facts.append(f"  • {k[5:]}: {m.group(1)} lines (header is authoritative)")
            facts_block = "\n".join(line_facts) if line_facts else (
                "  • (no [CODE:] reads recorded yet)"
            )
            full_response += result
            correction = (
                "\n\n[SYSTEM NOTE: You wrote a phrase claiming the [CODE:] "
                "output was 'partial' / 'truncated' / 'incomplete'. That is a "
                "HALLUCINATION. The runtime always names the total line count "
                "in the header — `=== Code: <path> (N lines) ===` — and that "
                "number is authoritative.\n\n"
                "Your actual reads this session:\n"
                f"{facts_block}\n\n"
                "If the header says N lines and you see N numbered lines, the "
                "file is COMPLETE. Short files are short, not partial. The "
                "only legitimate truncation markers are 'SKELETON ONLY' and "
                "'KEPT N/M lines' — neither is present in your current reads.\n\n"
                "Do NOT re-request the same file. Reason from the content you "
                "have. If you genuinely needed a different file, name THAT "
                "file in your next tool call.]\n\n"
                "Continue your verification using the content you already have."
            )
            current_prompt = (
                current_prompt + "\n\nASSISTANT: " + full_response + correction
            )
            full_response = ""
            continue

        # ── Plan-completion guard ────────────────────────────────────
        # If the response contains BOTH tool tags AND plan-format headers
        # (## GOAL, ## REQUIREMENTS, ## IMPLEMENTATION STEPS, BEST: Plan #N),
        # the model has effectively committed to writing the plan. The tool
        # tags are stray — likely from "let me also check..." mid-plan. If we
        # honor them, the next round asks the model to "continue" and it
        # rewrites the entire plan from scratch (observed in deepseek-v4-pro
        # logs: R2 wrote a plan with stray tags, R3 rewrote it verbatim).
        # Treat the response as final: keep everything, skip tool processing.
        _PLAN_HEADERS = re.compile(
            r'(?m)^(?:#{1,3}\s*(?:GOAL|REQUIREMENTS|IMPLEMENTATION\s+STEPS'
            r'|SHARED\s+INTERFACES|EDGE\s+CASES|VERIFICATION'
            r'|TEST\s+CRITERIA)\b'
            r'|BEST:\s*Plan\s*#?\d+'
            r'|###\s*STEP\s*\d+:)',
            re.IGNORECASE,
        )
        _plan_committed = bool(_PLAN_HEADERS.search(result))
        if _plan_committed and has_tags:
            warn(
                f"  [{model.split('/')[-1]}] round {round_num}: "
                f"plan headers + stray tool tags detected — "
                f"treating response as final (no tool execution this round)"
            )
            full_response += result
            break

        # Detect the specific mistake: model wrote [SEARCH: N-M] expecting a
        # tool result, but that's edit syntax not a tool.  The filter removed
        # it from code_tags so has_tags is False — which would break the loop
        # and return a partial response with no edit content.
        # Only trigger when [SEARCH: N-M] appears WITHOUT a preceding === EDIT:
        # line (i.e., as a standalone tool call, not inside an edit block).
        _LINE_RANGE_TAG = re.compile(r'\[SEARCH:\s*\d+\s*-\s*\d+\s*\]')
        _EDIT_BLOCK = re.compile(r'===\s*EDIT:', re.IGNORECASE)
        has_misused_search = (
            bool(_LINE_RANGE_TAG.search(result))
            and not bool(_EDIT_BLOCK.search(result))
            and not bool(_EDIT_BLOCK.search(full_response[-500:]))  # not recently in an edit block
        )
        if has_misused_search and not has_tags:
            warn("  ⚠️  Model used [SEARCH: N-M] as a tool call — injecting correction")
            correction = (
                "\n\n[SYSTEM NOTE: [SEARCH: N-M] is EDIT SYNTAX, not a tool call. "
                "It belongs inside an === EDIT: file.py === block like this:\n"
                "=== EDIT: file.py ===\n"
                "[SEARCH: 45-49]\n"
                "exact code to find\n"
                "[/SEARCH]\n"
                "[REPLACE]\n"
                "new code\n"
                "[/REPLACE]\n"
                "Continue writing your edit blocks now. Do NOT write [SEARCH: N-M] "
                "as a standalone tag expecting a result.]\n\n"
            )
            full_response += result
            current_prompt = current_prompt + "\n\nASSISTANT: " + full_response + correction + "\n\nContinue:"
            result = ""
            full_response = ""
            continue

        if has_tags:
            # Trim result to end at the last tag — anything the model
            # wrote after the last ] is speculation without results.
            last_bracket = result.rfind(']')
            if last_bracket >= 0:
                result = result[:last_bracket + 1]

        full_response += result

        if not has_tags:
            break  # No tool requests — done

        # ── Apply pending edits BEFORE tool lookups ──────────────────
        # When a coder writes edit blocks then [STOP] + [CODE: file],
        # they want to verify their edits. on_stop applies the edits
        # to the sandbox so [CODE:] reads return the post-edit state.
        # The callback may return a feedback string describing which
        # edits applied vs were skipped — captured here and surfaced
        # in the next-round prompt so the model sees explicit results.
        if on_stop is not None:
            try:
                feedback = on_stop(full_response)
                if feedback and isinstance(feedback, str) and feedback.strip():
                    _last_edit_feedback = feedback.strip()
                    # Parse the feedback to update per-file attempt history.
                    # Lines starting with "✓" = success on that file; "✗" = miss.
                    # We use this for stall detection below.
                    #
                    # Feedback line shapes we must support:
                    #   ✓ CREATED  path/to/file.py (N lines written)
                    #   ✓ MODIFIED path/to/file.py (A → B lines)
                    #   ✗ REJECTED edit on path/to/file.py: SEARCH anchor …
                    #   ✗ REJECTED SEARCH starting with 'def foo' had 3 matches…
                    #   ↺ REVERTED path/to/file.py to prior snapshot
                    # The first token after the marker is a verb; the filepath
                    # is the first token-that-LOOKS-like-a-path on the line.
                    _path_like = re.compile(
                        r'(?<![\w/.])([\w./-]+\.(?:py|js|ts|jsx|tsx|html|css|json|lean|c|cpp|h|rs|java|go|rb|toml|yaml|yml|md|mjs|cjs|svelte|vue|lua|sh))(?![\w/.])'
                    )
                    for ln in _last_edit_feedback.splitlines():
                        s = ln.strip()
                        if not s or s[0] not in '✓✗↺':
                            continue
                        m = _path_like.search(s)
                        if not m:
                            continue
                        fp = m.group(1).rstrip(':,')
                        applied = s.startswith('✓')
                        _edit_attempts_per_file.setdefault(fp, []).append(applied)
                else:
                    _last_edit_feedback = None
            except Exception as e:
                warn(f"  on_stop callback error: {e}")
                _last_edit_feedback = None

        # ── Edit-flailing detection ──────────────────────────────────
        # If 3 consecutive rounds have attempted edits on the SAME file
        # and ALL failed, the model is flailing — it'll keep writing
        # variations of the same broken SEARCH forever. Surface a
        # strong correction telling it to take a different approach
        # (REPLACE LINES, REVERT then redo, or give up cleanly).
        # Observed: 19-round step-3 loop on domains/prompts.py where
        # the model kept tweaking SEARCH anchors that never matched.
        for fp, history in _edit_attempts_per_file.items():
            if len(history) >= 3 and not any(history[-3:]):
                _last_edit_feedback = (
                    (_last_edit_feedback + "\n\n") if _last_edit_feedback else ""
                ) + (
                    f"🛑 EDIT-FLAILING DETECTED on {fp}\n"
                    f"  You've attempted edits on {fp} in {len(history)} rounds and "
                    f"the last 3 all failed. Variations on the same SEARCH approach "
                    f"won't work — STOP retrying it.\n"
                    f"  CHOOSE ONE of these recovery paths:\n"
                    f"    1. [CODE: {fp}] to re-read the file fresh, then write a\n"
                    f"       FUNDAMENTALLY DIFFERENT SEARCH anchor (different lines,\n"
                    f"       not just different whitespace).\n"
                    f"    2. Use [REPLACE LINES N-M] with line numbers from the\n"
                    f"       most recent [CODE:] read of {fp} — bypasses SEARCH\n"
                    f"       entirely. Safe for unique line ranges.\n"
                    f"    3. If a prior edit DID apply but landed wrong, use\n"
                    f"       [REVERT FILE: {fp}] to undo, then plan from clean.\n"
                    f"    4. If you've tried everything: write what you accomplished\n"
                    f"       and [DONE][CONFIRM_DONE] — the next pass will retry."
                )
                # Reset the history so we don't fire again next round on
                # the same trigger — the model gets one strong nudge per
                # 3-failure streak, not repeated nags.
                _edit_attempts_per_file[fp] = []

        # Run requested lookups — check cache first
        round_output = ""  # results from THIS round only (for logging)

        # Filter out tags this model already ran in a previous round.
        # NOTE: we only check local_research (this model's own history), NOT
        # the shared research_cache. Results from other parallel models are NOT
        # "cached" from this model's perspective — it must request them itself.
        # This prevents false stall detection and stops the model from thinking
        # it has seen content it never actually requested.
        def _cached_or_run(tag_type: str, tags: list[str]) -> tuple[list[str], str]:
            """Returns (new_tags_to_run, cached_output_for_already_run_tags).
            CODE and KEEP always re-run — file content may have changed."""
            if tag_type in ("CODE", "KEEP"):
                return tags, ""
            cached_out = ""
            new_tags = []
            for tag in tags:
                clean_tag, label = _strip_label(tag)
                key = f"{tag_type}:{clean_tag.strip().lower()}"
                if key in local_research:
                    # This model ran it in an earlier round — show cached result
                    rn = _manifest[key]["round"] if key in _manifest else "?"
                    cached_out += (
                        f"\n[CACHED — you already ran {tag_type}: {clean_tag} "
                        f"in round {rn}. Result is unchanged — do not re-request.]\n"
                        + local_research[key]
                    )
                    persistent_lookups[key] = local_research[key]
                    if label:
                        _label_to_keys.setdefault(label, []).append(key)
                else:
                    new_tags.append(tag)
            return new_tags, cached_out

        def _store(tag_type: str, tag: str, result: str):
            """Store a result in local_research, shared cache, and persistent lookups."""
            clean_tag, label = _strip_label(tag)
            key = f"{tag_type}:{clean_tag.strip().lower()}"
            local_research[key] = result
            persistent_lookups[key] = result
            if research_cache is not None:
                research_cache[key] = result
            if label:
                _label_to_keys.setdefault(label, []).append(key)
            _manifest[key] = {"round": round_num, "tag_type": tag_type, "arg": clean_tag}

        async def _locked_lookup(tag_type: str, tag: str, run_fn) -> str:
            """Run a lookup with a per-key lock to prevent duplicate concurrent executions.
            For non-CODE/KEEP types: if the result is already in the shared cache
            (from another parallel model), return it directly and record it as seen
            by this model — no re-execution, no wasted API call."""
            clean_tag, label = _strip_label(tag)
            key = f"{tag_type}:{clean_tag.strip().lower()}"
            if key not in _inflight_locks:
                _inflight_locks[key] = asyncio.Lock()
            lock = _inflight_locks[key]

            async with lock:
                if tag_type not in ("CODE", "KEEP"):
                    if research_cache is not None and key in research_cache:
                        result = research_cache[key]
                        # Record as seen by this model (local_research + manifest)
                        local_research[key] = result
                        persistent_lookups[key] = result
                        if label:
                            _label_to_keys.setdefault(label, []).append(key)
                        _manifest[key] = {"round": round_num,
                                          "tag_type": tag_type, "arg": clean_tag}
                        return result
                result = run_fn(clean_tag)
                if asyncio.iscoroutine(result):
                    result = await result
                _store(tag_type, tag, result)
                return result

        # ── Handle [DISCARD: #label] — remove labeled results from context ──
        if discard_tags:
            for label in discard_tags:
                if label in _label_to_keys:
                    for key in _label_to_keys[label]:
                        persistent_lookups.pop(key, None)
                        local_research.pop(key, None)
                    status(f"  Discarded #{label} ({len(_label_to_keys[label])} results)")
                    del _label_to_keys[label]
                else:
                    warn(f"  [DISCARD: #{label}] — label not found, ignoring")

        total = len(code_tags) + len(web_tags) + len(detail_tags) + len(file_tags) + len(refs_tags) + len(purpose_tags) + len(semantic_tags) + len(lsp_tags) + len(knowledge_tags) + len(keep_tags)
        mode = _describe_tool_mode(result)
        tags_desc = _tag_summary(
            code_tags, web_tags, detail_tags, file_tags, refs_tags,
            purpose_tags, semantic_tags, lsp_tags, knowledge_tags, keep_tags,
            research_cache, persistent_lookups,
        )
        status(f"  [{model.split('/')[-1]}] tool round {round_num}/{max_rounds}: "
               f"{total} lookup(s) — {mode}")
        status(f"    tags: {tags_desc}")

        # ── Stall detection ────────────────────────────────────────────
        # Build a stable key set for THIS round's tag requests so we can
        # tell whether the model is making progress or just re-requesting
        # already-cached lookups. If two consecutive rounds request only
        # tools whose results are already in persistent_lookups, the model
        # is spinning — break out and let it commit.
        def _norm_tag_key(tag_type: str, tag_arg: str) -> str:
            clean, _ = _strip_label(tag_arg)
            return f"{tag_type}:{clean.strip().lower()}"

        round_keys: set[str] = set()
        for t in code_tags:    round_keys.add(_norm_tag_key("SEARCH", t))
        for t in web_tags:     round_keys.add(_norm_tag_key("WEBSEARCH", t))
        for t in detail_tags:  round_keys.add(_norm_tag_key("DETAIL", t))
        for t in file_tags:    round_keys.add(_norm_tag_key("CODE", t))
        for t in refs_tags:    round_keys.add(_norm_tag_key("REFS", t))
        for t in purpose_tags: round_keys.add(_norm_tag_key("PURPOSE", t))
        for t in lsp_tags:     round_keys.add(_norm_tag_key("LSP", t))
        for t in knowledge_tags: round_keys.add(_norm_tag_key("KNOWLEDGE", t))
        for t in keep_tags:    round_keys.add(_norm_tag_key("KEEP", t))

        # A round is "stalled" if every key was already run by this model.
        # We do NOT check the shared research_cache — results from other parallel
        # models are new to this model and must not trigger false stall detection.
        # CODE/KEEP exception: file content can change, so a single re-read is
        # legitimate. But the SECOND identical re-read in a row is always a loop.
        def _is_cached(k: str) -> bool:
            tt = k.split(':', 1)[0]
            if tt in ('CODE', 'KEEP'):
                # Identical repeats count as "cached" once we've seen them
                # at least twice — the model is re-requesting the same lines.
                return _reread_count.get(k, 0) >= 2
            return k in local_research

        # Update re-read counters BEFORE stall detection. A key fires here on
        # every round it appears; non-CODE/KEEP keys never enter (they were
        # already served from cache and don't reach this round_keys set with
        # any meaning anyway).
        for k in round_keys:
            tt = k.split(':', 1)[0]
            if tt in ('CODE', 'KEEP') and k in _manifest:
                _reread_count[k] = _reread_count.get(k, 0) + 1

        if round_keys and all(_is_cached(k) for k in round_keys):
            _stall_rounds += 1
        elif round_keys == _last_round_keys and round_keys:
            _stall_rounds += 1
        else:
            _stall_rounds = 0
        _last_round_keys = round_keys

        if _stall_rounds >= 2:
            warn(
                f"  Stall: round {round_num} repeated already-cached lookups. "
                f"Forcing the model to commit instead of looping."
            )
            # Inject a hard commit instruction and let the loop run ONE more
            # turn so the model can emit its plan/code, then break.
            stall_note = (
                "\n\n══════════════════════════════════════════════════════════════════════\n"
                "🛑 STOP INVESTIGATING — COMMIT NOW\n"
                "══════════════════════════════════════════════════════════════════════\n"
                "You have spent multiple rounds re-requesting the SAME tool results.\n"
                "Investigation is over. Write your final answer NOW using only what\n"
                "you already know. Do NOT use any more tool tags. Do NOT write [STOP].\n"
                "If you are a planner: WRITE THE PLAN.\n"
                "If you are a coder: WRITE THE EDIT BLOCKS, then [DONE].\n"
                "══════════════════════════════════════════════════════════════════════\n"
            )
            current_prompt = current_prompt + stall_note
            # Force one final round with no tool processing
            try:
                final_result = await call_with_retry(
                    model, current_prompt, max_tokens=max_tokens,
                    stop_check=None,  # no early stop — let it write everything
                    log_label=log_label + " (commit)",
                )
                # Strip any remaining signals
                final_result = DONE_TAG.sub('', final_result)
                final_result = STOP_TAG.sub('', final_result).rstrip()
                full_response += "\n" + final_result
            except Exception as e:
                warn(f"  Forced-commit round failed: {e}")
            break

        # Lookup runners — defined ONCE outside the loop so each closure
        # captures fresh `tag` via its parameter only, never via the loop
        # variable. The previous inline-`async def` form was correct only
        # because we awaited every result before the next iteration; making
        # it explicit removes the foot-gun.
        def _run_search(tag): return _run_code_searches([tag], _search_root)
        def _run_web(tag):    return _run_web_searches([tag])
        def _run_detail(tag): return _run_detail_lookups([tag], detailed_map)
        def _run_code(tag):
            return _run_code_reads([tag], project_root, viewed_versions=viewed_versions)
        def _run_refs(tag):   return _run_refs_searches([tag], _search_root)
        def _run_purpose(tag): return _run_purpose_lookups([tag], purpose_map, project_root)
        async def _run_semantic(tag):
            from tools.code_index import _maps_dir, _load_all_code
            from tools.embeddings import semantic_retrieve
            maps_dir = _maps_dir(project_root)
            _, file_hash = _load_all_code(project_root)
            return await semantic_retrieve(
                tag, purpose_map, project_root, maps_dir, file_hash, top_n=10
            )
        def _run_lsp(tag): return _run_lsp_searches([tag], project_root)
        def _run_knowledge(tag): return _run_knowledge_lookups([tag])

        if code_tags and project_root:
            new_tags, cached = _cached_or_run("SEARCH", code_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("SEARCH", t, _run_search)
                round_output += r

        if web_tags:
            new_tags, cached = _cached_or_run("WEBSEARCH", web_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("WEBSEARCH", t, _run_web)
                round_output += r

        if detail_tags and detailed_map:
            new_tags, cached = _cached_or_run("DETAIL", detail_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("DETAIL", t, _run_detail)
                round_output += r

        if file_tags and project_root:
            new_tags, cached = _cached_or_run("CODE", file_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("CODE", t, _run_code)
                round_output += r

        if refs_tags and project_root:
            new_tags, cached = _cached_or_run("REFS", refs_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("REFS", t, _run_refs)
                round_output += r

        if purpose_tags and purpose_map and project_root:
            new_tags, cached = _cached_or_run("PURPOSE", purpose_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("PURPOSE", t, _run_purpose)
                round_output += r

        if semantic_tags and purpose_map and project_root:
            new_tags, cached = _cached_or_run("SEMANTIC", semantic_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("SEMANTIC", t, _run_semantic)
                round_output += r

        if lsp_tags and project_root:
            new_tags, cached = _cached_or_run("LSP", lsp_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("LSP", t, _run_lsp)
                round_output += r

        if knowledge_tags:
            new_tags, cached = _cached_or_run("KNOWLEDGE", knowledge_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("KNOWLEDGE", t, _run_knowledge)
                round_output += r

        # ── KEEP handler — replaces CODE entries in persistent_lookups ──
        if keep_tags and project_root:
            def _on_keep_seen(canonical_key: str, raw_arg: str) -> None:
                # Register the KEEP in the manifest + local_research so
                # the loop detector and cache annotations work for KEEP
                # the same way they do for CODE. KEEP content itself
                # already lives in persistent_lookups (replacing the
                # corresponding CODE entry).
                local_research[canonical_key] = persistent_lookups.get(
                    canonical_key, ""
                )
                _manifest[canonical_key] = {
                    "round": round_num,
                    "tag_type": "KEEP",
                    "arg": raw_arg.strip(),
                }
            keep_result = await _run_keep(
                keep_tags, project_root,
                persistent_lookups, research_cache,
                viewed_versions=viewed_versions,
                on_keep_seen=_on_keep_seen,
            )
            round_output += keep_result

        # Rebuild search_output from ALL persistent lookups.
        # This is the key mechanism: if KEEP replaced a CODE entry,
        # the full file is gone — only the kept ranges remain.
        # CONTEXT BUDGET — if the cumulative tool-results would exceed
        # ~80k chars (≈ 20k tokens), drop the LEAST-recent entries to
        # stay within the model context. The KEEP mechanism normally
        # keeps things small, but a coder reading 8 large files in one
        # session can still pile on. We always keep the entries touched
        # THIS round so the model never loses the result of what it
        # just asked for.
        TOOL_OUTPUT_BUDGET = 80_000  # chars
        this_round_keys = round_keys  # set built earlier this round
        all_entries = list(persistent_lookups.items())
        total_chars = sum(len(v) for _, v in all_entries)
        if total_chars > TOOL_OUTPUT_BUDGET and len(all_entries) > 1:
            # ── Recency scoring ─────────────────────────────────────────
            # Bump every entry the model has been REFERENCING in its
            # recent prose (by tag argument substring match against
            # _round_texts[-1]). Without this, a CODE result the model
            # keeps citing across 6 rounds loses to fresh lookups even
            # though it's actively used. We treat "named in last round"
            # as equivalent to "just looked up this round".
            recent_text = _round_texts[-1] if _round_texts else ""
            recent_text_lower = recent_text.lower()
            def _entry_score(k: str) -> tuple[int, int]:
                # This-round entries are SACRED — they were JUST asked for.
                if k in this_round_keys:
                    return (10_000_000, round_num)
                info = _manifest.get(k)
                base_round = info["round"] if info else 0
                # Boost if the model mentioned the arg by name in last round
                bump = 0
                if info:
                    arg = (info.get("arg") or "").strip().lower()
                    if arg and len(arg) >= 3 and arg in recent_text_lower:
                        bump = round_num  # "as if it were touched this round"
                return (base_round + bump, base_round)
            all_entries.sort(key=lambda kv: _entry_score(kv[0]), reverse=True)
            kept_entries: list[tuple[str, str]] = []
            running = 0
            for k, v in all_entries:
                if running + len(v) > TOOL_OUTPUT_BUDGET and (
                    k not in this_round_keys
                ):
                    continue
                kept_entries.append((k, v))
                running += len(v)
            dropped = len(persistent_lookups) - len(kept_entries)
            if dropped > 0:
                warn(
                    f"  [{model.split('/')[-1]}] round {round_num}: "
                    f"tool-results over {TOOL_OUTPUT_BUDGET:,}-char budget — "
                    f"dropped {dropped} oldest lookup(s) from prompt "
                    f"(still tracked in manifest)"
                )
            search_output = "\n".join(v for _, v in kept_entries)
        else:
            search_output = "\n".join(persistent_lookups.values())

        if not search_output.strip():
            search_output = (
                "\n=== LOOKUP RESULTS: all lookups returned empty or failed ===\n"
                "The items you searched for were not found. Continue with what you know.\n"
                "Do NOT re-request the same lookups — they will fail again.\n"
            )

        # Build continuation prompt — include explicit round budget so the
        # model feels time pressure to commit instead of looping. After the
        # halfway mark we escalate: "wrap up". Past the threshold, we say
        # "STOP investigating, COMMIT NOW".
        budget_msg = ""
        rounds_used = round_num
        rounds_left = max_rounds - rounds_used
        if rounds_left <= 0:
            budget_msg = (
                "\n⛔ NO TOOL ROUNDS LEFT. This is your FINAL response. "
                "Write your plan/edits NOW. Do NOT use any more tool tags."
            )
        elif rounds_used >= max(3, max_rounds * 2 // 3):
            budget_msg = (
                f"\n⚠ Round {rounds_used}/{max_rounds}. {rounds_left} round(s) left. "
                "Wrap up investigation and commit to your answer. Use tools ONLY if "
                "absolutely required to fill a remaining gap."
            )
        elif rounds_used >= max(2, max_rounds // 2):
            budget_msg = (
                f"\n• Round {rounds_used}/{max_rounds}. You have {rounds_left} round(s) left. "
                "Prefer committing over more investigation when in doubt."
            )

        # ── Build the model's own work-so-far as ONE continuous stream ───
        # No "Round N" labels, no opening/closing wrappers — the model
        # reads its own prior output as a single flowing narrative, the
        # way it would experience tool use inside one Claude/Anthropic
        # API turn. This is the difference between "I am restarting in
        # round 2" (current failure mode) and "I am continuing my work."
        # Each prior round's output is separated by a single horizontal
        # rule so the model can see where streaming stopped/resumed if
        # it needs that signal, but the rule is minimal — not a banner.
        round_history = "\n\n────────\n\n".join(_round_texts)

        # ── Build context manifest ────────────────────────────────────
        def _manifest_line(k: str, v: dict) -> str:
            arg = v["arg"]
            tt  = v["tag_type"]
            rn  = v["round"]
            result_text = persistent_lookups.get(k, "")
            lines_hint = ""
            m = re.search(r'\((\d+) lines\)', result_text)
            if m:
                lines_hint = f" — {m.group(1)} lines"
            elif "KEPT" in result_text:
                m2 = re.search(r'KEPT (\d+/\d+ lines)', result_text)
                if m2:
                    lines_hint = f" — {m2.group(1)} kept"
            return f"  [{tt}: {arg}] (R{rn}{lines_hint})"

        manifest_lines = ["══ CONTEXT MANIFEST — what you have actually read ══"]
        if _manifest:
            for k, v in _manifest.items():
                base = _manifest_line(k, v)
                rcount = _reread_count.get(k, 0)
                if rcount >= 2:
                    base += f"  ⛔ RE-READ {rcount}× — DO NOT request this again"
                elif rcount == 1:
                    base += "  ⚠ already re-read once"
                manifest_lines.append(base)
        else:
            manifest_lines.append("  (no tool results yet)")
        # If the model just re-issued a CODE/KEEP for an identical key this
        # round, surface that as a separate top-level warning — easy to miss
        # buried in the manifest list.
        repeat_offenders = [k for k in round_keys
                            if k.split(':', 1)[0] in ('CODE', 'KEEP')
                            and _reread_count.get(k, 0) >= 2]
        if repeat_offenders:
            manifest_lines.append(
                "🛑 LOOP DETECTED: you just re-requested " +
                ", ".join(f"[{k}]" for k in repeat_offenders) +
                " — the result is IDENTICAL to your previous reads. "
                "STOP re-investigating. Use what you already have and COMMIT."
            )
        manifest_lines.append(
            "⚠ HARD RULE: Any file or result NOT listed above is UNKNOWN to you.\n"
            "  Do NOT reference, quote, or reason about content you have not seen.\n"
            "  If you need a file that is not listed — request it with a tool tag.\n"
            "  If a file IS listed — DO NOT re-read it. Reason from what you have.\n"
            "══"
        )
        manifest_str = "\n".join(manifest_lines)

        # ── Build the "dropped tool calls" block ──────────────────────
        # When the model wrote tool tags OUTSIDE [tool use] blocks, the
        # strict masker drops them. We detected those in `dropped_tags`
        # above. Show the model exactly what fired vs. what didn't so it
        # doesn't hallucinate results for the dropped ones. This was the
        # bug behind the "writes 3 [DETAIL:] tags, only 1 fires, model
        # invents results for the other 2" failure.
        if dropped_tags:
            dropped_lines = []
            for tt, arg in dropped_tags:
                dropped_lines.append(f"  ✗ [{tt}: {arg}]")
            dropped_block = (
                "══════════════════════════════════════════════════════════════════════\n"
                "TOOL CALLS THAT DID NOT FIRE — must be inside [tool use]...[/tool use]\n"
                "══════════════════════════════════════════════════════════════════════\n"
                "These tags appeared in your response but were NOT executed,\n"
                "because they were written OUTSIDE a [tool use]...[/tool use]\n"
                "block. The runtime only fires tags that are deliberately\n"
                "wrapped, to prevent accidental execution of tags discussed in\n"
                "prose or in examples.\n\n"
                + "\n".join(dropped_lines) + "\n\n"
                "If you wanted these to fire, wrap EACH tool call in its own\n"
                "[tool use]...[/tool use] block, then signal once at the end:\n"
                "  [tool use]\n"
                "  [DETAIL: Ensemble]\n"
                "  [/tool use]\n"
                "  [tool use]\n"
                "  [DETAIL: Debate]\n"
                "  [/tool use]\n"
                "  [tool use]\n"
                "  [DETAIL: Synthesizer]\n"
                "  [/tool use]\n"
                "  [STOP]\n"
                "  [CONFIRM_STOP]\n\n"
                "OR put all calls in ONE [tool use] block:\n"
                "  [tool use]\n"
                "  [DETAIL: Ensemble]\n"
                "  [DETAIL: Debate]\n"
                "  [DETAIL: Synthesizer]\n"
                "  [/tool use]\n"
                "  [STOP]\n"
                "  [CONFIRM_STOP]\n\n"
                "⚠ Do NOT assume the dropped calls produced results — they did\n"
                "  not. The TOOL RESULTS section below contains ONLY what fired.\n"
                "══════════════════════════════════════════════════════════════════════\n\n"
            )
        else:
            dropped_block = ""

        # ── Build the "edit application results" block ────────────────
        # The dominant cause of multi-round edit loops is the model
        # writing an edit, then [CODE:] verifying it, but never seeing
        # explicit "edit applied" or "edit skipped" feedback from the
        # runtime. Without that signal the model has to infer success
        # by reading the file diff — which it consistently gets wrong
        # (see the 19-round step-3 loop on domains/prompts.py). Here
        # we inject the on_stop feedback at the TOP of the continuation
        # prompt so it's the first thing the model reads next round.
        if _last_edit_feedback:
            edit_results_block = (
                "══════════════════════════════════════════════════════════════════════\n"
                "EDIT APPLICATION RESULTS — what the runtime did with your last edits\n"
                "══════════════════════════════════════════════════════════════════════\n"
                f"{_last_edit_feedback}\n\n"
                "▸ If an edit applied, the file IS now different — don't re-write\n"
                "  the same edit thinking it failed. Look at the post-edit content\n"
                "  in the [CODE:] result below to confirm.\n"
                "▸ If an edit was REJECTED, the reason tells you what to fix.\n"
                "  Common causes:\n"
                "    • SEARCH anchor not unique — multiple regions matched;\n"
                "      add more context lines to the SEARCH to disambiguate.\n"
                "    • SEARCH anchor not matched — your SEARCH text doesn't\n"
                "      appear verbatim in the file. Re-read the file with\n"
                "      [CODE:] and copy the exact lines.\n"
                "    • Catastrophic-shrink tripwire — your SEARCH would have\n"
                "      removed >50%% of the file. Use a smaller, more unique\n"
                "      anchor on the specific change point, not a big sweep.\n"
                "▸ DO NOT re-issue an identical SEARCH/REPLACE that already\n"
                "  applied. Re-applying creates duplicates.\n"
                "══════════════════════════════════════════════════════════════════════\n\n"
            )
        else:
            edit_results_block = ""

        # The "preamble already done" reminder is now woven into the
        # final "continue your work" cue at the bottom of the prompt
        # rather than appearing as its own banner. A separate banner
        # made the prompt feel like a fresh round; embedding the
        # reminder in the continue cue makes it feel like a single
        # ongoing thought stream.
        preamble_block = ""

        # The continuation prompt is structured to FEEL like one
        # continuous call to the model, not a series of restarts:
        #
        #   [Full system prompt — kept intact, the model needs it every
        #    round so it doesn't forget its role, rules, and tools.]
        #
        #   [USER REQUEST — also kept intact for the same reason.]
        #
        #   [Edit feedback / manifest — only when relevant, framed as
        #    "since you last wrote, here's what changed in the world."]
        #
        #   [YOUR WORK SO FAR — the model's own prior output streamed
        #    together with no round labels, like a Claude turn that just
        #    happens to have used tools partway through.]
        #
        #   [TOOL RESULTS — cumulative, deduped via persistent_lookups.]
        #
        #   [A single-line "continue from your last sentence" cue.]
        #
        # The system prompt and USER REQUEST stay full every round so the
        # model never forgets its instructions; only the framing around
        # the model's own previous work changes to emphasize continuity.

        # Preamble continuity cue, embedded in the continue line below.
        # If we tracked completed orient/preamble sections, we mention
        # them in passing — no banner, just an inline reminder.
        if _preamble_done:
            preamble_cue = (
                f"  (You already did your initial orient — "
                f"{', '.join(_preamble_done[:3])}"
                f"{'…' if len(_preamble_done) > 3 else ''}. Revise these only "
                f"if new evidence demands; don't restate them.)"
            )
        else:
            preamble_cue = ""

        current_prompt = f"""{dropped_block}{edit_results_block}{prompt}

══════════════════════════════════════════════════════════════════════
YOUR WORK SO FAR (continuous — keep writing from where you stopped)
══════════════════════════════════════════════════════════════════════
{round_history}

══════════════════════════════════════════════════════════════════════
TOOL RESULTS (cumulative across all tool calls THAT ACTUALLY FIRED)
══════════════════════════════════════════════════════════════════════
Only results below ARE real. Anything you "remember" requesting that
isn't listed here either was dropped (see TOOL CALLS THAT DID NOT FIRE
above, if present) or you never requested it. Do not reason from
results that aren't shown.
{search_output}

{manifest_str}
{budget_msg}

──────────────────────────────────────────────────────────────────────
↓ Continue writing from where you stopped — this is the same response,
  not a new round. Pick up at the next sentence.
{preamble_cue}
──────────────────────────────────────────────────────────────────────"""

    return {
        "model": model,
        "answer": full_response,
        "done": _done_signaled,
        "research": local_research,
        # `persistent_lookups` reflects the FINAL view the model had: CODE
        # entries replaced by their KEEP-filtered version, DISCARDed entries
        # removed. Callers that want to know exactly what the model was
        # looking at when it produced `answer` should read this, not
        # `research` (which only holds NEW results from this run).
        "persistent_lookups": dict(persistent_lookups),
    }
