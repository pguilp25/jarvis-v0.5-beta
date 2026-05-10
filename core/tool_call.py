"""
Tool Call Loop — shared by all workflows.

Any AI can pause mid-thought to search:
  [SEARCH: pattern]    → ripgrep code search (coding agent)
  [WEBSEARCH: query]   → web search (research, chat)

JARVIS detects the tags, runs the searches, feeds results back,
and the AI continues from where it left off. Up to 5 rounds.
"""

import asyncio
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
# The model writes tool tags, then [STOP]. We process tools and feed back.
STOP_TAG = re.compile(r'\[STOP\]', re.IGNORECASE)
# DONE signals the model is completely finished — apply edits and exit.
# Only used by coders/reviewers/self-checkers who write edit blocks.
DONE_TAG = re.compile(r'\[DONE\]', re.IGNORECASE)
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

# === EDIT: ... === end FILE === (full file creation)
_EDIT_FILE_SPAN = re.compile(
    r'===\s*FILE:.*?===\s*END\s+FILE\s*===',
    re.DOTALL | re.IGNORECASE,
)
# [SEARCH]...[/SEARCH] — code the coder is searching for
_SEARCH_BLOCK = re.compile(r'\[SEARCH[^\]]*\](.*?)\[/SEARCH\]', re.DOTALL | re.IGNORECASE)
# [REPLACE]...[/REPLACE] — replacement code
_REPLACE_BLOCK = re.compile(r'\[REPLACE[^\]]*\](.*?)\[/REPLACE\]', re.DOTALL | re.IGNORECASE)
# [INSERT AFTER LINE N]...[/INSERT] — inserted code
_INSERT_BLOCK = re.compile(r'\[INSERT[^\]]*\](.*?)\[/INSERT\]', re.DOTALL | re.IGNORECASE)
# Legacy: === EDIT: ... [/REPLACE] span (catches multi-block edits partially)
_EDIT_BLOCK_SPAN = re.compile(
    r'===\s*(?:EDIT|FILE):.*?'
    r'(?:'
        r'\[/REPLACE\]|\[/INSERT\]|===\s*END\s+FILE\s*==='
    r')',
    re.DOTALL | re.IGNORECASE,
)
_BACKSLASH_BRACKET = re.compile(r'\\\[')


def _mask_quoted_tags(text: str) -> str:
    """Return text with `[` replaced by NUL inside any region where tags
    should NOT be interpreted as tool calls.

    The result is ONLY used to drive tag extraction. The model's actual
    response is unchanged. NUL is a safe sentinel — it never appears in
    legitimate model output and prevents any `\\[(SEARCH|...)` regex from
    matching across the masked spans.
    """
    if not text or '[' not in text:
        return text

    masked = list(text)

    def _blank(start: int, end: int) -> None:
        for i in range(start, min(end, len(masked))):
            if masked[i] == '[':
                masked[i] = '\x00'

    # 0. <think>...</think> blocks — model's internal reasoning.
    # Tags written inside thinking are NOT intended as tool calls.
    for m in _THINK_BLOCK.finditer(text):
        _blank(m.start(), m.end())

    # 1. Fenced code blocks (```...```)
    for m in _FENCED_CODE_BLOCK.finditer(text):
        _blank(m.start(), m.end())

    # 2. Inline backtick spans (`...`)
    for m in _INLINE_BACKTICK.finditer(text):
        _blank(m.start(), m.end())

    # 3. Code-writing blocks — mask all forms where the model writes actual code.
    # Tool tags inside written code must NEVER fire as real tool calls.
    for pattern in (_EDIT_FILE_SPAN, _SEARCH_BLOCK, _REPLACE_BLOCK,
                    _INSERT_BLOCK, _EDIT_BLOCK_SPAN):
        for m in pattern.finditer(text):
            _blank(m.start(), m.end())

    # 3b. Legacy edit block span (belt-and-suspenders for === EDIT: blocks)
    for m in _EDIT_BLOCK_SPAN.finditer(text):
        _blank(m.start(), m.end())

    # 4. Explicit escape: `\[TAG: ...]` → mask just the leading `[`
    for m in _BACKSLASH_BRACKET.finditer(text):
        # The `[` is at m.end() - 1
        idx = m.end() - 1
        if 0 <= idx < len(masked) and masked[idx] == '[':
            masked[idx] = '\x00'

    # 5. [tool use]...[/tool use] enforcement:
    # If the response contains ANY deliberate tool-use blocks, mask every `[`
    # that is OUTSIDE those blocks. This ensures accidental tag mentions in
    # explanatory text, examples, or mid-thought analysis never fire as real
    # tool calls — only explicitly wrapped calls execute.
    # If no [tool use] blocks are present, fall through unchanged (backward compat).
    tool_use_blocks = list(_TOOL_USE_BLOCK.finditer(text))
    if tool_use_blocks:
        # Build the set of character positions that are INSIDE a block's content
        inside = set()
        for m in tool_use_blocks:
            inside.update(range(m.start(1), m.end(1)))
        # Mask every [ that is outside all blocks (skipping already-NUL positions)
        for i in range(len(masked)):
            if masked[i] == '[' and i not in inside:
                masked[i] = '\x00'

    return ''.join(masked)


def extract_search_tags(text: str) -> list[str]:
    # Exclude patterns that are edit syntax, not code search queries:
    # 1. Pure line-range patterns like "339-342" — [SEARCH: N-M] anchored edit
    # 2. File paths like "ui/index.html" — [SEARCH: filepath] edit reference
    # Routing these to ripgrep produces garbage results and causes the model to loop.
    _LINE_RANGE = re.compile(r'^\d+\s*-\s*\d+$')
    # File-path heuristic: contains a dot-extension (2-5 chars) and optionally
    # path separators. Real search queries are identifiers/patterns, not paths.
    _FILE_PATH = re.compile(r'\.\w{1,5}$')
    masked = _mask_quoted_tags(text)
    results = []
    for q in SEARCH_TAG.findall(masked):
        clean, _ = _strip_label(q)
        stripped = clean.strip()
        if _LINE_RANGE.match(stripped):
            continue  # anchored edit syntax [SEARCH: 45-49]
        if _FILE_PATH.search(stripped) and not ' ' in stripped:
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

    KEEP_HINT_THRESHOLD = 400  # lines — suggest KEEP above this
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

        # Always read from sandbox first
        sandbox_path = os.path.join(sandbox_dir, fpath)
        if os.path.isfile(sandbox_path):
            try:
                with open(sandbox_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                content = None

        # Fallback: project root, then CWD
        if content is None:
            full_path = os.path.join(project_root, fpath)
            content = read_file(full_path)
        if not content or content.startswith("["):
            content = read_file(fpath)

        # Binary / unreadable files return a [... — skipped] string — treat as missing
        if content and content.startswith("[BINARY") or (content and content.startswith("[READ ERROR")):
            output_parts.append(f"\n=== Code: {fpath} — {content.strip()} ===")
            continue

        if content is not None and not content.startswith("["):
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
                output_parts.append(
                    f"\n=== Code: {fpath} (lines {range_str} of {total_lines}) ===\n{combined}"
                )
            else:
                numbered = add_line_numbers(content)
                large_note = ""
                if total_lines > KEEP_HINT_THRESHOLD:
                    large_note = (
                        f"⚠ Big file ({total_lines} lines) — "
                        f"[KEEP: {fpath} X-Y, A-B] recommended to select only "
                        f"the lines you need.\n"
                    )
                output_parts.append(
                    f"\n=== Code: {fpath} ({total_lines} lines) ===\n"
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
) -> str:
    """Process [KEEP: filepath X-Y, A-B] tags.

    1. Parse filepath + line ranges from the tag argument
    2. Find the original file content (from persistent_lookups or disk)
    3. Build filtered view with preserved line numbers
    4. Run auto-RAG on kept lines
    5. REPLACE the CODE entry in persistent_lookups with the filtered view
    6. Return the filtered view + dependency summary
    """
    import os
    from tools.codebase import read_file, norm_path
    from workflows.code import _parse_keep_ranges, _filter_by_ranges, _auto_rag

    output_parts = []

    for arg in keep_args:
        arg = arg.strip()

        # Parse: "filepath X-Y, A-B" or "filepath X-Y A-B"
        # The filepath is everything before the first digit-dash-digit pattern
        range_match = re.search(r'(\d+)\s*-\s*(\d+)', arg)
        if not range_match:
            output_parts.append(f"=== KEEP: invalid format '{arg}' — use [KEEP: filepath X-Y, A-B] ===")
            continue

        filepath = arg[:range_match.start()].strip()
        ranges_text = arg[range_match.start():]
        filepath = norm_path(filepath)

        status(f"    KEEP: {filepath}")

        # Find original content — check persistent_lookups first
        original_content = None
        code_key = f"CODE:{filepath.strip().lower()}"

        # Search persistent_lookups for a matching CODE entry
        matched_key = None
        for key in persistent_lookups:
            if key.startswith("CODE:"):
                key_path = key[5:]
                if (key_path == filepath.strip().lower() or
                    key_path.endswith(filepath.strip().lower()) or
                    filepath.strip().lower().endswith(key_path)):
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

        if not raw_content:
            full_path = os.path.join(project_root, filepath)
            raw_content = read_file(full_path)
        if not raw_content or raw_content.startswith("["):
            raw_content = read_file(filepath)

        if not raw_content or raw_content.startswith("["):
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
    r'|\[STOP\]',
    re.IGNORECASE,
)


def _text_has_complete_tag(text: str) -> bool:
    """Return True if text contains at least one complete tool tag."""
    return bool(_ALL_TAGS.search(text))


def _strip_after_last_tag(text: str) -> str:
    """
    Strip any text the model wrote AFTER the last complete tool tag.
    That text was written without results — it's speculation.
    """
    matches = list(_ALL_TAGS.finditer(text))
    if not matches:
        return text
    last_match = matches[-1]
    # Keep everything up to and including the last tag
    return text[:last_match.end()]


_TOOL_USE_OPEN  = re.compile(r'\[tool use\]',  re.IGNORECASE)
_TOOL_USE_CLOSE = re.compile(r'\[/tool use\]', re.IGNORECASE)


def _autocomplete_tool_blocks(text: str) -> tuple[str, int]:
    """Close any unclosed [tool use] blocks by appending [/tool use] after
    the last tool tag or [STOP] in each block.

    Returns (fixed_text, number_of_blocks_fixed).
    A non-zero count means the model wrote [tool use] but forgot [/tool use].
    """
    opens  = list(_TOOL_USE_OPEN.finditer(text))
    closes = list(_TOOL_USE_CLOSE.finditer(text))
    unclosed = len(opens) - len(closes)
    if unclosed <= 0:
        return text, 0

    # Find the rightmost complete tool tag or [STOP] to insert after.
    all_markers = list(_ALL_TAGS.finditer(text))
    stop_markers = list(STOP_TAG.finditer(text))
    all_end_positions = [m.end() for m in all_markers + stop_markers]
    insert_pos = max(all_end_positions) if all_end_positions else len(text)

    closing = (' [/tool use]' * unclosed)
    fixed = text[:insert_pos] + closing + text[insert_pos:]
    return fixed, unclosed


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
    on_stop: "Callable[[str], None] | None" = None,
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

    # Per-round response text — used to build tagged round history in prompt.
    _round_texts: list[str] = []  # _round_texts[i] = text produced in round i+1

    def _stop_check(accumulated: str) -> bool:
        # [STOP] and [DONE] are written OUTSIDE [tool use] blocks.
        # Check them against raw unmasked text — block enforcement masking
        # converts their '[' to \x00, which makes STOP_TAG.search(masked) fail.
        if DONE_TAG.search(accumulated):
            return True
        if STOP_TAG.search(accumulated):
            return True

        # For planners/merger only: fire when a COMPLETE [tool use]...[/tool use]
        # block is present. Coders/reviewers/self-checkers must use [STOP] or
        # [DONE] — they write tool blocks mid-response (to read a file) and then
        # continue writing edits, so stopping on the block alone would cut them off.
        if stop_on_tool_block:
            masked = _mask_quoted_tags(accumulated)
            if _TOOL_USE_BLOCK.search(masked):
                return True

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

        # ── Check for [DONE] — model signals it's completely finished ──
        # If [DONE] appears, the model is done. Strip [DONE] and break.
        if DONE_TAG.search(result):
            result = DONE_TAG.sub('', result).rstrip()
            # Also strip [STOP] if present in same response
            result = STOP_TAG.sub('', result).rstrip()
            _round_texts.append(result)
            full_response += result
            _done_signaled = True
            break

        # ── Strip [STOP] tag from response (it's a signal, not content) ──
        has_stop = bool(STOP_TAG.search(result))
        result = STOP_TAG.sub('', result)
        _round_texts.append(result)

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
        if on_stop is not None:
            try:
                on_stop(full_response)
            except Exception as e:
                warn(f"  on_stop callback error: {e}")

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
        def _is_cached(k: str) -> bool:
            if k.split(':', 1)[0] in ('CODE', 'KEEP'):
                return False  # always fresh — file may have changed
            return k in local_research

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

        if code_tags and project_root:
            new_tags, cached = _cached_or_run("SEARCH", code_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("SEARCH", t,
                    lambda tag: _run_code_searches([tag], project_root))
                round_output += r

        if web_tags:
            new_tags, cached = _cached_or_run("WEBSEARCH", web_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("WEBSEARCH", t,
                    lambda tag: _run_web_searches([tag]))
                round_output += r

        if detail_tags and detailed_map:
            new_tags, cached = _cached_or_run("DETAIL", detail_tags)
            round_output += cached
            for t in new_tags:
                async def _detail_fn(tag):
                    return _run_detail_lookups([tag], detailed_map)
                r = await _locked_lookup("DETAIL", t, _detail_fn)
                round_output += r

        if file_tags and project_root:
            new_tags, cached = _cached_or_run("CODE", file_tags)
            round_output += cached
            for t in new_tags:
                async def _code_fn(tag):
                    return _run_code_reads([tag], project_root, viewed_versions=viewed_versions)
                r = await _locked_lookup("CODE", t, _code_fn)
                round_output += r

        if refs_tags and project_root:
            new_tags, cached = _cached_or_run("REFS", refs_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("REFS", t,
                    lambda tag: _run_refs_searches([tag], project_root))
                round_output += r

        if purpose_tags and purpose_map and project_root:
            new_tags, cached = _cached_or_run("PURPOSE", purpose_tags)
            round_output += cached
            for t in new_tags:
                async def _purpose_fn(tag):
                    return _run_purpose_lookups([tag], purpose_map, project_root)
                r = await _locked_lookup("PURPOSE", t, _purpose_fn)
                round_output += r

        if semantic_tags and purpose_map and project_root:
            new_tags, cached = _cached_or_run("SEMANTIC", semantic_tags)
            round_output += cached
            for t in new_tags:
                async def _semantic_fn(tag):
                    from pathlib import Path as _Path
                    from tools.code_index import _maps_dir, _load_all_code
                    from tools.embeddings import semantic_retrieve
                    maps_dir = _maps_dir(project_root)
                    _, file_hash = _load_all_code(project_root)
                    return await semantic_retrieve(
                        tag, purpose_map, project_root, maps_dir, file_hash, top_n=10
                    )
                r = await _locked_lookup("SEMANTIC", t, _semantic_fn)
                round_output += r

        if lsp_tags and project_root:
            new_tags, cached = _cached_or_run("LSP", lsp_tags)
            round_output += cached
            for t in new_tags:
                r = await _locked_lookup("LSP", t,
                    lambda tag: _run_lsp_searches([tag], project_root))
                round_output += r

        if knowledge_tags:
            new_tags, cached = _cached_or_run("KNOWLEDGE", knowledge_tags)
            round_output += cached
            for t in new_tags:
                async def _knowledge_fn(tag):
                    return _run_knowledge_lookups([tag])
                r = await _locked_lookup("KNOWLEDGE", t, _knowledge_fn)
                round_output += r

        # ── KEEP handler — replaces CODE entries in persistent_lookups ──
        if keep_tags and project_root:
            keep_result = await _run_keep(
                keep_tags, project_root,
                persistent_lookups, research_cache,
                viewed_versions=viewed_versions,
            )
            round_output += keep_result

        # Rebuild search_output from ALL persistent lookups.
        # This is the key mechanism: if KEEP replaced a CODE entry,
        # the full file is gone — only the kept ranges remain.
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

        # ── Build tagged round history ────────────────────────────────
        round_history_parts = []
        for i, rt in enumerate(_round_texts):
            rn = i + 1
            round_history_parts.append(
                f"[Thinking — Round {rn}]\n{rt}\n[/Thinking — Round {rn}]"
            )
        round_history = "\n\n".join(round_history_parts)

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
            manifest_lines.extend(_manifest_line(k, v) for k, v in _manifest.items())
        else:
            manifest_lines.append("  (no tool results yet)")
        manifest_lines.append(
            "⚠ HARD RULE: Any file or result NOT listed above is UNKNOWN to you.\n"
            "  Do NOT reference, quote, or reason about content you have not seen.\n"
            "  If you need a file that is not listed — request it with a tool tag.\n"
            "══"
        )
        manifest_str = "\n".join(manifest_lines)

        current_prompt = f"""{prompt}

══════════════════════════════════════════════════════════════════════
YOUR THINKING SO FAR — by round
══════════════════════════════════════════════════════════════════════
{round_history}

══════════════════════════════════════════════════════════════════════
RESULTS YOU REQUESTED — Round {round_num} results now available
══════════════════════════════════════════════════════════════════════
{search_output}

{manifest_str}
{budget_msg}

▶ Continue naturally depending on where you were:
  • Mid-plan or mid-analysis? Keep writing it — use the results above to fill the gap.
  • Your thinking ended with "waiting for results" or "let me check X"? You have the
    results now — no need to wait, proceed directly.
  • Just needed a quick fact? Use it and move on.

  Do NOT repeat or restate what you wrote above — pick up from where you left off.
  Do NOT re-request lookups already shown above — they are cached.
  Need more lookups? Write new tags and [STOP].
  Ready to submit? Write edits then [DONE] (or finish naturally if no edits)."""

    return {"model": model, "answer": full_response, "done": _done_signaled, "research": local_research}
