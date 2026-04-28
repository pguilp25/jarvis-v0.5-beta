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
LSP_TAG = re.compile(r'\[LSP:\s*(.+?)\]', re.IGNORECASE)
KNOWLEDGE_TAG = re.compile(r'\[KNOWLEDGE:\s*(.+?)\]', re.IGNORECASE)
# KEEP strips a previously-loaded [CODE:] result to only the specified line
# ranges, removing the full file from context.  Format:
#   [KEEP: filepath 10-50, 80-120]
KEEP_TAG = re.compile(r'\[KEEP:\s*(.+?)\]', re.IGNORECASE)
# DONE signals the model is finished — no more tool processing.
DONE_TAG = re.compile(r'\[DONE\]', re.IGNORECASE)


def extract_search_tags(text: str) -> list[str]:
    # Exclude pure line-range patterns like "339-342" or "25-28" — these are
    # [SEARCH: N-M] anchored edit syntax, not code search queries.  Routing
    # them to ripgrep produces garbage results and causes the model to loop.
    _LINE_RANGE = re.compile(r'^\d+\s*-\s*\d+$')
    return [q for q in SEARCH_TAG.findall(text) if not _LINE_RANGE.match(q.strip())]

def extract_websearch_tags(text: str) -> list[str]:
    return WEBSEARCH_TAG.findall(text)

def extract_detail_tags(text: str) -> list[str]:
    return DETAIL_TAG.findall(text)

def extract_code_tags(text: str) -> list[str]:
    return CODE_TAG.findall(text)

def extract_refs_tags(text: str) -> list[str]:
    return REFS_TAG.findall(text)

def extract_purpose_tags(text: str) -> list[str]:
    return PURPOSE_TAG.findall(text)

def extract_lsp_tags(text: str) -> list[str]:
    return LSP_TAG.findall(text)

def extract_knowledge_tags(text: str) -> list[str]:
    return KNOWLEDGE_TAG.findall(text)

def extract_keep_tags(text: str) -> list[str]:
    return KEEP_TAG.findall(text)

def has_tool_tags(text: str) -> bool:
    return bool(SEARCH_TAG.search(text) or WEBSEARCH_TAG.search(text)
                or DETAIL_TAG.search(text) or CODE_TAG.search(text)
                or REFS_TAG.search(text) or PURPOSE_TAG.search(text)
                or LSP_TAG.search(text) or KNOWLEDGE_TAG.search(text)
                or KEEP_TAG.search(text))


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

def _run_code_reads(filepaths: list[str], project_root: str) -> str:
    """Read source code files from the sandbox.

    Always reads from .jarvis_sandbox/ — that's the working copy where
    all edits are applied. The real project is untouched.
    """
    import os
    from tools.codebase import read_file, norm_path, add_line_numbers

    KEEP_HINT_THRESHOLD = 400  # lines — suggest KEEP above this
    sandbox_dir = os.path.join(project_root, ".jarvis_sandbox")

    output_parts = []
    for fpath in filepaths:
        fpath = norm_path(fpath.strip())
        status(f"    Reading code: {fpath}")

        content = None

        # Always read from sandbox
        sandbox_path = os.path.join(sandbox_dir, fpath)
        if os.path.isfile(sandbox_path):
            try:
                with open(sandbox_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                content = None

        # Fallback: sandbox might not have this file (not in project)
        if content is None:
            full_path = os.path.join(project_root, fpath)
            content = read_file(full_path)
        if not content or content.startswith("["):
            content = read_file(fpath)

        if content and not content.startswith("["):
            numbered = add_line_numbers(content)
            line_count = content.count('\n') + 1
            output_parts.append(f"\n=== Code: {fpath} ({line_count} lines) ===\n{numbered}")

            if line_count > KEEP_HINT_THRESHOLD:
                output_parts.append(
                    f"\n⚠ {fpath} is large ({line_count} lines). "
                    f"Use [KEEP: {fpath} X-Y, A-B] to select only the line "
                    f"ranges you need — the full file will be replaced with just "
                    f"those ranges, freeing context for your actual work.\n"
                    f"Example: [KEEP: {fpath} 1-15, 200-250, 310-340]"
                )
        else:
            output_parts.append(f"\n=== Code: {fpath} — FILE NOT FOUND ===")
    return "\n".join(output_parts)


# ─── KEEP Handler ────────────────────────────────────────────────────────────

async def _run_keep(
    keep_args: list[str], project_root: str,
    persistent_lookups: dict[str, str],
    research_cache: dict | None = None,
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
    r'\[(SEARCH|WEBSEARCH|DETAIL|CODE|REFS|PURPOSE|LSP|KNOWLEDGE|KEEP):\s*.+?\]',
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


# ─── Main Tool Call Loop ────────────────────────────────────────────────────

async def call_with_tools(
    model: str,
    prompt: str,
    project_root: str | None = None,
    max_tokens: int = 16384,
    max_rounds: int = 30,
    enable_code_search: bool = True,
    enable_web_search: bool = True,
    detailed_map: str | None = None,
    purpose_map: str | None = None,
    research_cache: dict | None = None,
    log_label: str = "",
) -> dict:
    """
    Call a model with mid-thought tool use.

    The AI thinks, writes tool tags (e.g. [CODE: ...], [REFS: ...]).
    JARVIS runs ALL requested lookups at once and feeds results back.

    Tool tags:
      [SEARCH: pattern]       → code search
      [WEBSEARCH: query]      → web search
      [DETAIL: section name]  → detailed code map lookup
      [CODE: path/to/file]    → read actual source code file
      [REFS: name]            → find all definitions, imports, usages
      [PURPOSE: category]     → all code serving a purpose

    research_cache: shared dict that accumulates all lookup results across
    multiple AI calls. Same tag won't re-run if cached.

    Returns {"model": str, "answer": str, "research": {tag_key: result}}.
    """
    full_response = ""
    current_prompt = prompt
    # Track this call's research (also writes to shared cache if provided)
    local_research: dict[str, str] = {}
    # Persistent lookup results — survives across rounds. Keyed by "TYPE:arg".
    # When [KEEP:] fires, it REPLACES the corresponding [CODE:] entry, removing
    # the full file from context and inserting only the kept ranges.
    persistent_lookups: dict[str, str] = {}

    def _stop_check(accumulated: str) -> bool:
        # Check for [DONE] — model is finished, stop immediately
        if DONE_TAG.search(accumulated):
            return True

        # Check for STOP keyword after a tag (legacy, some models still write it)
        if _text_has_complete_tag(accumulated):
            last_tag = list(_ALL_TAGS.finditer(accumulated))
            if last_tag:
                after = accumulated[last_tag[-1].end():]
                if re.search(r'\bSTOP\b', after, re.IGNORECASE):
                    return True

        # Check if response ends with a complete tool tag + minimal trailing text.
        # This catches models that write "[CODE: file.py]\n" without STOP.
        stripped = accumulated.rstrip()
        if stripped.endswith(']') and _text_has_complete_tag(stripped):
            last_tag = list(_ALL_TAGS.finditer(stripped))
            if last_tag:
                tag = last_tag[-1]
                tag_text = stripped[tag.start():tag.end()]
                after_tag = stripped[tag.end():].strip()
                if not after_tag:
                    # Edit-syntax exception 1: [SEARCH: N-M] is an edit anchor,
                    # not a tool call. The post-stream parser already handles this
                    # via `has_misused_search`, but if we stop here the model never
                    # gets to write the [/SEARCH] [REPLACE] [/REPLACE] that follows.
                    if re.match(r'\[SEARCH:\s*\d+\s*-\s*\d+\s*\]\Z',
                                tag_text, re.IGNORECASE):
                        return False
                    # Edit-syntax exception 2: ANY tag that appears inside an
                    # open === EDIT: === block is edit content (the model may
                    # be editing a file that literally contains [CODE:] etc.).
                    # An edit block is "open" when the most recent === EDIT: marker
                    # is more recent than the most recent [/REPLACE] closer.
                    last_edit_open = stripped.rfind('=== EDIT:')
                    last_replace_close = stripped.rfind('[/REPLACE]')
                    if last_edit_open != -1 and last_edit_open > last_replace_close:
                        return False
                    return True

        return False

    for round_num in range(1, max_rounds + 1):
        result = await call_with_retry(
            model, current_prompt, max_tokens=max_tokens,
            stop_check=_stop_check,
            log_label=log_label,
        )

        # ── Check for [DONE] — model signals it's finished ───────────
        # If [DONE] appears anywhere, the model is done. Any tag-like text
        # in the response (e.g. plan mentioning "[CODE: path]") is content,
        # not a tool request. Strip [DONE] and break.
        if DONE_TAG.search(result):
            result = DONE_TAG.sub('', result).rstrip()
            full_response += result
            break

        # ── Detect tool tags anywhere in the response ────────────────
        # If the model wrote [CODE: ...] or [REFS: ...] anywhere in its
        # response, we process them.
        code_tags = list(dict.fromkeys(extract_search_tags(result))) if enable_code_search else []
        web_tags = list(dict.fromkeys(extract_websearch_tags(result))) if enable_web_search else []
        detail_tags = list(dict.fromkeys(extract_detail_tags(result))) if detailed_map else []
        file_tags = list(dict.fromkeys(extract_code_tags(result))) if project_root else []
        refs_tags = list(dict.fromkeys(extract_refs_tags(result))) if project_root else []
        purpose_tags = list(dict.fromkeys(extract_purpose_tags(result))) if purpose_map else []
        lsp_tags = list(dict.fromkeys(extract_lsp_tags(result))) if project_root else []
        knowledge_tags = list(dict.fromkeys(extract_knowledge_tags(result)))
        keep_tags = list(dict.fromkeys(extract_keep_tags(result))) if project_root else []

        has_tags = bool(code_tags or web_tags or detail_tags or file_tags
                        or refs_tags or purpose_tags or lsp_tags
                        or knowledge_tags or keep_tags)

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

        # Run requested lookups — check cache first
        round_output = ""  # results from THIS round only (for logging)

        # Filter out tags that are already in the cache
        def _cached_or_run(tag_type: str, tags: list[str]) -> tuple[list[str], str]:
            """Returns (uncached_tags, cached_output).
            CODE and KEEP always re-run — files may have changed between rounds."""
            # CODE and KEEP always re-run (file content may have changed)
            if tag_type in ("CODE", "KEEP"):
                return tags, ""
            if research_cache is None:
                return tags, ""
            cached_out = ""
            new_tags = []
            for tag in tags:
                key = f"{tag_type}:{tag.strip().lower()}"
                if key in research_cache:
                    cached_out += research_cache[key]
                    # Also populate persistent_lookups so the cached result
                    # ends up in search_output and actually reaches the model.
                    # Without this, cached REFS/LSP/etc. results silently
                    # disappear and the model loops asking for the same tool.
                    persistent_lookups[key] = research_cache[key]
                else:
                    new_tags.append(tag)
            return new_tags, cached_out

        def _store(tag_type: str, tag: str, result: str):
            """Store a result in the shared cache, local research, and persistent lookups."""
            key = f"{tag_type}:{tag.strip().lower()}"
            local_research[key] = result
            persistent_lookups[key] = result
            if research_cache is not None:
                research_cache[key] = result

        async def _locked_lookup(tag_type: str, tag: str, run_fn) -> str:
            """Run a lookup with a per-key lock to prevent duplicate concurrent executions."""
            key = f"{tag_type}:{tag.strip().lower()}"
            if key not in _inflight_locks:
                _inflight_locks[key] = asyncio.Lock()
            lock = _inflight_locks[key]

            async with lock:
                # CODE always re-runs — file may have changed (e.g. after edits)
                if tag_type not in ("CODE", "KEEP"):
                    if research_cache is not None and key in research_cache:
                        result = research_cache[key]
                        persistent_lookups[key] = result
                        return result
                # Run the lookup
                result = run_fn(tag)
                if asyncio.iscoroutine(result):
                    result = await result
                _store(tag_type, tag, result)
                return result

        total = len(code_tags) + len(web_tags) + len(detail_tags) + len(file_tags) + len(refs_tags) + len(purpose_tags) + len(lsp_tags) + len(knowledge_tags) + len(keep_tags)
        status(f"  Tool use round {round_num}: {total} lookups")

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
                    return _run_code_reads([tag], project_root)
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

        # Build continuation prompt
        current_prompt = f"""{prompt}

YOUR PREVIOUS THINKING (you wrote this — continue from where you stopped):
{full_response}

LOOKUP RESULTS (you requested these mid-thought):
{search_output}

Continue from where you left off. You now have the results.
If you need MORE info, write new tags. You can re-read files if needed.
When you are FINISHED (your final answer is complete), write [DONE] at the end.
Do NOT repeat what you already wrote above — just continue."""

    return {"model": model, "answer": full_response, "research": local_research}
