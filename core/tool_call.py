"""
Tool Call Loop — shared by all workflows.

Any AI can pause mid-thought to search:
  [SEARCH: pattern]    → ripgrep code search (coding agent)
  [WEBSEARCH: query]   → web search (research, chat)

JARVIS detects the tags, runs the searches, feeds results back,
and the AI continues from where it left off. Up to 5 rounds.
"""

import re
from core.retry import call_with_retry
from core.cli import step, status, warn


# ─── Tag Patterns ────────────────────────────────────────────────────────────

SEARCH_TAG = re.compile(r'\[SEARCH:\s*(.+?)\]', re.IGNORECASE)
WEBSEARCH_TAG = re.compile(r'\[WEBSEARCH:\s*(.+?)\]', re.IGNORECASE)
DETAIL_TAG = re.compile(r'\[DETAIL:\s*(.+?)\]', re.IGNORECASE)
CODE_TAG = re.compile(r'\[CODE:\s*(.+?)\]', re.IGNORECASE)
REFS_TAG = re.compile(r'\[REFS:\s*(.+?)\]', re.IGNORECASE)
PURPOSE_TAG = re.compile(r'\[PURPOSE:\s*(.+?)\]', re.IGNORECASE)
LSP_TAG = re.compile(r'\[LSP:\s*(.+?)\]', re.IGNORECASE)
KNOWLEDGE_TAG = re.compile(r'\[KNOWLEDGE:\s*(.+?)\]', re.IGNORECASE)


def extract_search_tags(text: str) -> list[str]:
    return SEARCH_TAG.findall(text)

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

def has_tool_tags(text: str) -> bool:
    return bool(SEARCH_TAG.search(text) or WEBSEARCH_TAG.search(text)
                or DETAIL_TAG.search(text) or CODE_TAG.search(text)
                or REFS_TAG.search(text) or PURPOSE_TAG.search(text)
                or LSP_TAG.search(text) or KNOWLEDGE_TAG.search(text))


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
    """Read actual source code files. [CODE: path/to/file.py] → full file content."""
    import os
    from tools.codebase import read_file, norm_path

    output_parts = []
    for fpath in filepaths:
        fpath = norm_path(fpath.strip())
        status(f"    Reading code: {fpath}")
        # Try relative to project root
        full_path = os.path.join(project_root, fpath)
        content = read_file(full_path)
        if content and not content.startswith("["):
            output_parts.append(f"\n=== Code: {fpath} ===\n{content}")
        else:
            # Try as-is (absolute or different relative)
            content = read_file(fpath)
            if content and not content.startswith("["):
                output_parts.append(f"\n=== Code: {fpath} ===\n{content}")
            else:
                output_parts.append(f"\n=== Code: {fpath} — FILE NOT FOUND ===")
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
    r'\[(SEARCH|WEBSEARCH|DETAIL|CODE|REFS|PURPOSE|LSP|KNOWLEDGE):\s*.+?\]',
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
    max_rounds: int = 10,
    enable_code_search: bool = True,
    enable_web_search: bool = True,
    detailed_map: str | None = None,
    purpose_map: str | None = None,
    research_cache: dict | None = None,
    log_label: str = "",
) -> dict:
    """
    Call a model with mid-thought tool use.

    The AI thinks, writes tool tags, then writes STOP.
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

    # ── Cycle Detector — finds repeating patterns of any length ─────────

    class CycleDetector:
        """
        Detects repeating sequences in streaming text.
        Tracks lines as they arrive. Checks if the last N lines have appeared
        before — for window sizes from 1 to 50. Triggers when a pattern
        repeats 3 times (exact or 90% word match).
        """
        def __init__(self):
            self.lines: list[str] = []
            self.raw = ""
            self.triggered = False
            self._check_interval = 0  # only check every 5 new lines

        def _words(self, text: str) -> set[str]:
            return set(text.lower().split())

        def _similarity(self, a: str, b: str) -> float:
            """Word-level overlap ratio between two strings."""
            wa, wb = self._words(a), self._words(b)
            if not wa or not wb:
                return 0.0
            return len(wa & wb) / max(len(wa), len(wb))

        def _block_match(self, block_a: list[str], block_b: list[str]) -> bool:
            """Check if two blocks of lines match (exact or 90% similar)."""
            if len(block_a) != len(block_b):
                return False
            # Try exact match first (fast)
            if block_a == block_b:
                return True
            # Word-level similarity on joined text
            text_a = "\n".join(block_a)
            text_b = "\n".join(block_b)
            return self._similarity(text_a, text_b) >= 0.90

        def feed(self, chunk: str) -> bool:
            """Feed a new text chunk. Returns True if loop detected."""
            self.raw += chunk
            # Split into lines, keep accumulating partial last line
            new_lines = self.raw.split('\n')
            if len(new_lines) <= 1:
                return False  # no complete new line yet

            # All complete lines except the last partial one
            complete = new_lines[:-1]
            self.raw = new_lines[-1]  # keep partial line

            for line in complete:
                stripped = line.strip()
                if stripped:  # skip empty lines
                    self.lines.append(stripped)

            self._check_interval += len(complete)
            if self._check_interval < 5:
                return False  # don't check on every single line
            self._check_interval = 0

            return self._check_cycles()

        def _check_cycles(self) -> bool:
            """Check if any pattern of length 1-50 has repeated 3 times."""
            n = len(self.lines)
            if n < 3:
                return False

            # Check window sizes: 1, 2, 3, 5, 8, 12, 18, 25, 35, 50
            for win_size in [1, 2, 3, 5, 8, 12, 18, 25, 35, 50]:
                if n < win_size * 3:
                    continue  # need at least 3 repetitions

                # Current window = last win_size lines
                current = self.lines[-win_size:]
                matches = 0

                # Slide backwards through history, step by win_size
                pos = n - win_size
                while pos >= win_size:
                    pos -= win_size
                    candidate = self.lines[pos:pos + win_size]
                    if self._block_match(current, candidate):
                        matches += 1
                        if matches >= 2:  # current + 2 earlier = 3 total
                            self.triggered = True
                            return True
                    else:
                        break  # non-matching block breaks the chain

            return False

    detector = CycleDetector()
    _loop_broken = False

    def _stop_on_stop_or_loop(accumulated: str) -> bool:
        nonlocal _loop_broken

        # Check for STOP keyword (normal tool use)
        if _text_has_complete_tag(accumulated):
            last_tag = list(_ALL_TAGS.finditer(accumulated))
            if last_tag:
                after = accumulated[last_tag[-1].end():]
                if re.search(r'\bSTOP\b', after, re.IGNORECASE):
                    _loop_broken = False
                    return True

        return False

    # We check cycles separately via the detector fed from streaming chunks
    # The stop_check only sees accumulated text, but we need per-chunk feeding
    # So we wrap the stop_check to also feed the detector

    _last_len = 0

    def _stop_check_with_cycle(accumulated: str) -> bool:
        nonlocal _loop_broken, _last_len

        # Feed new chunk to cycle detector
        new_text = accumulated[_last_len:]
        _last_len = len(accumulated)

        if new_text and detector.feed(new_text):
            warn(f"Cycle detected: repeating pattern found ({len(detector.lines)} lines analyzed) — interrupting")
            _loop_broken = True
            return True

        return _stop_on_stop_or_loop(accumulated)

    for round_num in range(1, max_rounds + 1):
        _last_len = 0  # reset per round
        result = await call_with_retry(
            model, current_prompt, max_tokens=max_tokens,
            stop_check=_stop_check_with_cycle,
            log_label=log_label,
        )

        # If we interrupted a cycle, inject a break prompt and continue
        if _loop_broken:
            _loop_broken = False
            detector.lines.clear()  # reset detector for fresh start
            detector.raw = ""
            full_response += result
            status(f"  Injecting loop-break, continuing from round {round_num}...")

            current_prompt = f"""{prompt}

YOUR PREVIOUS THINKING (you wrote this — continue from where you stopped):
{full_response}

IMPORTANT: You were repeating yourself — writing the same lines in a loop.
You already covered this. Move forward to your NEXT step.
Do NOT re-analyze what you already analyzed above.
Pick up from where you left off and produce your output."""
            continue

        # Only process tags if the AI explicitly wrote STOP after them.
        # If the stream completed naturally (no STOP), any tags in the text
        # are part of the content (e.g. plan text mentioning "[SEARCH: ...]"),
        # not actual tool requests.
        stopped_by_stop = bool(
            _text_has_complete_tag(result)
            and re.search(r'\bSTOP\b', result[result.rfind(']'):], re.IGNORECASE)
        )

        if stopped_by_stop:
            code_tags = list(dict.fromkeys(extract_search_tags(result))) if enable_code_search else []
            web_tags = list(dict.fromkeys(extract_websearch_tags(result))) if enable_web_search else []
            detail_tags = list(dict.fromkeys(extract_detail_tags(result))) if detailed_map else []
            file_tags = list(dict.fromkeys(extract_code_tags(result))) if project_root else []
            refs_tags = list(dict.fromkeys(extract_refs_tags(result))) if project_root else []
            purpose_tags = list(dict.fromkeys(extract_purpose_tags(result))) if purpose_map else []
            lsp_tags = list(dict.fromkeys(extract_lsp_tags(result))) if project_root else []
            knowledge_tags = list(dict.fromkeys(extract_knowledge_tags(result)))

            # Cap: max 8 tags per type per round
            MAX_TAGS = 8
            code_tags = code_tags[:MAX_TAGS]
            web_tags = web_tags[:MAX_TAGS]
            detail_tags = detail_tags[:MAX_TAGS]
            file_tags = file_tags[:MAX_TAGS]
            refs_tags = refs_tags[:MAX_TAGS]
            purpose_tags = purpose_tags[:MAX_TAGS]
            lsp_tags = lsp_tags[:MAX_TAGS]
            knowledge_tags = knowledge_tags[:MAX_TAGS]

            # Trim everything after STOP
            last_bracket = result.rfind(']')
            result = result[:last_bracket + 1]
        else:
            code_tags = web_tags = detail_tags = file_tags = refs_tags = purpose_tags = lsp_tags = knowledge_tags = []

        has_tags = bool(code_tags or web_tags or detail_tags or file_tags or refs_tags or purpose_tags or lsp_tags or knowledge_tags)

        full_response += result

        if not has_tags:
            break  # No more tool requests — done

        # Run requested lookups — check cache first
        search_output = ""

        # Filter out tags that are already in the cache
        def _cached_or_run(tag_type: str, tags: list[str]) -> tuple[list[str], str]:
            """Returns (uncached_tags, cached_output)."""
            if research_cache is None:
                return tags, ""
            cached_out = ""
            new_tags = []
            for tag in tags:
                key = f"{tag_type}:{tag.strip().lower()}"
                if key in research_cache:
                    cached_out += research_cache[key]
                else:
                    new_tags.append(tag)
            return new_tags, cached_out

        def _store(tag_type: str, tag: str, result: str):
            """Store a result in the shared cache and local research."""
            key = f"{tag_type}:{tag.strip().lower()}"
            local_research[key] = result
            if research_cache is not None:
                research_cache[key] = result

        total = len(code_tags) + len(web_tags) + len(detail_tags) + len(file_tags) + len(refs_tags) + len(purpose_tags) + len(lsp_tags) + len(knowledge_tags)
        status(f"  Tool use round {round_num}: {total} lookups")

        if code_tags and project_root:
            new_tags, cached = _cached_or_run("SEARCH", code_tags)
            search_output += cached
            for t in new_tags:
                r = await _run_code_searches([t], project_root)
                _store("SEARCH", t, r)
                search_output += r

        if web_tags:
            new_tags, cached = _cached_or_run("WEBSEARCH", web_tags)
            search_output += cached
            for t in new_tags:
                r = await _run_web_searches([t])
                _store("WEBSEARCH", t, r)
                search_output += r

        if detail_tags and detailed_map:
            new_tags, cached = _cached_or_run("DETAIL", detail_tags)
            search_output += cached
            for t in new_tags:
                r = _run_detail_lookups([t], detailed_map)
                _store("DETAIL", t, r)
                search_output += r

        if file_tags and project_root:
            new_tags, cached = _cached_or_run("CODE", file_tags)
            search_output += cached
            for t in new_tags:
                r = _run_code_reads([t], project_root)
                _store("CODE", t, r)
                search_output += r

        if refs_tags and project_root:
            new_tags, cached = _cached_or_run("REFS", refs_tags)
            search_output += cached
            for t in new_tags:
                r = await _run_refs_searches([t], project_root)
                _store("REFS", t, r)
                search_output += r

        if purpose_tags and purpose_map and project_root:
            new_tags, cached = _cached_or_run("PURPOSE", purpose_tags)
            search_output += cached
            for t in new_tags:
                r = _run_purpose_lookups([t], purpose_map, project_root)
                _store("PURPOSE", t, r)
                search_output += r

        if lsp_tags and project_root:
            new_tags, cached = _cached_or_run("LSP", lsp_tags)
            search_output += cached
            for t in new_tags:
                r = await _run_lsp_searches([t], project_root)
                _store("LSP", t, r)
                search_output += r

        if knowledge_tags:
            new_tags, cached = _cached_or_run("KNOWLEDGE", knowledge_tags)
            search_output += cached
            for t in new_tags:
                r = _run_knowledge_lookups([t])
                _store("KNOWLEDGE", t, r)
                search_output += r

        if not search_output.strip():
            break

        # Build continuation prompt
        current_prompt = f"""{prompt}

YOUR PREVIOUS THINKING (you wrote this — continue from where you stopped):
{full_response}

LOOKUP RESULTS (you requested these mid-thought):
{search_output}

Continue from where you left off. You now have the results.
If you need MORE info, write new tags then STOP. Do NOT repeat tags you already used.
Do NOT repeat what you already wrote above — just continue."""

    return {"model": model, "answer": full_response, "research": local_research}
