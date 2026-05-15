"""
Code Indexer — Two-Level Map System

Processes the codebase in ~100K token batches using Nemotron Super (128K context).
Each batch generates partial maps, which are then merged into:

1. GENERAL MAP (~2-4K tokens):
   High-level overview by feature/component.
   Every AI gets this in context — cheap, always available.

2. DETAILED MAP (~20-50K tokens):
   Function signatures, logic flow, variables, dependencies.
   Split into named SECTIONS that AIs can request on demand:
   [DETAIL: snake movement] → returns just that section.

Small projects (< 100K tokens): single batch, no merge needed.
Large projects: batches processed in parallel, then merged.

Maps are cached in .jarvis/maps/ — regenerated only when files change.
"""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from core.retry import call_with_retry
from core.cli import step, status, success, warn
from tools.codebase import scan_project, read_file, IGNORE_DIRS, IGNORE_EXTENSIONS, _is_ignored_dir


# ─── Configuration ───────────────────────────────────────────────────────────

INDEX_MODEL = "nvidia/glm-5.1"       # 128K context — process in batches
INDEX_FALLBACK = "nvidia/deepseek-v4-pro"    # Fallback
BATCH_TOKEN_LIMIT = 100_000                # Max tokens per batch
MAPS_DIR_NAME = ".jarvis/maps"


# ─── Prompts ─────────────────────────────────────────────────────────────────

GENERAL_MAP_PROMPT = """You are a senior architect analyzing an entire codebase.

WHY YOU EXIST: You are the FIRST step in a multi-AI coding pipeline.
Your general map will be shown to EVERY AI in the system — planners, coders,
and reviewers. They use it to understand what the project does and which
files matter for any given task. If you miss a component, the planner won't
know it exists and the coder won't touch it. If you mislabel a feature,
the planner will misunderstand the architecture.

YOUR OUTPUT is shown at the top of every prompt as "PROJECT OVERVIEW".
It needs to be SHORT (2000-4000 tokens) because it's repeated many times.
But it must be COMPLETE — every feature and component must be listed.

Read ALL the code below and produce a HIGH-LEVEL OVERVIEW of the project.

Describe the project by FEATURES and COMPONENTS, not by files.
Think like a product manager explaining what this software does.

For each feature/component, write:
- What it does (in plain English)
- Which files are involved
- How it connects to other components

Example format:
## Snake Movement
Controls how the snake moves on the grid. Handles direction changes,
speed increases, and wall collision. Files: game.js (moveSnake, checkCollision)

## Score System
Tracks and displays player score. Increases when food is eaten.
Files: game.js (updateScore), index.html (score display)

Keep it SHORT but COMPLETE. Target: 2000-4000 tokens maximum.

═══ FULL PROJECT CODE ═══
{all_code}
"""

DETAILED_MAP_PROMPT = """You are a senior architect creating a DETAILED technical map of a codebase.

WHY YOU EXIST: You are part of a multi-AI coding pipeline. When a planner or
coder needs to understand a specific feature, they write [DETAIL: feature name]
and your map is looked up to give them the answer. Your sections are the
REFERENCE that other AIs use instead of reading raw source code.

If your map is wrong about a function's parameters, the coder will write
broken code. If you miss a dependency, the planner won't account for side
effects. If your logic descriptions are vague, the coder will guess.

YOUR OUTPUT is stored and queried by section name. Each section must be
self-contained — an AI reading just that section should understand everything
about that feature without needing the source code.

Read ALL the code below and produce a structured breakdown.

For each feature/component in the project, create a SECTION with:
- Section name (matching the general map)
- Every function/class involved, with:
  - File path and line numbers (approximate)
  - Parameters and return types
  - What it does (1-2 sentences)
  - Key variables and their purpose
  - Dependencies (what it calls, what calls it)

Format EACH section like this:
=== SECTION: Feature Name ===
### file.py — function_name(params)
  Purpose: what it does
  Variables: key_var (type) — what it stores
  Calls: other_function, third_function
  Called by: main_loop
  Logic: brief description of the algorithm

=== SECTION: Another Feature ===
...

Be THOROUGH — include enough detail that an AI can modify the code without
reading the original file. Get function signatures RIGHT. Get parameters RIGHT.

═══ FULL PROJECT CODE ═══
{all_code}
"""

PURPOSE_MAP_PROMPT = """You are creating a PURPOSE-BASED INDEX of a codebase.

WHAT THIS IS FOR: When another AI needs to modify "all the colors" or "all the
API calls" or "all the error handling", it queries this index and gets back the
EXACT code snippets. If you miss a location, the AI will NOT find it and will
NOT update it. Missing entries cause bugs.

YOUR JOB: Read ALL the code below. Every line has a number. For each meaningful
purpose you find, list EVERY location that serves that purpose with EXACT
first and last line numbers.

CRITICAL RULES:

1. LINE NUMBERS MUST BE EXACT.
   Look at the number at the start of each line (e.g. "  42 | def foo():").
   Use THOSE numbers. The system will extract code at these exact lines.
   Wrong line numbers = wrong code shown to the AI = broken edits.

2. INCLUDE THE FULL RANGE.
   If a function that does API calls spans lines 50-120, write LINES: 50-120.
   Do NOT write just LINES: 50-55 to "summarize". The AI needs to see ALL of it.
   Include the function signature, the body, and the closing.

3. MISS NOTHING.
   The AI will ASSUME that your list is COMPLETE. If you list 3 places where
   colors are defined but there's a 4th you missed, the AI will change 3 and
   leave the 4th inconsistent. Scan the ENTIRE code for each category.
   Check every file. Check imports, constants, inline values, comments.

4. CATEGORIES MUST BE PRECISE AND NARROW.
   Bad: "UI" (too broad — would return half the code)
   Bad: "styles" (too vague)
   Good: "CSS color variables and theme colors"
   Good: "WebSocket message handlers"
   Good: "API endpoint URL definitions"
   Good: "error messages shown to user"
   A category should return 5-50 specific locations, not 200.
   If a category would be too large, split it.

5. EVERY LINE OF CODE BELONGS TO AT LEAST ONE CATEGORY.
   If you can't categorize something, create a more specific category for it.
   Don't leave code uncategorized.

6. OVERLAPPING IS OK.
   The same lines can appear in multiple categories if they serve multiple purposes.
   A function that does an API call AND handles errors can appear in both
   "API client calls" and "error handling".

FORMAT:

=== PURPOSE: Category Name ===
Description: one sentence explaining exactly what this category covers.

FILE: path/to/file.ext
  LINES: 10-25 — what these lines do (be specific)
  LINES: 88-102 — what these lines do

FILE: path/to/other.ext
  LINES: 5-18 — what these lines do

=== PURPOSE: Another Category ===
Description: ...
...

PROCESS:
1. First, scan all the code and identify what this project does
2. Create a list of precise categories that cover everything
3. Go through EVERY file, EVERY function, EVERY block of code
4. For each, ask: "what purpose does this serve?" and add it to the right category
5. Double-check: for each category, did I find ALL occurrences across ALL files?

═══ FULL PROJECT CODE (with line numbers) ═══
{all_code}
"""


# ─── Load All Code ──────────────────────────────────────────────────────────

def _load_all_code(project_root: str) -> tuple[str, str]:
    """
    Load all code files from the project into a single string.
    Returns (all_code_text, file_hash) where hash detects changes.
    """
    root = Path(project_root).resolve()
    parts = []
    hasher = hashlib.md5()

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames) if not _is_ignored_dir(d)]

        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            if fpath.suffix in IGNORE_EXTENSIONS:
                continue
            if fpath.stat().st_size > 100_000:  # Skip files > 100KB
                continue

            rel = fpath.relative_to(root)
            content = read_file(str(fpath))
            if content and not content.startswith("["):
                header = f"\n{'═' * 60}\n══ {rel} ══\n{'═' * 60}\n"
                parts.append(header + content)
                hasher.update(content.encode("utf-8", errors="replace"))

    all_code = "\n".join(parts)
    file_hash = hasher.hexdigest()
    return all_code, file_hash


def _load_all_code_numbered(project_root: str) -> str:
    """Load all code with line numbers prepended to each line."""
    root = Path(project_root).resolve()
    parts = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames) if not _is_ignored_dir(d)]

        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            if fpath.suffix in IGNORE_EXTENSIONS:
                continue
            if fpath.stat().st_size > 100_000:
                continue

            rel = fpath.relative_to(root)
            content = read_file(str(fpath))
            if content and not content.startswith("["):
                header = f"\n{'═' * 60}\n══ {rel} ══\n{'═' * 60}"
                lines = content.split('\n')
                numbered = '\n'.join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
                parts.append(header + '\n' + numbered)

    return "\n".join(parts)


def _chunk_code(all_code: str, max_tokens: int = BATCH_TOKEN_LIMIT) -> list[str]:
    """
    Split code into chunks that fit within token limits.
    Splits at file boundaries (═══ markers) so files stay whole.
    Returns list of chunks, each under max_tokens.
    """
    from core.tokens import count_tokens

    # Split into individual files
    file_marker = "═" * 60
    raw_parts = all_code.split(file_marker)

    # Reassemble into file blocks (marker + content + marker)
    files = []
    i = 0
    while i < len(raw_parts):
        part = raw_parts[i]
        if part.strip().startswith("══ ") and part.strip().endswith(" ══"):
            # This is a header like "══ path/to/file ══"
            file_block = file_marker + part + file_marker
            if i + 1 < len(raw_parts):
                file_block += raw_parts[i + 1]
                i += 2
            else:
                i += 1
            files.append(file_block)
        else:
            i += 1

    if not files:
        return [all_code] if all_code.strip() else []

    # Pack files into chunks under the token limit
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for file_block in files:
        file_tokens = count_tokens(file_block)

        # Single file exceeds limit — truncate it and put it alone
        if file_tokens > max_tokens:
            if current_chunk.strip():
                chunks.append(current_chunk)
                current_chunk = ""
                current_tokens = 0
            # Truncate to roughly max_tokens (4 chars per token estimate)
            truncated = file_block[:max_tokens * 4]
            chunks.append(truncated)
            continue

        # Would this file push us over the limit?
        if current_tokens + file_tokens > max_tokens and current_chunk.strip():
            chunks.append(current_chunk)
            current_chunk = ""
            current_tokens = 0

        current_chunk += file_block
        current_tokens += file_tokens

    if current_chunk.strip():
        chunks.append(current_chunk)

    return chunks if chunks else [all_code]


# ─── Cache Management ───────────────────────────────────────────────────────

def _maps_dir(project_root: str) -> Path:
    p = Path(project_root) / MAPS_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_cached_maps(project_root: str) -> dict | None:
    """Load cached maps if they exist and hash matches."""
    maps_dir = _maps_dir(project_root)
    meta_path = maps_dir / "meta.json"

    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
        general_path = maps_dir / "general_map.md"
        detailed_path = maps_dir / "detailed_map.md"
        purpose_path = maps_dir / "purpose_map.md"

        if not general_path.exists() or not detailed_path.exists():
            return None

        return {
            "hash": meta.get("hash", ""),
            "general": general_path.read_text(),
            "detailed": detailed_path.read_text(),
            "purpose": purpose_path.read_text() if purpose_path.exists() else "",
            "timestamp": meta.get("timestamp", 0),
        }
    except Exception:
        return None


def _save_maps(project_root: str, file_hash: str, general: str, detailed: str, purpose: str = ""):
    """Save maps to cache."""
    maps_dir = _maps_dir(project_root)

    (maps_dir / "general_map.md").write_text(general, encoding="utf-8")
    (maps_dir / "detailed_map.md").write_text(detailed, encoding="utf-8")
    if purpose:
        (maps_dir / "purpose_map.md").write_text(purpose, encoding="utf-8")
    (maps_dir / "meta.json").write_text(json.dumps({
        "hash": file_hash,
        "timestamp": time.time(),
    }))


def patch_maps(project_root: str, general: str, detailed: str):
    """
    Save AI-updated maps and stamp them with the CURRENT code hash.
    Called after applying changes — the coding AI already updated the maps
    based on what it changed, so no need to re-read all the code.
    
    Next time generate_maps() is called:
      - If no external changes → hash matches → uses these patched maps (instant)
      - If external changes → hash mismatch → full rescan with UltraLong-8B
    """
    _, current_hash = _load_all_code(project_root)
    _save_maps(project_root, current_hash, general, detailed)
    return current_hash


# ─── Generate Maps ──────────────────────────────────────────────────────────

async def generate_maps(project_root: str, force: bool = False) -> dict:
    """
    Generate the three-level code maps.
    Processes codebase in ~100K token batches, then merges.
    Returns {"general": str, "detailed": str, "purpose": str}.
    Caches results — only regenerates when files change.
    """
    step("Code Indexer")

    # Load all code
    all_code, file_hash = _load_all_code(project_root)

    if not all_code.strip():
        status("Empty project — no maps to generate")
        return {"general": "(empty project)", "detailed": "(empty project)", "purpose": ""}

    code_chars = len(all_code)
    status(f"Project: {code_chars:,} chars of code loaded")

    # Check cache
    if not force:
        cached = _load_cached_maps(project_root)
        if cached and cached["hash"] == file_hash:
            status("Maps cached and up-to-date — using cache")
            return {"general": cached["general"], "detailed": cached["detailed"],
                    "purpose": cached.get("purpose", "")}
        elif cached:
            status("Files changed since last index — regenerating")

    # Chunk the code (for general + detailed maps)
    chunks = _chunk_code(all_code)
    status(f"Split into {len(chunks)} batch(es)")

    # Load numbered code for purpose map (separate chunking)
    numbered_code = _load_all_code_numbered(project_root)
    purpose_chunks = _chunk_code(numbered_code)

    model = INDEX_MODEL

    if len(chunks) == 1:
        # ── Single batch — generate all 3 maps in parallel ──
        step(f"Generating maps (single batch, {code_chars:,} chars)...")

        general_task = call_with_retry(
            model, GENERAL_MAP_PROMPT.format(all_code=chunks[0]), max_tokens=8192,
            log_label="indexing: general map"
        )
        detailed_task = call_with_retry(
            model, DETAILED_MAP_PROMPT.format(all_code=chunks[0]), max_tokens=16384,
            log_label="indexing: detailed map"
        )
        purpose_task = call_with_retry(
            model, PURPOSE_MAP_PROMPT.format(all_code=purpose_chunks[0]), max_tokens=32768,
            log_label="indexing: purpose map"
        )
        general, detailed, purpose = await asyncio.gather(
            general_task, detailed_task, purpose_task
        )

    else:
        # ── Multi-batch — generate partial maps, then merge ──
        step(f"Generating partial maps ({len(chunks)} batches in parallel)...")

        general_tasks = [
            call_with_retry(model, GENERAL_MAP_PROMPT.format(all_code=chunk), max_tokens=4096,
                            log_label=f"indexing: general map (batch {i+1})")
            for i, chunk in enumerate(chunks)
        ]
        detailed_tasks = [
            call_with_retry(model, DETAILED_MAP_PROMPT.format(all_code=chunk), max_tokens=8192,
                            log_label=f"indexing: detailed map (batch {i+1})")
            for i, chunk in enumerate(chunks)
        ]
        purpose_tasks = [
            call_with_retry(model, PURPOSE_MAP_PROMPT.format(all_code=pc), max_tokens=16384,
                            log_label=f"indexing: purpose map (batch {i+1})")
            for i, pc in enumerate(purpose_chunks)
        ]

        all_results = await asyncio.gather(
            *general_tasks, *detailed_tasks, *purpose_tasks, return_exceptions=True
        )

        ng = len(chunks)
        nd = len(chunks)
        np_ = len(purpose_chunks)

        general_parts = [r for r in all_results[:ng] if isinstance(r, str) and r.strip()]
        detailed_parts = [r for r in all_results[ng:ng+nd] if isinstance(r, str) and r.strip()]
        purpose_parts = [r for r in all_results[ng+nd:] if isinstance(r, str) and r.strip()]

        status(f"Got {len(general_parts)} general + {len(detailed_parts)} detailed + {len(purpose_parts)} purpose partial maps")

        if not general_parts:
            raise RuntimeError("All general map batches failed")

        # Merge general maps
        if len(general_parts) == 1:
            general = general_parts[0]
        else:
            step("Merging general maps...")
            all_general = "\n\n---\n\n".join(
                f"=== PARTIAL MAP {i+1}/{len(general_parts)} ===\n{p}"
                for i, p in enumerate(general_parts)
            )
            general = await call_with_retry(model, f"""Merge these partial maps into ONE general overview.
Each covers different files from the same project. Remove duplicates, merge related features. Keep it concise (2000-4000 tokens).

PARTIAL MAPS:
{all_general}

Write the merged GENERAL MAP:""", max_tokens=8192, log_label="merging general maps")

        # Merge detailed maps
        if len(detailed_parts) == 1:
            detailed = detailed_parts[0]
        else:
            step("Merging detailed maps...")
            all_detailed = "\n\n".join(detailed_parts)
            import re as _re
            sections = _re.split(r'(===\s*SECTION:\s*)', all_detailed)
            seen = {}
            for i in range(len(sections)):
                if sections[i].strip().startswith("=== SECTION:"):
                    if i + 1 < len(sections):
                        content = sections[i] + sections[i + 1]
                        name_match = _re.match(r'===\s*SECTION:\s*(.+?)===', content)
                        name = name_match.group(1).strip().lower() if name_match else str(i)
                        if name not in seen or len(content) > len(seen[name]):
                            seen[name] = content
            detailed = "\n\n".join(seen.values()) if seen else all_detailed

        # Merge purpose maps — combine entries from same category
        if len(purpose_parts) <= 1:
            purpose = purpose_parts[0] if purpose_parts else ""
        else:
            step("Merging purpose maps...")
            all_purpose = "\n\n".join(purpose_parts)
            import re as _re2

            # Parse all categories from all batches
            cat_blocks = _re2.split(r'===\s*PURPOSE:\s*', all_purpose)
            merged = {}  # name -> {description, entries}

            for block in cat_blocks:
                if not block.strip():
                    continue
                # Extract category name (up to next ===)
                title_end = block.find("===")
                if title_end == -1:
                    name = block.split("\n")[0].strip()
                    body = block
                else:
                    name = block[:title_end].strip()
                    body = block[title_end + 3:].strip()

                name_lower = name.lower()

                if name_lower not in merged:
                    merged[name_lower] = {"name": name, "body": body}
                else:
                    # Append new FILE: entries, skip duplicate description
                    existing_body = merged[name_lower]["body"]
                    # Extract just the FILE: blocks from the new body
                    file_blocks = _re2.findall(r'(FILE:.*?)(?=FILE:|$)', body, _re2.DOTALL)
                    for fb in file_blocks:
                        fb = fb.strip()
                        if fb and fb not in existing_body:
                            merged[name_lower]["body"] += "\n\n" + fb

            purpose_parts_merged = []
            for data in merged.values():
                purpose_parts_merged.append(f"=== PURPOSE: {data['name']} ===\n{data['body']}")
            purpose = "\n\n".join(purpose_parts_merged) if purpose_parts_merged else all_purpose

    status(f"General map: {len(general):,} chars")
    status(f"Detailed map: {len(detailed):,} chars")
    status(f"Purpose map: {len(purpose):,} chars")

    # Save to cache
    _save_maps(project_root, file_hash, general, detailed, purpose)

    # Build semantic embeddings index in the background (non-blocking)
    if purpose:
        try:
            import asyncio as _asyncio
            from tools.embeddings import build_embeddings as _build_embeddings
            maps_dir = _maps_dir(project_root)
            _asyncio.ensure_future(_build_embeddings(purpose, maps_dir, file_hash))
            status("Semantic index: building in background...")
        except Exception as _e:
            warn(f"Semantic index build skipped: {_e}")

    success(f"Maps generated and cached ({len(chunks)} batch(es))")

    return {"general": general, "detailed": detailed, "purpose": purpose}


# ─── Query Detailed Map ─────────────────────────────────────────────────────

def get_detail_section(detailed_map: str, section_name: str) -> str:
    """
    Extract a specific section from the detailed map.
    AI writes [DETAIL: section name] → this returns that section.
    Fuzzy matches section names.
    """
    section_name_lower = section_name.lower().strip()

    # Split by === SECTION: ... ===
    import re
    sections = re.split(r'===\s*SECTION:\s*', detailed_map)

    best_match = ""
    best_score = 0

    for section in sections:
        if not section.strip():
            continue

        # Get section title (first line up to ===)
        title_end = section.find("===")
        if title_end == -1:
            title = section.split("\n")[0].strip()
            body = section
        else:
            title = section[:title_end].strip()
            body = section[title_end + 3:].strip()

        # Fuzzy match
        title_lower = title.lower()
        score = 0

        # Exact match
        if section_name_lower == title_lower:
            return f"=== SECTION: {title} ===\n{body}"

        # Keyword overlap
        query_words = set(section_name_lower.split())
        title_words = set(title_lower.split())
        overlap = len(query_words & title_words)
        if overlap > best_score:
            best_score = overlap
            best_match = f"=== SECTION: {title} ===\n{body}"

        # Substring match
        if section_name_lower in title_lower or title_lower in section_name_lower:
            return f"=== SECTION: {title} ===\n{body}"

    if best_match:
        return best_match

    return f"(no section found matching '{section_name}')"


def list_sections(detailed_map: str) -> list[str]:
    """List all section names in the detailed map."""
    import re
    return re.findall(r'===\s*SECTION:\s*(.+?)\s*===', detailed_map)


# ─── Detail Tag Detection ───────────────────────────────────────────────────

import re
DETAIL_TAG = re.compile(r'\[DETAIL:\s*(.+?)\]', re.IGNORECASE)


def extract_detail_tags(text: str) -> list[str]:
    """Extract [DETAIL: section name] tags from AI response."""
    return DETAIL_TAG.findall(text)


# ─── Purpose Map Query ────────────────────────────────────────────────────

def get_purpose_snippets(purpose_map: str, category_name: str, project_root: str) -> str:
    """
    Extract a category from the purpose map and return actual code snippets
    with 10 lines of context above and below each referenced range.
    AI writes [PURPOSE: category name] -> this returns the code.
    """
    import re as _re
    import difflib

    category_lower = category_name.lower().strip()

    # Split by === PURPOSE: ... ===
    cats = _re.split(r'===\s*PURPOSE:\s*', purpose_map)

    best_match = ""
    best_score = 0

    for cat in cats:
        if not cat.strip():
            continue

        title_end = cat.find("===")
        if title_end == -1:
            title = cat.split("\n")[0].strip()
            body = cat
        else:
            title = cat[:title_end].strip()
            body = cat[title_end + 3:].strip()

        title_lower = title.lower()

        if category_lower == title_lower or category_lower in title_lower or title_lower in category_lower:
            best_match = body
            best_score = 999
            break

        query_words = set(category_lower.split())
        title_words = set(title_lower.split())
        overlap = len(query_words & title_words)
        if overlap > best_score:
            best_score = overlap
            best_match = body

    if not best_match:
        # List available categories
        all_cats = list_purposes(purpose_map)
        return f"(no category matching '{category_name}')\nAvailable: {', '.join(all_cats)}"

    # Parse FILE: and LINES: references from the matched category
    # Extract description if present
    desc_line = ""
    for line in best_match.split('\n'):
        if line.strip().lower().startswith("description:"):
            desc_line = line.strip()
            break

    output_parts = [f"=== PURPOSE: {category_name} ==="]
    if desc_line:
        output_parts.append(desc_line)

    current_file = None
    file_pattern = _re.compile(r'^FILE:\s*(.+)$', _re.MULTILINE)
    line_pattern = _re.compile(r'^\s*LINES:\s*(\d+)\s*-\s*(\d+)\s*(?:—|-)\s*(.*)$', _re.MULTILINE)

    # Extract all file + line references
    refs = []
    for line in best_match.split('\n'):
        file_match = file_pattern.match(line.strip())
        if file_match:
            current_file = file_match.group(1).strip()
            continue
        line_match = line_pattern.match(line)
        if line_match and current_file:
            start = int(line_match.group(1))
            end = int(line_match.group(2))
            desc = line_match.group(3).strip()
            refs.append((current_file, start, end, desc))

    if not refs:
        return f"=== PURPOSE: {category_name} ===\n{best_match}\n(no parseable line references found)"

    output_parts.append(f"({len(refs)} locations — this is the COMPLETE list, nothing else in the project serves this purpose)")

    # Read actual code and extract snippets with context
    for filepath, start, end, desc in refs:
        full_path = os.path.join(project_root, filepath)
        content = read_file(full_path)
        if not content or content.startswith("["):
            output_parts.append(f"\n-- {filepath}:{start}-{end} ({desc}) -- FILE NOT FOUND")
            continue

        lines = content.split('\n')
        total_lines = len(lines)

        # Add 10 lines context above and below
        ctx_start = max(0, start - 1 - 10)  # -1 for 0-indexed
        ctx_end = min(total_lines, end + 10)

        output_parts.append(f"\n-- {filepath}:{start}-{end} ({desc}) --")
        for i in range(ctx_start, ctx_end):
            marker = ">>>" if start - 1 <= i < end else "   "
            output_parts.append(f"{marker} {i+1:4d} | {lines[i]}")

    return "\n".join(output_parts)


def list_purposes(purpose_map: str) -> list[str]:
    """List all category names in the purpose map."""
    import re
    return re.findall(r'===\s*PURPOSE:\s*(.+?)\s*===', purpose_map)


def semantic_search(query: str, purpose_map: str, project_root: str, top_n: int = 10) -> str:
    """Synchronous wrapper — semantic search is handled async via tool_call.py.
    This fallback is only used when called outside the async context.
    """
    return f"(semantic search requires async context — use [SEMANTIC: {query}] in the tool loop)"
