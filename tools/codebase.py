"""
Codebase tools — project scanning, code search, file operations.

Used by the coding agent at every phase:
  - Pre-search: scan project structure + find relevant files
  - On-demand: any AI writes [SEARCH: pattern] → auto-detected, results fed back
"""

import asyncio
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from core.cli import step, status, warn
from clients.gemini import call_flash


# ─── Path Normalization (Windows + Linux) ──────────────────────────────────

def norm_path(p: str) -> str:
    """Normalize path separators to OS native. Accepts both / and \\."""
    return str(Path(p))


def to_forward_slash(p: str) -> str:
    """Convert to forward slashes (for display, ripgrep globs, etc.)."""
    return p.replace("\\", "/")


# ─── Configuration ───────────────────────────────────────────────────────────

LARGE_FILE_THRESHOLD = 50_000  # chars — warn to use KEEP above this
MAX_SEARCH_RESULTS = 30      # ripgrep matches
IGNORE_DIRS = {
    # Python
    "__pycache__", ".venv", "venv", "env", ".env", "virtualenv",
    "site-packages", ".eggs",
    ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".pytype", ".pyre",
    # Node / JS
    "node_modules", "bower_components", ".npm", ".yarn",
    ".next", ".nuxt", ".svelte-kit", ".astro",
    # Build / dist
    "build", "dist", "out", "_build", "target", "release", "debug",
    "bin", "obj", "cmake-build-debug", "cmake-build-release",
    # Version control
    ".git", ".svn", ".hg", ".bzr",
    # IDE / editor
    ".idea", ".vscode", ".vs", ".eclipse", ".settings",
    # Package managers / caches
    ".cache", ".gradle", ".m2", ".cargo",
    "vendor", "third_party", "3rdparty", "external", "deps",
    "lib", "libs", "packages",
    # Rust
    ".rustup",
    # Go
    "pkg",
    # Ruby
    ".bundle",
    # Docker
    ".docker",
    # Misc
    ".jarvis", ".jarvis_sandbox", "coverage", "htmlcov",
    ".terraform", ".serverless",
    "logs", "tmp", "temp",
}

# Directories matching these suffixes are also ignored
IGNORE_DIR_SUFFIXES = {".egg-info", ".dist-info"}


def _is_ignored_dir(dirname: str) -> bool:
    """Check if a directory should be ignored (exact match or suffix)."""
    if dirname in IGNORE_DIRS:
        return True
    for suffix in IGNORE_DIR_SUFFIXES:
        if dirname.endswith(suffix):
            return True
    return False

IGNORE_EXTENSIONS = {
    # Compiled
    ".pyc", ".pyo", ".so", ".o", ".a", ".class", ".dll", ".exe", ".dylib",
    ".wasm",
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg", ".bmp", ".webp", ".tiff",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
    # Data / binary
    ".db", ".sqlite", ".sqlite3", ".bin", ".dat", ".pickle", ".pkl",
    ".h5", ".hdf5", ".parquet", ".arrow", ".feather",
    # Media
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Lock files (large, not useful for planning)
    ".lock",
    # Maps / minified
    ".map", ".min.js", ".min.css",
    # Models
    ".onnx", ".pb", ".pt", ".pth", ".safetensors", ".gguf",
}


# ─── Project Structure ───────────────────────────────────────────────────────

def scan_project(root: str, max_depth: int = 4) -> str:
    """
    Get the full project tree (files + dirs, up to max_depth).
    Returns a formatted string for AI consumption.
    """
    root = Path(root).resolve()
    if not root.exists():
        return f"Directory not found: {root}"

    lines = [f"Project: {root.name}/"]
    file_count = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Filter ignored dirs
        dirnames[:] = [d for d in sorted(dirnames) if not _is_ignored_dir(d)]

        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            dirnames.clear()
            continue

        indent = "  " * depth
        if depth > 0:
            lines.append(f"{indent}{rel.name}/")

        for f in sorted(filenames):
            fpath = Path(dirpath) / f
            if fpath.suffix in IGNORE_EXTENSIONS:
                continue
            size = fpath.stat().st_size if fpath.exists() else 0
            size_str = f" ({_human_size(size)})" if size > 10000 else ""
            lines.append(f"{indent}  {f}{size_str}")
            file_count += 1

    lines.append(f"\n({file_count} files)")
    return "\n".join(lines)


def _human_size(n: int) -> str:
    for unit in ["B", "KB", "MB"]:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


# ─── File Reading ────────────────────────────────────────────────────────────

def read_file(path: str) -> str | None:
    """Read a file, return raw content or None if binary / unreadable.
    No size limit — callers add KEEP hints based on LARGE_FILE_THRESHOLD."""
    p = Path(norm_path(path))
    if not p.exists():
        return None
    if p.suffix in IGNORE_EXTENSIONS:
        return f"[BINARY FILE: {p.suffix} — skipped]"
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[READ ERROR: {e}]"


def read_files(paths: list[str]) -> dict[str, str]:
    """Read multiple files. Returns {path: content}."""
    return {p: read_file(p) or f"[NOT FOUND: {p}]" for p in paths}


# ─── Code Search (ripgrep) ──────────────────────────────────────────────────


def add_line_numbers(content: str) -> str:
    """Add line numbers to code content in the iN| format the model expects.

    Format per line:  iN|{code without leading whitespace} {lineno}
    where N is the number of leading spaces (tabs are expanded to TAB_WIDTH=4)
    and {lineno} is separated from the code by a single space — matching the
    format the prompts document and the format _filter_by_ranges (used by KEEP)
    produces. Reading and writing are symmetric across [CODE:] and [KEEP:].

    Examples:
       def foo():        →  i0|def foo() 10
           return 1      →  i4|return 1 11
               pass      →  i8|pass 12
       (blank)           →  i0| 13
    """
    TAB_WIDTH = 4
    lines = content.split('\n')
    out = []
    for i, line in enumerate(lines):
        expanded = line.expandtabs(TAB_WIDTH)
        stripped = expanded.lstrip(' ')
        n_indent = len(expanded) - len(stripped)
        out.append(f"i{n_indent}|{stripped} {i+1}")
    return '\n'.join(out)


def _make_whitespace_visible(line: str) -> str:
    """Legacy: render leading whitespace visibly. Kept for backwards
    compatibility with any callers that still want the old format."""
    stripped = line.lstrip()
    prefix = line[:len(line) - len(stripped)]
    visible = prefix.replace('\t', '→').replace(' ', '⁃')
    return visible + stripped


def extract_relevant_sections(
    content: str, hints: str, context_lines: int = 100, max_short_file: int = 200,
) -> str:
    """Show only the relevant parts of a file with context_lines padding.

    For short files (<= max_short_file lines), returns the whole file with
    line numbers. For larger files, searches for keywords from `hints`
    (plan details, function names, etc.), finds matching lines, expands
    each match by context_lines above and below, merges overlapping
    ranges, and returns the sections with line numbers + gap markers.

    Always preserves original line numbers so edits can reference them.
    """
    lines = content.split('\n')
    total = len(lines)

    if total <= max_short_file:
        return add_line_numbers(content)

    # Extract searchable keywords from hints (identifiers, function names, etc.)
    # Match words that look like identifiers (2+ chars, not common English)
    keywords = set()
    for word in re.findall(r'[A-Za-z_]\w{2,}', hints):
        low = word.lower()
        # Skip common English words and plan-format words
        if low in {
            'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'has',
            'will', 'should', 'must', 'into', 'when', 'then', 'each', 'make',
            'use', 'new', 'old', 'add', 'not', 'all', 'but', 'can', 'get',
            'set', 'put', 'also', 'any', 'file', 'code', 'line', 'step',
            'current', 'behavior', 'logic', 'details', 'modify', 'create',
            'change', 'update', 'function', 'method', 'class', 'return',
            'import', 'export', 'edit', 'replace', 'none', 'existing',
        }:
            continue
        keywords.add(word)

    if not keywords:
        # No useful keywords — show first and last context_lines
        return add_line_numbers(content)

    # Find all lines that match any keyword
    matched_lines: set[int] = set()
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw in line:
                matched_lines.add(i)
                break

    if not matched_lines:
        # No matches — show the whole file
        return add_line_numbers(content)

    # Build ranges with context_lines padding, merge overlapping
    ranges: list[tuple[int, int]] = []
    for line_idx in sorted(matched_lines):
        start = max(0, line_idx - context_lines)
        end = min(total - 1, line_idx + context_lines)
        if ranges and start <= ranges[-1][1] + 1:
            # Merge with previous range
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    # Build output with line numbers and gap markers
    width = len(str(total))
    output_parts = []
    output_parts.append(f"(file has {total} lines total, showing relevant sections)")

    for i, (start, end) in enumerate(ranges):
        if i == 0 and start > 0:
            output_parts.append(f"{'·' * 40} (lines 1-{start} omitted)")
        elif i > 0:
            prev_end = ranges[i - 1][1]
            output_parts.append(f"{'·' * 40} (lines {prev_end + 2}-{start} omitted)")

        for j in range(start, end + 1):
            output_parts.append(f"{j+1:>{width}}\t{lines[j]}")

        if i == len(ranges) - 1 and end < total - 1:
            output_parts.append(f"{'·' * 40} (lines {end + 2}-{total} omitted)")

    return '\n'.join(output_parts)


def search_code(pattern: str, root: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """
    Search codebase using ripgrep (rg) or grep fallback.
    Returns list of {file, line_num, line, context}.
    """
    root = str(Path(root).resolve())

    # Build ignore args
    ignore_args = []
    for d in IGNORE_DIRS:
        ignore_args.extend(["--glob", f"!{d}/"])
    for suffix in IGNORE_DIR_SUFFIXES:
        ignore_args.extend(["--glob", f"!*{suffix}/"])

    # Try ripgrep first
    try:
        cmd = [
            "rg", "--line-number", "--no-heading", "--color=never",
            "--max-count=5",  # max 5 matches per file
            # No --max-filesize: old 100K cap silently skipped large files
            # (e.g. workflows/code.py at 260KB) — giving false "no results".
            "-C", "5",  # 5 lines context (was 25 — most was wasted)
            *ignore_args,
            pattern, root,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode <= 1:  # 0 = found, 1 = not found
            return _parse_rg_output(result.stdout, max_results)
    except FileNotFoundError:
        pass  # ripgrep not installed, fall through to grep
    except subprocess.TimeoutExpired:
        warn("Code search timed out")
        return []

    # Fallback: grep
    try:
        cmd = [
            "grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts",
            "--include=*.lean", "--include=*.c", "--include=*.cpp", "--include=*.h",
            "--include=*.rs", "--include=*.java",
            "-C", "25",
            pattern, root,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return _parse_grep_output(result.stdout, max_results)
    except Exception:
        return []


def _parse_rg_output(output: str, max_results: int) -> list[dict]:
    """Parse ripgrep output into structured results."""
    results = []
    current_file = ""

    for line in output.split("\n"):
        if not line.strip():
            continue
        # rg format: file:line:content  or  file-line-content (context)
        match = re.match(r'^(.+?)[:-](\d+)[:-](.*)$', line)
        if match and len(results) < max_results:
            filepath, line_num, content = match.groups()
            results.append({
                "file": filepath,
                "line_num": int(line_num),
                "line": content.strip(),
            })
            current_file = filepath

    return results


def _parse_grep_output(output: str, max_results: int) -> list[dict]:
    """Parse grep output into structured results."""
    results = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        match = re.match(r'^(.+?)[:-](\d+)[:-](.*)$', line)
        if match and len(results) < max_results:
            filepath, line_num, content = match.groups()
            results.append({
                "file": filepath,
                "line_num": int(line_num),
                "line": content.strip(),
            })
    return results


def format_search_results(results: list[dict]) -> str:
    """Format search results for AI consumption."""
    if not results:
        return "(no matches found)"

    lines = []
    current_file = ""
    for r in results:
        if r["file"] != current_file:
            current_file = r["file"]
            lines.append(f"\n── {current_file} ──")
        lines.append(f"  {r['line_num']:>4}: {r['line']}")

    return "\n".join(lines)


# ─── On-Demand Search Tag Detection ─────────────────────────────────────────

SEARCH_TAG_PATTERN = re.compile(r'\[SEARCH:\s*(.+?)\]', re.IGNORECASE)


def extract_search_requests(text: str) -> list[str]:
    """Extract [SEARCH: pattern] tags from AI response."""
    return SEARCH_TAG_PATTERN.findall(text)


async def run_on_demand_searches(text: str, project_root: str) -> str:
    """
    Detect [SEARCH: pattern] tags in AI output, run them, return results.
    Returns empty string if no tags found.
    """
    patterns = extract_search_requests(text)
    if not patterns:
        return ""

    all_results = []
    for pattern in patterns[:5]:  # Cap at 5 searches
        status(f"On-demand search: {pattern}")
        results = search_code(pattern, project_root)
        if results:
            all_results.append(f"\n=== Search: {pattern} ===")
            all_results.append(format_search_results(results))

    return "\n".join(all_results)


# ─── Pre-Search (Gemini Flash) ──────────────────────────────────────────────

async def pre_search(task_description: str, project_structure: str, project_root: str) -> str:
    """
    Gemini Flash analyzes the task + project structure and searches
    for likely relevant files/patterns BEFORE the main AIs start.
    """
    step("Pre-search: Gemini Flash scanning project...")

    prompt = f"""You are a code search assistant. Given a coding task and project structure,
identify the most relevant files and code patterns to search for.

TASK: {task_description}

PROJECT STRUCTURE:
{project_structure[:8000]}

Output ONLY a JSON list of search queries (strings to grep for):
["pattern1", "pattern2", "pattern3"]

Think about:
- Function names that would be involved
- Class names / module names
- Variable names
- Import statements
- File paths to read directly

Keep it to 5-8 specific, targeted patterns."""

    try:
        result = await call_flash(prompt, max_tokens=512)
        # Parse patterns
        import json
        # Clean markdown fences
        cleaned = result.strip().strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        patterns = json.loads(cleaned)

        if not isinstance(patterns, list):
            return ""

        # Run all searches
        all_results = []
        files_found = set()

        for pattern in patterns[:8]:
            results = search_code(str(pattern), project_root)
            if results:
                all_results.append(f"\n=== Pre-search: {pattern} ===")
                all_results.append(format_search_results(results))
                for r in results:
                    files_found.add(r["file"])

        # Also read the most relevant files in full
        file_contents = []
        for fpath in sorted(files_found)[:10]:  # Cap at 10 files
            content = read_file(fpath)
            if content and not content.startswith("["):
                file_contents.append(f"\n══ {fpath} ══\n{content}")

        status(f"Pre-search: {len(patterns)} patterns, {len(files_found)} files found")

        pre_search_output = "\n".join(all_results)
        if file_contents:
            pre_search_output += "\n\n=== FULL FILE CONTENTS ===\n" + "\n".join(file_contents)

        return pre_search_output

    except Exception as e:
        warn(f"Pre-search failed: {e}")
        return ""


# ─── Reference Search ──────────────────────────────────────────────────────

def search_refs(name: str, root: str, max_results: int = 30) -> str:
    """
    Find all references to a function/class/variable by name.
    Uses word-boundary matching so searching "render" won't match "prerender".
    Groups results by: definitions, imports, and usages.
    Returns formatted string.
    """
    root = str(Path(root).resolve())

    # Build ignore args
    ignore_args = []
    for d in IGNORE_DIRS:
        ignore_args.extend(["--glob", f"!{d}/"])
    for suffix in IGNORE_DIR_SUFFIXES:
        ignore_args.extend(["--glob", f"!*{suffix}/"])

    # Use ripgrep with word boundary.
    # No --max-filesize: the old 200K cap silently skipped large files like
    # workflows/code.py (260KB), causing REFS to return empty results even
    # when the symbol was clearly defined there.
    # No -C context: ripgrep context lines use a dash separator which the old
    # split(':', 2) parser dropped entirely. We now use _parse_rg_output which
    # handles both separators, so context works correctly.
    try:
        cmd = [
            "rg", "--line-number", "--no-heading", "--color=never",
            "--max-count=10",
            "-w",  # word boundary match
            "-C", "3",  # 3 lines context — enough to see usage without bloat
            *ignore_args,
            name, root,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode > 1:
            return f"Search for '{name}': ripgrep error (rc={result.returncode})"
    except FileNotFoundError:
        # Fallback to grep
        try:
            cmd = [
                "grep", "-rnw", "--include=*.py", "--include=*.js", "--include=*.ts",
                "--include=*.jsx", "--include=*.tsx", "--include=*.c", "--include=*.cpp",
                "--include=*.h", "--include=*.rs", "--include=*.java", "--include=*.lean",
                "-C", "3",
                name, root,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except Exception:
            return f"Search for '{name}': search failed"
    except subprocess.TimeoutExpired:
        return f"Search for '{name}': timed out"

    if not result.stdout.strip():
        return f"Search for '{name}': no matches found"

    # Parse with _parse_rg_output which handles both match lines (`:` sep) and
    # context lines (`-` sep). Old split(':', 2) silently dropped context lines.
    raw_results = _parse_rg_output(result.stdout, max_results)

    if not raw_results:
        return f"Search for '{name}': no matches found"

    # Categorize results
    definitions = []
    imports = []
    usages = []

    for item in raw_results:
        filepath = item["file"]
        linenum  = item["line_num"]
        content  = item["line"]

        # Make path relative
        try:
            rel = str(Path(filepath).relative_to(root))
        except ValueError:
            rel = filepath

        entry = f"  {rel}:{linenum}  {content}"

        # Categorize based on line content
        stripped = content.lstrip()
        if any(stripped.startswith(kw) for kw in [
            f"def {name}", f"class {name}", f"async def {name}",
            f"function {name}", f"const {name}", f"let {name}", f"var {name}",
            f"export function {name}", f"export const {name}",
            f"export default function {name}",
            f"fn {name}", f"pub fn {name}", f"struct {name}", f"enum {name}",
        ]):
            definitions.append(entry)
        elif "import" in stripped.lower() or "require" in stripped.lower():
            imports.append(entry)
        else:
            usages.append(entry)

    parts = [f"=== References for '{name}' ==="]
    if definitions:
        parts.append(f"\nDEFINED ({len(definitions)}):")
        parts.extend(definitions)
    if imports:
        parts.append(f"\nIMPORTED ({len(imports)}):")
        parts.extend(imports)
    if usages:
        parts.append(f"\nUSED ({len(usages)}):")
        parts.extend(usages)

    return "\n".join(parts)
