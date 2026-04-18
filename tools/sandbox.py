"""
Sandbox — safe file editing environment for the coding agent.

All changes happen in a sandbox copy. Original files are NEVER touched
until the user explicitly approves the diff.
"""

import os
import shutil
import difflib
from pathlib import Path
from core.cli import step, status, warn, success


class Sandbox:
    """
    Manages a temporary copy of files for safe editing.
    Original files → sandbox copies → user approves → apply to originals.
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.sandbox_dir = self.project_root / ".jarvis_sandbox"
        self.original_files: dict[str, str] = {}   # rel_path → original content
        self.modified_files: dict[str, str] = {}    # rel_path → new content
        self.new_files: dict[str, str] = {}         # rel_path → content (files that don't exist yet)

    def setup(self):
        """Create sandbox directory."""
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        status(f"Sandbox created: {self.sandbox_dir}")

    def _norm(self, rel_path: str) -> str:
        """Normalize path separators (accept both / and \\)."""
        return str(Path(rel_path))

    def load_file(self, rel_path: str) -> str | None:
        """Load a file into the sandbox. Returns content or None."""
        rel_path = self._norm(rel_path)
        src = self.project_root / rel_path
        if not src.exists():
            return None
        try:
            content = src.read_text(encoding="utf-8", errors="replace")
            self.original_files[rel_path] = content

            # Copy to sandbox
            dest = self.sandbox_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

            return content
        except Exception as e:
            warn(f"Failed to load {rel_path}: {e}")
            return None

    def write_file(self, rel_path: str, content: str):
        """Write modified content to sandbox."""
        rel_path = self._norm(rel_path)
        # Track whether this is a new file or modification
        if rel_path in self.original_files:
            self.modified_files[rel_path] = content
        else:
            src = self.project_root / rel_path
            if src.exists():
                # Load original first
                self.original_files[rel_path] = src.read_text(encoding="utf-8", errors="replace")
                self.modified_files[rel_path] = content
            else:
                self.new_files[rel_path] = content

        # Write to sandbox
        dest = self.sandbox_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    def get_diff(self, rel_path: str) -> str:
        """Get unified diff for a modified file."""
        original = self.original_files.get(rel_path, "")
        if rel_path in self.modified_files:
            modified = self.modified_files[rel_path]
        elif rel_path in self.new_files:
            original = ""
            modified = self.new_files[rel_path]
        else:
            return "(no changes)"

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
        return "\n".join(diff) or "(no changes)"

    def get_all_diffs(self) -> str:
        """Get diffs for all changed files."""
        diffs = []
        for rel_path in sorted(set(list(self.modified_files.keys()) + list(self.new_files.keys()))):
            diff = self.get_diff(rel_path)
            if diff != "(no changes)":
                diffs.append(f"═══ {rel_path} ═══\n{diff}")

        return "\n\n".join(diffs) if diffs else "(no changes)"

    def apply(self) -> list[str]:
        """Apply all sandbox changes to the actual project. Returns list of modified files."""
        applied = []

        for rel_path, content in self.modified_files.items():
            dest = self.project_root / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            applied.append(f"Modified: {rel_path}")

        for rel_path, content in self.new_files.items():
            dest = self.project_root / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            applied.append(f"Created: {rel_path}")

        success(f"Applied {len(applied)} changes to project")
        return applied

    def cleanup(self):
        """Remove sandbox directory."""
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def summary(self) -> str:
        """Human-readable summary of changes."""
        lines = []
        for p in self.modified_files:
            orig_lines = self.original_files.get(p, "").count("\n")
            new_lines = self.modified_files[p].count("\n")
            lines.append(f"  Modified: {p} ({orig_lines} → {new_lines} lines)")
        for p in self.new_files:
            new_lines = self.new_files[p].count("\n")
            lines.append(f"  New file: {p} ({new_lines} lines)")
        return "\n".join(lines) if lines else "  (no changes)"
