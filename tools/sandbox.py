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
    Manages a full copy of the project for safe editing.
    On setup, the entire project is copied to .jarvis_sandbox/.
    All reads and writes happen in the sandbox.
    Original files are NEVER touched until the user explicitly approves.
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.sandbox_dir = self.project_root / ".jarvis_sandbox"
        self.original_files: dict[str, str] = {}   # rel_path → original content
        self.modified_files: dict[str, str] = {}    # rel_path → new content
        self.new_files: dict[str, str] = {}         # rel_path → content (files that don't exist yet)

    def setup(self):
        """Create sandbox as a full copy of the project."""
        # Clean previous sandbox
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Copy all project files to sandbox (skip hidden dirs, node_modules, etc.)
        skip_dirs = {'.jarvis_sandbox', '.git', 'node_modules', '__pycache__',
                     '.venv', 'venv', '.tox', '.mypy_cache', '.pytest_cache'}
        skip_exts = {'.pyc', '.pyo', '.so', '.o', '.a', '.dylib'}
        copied = 0
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden and excluded directories
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
            rel_root = Path(root).relative_to(self.project_root)

            for fname in files:
                if Path(fname).suffix in skip_exts:
                    continue
                src = Path(root) / fname
                rel_path = str(rel_root / fname)
                dest = self.sandbox_dir / rel_path

                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                    copied += 1
                except Exception:
                    pass  # Skip files that can't be copied (binary, permissions, etc.)

        status(f"Sandbox created: {copied} files copied to {self.sandbox_dir}")

    def _norm(self, rel_path: str) -> str:
        """Normalize path separators (accept both / and \\)."""
        return str(Path(rel_path))

    def load_file(self, rel_path: str) -> str | None:
        """Load a file from the sandbox (working copy). Returns content or None."""
        rel_path = self._norm(rel_path)

        # Read from sandbox (working copy)
        sandbox_src = self.sandbox_dir / rel_path
        if sandbox_src.exists():
            try:
                content = sandbox_src.read_text(encoding="utf-8", errors="replace")
                # Track original if not already tracked
                if rel_path not in self.original_files:
                    real_src = self.project_root / rel_path
                    if real_src.exists():
                        self.original_files[rel_path] = real_src.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    else:
                        self.original_files[rel_path] = ""
                return content
            except Exception as e:
                warn(f"Failed to load {rel_path} from sandbox: {e}")
                return None

        # Fallback: file not in sandbox, try project root
        src = self.project_root / rel_path
        if not src.exists():
            return None
        try:
            content = src.read_text(encoding="utf-8", errors="replace")
            self.original_files[rel_path] = content

            # Copy to sandbox for future reads
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
