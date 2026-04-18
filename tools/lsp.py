"""
LSP (Language Server Protocol) client for smart code search.

Instead of 10 ripgrep searches, one LSP call gives exact semantic results:
  - find_references: every usage of a symbol (not just text matches)
  - find_definition: where a symbol is defined
  - get_hover: type info for a symbol

Auto-detects project language, starts the appropriate server, falls back
to ripgrep if no LSP server is installed.

Supported: Python (pylsp/pyright), JavaScript/TypeScript (typescript-language-server)
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from core.cli import status, warn


# ─── LSP Server Detection ──────────────────────────────────────────────────

LSP_SERVERS = {
    "python": [
        {"cmd": ["pylsp"], "name": "pylsp"},
        {"cmd": ["pyright-langserver", "--stdio"], "name": "pyright"},
    ],
    "javascript": [
        {"cmd": ["typescript-language-server", "--stdio"], "name": "tsserver"},
    ],
    "typescript": [
        {"cmd": ["typescript-language-server", "--stdio"], "name": "tsserver"},
    ],
}


def _detect_language(project_root: str) -> str | None:
    """Detect primary language by counting file extensions."""
    counts = {"python": 0, "javascript": 0, "typescript": 0}
    root = Path(project_root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {
            "__pycache__", ".git", "node_modules", ".venv", "venv", "build", "dist",
        }]
        for f in filenames:
            ext = Path(f).suffix
            if ext == ".py":
                counts["python"] += 1
            elif ext in (".js", ".jsx", ".mjs"):
                counts["javascript"] += 1
            elif ext in (".ts", ".tsx"):
                counts["typescript"] += 1
    if not any(counts.values()):
        return None
    return max(counts, key=counts.get)


def _find_lsp_server(language: str) -> dict | None:
    """Find an installed LSP server for the given language."""
    candidates = LSP_SERVERS.get(language, [])
    for server in candidates:
        try:
            # Check if the command exists
            result = subprocess.run(
                [server["cmd"][0], "--version"] if server["cmd"][0] != "pylsp" else ["pylsp", "--help"],
                capture_output=True, timeout=5
            )
            return server
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


# ─── JSON-RPC over stdio ──────────────────────────────────────────────────

class LSPClient:
    """Minimal LSP client — JSON-RPC over stdio."""

    def __init__(self, cmd: list[str], project_root: str, name: str = "lsp"):
        self.cmd = cmd
        self.project_root = Path(project_root).resolve()
        self.name = name
        self.process = None
        self._req_id = 0
        self._initialized = False
        self._lock = asyncio.Lock()

    async def start(self) -> bool:
        """Start LSP server and initialize."""
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root),
            )

            # Initialize handshake
            init_result = await self._request("initialize", {
                "processId": os.getpid(),
                "rootUri": self.project_root.as_uri(),
                "rootPath": str(self.project_root),
                "capabilities": {
                    "textDocument": {
                        "references": {"dynamicRegistration": False},
                        "definition": {"dynamicRegistration": False},
                        "hover": {"dynamicRegistration": False},
                    },
                },
            })

            if init_result is not None:
                await self._notify("initialized", {})
                self._initialized = True
                status(f"LSP: {self.name} started for {self.project_root.name}")
                return True
            return False
        except Exception as e:
            warn(f"LSP: failed to start {self.name}: {e}")
            return False

    async def stop(self):
        """Shutdown LSP server."""
        if self.process and self.process.returncode is None:
            try:
                await self._request("shutdown", None)
                await self._notify("exit", None)
            except Exception:
                pass
            try:
                self.process.kill()
            except Exception:
                pass

    async def _send(self, data: dict):
        """Send JSON-RPC message."""
        body = json.dumps(data)
        header = f"Content-Length: {len(body)}\r\n\r\n"
        self.process.stdin.write(header.encode() + body.encode())
        await self.process.stdin.drain()

    async def _recv(self) -> dict | None:
        """Read one JSON-RPC response."""
        try:
            # Read headers
            headers = {}
            while True:
                line = await asyncio.wait_for(
                    self.process.stdout.readline(), timeout=30
                )
                line = line.decode().strip()
                if not line:
                    break
                if ":" in line:
                    key, val = line.split(":", 1)
                    headers[key.strip().lower()] = val.strip()

            content_length = int(headers.get("content-length", 0))
            if content_length == 0:
                return None

            body = await asyncio.wait_for(
                self.process.stdout.readexactly(content_length), timeout=30
            )
            return json.loads(body)
        except (asyncio.TimeoutError, Exception):
            return None

    async def _request(self, method: str, params) -> dict | None:
        """Send request, wait for matching response (skip notifications)."""
        async with self._lock:
            self._req_id += 1
            req_id = self._req_id
            msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
            if params is not None:
                msg["params"] = params
            await self._send(msg)

            # Read responses until we get ours (skip notifications)
            for _ in range(50):  # max 50 messages before giving up
                resp = await self._recv()
                if resp is None:
                    return None
                if resp.get("id") == req_id:
                    return resp.get("result")
            return None

    async def _notify(self, method: str, params):
        """Send notification (no response expected)."""
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._send(msg)

    def _file_uri(self, path: str) -> str:
        """Convert file path to URI."""
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p
        return p.resolve().as_uri()

    async def _open_file(self, filepath: str):
        """Notify server about an opened file."""
        p = Path(filepath) if Path(filepath).is_absolute() else self.project_root / filepath
        try:
            text = p.read_text(errors="replace")
        except Exception:
            return
        # Guess language ID
        ext = p.suffix
        lang_id = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "javascriptreact", ".tsx": "typescriptreact",
        }.get(ext, "plaintext")

        await self._notify("textDocument/didOpen", {
            "textDocument": {
                "uri": self._file_uri(str(p)),
                "languageId": lang_id,
                "version": 1,
                "text": text,
            }
        })

    async def find_references(self, filepath: str, line: int, col: int) -> list[dict]:
        """Find all references to symbol at position. Returns [{file, line, col}]."""
        if not self._initialized:
            return []
        await self._open_file(filepath)
        result = await self._request("textDocument/references", {
            "textDocument": {"uri": self._file_uri(filepath)},
            "position": {"line": line, "character": col},
            "context": {"includeDeclaration": True},
        })
        if not result:
            return []
        refs = []
        for loc in result:
            uri = loc.get("uri", "")
            # Convert URI to relative path
            if uri.startswith("file://"):
                fpath = uri[7:]
                if os.name == "nt" and fpath.startswith("/"):
                    fpath = fpath[1:]  # Windows: remove leading /
                try:
                    fpath = str(Path(fpath).relative_to(self.project_root))
                except ValueError:
                    pass
            else:
                fpath = uri
            rng = loc.get("range", {}).get("start", {})
            refs.append({
                "file": fpath,
                "line": rng.get("line", 0) + 1,  # LSP is 0-indexed
                "col": rng.get("character", 0),
            })
        return refs

    async def find_definition(self, filepath: str, line: int, col: int) -> list[dict]:
        """Find definition of symbol at position."""
        if not self._initialized:
            return []
        await self._open_file(filepath)
        result = await self._request("textDocument/definition", {
            "textDocument": {"uri": self._file_uri(filepath)},
            "position": {"line": line, "character": col},
        })
        if not result:
            return []
        # Result can be Location or Location[]
        if isinstance(result, dict):
            result = [result]
        defs = []
        for loc in result:
            uri = loc.get("uri", "")
            if uri.startswith("file://"):
                fpath = uri[7:]
                if os.name == "nt" and fpath.startswith("/"):
                    fpath = fpath[1:]
                try:
                    fpath = str(Path(fpath).relative_to(self.project_root))
                except ValueError:
                    pass
            else:
                fpath = uri
            rng = loc.get("range", {}).get("start", {})
            defs.append({
                "file": fpath,
                "line": rng.get("line", 0) + 1,
                "col": rng.get("character", 0),
            })
        return defs


# ─── Session Manager ──────────────────────────────────────────────────────

_active_client: LSPClient | None = None
_active_root: str = ""


async def get_lsp_client(project_root: str) -> LSPClient | None:
    """Get or create an LSP client for the project. Returns None if no server available."""
    global _active_client, _active_root

    project_root = str(Path(project_root).resolve())

    # Reuse existing client if same project
    if (_active_client and _active_client._initialized
            and _active_root == project_root):
        return _active_client

    # Clean up old client
    if _active_client:
        await _active_client.stop()
        _active_client = None
        _active_root = ""

    # Detect language and find server
    lang = _detect_language(project_root)
    if not lang:
        return None

    server = _find_lsp_server(lang)
    if not server:
        return None

    client = LSPClient(server["cmd"], project_root, server["name"])
    if await client.start():
        _active_client = client
        _active_root = project_root
        return client

    return None


async def lsp_find_references(name: str, project_root: str) -> str | None:
    """
    Use LSP to find all references to a symbol name.
    Returns formatted string like search_refs, or None if LSP unavailable.
    """
    client = await get_lsp_client(project_root)
    if not client:
        return None

    # Find where the symbol is defined first (by grepping for its definition)
    # We need a file+line+col to query LSP
    root = Path(project_root).resolve()
    target_file = None
    target_line = 0
    target_col = 0

    # Quick scan for the definition
    import re
    def_patterns = [
        rf'^\s*(?:async\s+)?def\s+{re.escape(name)}\s*\(',
        rf'^\s*class\s+{re.escape(name)}\s*[\(:]',
        rf'^\s*(?:const|let|var|function|export\s+(?:default\s+)?function)\s+{re.escape(name)}\b',
        rf'^\s*(?:pub\s+)?fn\s+{re.escape(name)}\b',
    ]

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {
            "__pycache__", ".git", "node_modules", ".venv", "venv",
        }]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix not in {".py", ".js", ".ts", ".jsx", ".tsx"}:
                continue
            try:
                lines = fpath.read_text(errors="replace").split("\n")
            except Exception:
                continue
            for i, line in enumerate(lines):
                for pat in def_patterns:
                    m = re.match(pat, line)
                    if m:
                        target_file = str(fpath.relative_to(root))
                        target_line = i  # 0-indexed for LSP
                        target_col = line.index(name)
                        break
                if target_file:
                    break
            if target_file:
                break

    if not target_file:
        return None  # Couldn't find definition — fall back to ripgrep

    # Query LSP for references
    refs = await client.find_references(target_file, target_line, target_col)
    if not refs:
        return None

    # Format like search_refs output
    definitions = []
    imports = []
    usages = []

    for ref in refs:
        filepath = ref["file"]
        linenum = ref["line"]
        # Read the actual line
        try:
            full = root / filepath
            line_text = full.read_text(errors="replace").split("\n")[linenum - 1].strip()
        except Exception:
            line_text = ""

        entry = f"  {filepath}:{linenum}  {line_text}"

        if any(line_text.lstrip().startswith(kw) for kw in [
            f"def {name}", f"class {name}", f"async def {name}",
            f"function {name}", f"const {name}", f"let {name}",
            f"export function {name}", f"fn {name}", f"pub fn {name}",
        ]):
            definitions.append(entry)
        elif "import" in line_text.lower() or "require" in line_text.lower():
            imports.append(entry)
        else:
            usages.append(entry)

    parts = [f"=== References for '{name}' (via LSP: {client.name}) ==="]
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
