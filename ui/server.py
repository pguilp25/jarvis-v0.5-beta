"""
JARVIS Web UI — aiohttp server with WebSocket.
Multi-conversation support with persistent storage.
"""

import asyncio
import json
import os
import sys
import uuid
import webbrowser
from pathlib import Path

import aiohttp
from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import ConversationMemory
from core.persistence import save_session, load_session, clear_session
from core import thought_logger
from core.costs import cost_tracker

# ─── Global State ────────────────────────────────────────────────────────────

_conversations: dict = {}  # id -> {"name": str, "memory": ConversationMemory}
_active_conv: str = ""
_project_root: str = ""
_pending_sandbox = None
_pending_maps = None
_ws_clients: list = []  # use list to avoid set mutation issues
_current_task: asyncio.Task | None = None  # track current processing for stop

CONV_DIR = Path.home() / ".jarvis" / "conversations"


def _load_conversations():
    global _conversations, _active_conv
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CONV_DIR / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            _active_conv = meta.get("active", "")
            for cid, info in meta.get("conversations", {}).items():
                mem = ConversationMemory()
                mp = CONV_DIR / f"{cid}.json"
                if mp.exists():
                    try:
                        d = json.loads(mp.read_text())
                        mem.full_history = d.get("history", [])
                        mem.compressed_context = d.get("compressed")
                    except Exception:
                        pass
                _conversations[cid] = {"name": info.get("name", "Chat"), "memory": mem}
        except Exception:
            pass
    if not _conversations:
        cid = str(uuid.uuid4())[:8]
        _conversations[cid] = {"name": "New Chat", "memory": ConversationMemory()}
        _active_conv = cid
        _save_conv_meta()


def _save_conv_meta():
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "active": _active_conv,
        "conversations": {cid: {"name": c["name"]} for cid, c in _conversations.items()},
    }
    (CONV_DIR / "meta.json").write_text(json.dumps(meta))


def _save_conv_memory(cid):
    if cid not in _conversations:
        return
    mem = _conversations[cid]["memory"]
    d = {"history": mem.full_history, "compressed": mem.compressed_context}
    (CONV_DIR / f"{cid}.json").write_text(json.dumps(d))


def _get_memory():
    if _active_conv and _active_conv in _conversations:
        return _conversations[_active_conv]["memory"]
    return ConversationMemory()


def _build_history(cid):
    if cid not in _conversations:
        return []
    out = []
    for m in _conversations[cid]["memory"].full_history:
        if m.get("role") in ("user", "assistant"):
            out.append({"role": m["role"], "text": m.get("content", "")})
    return out


# ─── Broadcasting ────────────────────────────────────────────────────────────

async def _broadcast(msg_type, data):
    payload = json.dumps({"type": msg_type, **data})
    to_remove = []
    for i, ws in enumerate(_ws_clients):
        try:
            await ws.send_str(payload)
        except Exception:
            to_remove.append(i)
    for i in reversed(to_remove):
        _ws_clients.pop(i)


# ─── Hooks ───────────────────────────────────────────────────────────────────

_orig_header = thought_logger.write_header
_orig_chunk = thought_logger.write_chunk


def _hook_header(model_id, label=""):
    _orig_header(model_id, label)
    name = model_id.split("/")[-1]
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_broadcast("thinking_header", {"model": name, "label": label}))
    except RuntimeError:
        pass


def _hook_chunk(model_id, chunk):
    _orig_chunk(model_id, chunk)
    if chunk:
        name = model_id.split("/")[-1]
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_broadcast("thinking_chunk", {"model": name, "chunk": chunk}))
        except RuntimeError:
            pass


thought_logger.write_header = _hook_header
thought_logger.write_chunk = _hook_chunk

from core import cli as _cli
_orig_status = _cli.status


def _hook_status(msg, color="cyan"):
    _orig_status(msg, color)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_broadcast("status", {"text": msg, "color": color}))
    except RuntimeError:
        pass


_cli.status = _hook_status


# ─── WebSocket Handler ──────────────────────────────────────────────────────

async def handle_index(request):
    return web.FileResponse(Path(__file__).parent / "index.html")


async def handle_ws(request):
    global _active_conv, _project_root, _pending_sandbox, _pending_maps

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    _ws_clients.append(ws)

    await ws.send_json({
        "type": "init",
        "project": _project_root,
        "conversations": {c: _conversations[c]["name"] for c in _conversations},
        "active": _active_conv,
        "history": _build_history(_active_conv),
    })

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                a = data.get("action", "")
                if a == "message":
                    # Run processing in a separate task so WS loop stays responsive
                    global _current_task
                    if _current_task and not _current_task.done():
                        await ws.send_json({"type": "response", "text": "Already processing. Click stop first.", "is_system": True})
                    else:
                        _current_task = asyncio.create_task(_on_message(ws, data.get("text", "")))
                elif a == "stop":
                    if _current_task and not _current_task.done():
                        _current_task.cancel()
                        _current_task = None
                    else:
                        await ws.send_json({"type": "response", "text": "Nothing to stop.", "is_system": True})
                elif a == "sandbox_confirm":
                    await _on_sandbox(ws, data.get("confirm", False))
                elif a == "set_project":
                    await _on_project(ws, data.get("path", ""))
                elif a == "new_conversation":
                    await _on_new_conv(ws)
                elif a == "switch_conversation":
                    await _on_switch_conv(ws, data.get("id", ""))
                elif a == "rename_conversation":
                    await _on_rename_conv(ws, data.get("id", ""), data.get("name", ""))
                elif a == "delete_conversation":
                    await _on_delete_conv(ws, data.get("id", ""))
                elif a == "get_settings":
                    await _on_get_settings(ws)
                elif a == "save_settings":
                    await _on_save_settings(ws, data.get("settings", {}))
            elif msg.type == aiohttp.WSMsgType.ERROR:
                break
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
    return ws


async def _on_message(ws, text):
    global _pending_sandbox, _pending_maps
    if not text.strip():
        return

    memory = _get_memory()

    from main import handle_slash_command
    sr = handle_slash_command(text, memory)
    if sr is not None:
        await ws.send_json({"type": "response", "text": sr, "is_system": True})
        if _active_conv:
            _save_conv_memory(_active_conv)
        return

    if text.strip().lower().startswith("/index"):
        if not _project_root:
            await ws.send_json({"type": "response", "text": "No project set.", "is_system": True})
            return
        try:
            from tools.code_index import generate_maps, list_sections
            maps = await generate_maps(_project_root, force="force" in text.lower())
            secs = list_sections(maps["detailed"])
            await ws.send_json({"type": "response", "text": f"Indexed: {len(secs)} sections\n{', '.join(secs[:20])}", "is_system": True})
        except Exception as e:
            await ws.send_json({"type": "response", "text": f"Index failed: {e}", "is_system": True})
        return

    # Strip mode prefixes from displayed text (!!deep, !!deepcode, etc.)
    display_text = text
    for prefix in ["!!deepcode ", "!!simple ", "!!medium ", "!!hard ", "!!deep ",
                    "!!conjecture ", "!!compute ", "!!prove ", "!!image "]:
        if display_text.lower().startswith(prefix):
            display_text = display_text[len(prefix):].strip()
            break

    await _broadcast("thinking_start", {})
    try:
        import main as _m
        _m._project_root = _project_root

        answer = await _m.process_turn(text, memory)

        _pending_sandbox = _m._pending_sandbox
        _pending_maps = _m._pending_maps
        await _broadcast("thinking_end", {})
        await ws.send_json({"type": "response", "text": answer, "is_system": False})

        if _pending_sandbox:
            await ws.send_json({"type": "sandbox_prompt", "diff": _pending_sandbox.get_all_diffs()})

        # Auto-name conversation with AI
        if _active_conv and _conversations[_active_conv]["name"] == "New Chat":
            await _auto_name_conv(_active_conv, display_text)

        if _active_conv:
            _save_conv_memory(_active_conv)
    except asyncio.CancelledError:
        await _broadcast("thinking_end", {})
        await ws.send_json({"type": "response", "text": "Stopped.", "is_system": True})
        # Save whatever got added to memory before cancellation
        if _active_conv:
            _save_conv_memory(_active_conv)
            if _conversations[_active_conv]["name"] == "New Chat":
                await _auto_name_conv(_active_conv, display_text)
    except Exception as e:
        await _broadcast("thinking_end", {})
        await ws.send_json({"type": "response", "text": f"Error: {e}", "is_system": True})
        # Save whatever got added to memory before error
        if _active_conv:
            _save_conv_memory(_active_conv)
            if _conversations[_active_conv]["name"] == "New Chat":
                await _auto_name_conv(_active_conv, display_text)


async def _auto_name_conv(conv_id: str, user_text: str):
    """Name a conversation based on the first message using a fast model."""
    try:
        from core.retry import call_with_retry
        result = await call_with_retry(
            "groq/llama-3.1-8b",
            f"Give a short title (3-6 words, no quotes) for a conversation that starts with: \"{user_text[:200]}\"\nTitle:",
            max_tokens=20,
            temperature=0.3,
        )
        name = result.strip().strip('"\'').strip()[:50]
        if name and len(name) > 2:
            _conversations[conv_id]["name"] = name
            _save_conv_meta()
            await _broadcast("conv_renamed", {"id": conv_id, "name": name})
    except Exception:
        # Fallback: use first 40 chars of message
        name = user_text[:40] + ("..." if len(user_text) > 40 else "")
        _conversations[conv_id]["name"] = name
        _save_conv_meta()
        await _broadcast("conv_renamed", {"id": conv_id, "name": name})


async def _on_sandbox(ws, confirm):
    global _pending_sandbox, _pending_maps
    import main as _m
    if not _pending_sandbox:
        return
    if confirm:
        applied = _pending_sandbox.apply()
        t = "Changes applied:\n" + "\n".join(f"  {a}" for a in applied)
        if _project_root and _pending_maps:
            try:
                from tools.code_index import patch_maps
                h = patch_maps(_project_root, _pending_maps["general"], _pending_maps["detailed"])
                t += f"\nMaps updated ({h[:8]})"
            except Exception as e:
                t += f"\nMap patch failed: {e}"
    else:
        t = "Changes rejected."
    _pending_sandbox = None
    _pending_maps = None
    _m._pending_sandbox = None
    _m._pending_maps = None
    await ws.send_json({"type": "response", "text": t, "is_system": True})


async def _on_project(ws, path):
    global _project_root
    path = path.strip()
    if not path:
        await ws.send_json({"type": "response", "text": f"Project: {_project_root or 'none'}", "is_system": True})
        return
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        await ws.send_json({"type": "response", "text": f"Not a directory: {p}", "is_system": True})
        return
    _project_root = str(p)
    from main import _save_project_root
    _save_project_root(_project_root)
    await _broadcast("project_update", {"project": _project_root})


async def _on_new_conv(ws):
    global _active_conv
    cid = str(uuid.uuid4())[:8]
    _conversations[cid] = {"name": "New Chat", "memory": ConversationMemory()}
    _active_conv = cid
    _save_conv_meta()
    await _broadcast("conv_list", {
        "conversations": {c: _conversations[c]["name"] for c in _conversations},
        "active": _active_conv, "history": [],
    })


async def _on_switch_conv(ws, cid):
    global _active_conv
    if cid not in _conversations:
        return
    _active_conv = cid
    _save_conv_meta()
    await _broadcast("conv_list", {
        "conversations": {c: _conversations[c]["name"] for c in _conversations},
        "active": _active_conv, "history": _build_history(cid),
    })


async def _on_rename_conv(ws, cid, name):
    if cid not in _conversations or not name.strip():
        return
    _conversations[cid]["name"] = name.strip()
    _save_conv_meta()
    await _broadcast("conv_renamed", {"id": cid, "name": name.strip()})


async def _on_delete_conv(ws, cid):
    global _active_conv
    if cid not in _conversations or len(_conversations) <= 1:
        return
    del _conversations[cid]
    p = CONV_DIR / f"{cid}.json"
    if p.exists():
        p.unlink()
    if _active_conv == cid:
        _active_conv = next(iter(_conversations))
    _save_conv_meta()
    await _broadcast("conv_list", {
        "conversations": {c: _conversations[c]["name"] for c in _conversations},
        "active": _active_conv, "history": _build_history(_active_conv),
    })


# ─── Settings / API Keys ───────────────────────────────────────────────────

SETTINGS_FILE = Path.home() / ".jarvis" / "settings.json"


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  WARNING: Failed to read {SETTINGS_FILE}: {e}", flush=True)
    return {}


def _save_settings(settings: dict):
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"  ERROR saving settings to {SETTINGS_FILE}: {e}", flush=True)


async def _on_get_settings(ws):
    s = _load_settings()
    # Mask API keys for display (show last 6 chars)
    masked = {}
    for k, v in s.items():
        if v and len(v) > 6:
            masked[k] = "•" * (len(v) - 6) + v[-6:]
        else:
            masked[k] = v
    await ws.send_json({"type": "settings", "values": masked, "project": _project_root})


async def _on_save_settings(ws, settings: dict):
    current = _load_settings()
    changed = []
    for key in ["NVIDIA_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "OPENROUTER_API_KEY"]:
        val = settings.get(key, "")
        # Don't overwrite with masked value or empty
        if val and not val.startswith("•"):
            current[key] = val
            os.environ[key] = val
            changed.append(key.replace("_API_KEY", "").replace("_API_KEYS", ""))

    _save_settings(current)

    # Verify the file was actually written
    verify = _load_settings()
    for key in ["NVIDIA_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "OPENROUTER_API_KEY"]:
        if current.get(key) and current[key] != verify.get(key):
            print(f"  WARNING: {key} failed to persist!", flush=True)

    if changed:
        print(f"  Saved API keys: {', '.join(changed)}", flush=True)
    await ws.send_json({"type": "settings_saved"})


# ─── App ─────────────────────────────────────────────────────────────────────

def create_app():
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/ws", handle_ws)
    img = Path.home() / "jarvis_images"
    if img.exists():
        app.router.add_static("/images/", str(img))
    return app


async def start_server(port=3000):
    global _project_root
    _load_conversations()
    from main import _load_project_root
    _project_root = _load_project_root() or ""

    # Load saved API keys into environment — always override
    saved = _load_settings()
    loaded = []
    for key in ["NVIDIA_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "OPENROUTER_API_KEY"]:
        val = saved.get(key, "")
        if val:
            os.environ[key] = val
            loaded.append(key.replace("_API_KEY", "").replace("_API_KEYS", ""))
    if loaded:
        print(f"  Loaded API keys: {', '.join(loaded)}", flush=True)
    else:
        print(f"  No saved API keys found. Set them in Settings > API Keys.", flush=True)

    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    url = f"http://localhost:{port}"
    print(f"\n\033[1m\033[96m   ╔═══════════════════════════════════════╗")
    print(f"   ║       J.A.R.V.I.S. v0.5.1 — Web UI   ║")
    print(f"   ╠═══════════════════════════════════════╣")
    print(f"   ║  {url:<37s} ║")
    print(f"   ╚═══════════════════════════════════════╝\033[0m\n")
    webbrowser.open(url)
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(start_server())
