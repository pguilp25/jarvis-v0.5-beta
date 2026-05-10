"""
NVIDIA NIM embeddings client — uses llama-nemotron-embed-1b-v2.
Generates vector embeddings for semantic search over the code purpose map.
"""

import json
import math
import os
import time
from pathlib import Path

import aiohttp

EMBED_API_URL = "https://integrate.api.nvidia.com/v1/embeddings"
EMBED_MODEL   = "nvidia/llama-nemotron-embed-1b-v2"
EMBED_CACHE   = "embeddings.json"   # stored in the maps cache dir


def _get_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    return key


async def embed_texts(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Call NVIDIA NIM embeddings endpoint. Returns one vector per input text.

    input_type:
      "passage"  — for indexing code chunks
      "query"    — for embedding a user query at search time
    """
    headers = {
        "Authorization": f"Bearer {_get_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        "input_type": input_type,
        "encoding_format": "float",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            EMBED_API_URL, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Embed API HTTP {resp.status}: {body[:300]}")
            data = await resp.json()
    # data["data"] is a list of {"embedding": [...], "index": N}
    ordered = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in ordered]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _embed_cache_path(maps_dir: Path) -> Path:
    return maps_dir / EMBED_CACHE


def load_embed_cache(maps_dir: Path) -> dict | None:
    """Load embedding cache: {"hash": str, "chunks": [{"name": str, "text": str, "vec": [...]}]}"""
    p = _embed_cache_path(maps_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_embed_cache(maps_dir: Path, file_hash: str, chunks: list[dict]):
    """Save embedding cache to disk."""
    p = _embed_cache_path(maps_dir)
    p.write_text(
        json.dumps({"hash": file_hash, "chunks": chunks}),
        encoding="utf-8",
    )


# ─── Purpose-map chunker ─────────────────────────────────────────────────────

def parse_purpose_chunks(purpose_map: str) -> list[dict]:
    """Split purpose map into chunks: [{"name": str, "text": str}]"""
    import re
    chunks = []
    parts = re.split(r'===\s*PURPOSE:\s*', purpose_map)
    for part in parts:
        if not part.strip():
            continue
        title_end = part.find("===")
        if title_end == -1:
            title = part.split("\n")[0].strip()
            body  = part.strip()
        else:
            title = part[:title_end].strip()
            body  = part[title_end + 3:].strip()
        if not title:
            continue
        # Build indexable text: title + description + first few lines of body
        desc = ""
        for line in body.split("\n")[:5]:
            if line.strip():
                desc += line.strip() + " "
        text = f"{title}. {desc}".strip()
        chunks.append({"name": title, "text": text})
    return chunks


# ─── Main: build / retrieve ───────────────────────────────────────────────────

async def build_embeddings(
    purpose_map: str,
    maps_dir: Path,
    file_hash: str,
    batch_size: int = 32,
) -> list[dict]:
    """Embed all purpose-map chunks and save to cache.
    Returns the list of chunk dicts with "vec" populated.
    """
    from core.cli import status, warn
    chunks = parse_purpose_chunks(purpose_map)
    if not chunks:
        return []

    # Embed in batches (API limit)
    all_vecs: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch_texts = [c["text"] for c in chunks[i:i + batch_size]]
        try:
            vecs = await embed_texts(batch_texts, input_type="passage")
            all_vecs.extend(vecs)
            status(f"    Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
        except Exception as e:
            warn(f"    Embedding batch {i//batch_size + 1} failed: {e}")
            # Fill with zero vectors so indices stay aligned
            all_vecs.extend([[0.0] * 4096] * len(batch_texts))

    for chunk, vec in zip(chunks, all_vecs):
        chunk["vec"] = vec

    save_embed_cache(maps_dir, file_hash, chunks)
    return chunks


async def semantic_retrieve(
    query: str,
    purpose_map: str,
    project_root: str,
    maps_dir: Path,
    file_hash: str,
    top_n: int = 10,
) -> str:
    """Embed query, find top_n purpose chunks, return their code with ±10 lines context."""
    from tools.code_index import get_purpose_snippets
    from core.cli import status, warn

    # Load or build embedding cache
    cache = load_embed_cache(maps_dir)
    if cache and cache.get("hash") == file_hash and cache.get("chunks"):
        chunks = cache["chunks"]
    else:
        status("    Building semantic index (first time)...")
        try:
            chunks = await build_embeddings(purpose_map, maps_dir, file_hash)
        except Exception as e:
            warn(f"    Semantic index build failed: {e}")
            return f"(semantic search unavailable: {e})"

    if not chunks:
        return "(no purpose categories to search)"

    # Embed the query
    try:
        query_vecs = await embed_texts([query], input_type="query")
        qvec = query_vecs[0]
    except Exception as e:
        warn(f"    Query embedding failed: {e}")
        return f"(semantic search unavailable: {e})"

    # Rank by cosine similarity
    scored = []
    for chunk in chunks:
        vec = chunk.get("vec")
        if not vec or all(v == 0 for v in vec[:5]):
            continue
        sim = cosine_similarity(qvec, vec)
        scored.append((sim, chunk["name"]))

    scored.sort(reverse=True)
    top = scored[:top_n]

    if not top:
        return f"(no results for '{query}')"

    parts = [f"=== SEMANTIC: '{query}' — top {len(top)} matches ===\n"]
    for sim, cat_name in top:
        snippet = get_purpose_snippets(purpose_map, cat_name, project_root)
        parts.append(f"[similarity: {sim:.3f}]\n{snippet}\n")

    return "\n".join(parts)
