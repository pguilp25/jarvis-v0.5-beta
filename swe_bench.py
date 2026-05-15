#!/usr/bin/env python3
"""
SWE-bench Verified runner — runs the JARVIS coding agent against the
princeton-nlp/SWE-bench_Verified dataset, 5 instances in parallel by default.

Output: predictions.jsonl in the SWE-bench prediction format:
    {"instance_id": ..., "model_name_or_path": ..., "model_patch": ...}

After predictions are written, run the official evaluation harness with:
    python -m swebench.harness.run_evaluation \\
        --predictions_path predictions.jsonl \\
        --max_workers 5 \\
        --run_id jarvis_<timestamp> \\
        --cache_level env --clean True

The eval harness requires Docker. With 10–20 GB free, --cache_level env --clean True
keeps per-instance images from accumulating.

Usage:
    python swe_bench.py                       # 50 instances, parallel=5
    python swe_bench.py --limit 1             # smoke test
    python swe_bench.py --instance-ids id1,id2
    python swe_bench.py --limit 50 --parallel 5 --predictions predictions.jsonl
"""

from __future__ import annotations

import os

# Force HuggingFace datasets into offline mode BEFORE the library is imported.
# `datasets.config` reads HF_DATASETS_OFFLINE / HF_HUB_OFFLINE once at import
# time. If IPv6 is advertised but not routable on this host (we saw that
# here), an online check stalls in SYN-SENT for minutes. The dataset is
# cached locally after the first online load, so offline is correct for any
# subsequent run; if the cache is missing we fall back to online explicitly.
_HF_OFFLINE_DEFAULT = "1"
os.environ.setdefault("HF_DATASETS_OFFLINE", _HF_OFFLINE_DEFAULT)
os.environ.setdefault("HF_HUB_OFFLINE", _HF_OFFLINE_DEFAULT)

import argparse
import asyncio
import contextvars
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load saved API keys from ~/.jarvis/settings.json (same source as the
# interactive jarvis UI). The settings file is treated as the source of
# truth here — a stale shell-exported key would otherwise silently override
# the working one the user rotated through the UI, killing every model call
# with a 403 several minutes into the run.
try:
    _settings_file = Path.home() / ".jarvis" / "settings.json"
    if _settings_file.exists():
        _saved = json.loads(_settings_file.read_text(encoding="utf-8"))
        for _k in ("NVIDIA_API_KEY", "LIGHTNING_API_KEY", "DEEPINFRA_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
            _v = _saved.get(_k, "")
            if _v:
                os.environ[_k] = _v
except Exception:
    pass

# ─── Instance-tagged stderr ─────────────────────────────────────────────────
# Five parallel agents share one stderr — without tagging the output is
# unreadable. We install a proxy that prepends [instance_id] to every line
# based on a contextvar set by run_one_instance(). asyncio gives each Task
# its own copy of the context, so the tag follows the task.

_inst_tag: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "inst_tag", default=None
)
_orig_stderr = sys.stderr


class _TaggedStream:
    def __init__(self, underlying):
        self._u = underlying
        self._partial = ""

    def write(self, data: str) -> int:
        tag = _inst_tag.get()
        if tag is None:
            return self._u.write(data)
        # Tag at line boundaries so multi-write prints stay coherent.
        self._partial += data
        out = []
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            out.append(f"[{tag}] {line}\n")
        if out:
            self._u.write("".join(out))
        return len(data)

    def flush(self):
        if self._partial:
            tag = _inst_tag.get()
            if tag is None:
                self._u.write(self._partial)
            else:
                self._u.write(f"[{tag}] {self._partial}")
            self._partial = ""
        self._u.flush()

    def __getattr__(self, name):
        return getattr(self._u, name)


sys.stderr = _TaggedStream(_orig_stderr)
sys.stdout = _TaggedStream(sys.stdout)

# ─── JARVIS imports (after stream patching so their prints get tagged) ──────

from workflows.code import code_agent  # noqa: E402
from core.state import new_state  # noqa: E402
from core import thought_logger  # noqa: E402

# ─── Bypass full code indexing for SWE-bench ────────────────────────────────
# `tools.code_index.generate_maps` LLM-summarizes the entire codebase. For
# SWE-bench repos (astropy is 18 MB, django is larger) this spawns 60+
# concurrent NIM calls that swamp the 40-RPM rate limiter and time out —
# we observed every map batch failing on the smoke run. The agent still has
# [SEARCH], [REFS], [CODE] tools; that's exactly how SWE-agent and Agentless
# work. We stub generate_maps with a hint pointing the agent at its tools,
# and keep is_new_project=False (non-empty `detailed` string, not the
# sentinel "(empty project)") so workflows/code.py stays on the
# existing-project branch (not the empty-project "write from scratch" branch
# at workflows/code.py around line 7990).
#
# IMPORTANT: workflows/code.py imports generate_maps *inside* the code_agent
# body (`from tools.code_index import generate_maps`). That's a local import
# resolved at call time from the source module, so patching `workflows.code`
# alone doesn't work — we must replace the symbol on `tools.code_index`
# itself.

import tools.code_index as _code_index  # noqa: E402


async def _stub_generate_maps(project_root: str, force: bool = False) -> dict:
    name = Path(project_root).name
    general = (
        f"## {name}\n"
        "## Large existing codebase. Structural map skipped for SWE-bench performance.\n"
        "## Use the on-demand tools to discover code:\n"
        "##   [SEARCH: pattern]   — ripgrep over the repo\n"
        "##   [REFS: name]        — definitions / usages of a symbol\n"
        "##   [CODE: path/to/file] — read a file\n"
    )
    detailed = (
        f"## {name}\n"
        "(Detailed code index bypassed — read files directly with [CODE: path].)\n"
    )
    return {"general": general, "detailed": detailed, "purpose": ""}


_code_index.generate_maps = _stub_generate_maps

# ─── Config ─────────────────────────────────────────────────────────────────

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"
DEFAULT_MODEL_NAME = "jarvis-v0.6"
REPO_CACHE = Path.home() / ".swebench_repos"
WORK_BASE = Path.home() / ".swebench_work"
DEFAULT_TIMEOUT = 14400  # 4 h per instance, hard ceiling — tasks routinely take an hour+


# ─── Prompt construction ────────────────────────────────────────────────────


def build_task_prompt(instance: dict) -> str:
    """Format a SWE-bench instance as a task prompt for the coding agent."""
    problem = (instance.get("problem_statement") or "").strip()
    hints = (instance.get("hints_text") or "").strip()
    repo = instance["repo"]
    iid = instance["instance_id"]

    parts = [
        f"You are fixing a real GitHub issue in the {repo} repository.",
        f"Task id: {iid}",
        "",
        "RULES:",
        "  1. Produce a MINIMAL patch that resolves the issue.",
        "  2. Only edit source files. Do NOT add or modify any test files.",
        "  3. The repository's existing test suite must still pass after your fix.",
        "  4. Do NOT introduce new dependencies. Do NOT change unrelated code.",
        "  5. Stay inside the working tree at the current checkout — no shell, no git operations.",
        "",
        "=== ISSUE ===",
        problem,
    ]
    if hints:
        parts.extend(["", "=== MAINTAINER HINTS (from PR discussion) ===", hints])
    return "\n".join(parts)


# ─── Repo handling ──────────────────────────────────────────────────────────


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )


_clone_locks: dict[str, asyncio.Lock] = {}


def _clone_lock_for(repo: str) -> asyncio.Lock:
    if repo not in _clone_locks:
        _clone_locks[repo] = asyncio.Lock()
    return _clone_locks[repo]


async def ensure_repo_cached(repo: str, base_commit: str) -> Path:
    """One shared bare clone per repo. Per-instance clones hardlink from it.

    Bare = no working tree, half the disk of a regular clone. We do NOT use
    `--filter=blob:none` here: a partial clone breaks `git clone --no-local`
    later because lazy fetches can fail mid-clone. Serialized per-repo so
    5 parallel tasks don't race on the same .git.
    """
    owner, name = repo.split("/", 1)
    cache_path = REPO_CACHE / f"{owner}__{name}.git"

    async with _clone_lock_for(repo):
        if not cache_path.exists():
            REPO_CACHE.mkdir(parents=True, exist_ok=True)
            url = f"https://github.com/{repo}.git"
            r = _run(
                ["git", "clone", "--bare", url, str(cache_path)],
                timeout=900,
            )
            if r.returncode != 0:
                raise RuntimeError(f"git clone --bare {url} failed: {r.stderr[:400]}")

        # Ensure the base_commit is present (some new instances may sit on
        # commits the bare clone didn't pull originally — fetch on demand).
        r = _run(["git", "-C", str(cache_path), "cat-file", "-e", base_commit])
        if r.returncode != 0:
            r = _run(
                ["git", "-C", str(cache_path), "fetch", "origin", base_commit],
                timeout=300,
            )
            if r.returncode != 0:
                r = _run(["git", "-C", str(cache_path), "fetch", "--all"], timeout=600)
                if r.returncode != 0:
                    raise RuntimeError(f"fetch {base_commit} failed: {r.stderr[:400]}")

    return cache_path


def prepare_instance_workdir(instance: dict, cache_path: Path) -> Path:
    """Fresh local clone of the cached repo at base_commit, isolated per instance."""
    inst_dir = WORK_BASE / instance["instance_id"]
    if inst_dir.exists():
        shutil.rmtree(inst_dir)
    inst_dir.parent.mkdir(parents=True, exist_ok=True)

    # Default `git clone` from a local bare repo uses hardlinks for pack files:
    # disk-efficient and safe because the agent never writes inside .git/.
    r = _run(
        ["git", "clone", str(cache_path), str(inst_dir)],
        timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(f"local clone failed: {r.stderr[:400]}")

    r = _run(["git", "-C", str(inst_dir), "checkout", instance["base_commit"]])
    if r.returncode != 0:
        raise RuntimeError(f"checkout {instance['base_commit']} failed: {r.stderr[:400]}")

    # Identity so any incidental commits work; also config for clean diffs.
    _run(["git", "-C", str(inst_dir), "config", "user.email", "jarvis@localhost"])
    _run(["git", "-C", str(inst_dir), "config", "user.name", "JARVIS"])
    _run(["git", "-C", str(inst_dir), "config", "core.autocrlf", "false"])
    return inst_dir


def extract_patch(inst_dir: Path) -> str:
    """Run `git diff` on the instance dir after sandbox.apply() — git-compatible patch."""
    r = _run(
        ["git", "-C", str(inst_dir), "diff", "--no-color", "--no-ext-diff"],
        timeout=60,
    )
    return r.stdout or ""


# ─── Per-instance agent run ─────────────────────────────────────────────────


async def run_one_instance(
    instance: dict,
    sem: asyncio.Semaphore,
    model_name: str,
    timeout: int,
    summary_log,
) -> dict:
    iid = instance["instance_id"]
    tok = _inst_tag.set(iid)
    inst_dir: Path | None = None
    patch_text = ""
    error_msg = ""
    t0 = time.time()

    try:
        async with sem:
            summary_log(f"START  {iid}  ({instance['repo']})")
            try:
                cache_path = await ensure_repo_cached(instance["repo"], instance["base_commit"])
                inst_dir = prepare_instance_workdir(instance, cache_path)
            except Exception as e:
                error_msg = f"prep_failed: {e}"
                summary_log(f"PREP-FAIL {iid}  {error_msg}")
                return {
                    "instance_id": iid,
                    "model_name_or_path": model_name,
                    "model_patch": "",
                }

            state = new_state(raw_input=build_task_prompt(instance))
            state["processed_input"] = state["raw_input"]
            state["classification"] = {
                "complexity": 7,         # deep coding path (plan + parallel coders + review)
                "domain": "code",
                "agent": "code",
                "intent": "fix bug",
            }
            state["forced_complexity"] = 7
            state["project_root"] = str(inst_dir)

            try:
                state = await asyncio.wait_for(code_agent(state), timeout=timeout)
            except asyncio.TimeoutError:
                error_msg = f"timeout({timeout}s)"
                summary_log(f"TIMEOUT {iid}  {error_msg}")
            except Exception as e:
                error_msg = f"agent_exc: {type(e).__name__}: {e}"
                summary_log(f"AGENT-EXC {iid}  {error_msg}")
                traceback.print_exc(file=_orig_stderr)

            sandbox = state.get("pending_sandbox") if isinstance(state, dict) else None
            if sandbox is not None:
                try:
                    sandbox.apply()
                except Exception as e:
                    error_msg = error_msg or f"apply_failed: {e}"
                    summary_log(f"APPLY-FAIL {iid}  {error_msg}")

                try:
                    patch_text = extract_patch(inst_dir)
                except Exception as e:
                    error_msg = error_msg or f"diff_failed: {e}"

            elapsed = time.time() - t0
            size = len(patch_text)
            tag = f"err={error_msg}" if error_msg else "ok"
            summary_log(f"DONE   {iid}  elapsed={elapsed:6.0f}s  patch={size:6d}B  {tag}")

    finally:
        _inst_tag.reset(tok)
        if inst_dir is not None and inst_dir.exists():
            shutil.rmtree(inst_dir, ignore_errors=True)

    return {
        "instance_id": iid,
        "model_name_or_path": model_name,
        "model_patch": patch_text,
    }


# ─── Driver ─────────────────────────────────────────────────────────────────


def _select_instances(ds, limit: int, instance_ids: str, seed: int | None) -> list[dict]:
    if instance_ids:
        wanted = {s.strip() for s in instance_ids.split(",") if s.strip()}
        rows = [r for r in ds if r["instance_id"] in wanted]
        missing = wanted - {r["instance_id"] for r in rows}
        if missing:
            print(f"WARNING: {len(missing)} unknown instance_ids: {sorted(missing)[:5]}",
                  file=_orig_stderr)
        return rows
    n = min(limit, len(ds))
    # When --seed is set, shuffle the full dataset deterministically then
    # take the first n. This avoids the "first 50 are all astropy" trap
    # — SWE-bench_Verified is sorted by repo/issue id, so a head sample
    # is repo-biased. With a seed, the sample is representative AND
    # reproducible (same seed → same set, useful for A/B prompt runs).
    if seed is not None:
        import random as _random
        rng = _random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        return [ds[i] for i in indices[:n]]
    return [ds[i] for i in range(n)]


def _load_swe_bench_dataset():
    """Load SWE-bench Verified. Offline by default (see module top).

    If the local cache is missing, we re-import `datasets` with offline mode
    cleared so the first run can download. Subsequent runs hit the cache.
    """
    from datasets import load_dataset

    try:
        return load_dataset(DATASET_NAME, split=SPLIT)
    except Exception as off_exc:
        print(
            f"Cached offline load failed ({type(off_exc).__name__}: {off_exc}); "
            "retrying with online download (this can be slow if IPv6 is broken).",
            file=_orig_stderr,
        )
        os.environ["HF_DATASETS_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"
        # Force a re-read of datasets.config after toggling the env vars.
        import importlib, datasets, datasets.config  # noqa: WPS433
        importlib.reload(datasets.config)
        importlib.reload(datasets)
        from datasets import load_dataset as _online_load
        return _online_load(DATASET_NAME, split=SPLIT)


async def main_async(args) -> None:
    print(f"Loading {DATASET_NAME} split={SPLIT}...", file=_orig_stderr)
    ds = _load_swe_bench_dataset()

    instances = _select_instances(ds, args.limit, args.instance_ids, args.seed)
    if not instances:
        print("No instances selected — exiting.", file=_orig_stderr)
        return

    print(
        f"Running {len(instances)} instance(s), parallel={args.parallel}, "
        f"timeout={args.timeout}s, model_name={args.model_name}",
        file=_orig_stderr,
    )

    # One shared thought_logger session for the whole batch — model files
    # may interleave across instances; instance_id tagging stays on stderr.
    thought_logger.new_session(prompt=f"swe-bench run: {len(instances)} instances")

    sem = asyncio.Semaphore(args.parallel)

    def summary_log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        # Bypass the tagged proxy — these are the top-level run-summary lines.
        _orig_stderr.write(f"[{ts}] {msg}\n")
        _orig_stderr.flush()

    out_path = Path(args.predictions).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Stream predictions so a crash mid-run still leaves partial output.
    out_fh = out_path.open("w", buffering=1)

    write_lock = asyncio.Lock()

    async def run_and_record(inst: dict) -> None:
        pred = await run_one_instance(inst, sem, args.model_name, args.timeout, summary_log)
        async with write_lock:
            out_fh.write(json.dumps(pred) + "\n")
            out_fh.flush()

    t_start = time.time()
    await asyncio.gather(*(run_and_record(inst) for inst in instances))
    out_fh.close()
    total = time.time() - t_start

    # Quick result summary.
    succeeded = 0
    empty = 0
    with out_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["model_patch"]:
                succeeded += 1
            else:
                empty += 1
    summary_log(
        f"FINISHED  {len(instances)} instances in {total:.0f}s  "
        f"({succeeded} with patch, {empty} empty)  → {out_path}"
    )

    run_id = f"jarvis_{int(time.time())}"
    print(
        "\nTo run the official SWE-bench evaluation (requires Docker):",
        file=_orig_stderr,
    )
    print(
        f"  python -m swebench.harness.run_evaluation \\\n"
        f"    --predictions_path {out_path} \\\n"
        f"    --max_workers {args.parallel} \\\n"
        f"    --run_id {run_id} \\\n"
        f"    --cache_level env --clean True",
        file=_orig_stderr,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run JARVIS coding agent on SWE-bench Verified."
    )
    p.add_argument("--limit", type=int, default=50,
                   help="Number of instances to run (default 50).")
    p.add_argument("--parallel", type=int, default=5,
                   help="Concurrency for instance runs (default 5).")
    p.add_argument("--predictions", default="predictions.jsonl",
                   help="Output JSONL path (default predictions.jsonl).")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                   help="model_name_or_path field in predictions.")
    p.add_argument("--instance-ids", default="",
                   help="Comma-separated instance_ids; overrides --limit.")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help=f"Per-instance hard timeout in seconds (default {DEFAULT_TIMEOUT}).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed: shuffle the dataset before --limit so the "
                        "sample is representative (the dataset is sorted by repo, "
                        "so head-sampling is biased). Same seed → same sample.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted — partial predictions.jsonl may exist.", file=_orig_stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
