#!/usr/bin/env python3
"""
JARVIS Speed Benchmark — tests all models, measures latency and throughput.
Run: python3 benchmark.py
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clients.groq import call_groq
from clients.nvidia import call_nvidia
from core.cli import status
from core.tokens import count_tokens

# ─── Test Prompts ────────────────────────────────────────────────────────────

SIMPLE_PROMPT = "What is the Reynolds number? Answer in 2 sentences."
MEDIUM_PROMPT = "Explain the tradeoffs between LBM and finite volume for external aero at Re 500k. Be concise, 3-4 paragraphs."
CLASSIFY_PROMPT = """Classify this query. Reply with ONLY JSON, no markdown:
{"domain":"general|math|code|cfd","complexity":1-10,"agent":"chat|research|code"}

Query: Explain the tradeoffs between LBM and finite volume methods."""

# ─── Benchmark One Model ─────────────────────────────────────────────────────

async def bench_model(provider: str, model_id: str, prompt: str, label: str, max_tokens: int = 1024):
    """Benchmark a single model call. Returns dict with timing info."""
    try:
        start = time.time()
        if provider == "groq":
            result = await call_groq(model_id, prompt, max_tokens=max_tokens)
        elif provider == "nvidia":
            result = await call_nvidia(model_id, prompt, max_tokens=max_tokens)
        else:
            return None
        elapsed = time.time() - start

        output_tokens = count_tokens(result)
        tps = output_tokens / elapsed if elapsed > 0 else 0

        return {
            "model": model_id,
            "label": label,
            "elapsed": elapsed,
            "output_tokens": output_tokens,
            "tokens_per_sec": tps,
            "success": True,
            "preview": result[:100].replace("\n", " "),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "model": model_id,
            "label": label,
            "elapsed": elapsed,
            "output_tokens": 0,
            "tokens_per_sec": 0,
            "success": False,
            "preview": str(e)[:80],
        }


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    print("\n╔═══════════════════════════════════════════════════════════╗")
    print("║           JARVIS Speed Benchmark                         ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    # ── Groq Models ──────────────────────────────────────────────────
    groq_models = [
        ("groq/llama-3.1-8b",  "llama-3.1-8b-instant",                        "8B fast"),
        ("groq/gpt-oss-120b",  "openai/gpt-oss-120b",                         "GPT-OSS 120B"),
        ("groq/llama-4-scout", "meta-llama/llama-4-scout-17b-16e-instruct",   "Scout 17B"),
        ("groq/llama-3.3-70b", "llama-3.3-70b-versatile",                     "Llama 70B"),
        ("groq/qwen3-32b",     "qwen/qwen-3-32b",                             "Qwen3 32B"),
    ]

    nvidia_models = [
        ("nvidia/deepseek-v3.2",  "deepseek-ai/deepseek-v3.2",           "DeepSeek V3.2"),
        ("nvidia/glm-5",          "z-ai/glm5",                            "GLM-5"),
        ("nvidia/minimax-m2.7",   "minimaxai/minimax-m2.7",               "MiniMax M2.7"),
        ("nvidia/qwen-3.5",       "qwen/qwen3.5-397b-a17b",              "Qwen 3.5"),
        ("nvidia/kimi-k2.6",      "moonshotai/kimi-k2.6",                "Kimi K2.6"),
        ("nvidia/nemotron-super", "nvidia/nemotron-3-super-120b-a12b",   "Nemotron Super (legacy)"),
    ]

    results = []

    # ── Test 1: Classification speed (what decorticator needs) ───────
    print("═══ TEST 1: Classification Speed (decorticator task) ═══\n")
    print(f"  {'Model':<30} {'Time':>8} {'Tokens':>8} {'T/s':>8}  Status")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8}  {'─'*20}")

    for config_id, api_id, label in groq_models:
        r = await bench_model("groq", config_id, CLASSIFY_PROMPT, label, max_tokens=256)
        if r:
            results.append({**r, "test": "classify"})
            sym = "✓" if r["success"] else "✗"
            print(f"  {label:<30} {r['elapsed']:>7.1f}s {r['output_tokens']:>7} {r['tokens_per_sec']:>7.0f}  {sym} {r['preview'][:40]}")

    # Small pause
    await asyncio.sleep(1)

    for config_id, api_id, label in nvidia_models[:2]:  # Just test 2 NVIDIA for classify
        r = await bench_model("nvidia", config_id, CLASSIFY_PROMPT, label, max_tokens=256)
        if r:
            results.append({**r, "test": "classify"})
            sym = "✓" if r["success"] else "✗"
            print(f"  {label:<30} {r['elapsed']:>7.1f}s {r['output_tokens']:>7} {r['tokens_per_sec']:>7.0f}  {sym} {r['preview'][:40]}")

    # ── Test 2: Simple answer speed ──────────────────────────────────
    print(f"\n═══ TEST 2: Simple Answer Speed ═══\n")
    print(f"  {'Model':<30} {'Time':>8} {'Tokens':>8} {'T/s':>8}  Status")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8}  {'─'*20}")

    for config_id, api_id, label in groq_models:
        r = await bench_model("groq", config_id, SIMPLE_PROMPT, label, max_tokens=512)
        if r:
            results.append({**r, "test": "simple"})
            sym = "✓" if r["success"] else "✗"
            print(f"  {label:<30} {r['elapsed']:>7.1f}s {r['output_tokens']:>7} {r['tokens_per_sec']:>7.0f}  {sym}")

    # ── Test 3: Medium answer (parallel NVIDIA) ──────────────────────
    print(f"\n═══ TEST 3: Medium Answer — ALL NVIDIA in Parallel ═══\n")
    print("  Firing all 5 NVIDIA models simultaneously...")

    start_all = time.time()
    tasks = []
    for config_id, api_id, label in nvidia_models:
        tasks.append(bench_model("nvidia", config_id, MEDIUM_PROMPT, label, max_tokens=2048))

    nvidia_results = await asyncio.gather(*tasks)
    total_parallel = time.time() - start_all

    print(f"\n  {'Model':<30} {'Time':>8} {'Tokens':>8} {'T/s':>8}  Status")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8}  {'─'*20}")

    for r in nvidia_results:
        if r:
            results.append({**r, "test": "medium"})
            sym = "✓" if r["success"] else "✗"
            print(f"  {r['label']:<30} {r['elapsed']:>7.1f}s {r['output_tokens']:>7} {r['tokens_per_sec']:>7.0f}  {sym}")

    print(f"\n  Wall clock (parallel): {total_parallel:.1f}s")
    successful = [r for r in nvidia_results if r and r["success"]]
    if successful:
        sequential_est = sum(r["elapsed"] for r in successful)
        print(f"  Sequential estimate:   {sequential_est:.0f}s")
        print(f"  Speedup:               {sequential_est/total_parallel:.1f}x")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n═══ RECOMMENDATIONS ═══\n")

    # Find fastest classifier
    classifiers = [r for r in results if r["test"] == "classify" and r["success"]]
    if classifiers:
        fastest = min(classifiers, key=lambda r: r["elapsed"])
        print(f"  Best decorticator:  {fastest['label']} ({fastest['elapsed']:.1f}s, {fastest['tokens_per_sec']:.0f} t/s)")

    # Find fastest simple answerer
    simple = [r for r in results if r["test"] == "simple" and r["success"]]
    if simple:
        fastest = min(simple, key=lambda r: r["elapsed"])
        highest_tps = max(simple, key=lambda r: r["tokens_per_sec"])
        print(f"  Fastest answerer:   {fastest['label']} ({fastest['elapsed']:.1f}s)")
        print(f"  Highest throughput: {highest_tps['label']} ({highest_tps['tokens_per_sec']:.0f} t/s)")

    # NVIDIA speed ranking
    medium = [r for r in results if r["test"] == "medium" and r["success"]]
    if medium:
        by_speed = sorted(medium, key=lambda r: r["elapsed"])
        print(f"\n  NVIDIA speed ranking (medium answer):")
        for i, r in enumerate(by_speed):
            print(f"    {i+1}. {r['label']:<20} {r['elapsed']:>6.1f}s  ({r['tokens_per_sec']:.0f} t/s, {r['output_tokens']} tokens)")

    print()


if __name__ == "__main__":
    asyncio.run(main())
