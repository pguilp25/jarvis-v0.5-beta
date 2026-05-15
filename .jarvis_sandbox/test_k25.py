#!/usr/bin/env python3
"""Quick speed test for Kimi K2 Instruct on NVIDIA NIM."""
import asyncio, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clients.nvidia import call_nvidia
from core.tokens import count_tokens
from config import NVIDIA_MODEL_IDS

# Add the model for testing
NVIDIA_MODEL_IDS["nvidia/kimi-k2"] = "moonshotai/kimi-k2-instruct-0905"

PROMPTS = {
    "classify": "Classify this query as simple/medium/hard and pick a domain (math/code/cfd/general). Query: 'Explain LBM vs FVM tradeoffs at Re 500k'. Reply JSON only.",
    "short": "What is the Reynolds number? Answer in 2 sentences.",
    "medium": "Explain the tradeoffs between LBM and finite volume for external aero at Re 500k. Be concise, 3-4 paragraphs.",
}

async def test(name, prompt, max_tokens):
    print(f"\n  Test: {name} (max_tokens={max_tokens})")
    start = time.time()
    try:
        result = await call_nvidia("nvidia/kimi-k2", prompt, max_tokens=max_tokens)
        elapsed = time.time() - start
        tokens = count_tokens(result) if result else 0
        tps = tokens / elapsed if elapsed > 0 else 0
        print(f"  Time: {elapsed:.1f}s | Tokens: {tokens} | Speed: {tps:.0f} t/s")
        preview = (result or "")[:120].replace(chr(10), ' ')
        print(f"  Preview: {preview}")
        return {"name": name, "elapsed": elapsed, "tokens": tokens, "tps": tps}
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR after {elapsed:.1f}s: {e}")
        return {"name": name, "elapsed": elapsed, "tokens": 0, "tps": 0}

async def main():
    print("╔══════════════════════════════════════════════╗")
    print("║   Kimi K2 Instruct Speed Test (NVIDIA NIM)   ║")
    print("╚══════════════════════════════════════════════╝")

    results = []
    for name, prompt in PROMPTS.items():
        max_tok = 256 if name == "classify" else (512 if name == "short" else 2048)
        r = await test(name, prompt, max_tok)
        results.append(r)
        await asyncio.sleep(1)

    print(f"\n  ═══ Summary ═══")
    for r in results:
        print(f"  {r['name']:<10} {r['elapsed']:>6.1f}s  {r['tokens']:>5} tok  {r['tps']:>5.0f} t/s")

if __name__ == "__main__":
    asyncio.run(main())
