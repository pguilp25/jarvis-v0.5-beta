"""
JARVIS v5 Integration Guide
============================

These are the exact changes to make in your existing files.
Apply them in order.

═══════════════════════════════════════════════════════════════
1. config.py — Add !!compute override
═══════════════════════════════════════════════════════════════

In your OVERRIDE_MAP dict, add:

    "!!compute": 99,

So it looks like:

    OVERRIDE_MAP = {
        "!!simple": 2,
        "!!medium": 5,
        "!!hard": 10,
        "!!deep": 99,
        "!!conjecture": 99,
        "!!compute": 99,       # ← ADD THIS
    }


═══════════════════════════════════════════════════════════════
2. main.py — Route !!compute to deep thinking v5
═══════════════════════════════════════════════════════════════

REPLACE your import:
    from workflows.deep_thinking import chat_deep_thinking

WITH:
    from workflows.deep_thinking_v5 import chat_deep_thinking

That's it — deep_thinking_v5 auto-detects !!compute vs !!deep.


═══════════════════════════════════════════════════════════════
3. main.py — Update help text
═══════════════════════════════════════════════════════════════

Add to your help overrides section:

    !!compute   — Compute mode (code-execution research loop for combinatorics)

Full help should look like:

    Overrides:
      !!simple      — Force complexity 1-2 (fast)
      !!medium      — Force complexity 5-6 (intelligent)
      !!hard        — Force complexity 9-10 (full ensemble + debate)
      !!deep        — Deep thinking mode (100+ cycles, for hard conjectures)
      !!conjecture  — Same as !!deep
      !!compute     — Compute mode (code-execution + Z3 solver research loop)


═══════════════════════════════════════════════════════════════
4. core/retry.py — Add connectivity check (optional but recommended)
═══════════════════════════════════════════════════════════════

At the TOP of call_with_retry(), add:

    from tools.connectivity import is_online, wait_for_connection

    # Inside call_with_retry, before the retry loop:
    if not is_online():
        ok = await wait_for_connection(f"API call to {model_id}")
        if not ok:
            raise ConnectionError(f"Internet lost during call to {model_id}")


═══════════════════════════════════════════════════════════════
5. core/model_selector.py — Add NVIDIA_5_ENSEMBLE constant
═══════════════════════════════════════════════════════════════

If not already there, add:

    NVIDIA_5_ENSEMBLE = [
        "nvidia/deepseek-v3.2",
        "nvidia/glm-5",
        "nvidia/minimax-m2.5",
        "nvidia/qwen-3.5",
        "nvidia/nemotron-super",
    ]asdfghj


═══════════════════════════════════════════════════════════════
6. tools/__init__.py — Create if missing
═══════════════════════════════════════════════════════════════

    # tools/__init__.py
    # JARVIS tool modules


═══════════════════════════════════════════════════════════════
7. Install dependencies on Chromebook
═══════════════════════════════════════════════════════════════

    pip3 install z3-solver networkx numpy scipy --break-system-packages


═══════════════════════════════════════════════════════════════
8. TESTING
═══════════════════════════════════════════════════════════════

Test 1 — Text mode (should work exactly like v4):
    !!deep Prove that the sum of two even numbers is always even.

Test 2 — Compute mode (the new stuff):
    !!compute Find the adjacency matrix for a strongly regular graph srg(99, 14, 1, 2)

Test 3 — Connectivity (pull your WiFi cable during a cycle):
    Should pause, then resume when reconnected.

Test 4 — Timeout (should trigger within 60s):
    !!compute Find all Hamiltonian cycles in a random graph on 50 vertices


═══════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW — What happens on !!compute
═══════════════════════════════════════════════════════════════

    User: !!compute Find srg(99, 14, 1, 2) adjacency matrix

    ┌─────────────────────────────────────────────────────────┐
    │  Cycle 0: Initial analysis (5 AIs understand problem)   │
    └────────────────────────┬────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────┐
    │  Cycle N:                                               │
    │                                                         │
    │  Step A: 5 AIs hypothesize strategies (text)            │
    │    "Use vertex coloring to break symmetry"              │
    │    "Encode as binary matrix with row/col sum = 14"      │
    │    "Exploit eigenvalue constraints (2, -5)"             │
    │                                                         │
    │  Step B: 5 AIs write Python scripts (Z3 + networkx)    │
    │    Each gets their hypothesis + ALL 5 hypotheses        │
    │                                                         │
    │  Step C: Execute each script (subprocess, 60s timeout)  │
    │    ┌─ deepseek: ran 45s → PARTIAL (found 12/14 edges)  │
    │    ├─ glm-5:    ran 3s  → ERROR (import typo)          │
    │    ├─ minimax:  ran 60s → TIMEOUT → retry → TIMEOUT    │
    │    │            → try alternate code → TIMEOUT          │
    │    │            → SKIP minimax for next cycle           │
    │    ├─ qwen:     ran 28s → SUCCESS (candidate found!)   │
    │    └─ nemotron: ran 60s → TIMEOUT → retry → PARTIAL    │
    │                                                         │
    │  Step D: All results → attempt log → resolution check   │
    │    If qwen's candidate is valid → SOLVED → stop         │
    │    If not → next cycle with updated attempt log          │
    └────────────────────────┬────────────────────────────────┘
                             │
                     (loop up to 100 cycles)
                             │
    ┌────────────────────────▼────────────────────────────────┐
    │  Connectivity: if WiFi drops mid-cycle → PAUSE          │
    │  Poll every 5s → resume exactly where left off          │
    │  If offline > 10 min → save progress + stop             │
    └─────────────────────────────────────────────────────────┘

REDUNDANCY per script execution:
    Try 1: primary code ─── fail? ──→
    Try 2: primary retry ── fail? ──→
    Try 3: alternate code ─ fail? ──→
    Try 4: alternate retry ─ fail? ──→ skip this model this cycle
"""
