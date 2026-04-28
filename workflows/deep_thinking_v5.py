"""
Deep Thinking v5.1 — Text debate + Compute + Formal Proof Engine.

Three modes:
  !!deep / !!conjecture → Text debate (v4)
  !!compute             → Code-execution research loop (FunSearch-style)
  !!prove               → Lean 4 formal proof loop (NO self-eval, NO formatter)

!!prove mode:
  - The Lean 4 compiler IS the evaluator. Not Gemini Flash.
  - If compiler errors → feed errors back to MoA → loop
  - If compiler accepts → HALT. Save .lean to Desktop. Print raw proof. Done.
  - Self-eval is KILLED. Formatter is KILLED. Raw output only.
"""

import asyncio
import time

from core.retry import call_with_retry
from core.debate import STEP_A_PROMPT, STEP_B_PROMPT
from core.ensemble import run_ensemble, _build_prompt
from core.synthesizer import verify_agreement, merge_answers, synthesize
from core.state import AgentState
from core.cli import step, status, success, warn, error
from domains.prompts import get_assumption_prompt
from core.system_knowledge import SYSTEM_KNOWLEDGE
from tools.compute_node import (
    execute_with_redundancy, extract_python_code,
    format_result_for_moa, COMPUTE_TIMEOUT, cleanup_temp_files,
)
from tools.connectivity import is_online, wait_for_connection
from tools.lean_node import (
    execute_lean, extract_lean_code, format_lean_errors_for_moa,
    save_proof_to_desktop, cleanup_lean_files, LEAN_TIMEOUT,
)


# ─── Constants ───────────────────────────────────────────────────────────────

ATTEMPT_LOG_LIMIT = 12_000  # Larger for compute (code snippets + outputs)
MAX_CYCLES = 100

ALL_NVIDIA = [
    "nvidia/deepseek-v3.2",
    "nvidia/glm-5",
    "nvidia/minimax-m2.5",
    "nvidia/qwen-3.5",
    "nvidia/nemotron-super",
]

# ─── Text-mode verification (same as v4) ────────────────────────────────────

VERIFY_SOLUTION_PROMPT = SYSTEM_KNOWLEDGE + """

One AI claims to have SOLVED this problem. Rigorously verify their solution.

ORIGINAL PROBLEM:
{problem}

CLAIMED SOLUTION (by {solver_model}):
{claimed_solution}

YOUR TASK:
1. Check EVERY step — is each logically valid?
2. Look for: hidden assumptions, circular reasoning, gaps, arithmetic errors
3. Try to find a COUNTEREXAMPLE

Reply with ONLY:
CONFIRMED — the solution is correct. [one sentence why]
or
REJECTED — the solution has a flaw. [explain the flaw]"""


# ─── Compute-mode prompts ────────────────────────────────────────────────────

COMPUTE_STEP_A = SYSTEM_KNOWLEDGE + """

You are a computational mathematician in a 5-AI research team.
This problem CANNOT be solved by text reasoning — you must write CODE.

PROBLEM:
{problem}

ATTEMPT LOG (what was tried — DO NOT repeat failed approaches):
{attempt_log}

LAST EXECUTION RESULTS:
{last_execution_results}

CURRENT BEST APPROACH:
{current_best}

Step A — Hypothesize a NEW computational strategy:
1. Analyze WHY previous attempts failed (timeout? wrong constraints? too large?)
2. Propose a NEW approach. Think about:
   - Symmetry-breaking constraints to shrink the search space
   - Incremental solving (solver.push/pop)
   - Better variable encoding
   - Mathematical properties that constrain the solution
   - Problem decomposition

Format:
## HYPOTHESIS
[Your strategy]

## WHY IT SHOULD WORK
[Mathematical justification]

## ESTIMATED SEARCH SPACE REDUCTION
[How much smaller?]
"""

COMPUTE_STEP_B = SYSTEM_KNOWLEDGE + """

Based on 5 AI hypotheses, write a Python script to search for a solution.

PROBLEM:
{problem}

YOUR HYPOTHESIS:
{hypothesis}

ALL 5 HYPOTHESES:
{all_hypotheses}

ATTEMPT LOG:
{attempt_log}

RULES:
1. Use z3-solver (from z3 import *) and/or networkx
2. MUST print results to stdout
3. MUST complete within 60s or it gets KILLED
4. Set Z3 timeout: solver.set("timeout", 55000)  # 55s, 5s margin
5. Print partial results — if solver times out, print what was found
6. Use incremental solving where possible
7. At the END print:
   RESULT: FOUND <solution>
   or RESULT: NOT_FOUND <reason>
   or RESULT: PARTIAL <what was narrowed>
8. Single-threaded only. Keep memory reasonable (Chromebook).

Write ONLY the Python script inside a ```python block.
"""


# ─── Shared helpers ──────────────────────────────────────────────────────────

async def _call(model: str, prompt: str, max_tokens: int = 16384) -> dict:
    result = await call_with_retry(model, prompt, max_tokens=max_tokens)
    return {"model": model, "answer": result}


def _extract_attempt_log(answer: str) -> str:
    marker = "[ATTEMPT_LOG]"
    idx = answer.find(marker)
    if idx == -1:
        idx = answer.lower().find("[attempt_log]")
    if idx == -1:
        return ""
    log_text = answer[idx + len(marker):].strip()
    for stop in ["[CONTEXT_NOTES]", "[context_notes]", "[STATUS]", "[status]", "═══"]:
        stop_idx = log_text.find(stop)
        if stop_idx != -1:
            log_text = log_text[:stop_idx].strip()
    return log_text


def _extract_status(answer: str) -> str:
    marker = "[STATUS]"
    idx = answer.find(marker)
    if idx == -1:
        idx = answer.lower().find("[status]")
    if idx == -1:
        return "UNSOLVED"
    status_text = answer[idx + len(marker):].strip()
    first_line = status_text.split("\n")[0].strip().upper()
    return "SOLVED" if first_line.startswith("SOLVED") else "UNSOLVED"


def _trim_log(log: str, limit: int = ATTEMPT_LOG_LIMIT) -> str:
    parts = log.split("--- CYCLE")
    if len(parts) <= 2:
        return log[-limit:]
    while len("--- CYCLE".join(parts)) > limit and len(parts) > 2:
        parts.pop(1)
    trimmed = "--- CYCLE".join(parts)
    return trimmed[-limit:] if len(trimmed) > limit else trimmed


def _get_context(state: AgentState) -> str:
    history = state.get("conversation_history", [])
    if not history:
        return ""
    return "\n".join(m.get("content", "")[:2000] for m in history[-3:])


def _get_last_exchange(state: AgentState) -> str:
    history = state.get("conversation_history", [])
    if not history:
        return ""
    content = history[-1].get("content", "") if history else ""
    lines = content.split("\n") if content else []
    last_msgs = []
    for line in reversed(lines):
        if "] USER:" in line or "] ASSISTANT:" in line:
            last_msgs.insert(0, line)
            if len(last_msgs) >= 2:
                break
    return "\n".join(last_msgs) if last_msgs else ""


async def _get_best(query: str, answers: list[dict]) -> str:
    verdict = await verify_agreement(query, answers)
    if verdict == "agree":
        return answers[-1]["answer"]
    elif verdict == "complementary":
        result = await merge_answers(query, answers)
        return result["answer"]
    else:
        result = await synthesize(query, answers)
        return result["answer"]


async def _ensure_online(context: str):
    """Connectivity gate — blocks if offline."""
    if not is_online():
        ok = await wait_for_connection(context)
        if not ok:
            raise ConnectionError(f"Internet lost too long during {context}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT MODE — Identical to v4 deep_thinking.py
# ═══════════════════════════════════════════════════════════════════════════════

async def _text_loop(state: AgentState, query: str, context: str, answers: list[dict],
                     original_prompt: str, current_best: str) -> AgentState:
    """v4 text debate loop — 100 cycles with attempt log + SOLVED verification."""
    attempt_log = ""

    for cycle in range(1, MAX_CYCLES + 1):
        await _ensure_online(f"text cycle {cycle}")
        step(f"═══ Deep Thinking Cycle {cycle}/{MAX_CYCLES} ═══")

        all_answers_text = "\n\n".join(
            f"═══ AI-{i+1} ({a['model'].split('/')[-1]}) ═══\n{a['answer']}"
            for i, a in enumerate(answers)
        )

        # Step A: Hypothesize flaws
        step(f"Cycle {cycle} Step A: hypothesize flaws (all 5)")
        step_a_prompt = STEP_A_PROMPT.format(original_prompt=original_prompt, other_answers=all_answers_text)
        if attempt_log:
            step_a_prompt += f"\n\nATTEMPT LOG (DO NOT repeat failed approaches):\n{attempt_log}"

        step_a_results = list(await asyncio.gather(*[_call(m, step_a_prompt) for m in ALL_NVIDIA]))
        status(f"Step A: {len(step_a_results)} hypothesis sets")

        combined_hypotheses = "\n\n".join(
            f"═══ REVIEWER {i+1} ({r['model'].split('/')[-1]}) ═══\n{r['answer']}"
            for i, r in enumerate(step_a_results)
        )

        # Step B: Verify + improve
        await _ensure_online(f"text cycle {cycle} step B")
        step(f"Cycle {cycle} Step B: verify + improve (all 5)")
        step_b_prompt = STEP_B_PROMPT.format(original_prompt=original_prompt, hypotheses=combined_hypotheses)
        answers = list(await asyncio.gather(*[_call(m, step_b_prompt) for m in ALL_NVIDIA]))
        status(f"Step B: {len(answers)} improved answers")

        current_best = await _get_best(query, answers)

        # Extract attempt logs
        cycle_entries = []
        for a in answers:
            ai_name = a['model'].split('/')[-1]
            log_section = _extract_attempt_log(a['answer'])
            if log_section:
                cycle_entries.append(f"  {ai_name}:\n{log_section}")

        if cycle_entries:
            attempt_log += f"\n--- CYCLE {cycle} ---\n" + "\n".join(cycle_entries)
        else:
            fallback = "\n".join(f"  {a['model'].split('/')[-1]}: {a['answer'][-200:]}" for a in answers)
            attempt_log += f"\n--- CYCLE {cycle} ---\n{fallback}"

        if len(attempt_log) > ATTEMPT_LOG_LIMIT:
            attempt_log = _trim_log(attempt_log)

        status(f"Cycle {cycle}: best={len(current_best)} chars, log={len(attempt_log)} chars")

        # Check SOLVED claims
        solvers = [a for a in answers if _extract_status(a['answer']) == "SOLVED"]
        if solvers:
            solver = solvers[0]
            solver_name = solver['model'].split('/')[-1]
            step(f"🔍 {solver_name} claims SOLVED — all 5 verifying...")

            verify_prompt = VERIFY_SOLUTION_PROMPT.format(
                problem=query, solver_model=solver_name, claimed_solution=solver['answer'][:12000],
            )
            verifications = list(await asyncio.gather(*[_call(m, verify_prompt) for m in ALL_NVIDIA]))

            confirmed = sum(1 for v in verifications if v['answer'].strip().upper().startswith("CONFIRMED"))
            for v in verifications:
                v_name = v['model'].split('/')[-1]
                tag = "✓ CONFIRMED" if v['answer'].strip().upper().startswith("CONFIRMED") else "✗ REJECTED"
                status(f"  {tag}: {v_name}")

            if confirmed >= 3:
                success(f"VERIFIED SOLVED — {confirmed}/5 confirmed after {cycle} cycles!")
                state["final_answer"] = solver['answer']
                state["confidence"] = f"[SOLVED — verified {confirmed}/5 after {cycle} deep thinking cycles]"
                state["attempt_log"] = attempt_log
                return state
            else:
                warn(f"Claim rejected ({confirmed}/5) — continuing...")
                attempt_log += f"\n  ⚠️ {solver_name} claimed SOLVED but rejected ({confirmed}/5)"

        status(f"Cycle {cycle}: not yet solved — continuing...")

    warn(f"Max {MAX_CYCLES} cycles reached")
    state["final_answer"] = current_best
    state["confidence"] = f"[{MAX_CYCLES} cycles completed, unresolved]"
    state["attempt_log"] = attempt_log
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPUTE MODE — Code-execution research loop
# ═══════════════════════════════════════════════════════════════════════════════

async def _compute_loop(state: AgentState, query: str) -> AgentState:
    """FunSearch-style code-execution loop."""
    attempt_log = ""
    current_best = ""
    last_execution_results = ""
    skipped_models: set = set()

    # Initial analysis (text — understand the problem)
    step("Initial analysis: 5 AIs understanding the problem")
    await _ensure_online("initial analysis")
    init_prompt = (
        f"Analyze this problem. What are the key mathematical/computational challenges? "
        f"What makes it hard? What search space size? What constraints exist?\n\n"
        f"PROBLEM: {query}\n\nBe specific."
    )
    init_results = list(await asyncio.gather(
        *[_call(m, init_prompt) for m in ALL_NVIDIA], return_exceptions=True
    ))
    init_results = [r for r in init_results if isinstance(r, dict) and r.get("answer")]

    if init_results:
        current_best = max(init_results, key=lambda r: len(r["answer"]))["answer"]
        attempt_log = "--- CYCLE 0 (INIT) ---\nInitial problem analysis completed.\n"

    # Main compute loop
    for cycle in range(1, MAX_CYCLES + 1):
        t0 = time.monotonic()
        await _ensure_online(f"compute cycle {cycle}")
        step(f"═══ Compute Cycle {cycle}/{MAX_CYCLES} ═══")

        active_models = [m for m in ALL_NVIDIA if m not in skipped_models]
        if not active_models:
            warn("All models skipped — resetting skip list")
            skipped_models.clear()
            active_models = list(ALL_NVIDIA)

        # Step A: Hypothesize strategies (text)
        step(f"Step A: {len(active_models)} AIs hypothesizing strategies...")
        await _ensure_online(f"compute cycle {cycle} step A")

        step_a_tasks = []
        for model in active_models:
            prompt = COMPUTE_STEP_A.format(
                problem=query, attempt_log=attempt_log[-6000:],
                last_execution_results=last_execution_results[-4000:],
                current_best=current_best[-4000:],
            )
            step_a_tasks.append(_call(model, prompt))

        hypotheses = list(await asyncio.gather(*step_a_tasks, return_exceptions=True))
        hypotheses = [h for h in hypotheses if isinstance(h, dict) and h.get("answer")]

        if not hypotheses:
            warn("No hypotheses — all models failed Step A")
            attempt_log += f"\n--- CYCLE {cycle} ---\nAll models failed Step A\n"
            continue

        all_hypotheses_text = "\n\n".join(
            f"=== {h['model']} ===\n{h['answer'][:3000]}" for h in hypotheses
        )
        status(f"Step A: {len(hypotheses)} hypotheses")

        # Step B: Write code
        step(f"Step B: {len(hypotheses)} AIs writing code...")
        await _ensure_online(f"compute cycle {cycle} step B")

        step_b_tasks = []
        for h in hypotheses:
            prompt = COMPUTE_STEP_B.format(
                problem=query, hypothesis=h["answer"][:4000],
                all_hypotheses=all_hypotheses_text[:8000], attempt_log=attempt_log[-4000:],
            )
            step_b_tasks.append(_call(h["model"], prompt))

        code_responses = list(await asyncio.gather(*step_b_tasks, return_exceptions=True))
        code_responses = [c for c in code_responses if isinstance(c, dict) and c.get("answer")]

        if not code_responses:
            warn("No code generated — all models failed Step B")
            attempt_log += f"\n--- CYCLE {cycle} ---\nAll models failed Step B\n"
            continue

        status(f"Step B: {len(code_responses)} code scripts")

        # Step C: Execute each script with redundancy
        step(f"Step C: Executing {len(code_responses)} scripts...")
        execution_results = []
        cycle_found_solution = False

        for i, code_resp in enumerate(code_responses):
            model_name = code_resp["model"].split("/")[-1]
            primary_code = extract_python_code(code_resp["answer"])

            if not primary_code:
                warn(f"{model_name}: no valid Python code block — skipping")
                execution_results.append(f"{model_name}: NO CODE EXTRACTED")
                continue

            # Find alternate code from different model
            alt_code = None
            for j, other in enumerate(code_responses):
                if j != i:
                    alt = extract_python_code(other["answer"])
                    if alt and alt != primary_code:
                        alt_code = alt
                        break

            result = await execute_with_redundancy(
                code=primary_code, timeout=COMPUTE_TIMEOUT,
                label=f"c{cycle}_{model_name}", alternate_code=alt_code,
            )

            result_str = format_result_for_moa(result, primary_code)
            execution_results.append(f"### {model_name}:\n{result_str}")

            if result.get("skip_model"):
                skipped_models.add(code_resp["model"])
                warn(f"{model_name}: added to skip list")

            if result.get("success") and result.get("stdout") and "RESULT: FOUND" in result["stdout"]:
                success(f"SOLUTION FOUND by {model_name}!")
                cycle_found_solution = True
                current_best = result["stdout"]

        last_execution_results = "\n\n".join(execution_results)

        # Build cycle log
        cycle_log = f"\n--- CYCLE {cycle} (COMPUTE) ---\n"
        cycle_log += f"Models: {', '.join(m.split('/')[-1] for m in active_models)}\n"
        cycle_log += f"Skipped: {', '.join(s.split('/')[-1] for s in skipped_models) or 'none'}\n"
        for er in execution_results:
            for line in er.split("\n"):
                if "RESULT:" in line or "TIMEOUT" in line or "ERROR" in line:
                    cycle_log += f"  {line.strip()}\n"
                    break
        if cycle_found_solution:
            cycle_log += ">>> SOLUTION FOUND <<<\n"

        attempt_log += cycle_log
        if len(attempt_log) > ATTEMPT_LOG_LIMIT:
            attempt_log = _trim_log(attempt_log)

        duration = time.monotonic() - t0
        status(f"Cycle {cycle}: {duration:.1f}s | log={len(attempt_log)} | skipped={len(skipped_models)}")

        # Resolution check — if FOUND, verify
        if cycle_found_solution:
            success(f"Compute SOLVED after {cycle} cycles!")
            state["final_answer"] = current_best
            state["confidence"] = f"[COMPUTE SOLVED after {cycle} cycles, 5 models]"
            state["attempt_log"] = attempt_log
            cleanup_temp_files()
            return state

    warn(f"Max {MAX_CYCLES} compute cycles reached")
    state["final_answer"] = current_best
    state["confidence"] = f"[compute mode, {MAX_CYCLES} cycles, unresolved]"
    state["attempt_log"] = attempt_log
    cleanup_temp_files()
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  PROVE MODE — Lean 4 formal algebraic proof loop
# ═══════════════════════════════════════════════════════════════════════════════

PROVE_STEP_A = SYSTEM_KNOWLEDGE + """

You are a formal mathematician in a 5-AI research team.
You are writing a FORMAL PROOF in Lean 4. Not a text argument — COMPILABLE Lean 4 code.
The Lean compiler is the ONLY judge. If it compiles without errors, the proof is valid.

PROBLEM TO PROVE:
{problem}

ATTEMPT LOG (previous compilation errors — DO NOT repeat these mistakes):
{attempt_log}

LAST COMPILER OUTPUT:
{last_compiler_output}

Step A — Analyze the compiler errors and propose a FIX:
1. Read each error message carefully. What TYPE of error is it?
   - "unknown identifier" → wrong import or wrong name
   - "type mismatch" → wrong proof term for the goal type
   - "unsolved goals" → proof is incomplete
   - "failed to synthesize" → missing instance/typeclass
2. What specific change to the Lean code would fix this?
3. What Mathlib lemma or tactic might help?

Format:
## ERROR ANALYSIS
[What each error means]

## PROPOSED FIX
[Specific code changes]

## LEAN 4 TACTICS TO USE
[omega, simp, ring, norm_num, decide, exact, apply, etc.]
"""

PROVE_STEP_B = SYSTEM_KNOWLEDGE + """

Write a complete, compilable Lean 4 file that proves the stated theorem.
This will be compiled directly with `lean`. It must type-check with zero errors.

PROBLEM TO PROVE:
{problem}

YOUR FIX PROPOSAL (from Step A):
{hypothesis}

ALL 5 FIX PROPOSALS:
{all_hypotheses}

PREVIOUS COMPILER ERRORS:
{last_compiler_output}

ATTEMPT LOG:
{attempt_log}

RULES:
1. Write a COMPLETE .lean file — including all imports
2. Start with: import Mathlib  (or specific Mathlib imports if you know them)
3. Use `theorem`, `lemma`, or `example` declarations
4. Use tactics: omega, simp, ring, norm_num, decide, linarith, exact, apply, intro, cases, induction
5. Every `sorry` is a FAILURE. Do NOT use sorry.
6. Keep it as simple as possible. Lean compilation can be slow — avoid unnecessary complexity.
7. If the proof is about graph theory, define the graph structure explicitly.
8. Add `#check` statements after your proof to verify it compiled.

Write ONLY the Lean 4 code inside a ```lean block. No explanation outside.
"""


async def _prove_loop(state: AgentState, query: str) -> AgentState:
    """
    Lean 4 formal proof loop.

    NO self-eval. NO formatter. The Lean compiler is the evaluator.
    If it compiles → save to Desktop, print raw, HALT.
    If it errors → feed errors to MoA → loop.
    """
    attempt_log = ""
    last_compiler_output = "No compilation yet — first attempt."
    best_lean_code = ""

    # Initial analysis
    step("Prove mode: 5 AIs analyzing the problem for formal proof")
    await _ensure_online("prove init")
    init_prompt = (
        f"We need to write a formal proof in Lean 4 for this problem. "
        f"Analyze what mathematical structures, theorems, and Lean/Mathlib "
        f"tactics would be needed.\n\n"
        f"PROBLEM: {query}\n\n"
        f"Think about: what are the types? What Mathlib modules are relevant? "
        f"What's the proof strategy (induction, case analysis, decision procedure)?"
    )
    init_results = list(await asyncio.gather(
        *[_call(m, init_prompt) for m in ALL_NVIDIA], return_exceptions=True
    ))
    init_results = [r for r in init_results if isinstance(r, dict) and r.get("answer")]
    if init_results:
        attempt_log = "--- CYCLE 0 (INIT) ---\nProblem analyzed for formal proof.\n"

    # The init analysis becomes the first set of "hypotheses" for cycle 1
    # so cycle 1 skips Step A (no compiler errors to analyze yet) and goes
    # straight to writing Lean code
    first_cycle_hypotheses = init_results if init_results else None

    for cycle in range(1, MAX_CYCLES + 1):
        await _ensure_online(f"prove cycle {cycle}")
        t0 = time.monotonic()
        step(f"═══ Prove Cycle {cycle}/{MAX_CYCLES} ═══")

        # ── Step A: Analyze errors, propose fixes ────────────────────────
        #    SKIP on cycle 1 — no compiler output yet, use init analysis
        if first_cycle_hypotheses is not None:
            hypotheses = first_cycle_hypotheses
            first_cycle_hypotheses = None  # Only used once
            all_hypotheses_text = "\n\n".join(
                f"=== {h['model']} ===\n{h['answer'][:3000]}" for h in hypotheses
            )
            status(f"Cycle 1: using initial analysis as hypotheses ({len(hypotheses)} models)")
        else:
            step(f"Step A: 5 AIs analyzing compiler errors...")
            step_a_tasks = []
            for model in ALL_NVIDIA:
                prompt = PROVE_STEP_A.format(
                    problem=query, attempt_log=attempt_log[-6000:],
                    last_compiler_output=last_compiler_output[-6000:],
                )
                step_a_tasks.append(_call(model, prompt))

            hypotheses = list(await asyncio.gather(*step_a_tasks, return_exceptions=True))
            hypotheses = [h for h in hypotheses if isinstance(h, dict) and h.get("answer")]

            if not hypotheses:
                warn("All models failed Step A — retrying next cycle")
                attempt_log += f"\n--- CYCLE {cycle} ---\nAll models failed Step A\n"
                continue

            all_hypotheses_text = "\n\n".join(
                f"=== {h['model']} ===\n{h['answer'][:3000]}" for h in hypotheses
            )
            status(f"Step A: {len(hypotheses)} fix proposals")

        # ── Step B: Write Lean code ──────────────────────────────────────
        await _ensure_online(f"prove cycle {cycle} step B")
        step(f"Step B: 5 AIs writing Lean 4 code...")

        step_b_tasks = []
        for h in hypotheses:
            prompt = PROVE_STEP_B.format(
                problem=query, hypothesis=h["answer"][:4000],
                all_hypotheses=all_hypotheses_text[:8000],
                last_compiler_output=last_compiler_output[-4000:],
                attempt_log=attempt_log[-4000:],
            )
            step_b_tasks.append(_call(h["model"], prompt))

        code_responses = list(await asyncio.gather(*step_b_tasks, return_exceptions=True))
        code_responses = [c for c in code_responses if isinstance(c, dict) and c.get("answer")]

        if not code_responses:
            warn("All models failed Step B")
            attempt_log += f"\n--- CYCLE {cycle} ---\nAll models failed Step B\n"
            continue

        status(f"Step B: {len(code_responses)} Lean scripts")

        # ── Step C: Compile each .lean file ──────────────────────────────
        step(f"Step C: Compiling {len(code_responses)} Lean files...")
        compilation_results = []
        proof_found = False

        for i, code_resp in enumerate(code_responses):
            model_name = code_resp["model"].split("/")[-1]
            lean_code = extract_lean_code(code_resp["answer"])

            if not lean_code:
                warn(f"{model_name}: no Lean code block found — skipping")
                compilation_results.append(f"{model_name}: NO CODE EXTRACTED")
                continue

            result = await execute_lean(
                code=lean_code,
                timeout=LEAN_TIMEOUT,
                label=f"c{cycle}_{model_name}",
            )

            result_str = format_lean_errors_for_moa(result, lean_code)
            compilation_results.append(f"### {model_name}:\n{result_str}")

            if result["success"]:
                # ═══════════════════════════════════════════════════════════
                #  PROOF ACCEPTED — HARD OUTPUT BYPASS
                #  No self-eval. No formatter. Raw proof. Save and HALT.
                # ═══════════════════════════════════════════════════════════
                success(f"══ LEAN PROOF ACCEPTED BY COMPILER ══")
                success(f"Model: {model_name} | Cycle: {cycle} | Time: {result['duration']:.1f}s")

                # Save to desktop
                saved_path = save_proof_to_desktop(
                    result["lean_file"], lean_code,
                    label=query[:30].replace(" ", "_"),
                )

                # Set state with bypass flags
                state["final_answer"] = lean_code
                state["confidence"] = (
                    f"[LEAN PROOF VERIFIED by compiler — {model_name}, "
                    f"cycle {cycle}, {result['duration']:.1f}s]\n"
                    f"Proof saved to: {saved_path}"
                )
                state["attempt_log"] = attempt_log
                state["bypass_self_eval"] = True   # Signal to main.py
                state["bypass_formatter"] = True   # Signal to main.py
                state["proof_file"] = saved_path
                state["raw_lean_code"] = lean_code

                cleanup_lean_files()
                return state

            best_lean_code = lean_code  # Keep last attempt for reference

        # All compilations failed — update context for next cycle
        last_compiler_output = "\n\n".join(compilation_results)

        # Build cycle log
        cycle_log = f"\n--- CYCLE {cycle} (PROVE) ---\n"
        for cr in compilation_results:
            for line in cr.split("\n"):
                if "ERROR" in line or "ACCEPTED" in line or "TIMEOUT" in line:
                    cycle_log += f"  {line.strip()}\n"
                    break

        attempt_log += cycle_log
        if len(attempt_log) > ATTEMPT_LOG_LIMIT:
            attempt_log = _trim_log(attempt_log)

        duration = time.monotonic() - t0
        status(f"Cycle {cycle}: {duration:.1f}s | log={len(attempt_log)} chars")
        status(f"Cycle {cycle}: proof not yet accepted — continuing...")

    # Max cycles exhausted
    warn(f"Max {MAX_CYCLES} prove cycles reached — no accepted proof")
    state["final_answer"] = (
        f"After {MAX_CYCLES} cycles, no proof was accepted by the Lean compiler.\n\n"
        f"Last attempted Lean code:\n```lean\n{best_lean_code[:8000]}\n```\n\n"
        f"Last compiler output:\n{last_compiler_output[:4000]}"
    )
    state["confidence"] = f"[prove mode, {MAX_CYCLES} cycles, no accepted proof]"
    state["attempt_log"] = attempt_log
    state["bypass_self_eval"] = True   # Still bypass — self-eval can't judge proofs
    state["bypass_formatter"] = True
    cleanup_lean_files()
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def chat_deep_thinking(state: AgentState) -> AgentState:
    """
    Deep thinking v5.1 — auto-detects mode flag.
      !!prove   → Lean 4 formal proof loop (NO self-eval, NO formatter)
      !!compute → Python code-execution research loop
      !!deep / !!conjecture → text debate loop (v4)
    """
    step("Chat — deep thinking v5.1")
    query = state.get("processed_input", state["raw_input"])
    classification = state.get("classification", {})
    domain = classification.get("domain", "general")

    # Detect mode
    prove_mode = "!!prove" in query
    compute_mode = "!!compute" in query
    clean_query = query
    for prefix in ["!!prove", "!!compute", "!!deep", "!!conjecture"]:
        clean_query = clean_query.replace(prefix, "").strip()

    if prove_mode:
        step("PROVE MODE — Lean 4 formal proof loop (self-eval DISABLED)")
        return await _prove_loop(state, clean_query)

    if compute_mode:
        step("COMPUTE MODE — code-execution research loop")
        return await _compute_loop(state, clean_query)

    # Text mode (v4 behavior)
    step("TEXT MODE — iterative debate loop")
    complexity = 10
    ctx_tokens = state.get("context_tokens", 0)
    context = _get_context(state)

    await _ensure_online("initial ensemble")
    assumption = get_assumption_prompt(domain, complexity)
    last_exchange = _get_last_exchange(state)

    answers = await run_ensemble(
        query=clean_query, context=context, domain=domain,
        complexity=complexity, context_tokens=ctx_tokens,
        assumption_prompt=assumption, last_exchange=last_exchange,
    )

    if not answers:
        state["final_answer"] = "Failed to get initial responses."
        return state

    original_prompt = _build_prompt(clean_query, context, assumption, last_exchange, complexity)
    current_best = await _get_best(clean_query, answers)
    status(f"Initial answer: {len(current_best)} chars from {len(answers)} models")

    return await _text_loop(state, clean_query, context, answers, original_prompt, current_best)
