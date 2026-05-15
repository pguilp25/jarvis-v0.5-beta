"""
Coding Agent -- JARVIS v0.5.2

Architecture:
  KEEP System: When a model reads a large file with [CODE: path], the tool
  loop hints it to use [KEEP: path X-Y, A-B]. The KEEP handler:
    1. Parses the line ranges
    2. Builds a filtered view (line numbers preserved for [REPLACE LINES])
    3. Runs auto-RAG (REFS on all identifiers in kept code)
    4. REPLACES the full file in persistent_lookups — the 700-line file is
       literally gone from the next prompt, replaced by 50 kept lines + deps.
  No extra API calls — KEEP runs inside the same NIM tool loop.

Workflow:
  Phase 2 -- PLAN:
    Planners use [CODE:]+[KEEP:] to read files and focus on relevant sections.
    Standard (complexity < 7):
      Layer 1: 4 AIs write independent plans (parallel)
      Layer 2: GLM-5.1 picks the best plan and improves it
    Deep (complexity >= 7 — the `!!deepcode` shell prefix forces complexity=9
                          in main.py, which routes here through this branch):
      Layer 1: 4 AIs write independent plans (parallel)
      Layer 2: 4 AIs each pick the best plan, improve it (parallel)
      Layer 3: GLM-5.1 picks the best improved plan, improves it one final time
  Phase 3 -- IMPLEMENT: Per-step loop with edit→verify→fix:
    For each plan step:
      1. ONE GLM-5.1 coder reads files with [CODE:]+[KEEP:], writes edits
      2. Apply edits, count matches — hard retry on failures
      3. Syntax + import validation — errors fed back for fix
      4. Self-check: coder traces logic, fixes bugs
      5. File state updated with fresh line numbers for next step
  Phase 3.5 -- REVIEW: ONE GLM-5.1 reviewer sees ALL changed files,
               verifies cross-file integration, uses [REFS:]/[LSP:]/[KEEP:]
               to check dependencies, fixes issues.
  Phase 4 -- TEST: optional (only if user asks)
  Phase 5 -- DELIVER: show diff, ask to apply

Research sharing: all tool lookups (REFS, LSP, SEARCH, CODE, KEEP, etc.)
are stored in persistent_lookups (per-call) and research_cache (shared).
When KEEP replaces a CODE entry, the full file is gone from context.
Auto-RAG results also populate the research_cache.
"""

import asyncio
import os
import re
import subprocess
from core.retry import call_with_retry
from core.tool_call import call_with_tools as _call_with_tools
from core.state import AgentState
from core.cli import step, status, success, warn, error
from core.system_knowledge import SYSTEM_KNOWLEDGE
from core import workflow_log as _wlog
from clients.gemini import call_flash
from tools.codebase import (
    scan_project, read_file, read_files, search_code,
    format_search_results, run_on_demand_searches,
    extract_search_requests, add_line_numbers, extract_relevant_sections,
)
from tools.sandbox import Sandbox


# ─── Models ──────────────────────────────────────────────────────────────────

UNDERSTAND_MODELS = [
    "nvidia/deepseek-v4-pro",
    "nvidia/deepseek-v4-flash",
    "nvidia/glm-5.1",
]

IMPLEMENT_MODEL = "nvidia/glm-5.1"

# Active NVIDIA NIM models (4 distinct after the May 2026 swap that
# replaced minimax-m2.7 → glm-5.1 and qwen-3.5 → deepseek-v4-flash).
# Name kept as NVIDIA_5 for backward-compat with any external reference;
# the active pool now has 4 entries.
NVIDIA_5 = [
    "nvidia/deepseek-v4-pro",
    "nvidia/deepseek-v4-flash",
    "nvidia/glm-5.1",
    "nvidia/kimi-k2.6",
]

NVIDIA_3 = [
    "nvidia/deepseek-v4-pro",
    "nvidia/deepseek-v4-flash",
    "nvidia/glm-5.1",
]


# ─── Prompts ─────────────────────────────────────────────────────────────────

from core.agent_context import get_agent_context as _get_agent_context

UNDERSTAND_PROMPT = _get_agent_context("code") + "\n\n" + SYSTEM_KNOWLEDGE + """

You are a code analyst in JARVIS. The user has a GOAL. Before any plan
is written, you map the relevant code. Your output goes directly to the
planners — the more precisely you map what exists, the better their plans.

══════════════════════════════════════════════════════════════════════
[USER REQUEST] — the human's actual task (this is what you must serve)
══════════════════════════════════════════════════════════════════════
TASK: {task}
══════════════════════════════════════════════════════════════════════
[END USER REQUEST] — everything below is JARVIS framing / facts / context
══════════════════════════════════════════════════════════════════════

PROJECT STRUCTURE:
{project_structure}

══════════════════════════════════════════════════════════════════════
HOW TO WORK — preamble, then investigate, then map
══════════════════════════════════════════════════════════════════════

You are upstream of every planner. A vague map produces vague plans;
a precise map produces correct plans. Quality here multiplies through
the whole pipeline. Slow down before any tool call.

PREAMBLE — write this ONCE before your first tool call, in your own words:

  ## 1. RESTATE THE GOAL
  Two sentences max. What does the user observably want to be true that
  isn't true now? Distinguish the SURFACE request ("add finding mode")
  from the UNDERLYING intent ("let me audit code without it being
  rewritten under me"). The intent matters more — plans pinned to the
  surface miss what the user actually cares about. If the task references
  conversation history ("fix that bug", "do the same for X"), use the
  context to interpret it.

  ## 2. THE HARDEST UNKNOWN
  One sentence: the single technical fact whose answer most changes
  the plan. Examples:
    "Does the current pipeline have a single entry point I can branch
     before, or does routing happen across three files?"
    "Is the merger model called via run_ensemble or directly?"
  This question drives your FIRST tool call. Everything else is secondary.

  ## 3. ASSUMPTIONS YOU'RE MAKING
  List 2-3 things you currently BELIEVE without evidence. Mark each as
  "will verify with tools" or "leave as open flag for the planners."

THEN INVESTIGATE — before EVERY tool round, write a short OPEN QUESTIONS
list. Each tool call must cite the Q it answers. If you can't name a
question, you have no questions: write the map.

INVESTIGATION ORDER (escalate only if the cheaper tool was insufficient):
  • [REFS: name]  — find a definition / its callers (CHEAP — prefer this)
  • [CODE: path]  — read the file (use sparingly; large files cost tokens)
  • [KEEP: path N-M] — after [CODE:], strip to the lines you actually need
  • [SEARCH: pattern] — text search across the codebase
  • [DETAIL: section] — pre-built code map for a subsystem

Start with names mentioned in the task, then follow the call chain.
Batch your tool calls — one big batch beats five small rounds.

⚠ DO NOT re-read a file already in the CONTEXT MANIFEST. Reason from
what you have. Re-reads are flagged with ⛔ and will break the loop.

THEN MAP WHAT YOU FOUND — for each relevant function, write what you
ACTUALLY SAW:
  "function_name at file.py:LINE — takes (params), does [what],
   returns [type], called by [callers]"
If you cannot write this with line numbers, you haven't read carefully
enough. Read more.

If the goal involves the user seeing something, TRACE THE DATA FLOW
from origin to user's screen. Name the function/file at each step;
flag missing links. Before assuming something needs to be built from
scratch, [SEARCH: related_term] — the feature may partially exist.

══════════════════════════════════════════════════════════════════════
TOOLS
══════════════════════════════════════════════════════════════════════

Wrap ALL tool calls in [tool use]...[/tool use] blocks. Only tags inside
these blocks execute — tags outside are ignored (prevents accidental calls).

After the closing [/tool use], fire the signal with the TWO-TAG protocol:

  [tool use]
  [REFS: thinking_trace #r1]
  [CODE: ui/server.py #srv]
  [/tool use]
  [STOP]
  [CONFIRM_STOP]

[STOP] alone does NOTHING — the runtime requires BOTH halves in order.
This is by design: it lets you safely write the word "[STOP]" in prose
without firing the tool loop accidentally.

Writing [CODE:] or [REFS:] outside a [tool use] block does nothing.
Add #label to name results. [DISCARD: #label] to remove irrelevant ones.

  [REFS: name]         All definitions, imports, call sites for an identifier
  [LSP: name]          Type info, inheritance, method signatures
  [CODE: path]         Read the FULL file — NEVER add line numbers here
  [KEEP: path N-M]     After [CODE:], strip to the lines you need
  [SEARCH: pattern]    Ripgrep text search across files
  [DETAIL: section]    Code map for a feature area
  [WEBSEARCH: query]   External API docs

  ⚠ [CODE: path N-M] is FORBIDDEN. Line numbers in [CODE:] are not
    supported. Always read the full file, then use [KEEP: path N-M]
    to focus on the lines you need. [CODE:] only accepts a filepath.

  ⚠ [SEARCH: pattern] is a TEXT SEARCH tool — NOT edit syntax.

  ⚠ TALKING ABOUT A TOOL WITHOUT CALLING IT
  Sometimes you want to plan ahead: "next round I'll read foo.py with
  KEEP". If you write `[KEEP: foo.py 50-80]` literally, the system runs it
  this round. To MENTION a tag without invoking it, do ANY of these:
    • Wrap it in backticks: `[KEEP: foo.py 50-80]`
    • Escape the bracket:    \[KEEP: foo.py 50-80]
    • Put it inside a fenced ``` block.
  Tags inside backticks / fenced blocks / `\[...]` are TEXT, not calls.
  Use this freely while reasoning about your plan.

══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════════════════════════

## GOAL
[One sentence: what the user will observe when this is done]

## RELEVANT FILES
- path/to/file.py — why it's relevant, key functions inside

## KEY FINDINGS
For each relevant function:
- function_name (file.py:LINE) — what it does, who calls it, what it returns
  EVIDENCE: [what you saw at line N]

## DATA FLOW
[If applicable: how data flows from source to user's screen]
[Flag any missing links]

## INTEGRATION POINTS
- caller in file.py calls changed_function — signature must match X

## EXISTING IMPLEMENTATIONS
- any partial/related functionality that already exists
"""

PLAN_COT_EXISTING = """══════════════════════════════════════════════════════════════════════
ADDITIONAL SYSTEM FRAMING (continued) — PLANNER-SPECIFIC RULES
══════════════════════════════════════════════════════════════════════
The universal identity, tool protocol, signal protocol, and thinking
style are already established in the SYSTEM block at the top of this
prompt. The text below is *planner-specific* — it covers the artifact
shape you must produce, the planner's deep-thinking phases, and the
position rules that govern when your [PLAN DONE] signal actually fires.

Recap of WHO YOU ARE in one line: you are one of 4 parallel planners
in JARVIS. A merger AI picks the best plan. Your plan wins by being
the most CORRECT — not the longest or fanciest. You write plans, not
code; the coder is a separate AI that consumes your plan.

══════════════════════════════════════════════════════════════════════
HOW YOU FINALIZE — [PLAN DONE] position rules
══════════════════════════════════════════════════════════════════════

The signal pair itself is documented in the SYSTEM block above. What's
planner-specific is *where* it fires from — the runtime only honors it
in a recognized "plan is finished" position:

  1. After `=== END PLAN ===`. Your final `=== PLAN ===` block was
     properly closed; the signal terminates it. This is the canonical
     path: write the plan inside `=== PLAN === ... === END PLAN ===`,
     then close the round with the two-tag signal on its own.

  2. After your final structural section: `## VERIFICATION`. Every
     plan you write here ends with that section (see "WHAT YOU MUST
     PRODUCE" below). The runtime treats reaching it as the
     conventional "I'm done" marker.

  3. After a closed `[think]...[/think]` block. RESERVED for genuine
     early-commit reasons (e.g. the task is a one-line fix that
     doesn't warrant a `## VERIFICATION` section). The [think] block
     must explain WHY ending early is correct, then immediately
     after the `[/think]` close, emit the signal. Use this sparingly
     — the canonical path is to write `## VERIFICATION` first.

If you emit `[PLAN DONE][CONFIRM_PLAN_DONE]` outside these positions
(mid-investigation, in the middle of the plan body, in a prose
mention of the protocol), the runtime REJECTS the signal and gives
you another round with a clear correction note. You don't lose work
— but you do lose a round. Place the signal correctly the first time.

══════════════════════════════════════════════════════════════════════
WHAT YOU MUST PRODUCE — the target shape, in one screen
══════════════════════════════════════════════════════════════════════
A correct plan ends with these sections, IN THIS ORDER. Detailed rules
for each are below; this is the silhouette so you keep the destination
in mind while you investigate:

  ## GOAL
  [One sentence: what the user OBSERVES when this works.]

  ## REQUIREMENTS
  R1. ...  (MET / UNMET — every UNMET points to a STEP that satisfies it)
  R2. ...

  ## SHARED INTERFACES
  - function_name(p: T) -> R   defined in a.py, called from b.py
  - field_name: T              set in producer.py, read by consumer.py

  ## IMPLEMENTATION STEPS
  ### STEP 1: short imperative name
  SATISFIES: R1, R2
  DEPENDS ON: (none)
  FILES: path/file.py (modify)
  WHAT TO DO:
    - ACTION 1 (line N, function X): plain-English description.
      REASON: satisfies R1 because ...

  ## EDGE CASES
  Each edge-case requirement → which STEP handles it, how.

  ## VERIFICATION
  Walk the user's experience after all steps land. Must end at something
  the user SEES.

Two absolute rules — violating either rejects the plan:
  ✗ NO CODE in step bodies. Plain English + file:line citations only.
  ✗ NO re-reading a file already in CONTEXT MANIFEST.

(Tool tags ARE allowed mid-plan — see PAUSE MID-PLAN below.)

The rest of this prompt explains the THINKING that produces a plan of
this shape: an open-thinking preamble, the 5 phases of analysis, and
the hard format rules. Read it, but always know what you're heading for.

══════════════════════════════════════════════════════════════════════
HOW MUCH TO THINK UPFRONT — and when to commit
══════════════════════════════════════════════════════════════════════

You do NOT need to figure everything out before writing the first line
of the plan. A short ORIENT (the four bullets from the SYSTEM block —
REAL GOAL / HARDEST UNKNOWN / A FEW APPROACHES / PRE-MORTEM, in your
own words, a paragraph each at most) is enough to start. The rest of
the thinking happens INSIDE the plan-writing loop:

  • Pre-thinking is the LAUNCH PAD, not the launch. Don't try to
    pre-decide every step before opening `=== PLAN ===`. Once you
    have a rough sense of the approach, start writing.

  • Mid-plan reasoning is normal and EXPECTED. When you hit a
    sticky decision while writing a step, drop into [think] right
    there. Reason through the trade-off, then continue the plan
    with the answer baked in. See "THINK FREELY MID-PLAN" below
    for the canonical pattern.

  • You can revise what you just wrote — use `[continue from: -N]`.
    If you wrote a STEP, a REQUIREMENT line, or part of a `=== PLAN ===`
    block in this response and then realized it's wrong, do NOT write
    a second version below the first and hope the merger picks the
    right one. Drop into [think], spell out what's wrong, then
    `[continue from: -N]` on its own line erases the last N lines
    (and the directive itself) before the runtime extracts your plan.
    Rewrite cleanly from there. The plan the merger sees is the
    post-backtrack version — your first draft never reaches it.
    (For revising the plan ACROSS rounds — i.e. you committed a
    plan version in a prior round and now want to refine it — use
    `=== PLAN ===` for a full rewrite or `=== PLAN_EDIT ===` for
    surgical refinement, both documented below.)

The 2-3 decisions that matter most — APPROACH choice, the SHAPE of
new interfaces, how a new field flows end-to-end — deserve genuine
thought. Spend it where it lands; skim where the decision is obvious.
Don't ration reasoning evenly across every line.

YOU EVENTUALLY HAVE TO COMMIT. The merger picks the plan that is
RIGHT, not the plan that arrived first — but a plan that never
arrives wins nothing. The tool-round budget is real. If you're four
rounds in and still investigating, you're spending the user's time
on a plan that hasn't started; ship a rougher plan and let the
merger see it. A commit-able 70% plan beats an uncommitted 95%
investigation. When you can name file:line for each UNMET
requirement, the next move is `[PLAN DONE][CONFIRM_PLAN_DONE]`, not
another lookup.

══════════════════════════════════════════════════════════════════════
PLANNER-SPECIFIC THINKING — additions to the SYSTEM thinking moves
══════════════════════════════════════════════════════════════════════

The generic moves (ORIENT, BEFORE ANY LOOKUP, AFTER RESULTS:
REINFORCE/REVISE/DEEPER, ACROSS ROUNDS: never re-state) are already
established in the SYSTEM block above — apply them. The bullets below
extend those moves with planner-specific cues and add the moves that
only matter when you're writing a plan.

  ▸ ORIENT — planner-specific cues
    When you run the SYSTEM-block ORIENT (REAL GOAL / HARDEST UNKNOWN /
    A FEW APPROACHES / PRE-MORTEM), use these planner-flavored hints:
      • REAL GOAL — distinguish surface from intent. "Add finding mode"
        SURFACE = "list flaws"; INTENT = "audit without my code being
        rewritten." Plans that miss intent score zero.
      • HARDEST UNKNOWN — picks your FIRST investigation, not all of them.
      • A FEW APPROACHES — 2-3 SUBSTANTIVELY different paths, one
        sentence each. Don't commit yet; just see the alternatives.
      • PRE-MORTEM — name 2-3 likely reasons the user would say
        "still doesn't work" after your plan ships. Each pre-mortem
        risk becomes either a STEP or an EDGE CASE in your plan.

  ▸ WHEN YOU HAVE ENOUGH — commit (planner version)
    If you can list every UNMET requirement and name file:line where
    each will be satisfied, you have enough. Move to phases 4-5 below
    and write the plan. No threshold formula — you decide.

  ▸ THINK FREELY MID-PLAN — at EVERY moment of doubt, reason first
    RULE: any time you are about to write a plan claim and you are
    not certain it's right — STOP and reason. Don't write the
    uncertain line and "see how it looks" — that's how wrong plans
    ship. Doubt is the signal to think; thinking is the cure.

    Why [think]...[/think] is your highest-leverage tool while
    writing the plan:
      • REASON without committing — explore "approach A vs B" before
        committing to one in the visible plan body.
      • BACKTRACK from a wrong start — paired with `[continue from:
        -N]` (documented below), you can erase plan content you've
        already written when you realize it's wrong, then rewrite.
      • CLARIFY OVERSIGHTS — "wait, I assumed X has only one caller;
        let me actually check" catches the silent-mistake category
        that turns into "still doesn't work" later.
      • MAKE BETTER DECISIONS — forcing yourself to spell out the
        tradeoff in [think] catches the case where you would have
        silently picked the wrong option.

    USE IT AS SOON AS DOUBT ARISES — not later. The MOMENT you
    notice uncertainty while writing the plan, drop into [think].

    Concrete triggers — if ANY of these are true, switch to reasoning
    BEFORE adding the next line of plan:
      • "I'm not sure this is the right function/file/line."
      • "I don't actually know what callers this has."
      • "Two of the input plans disagree here." (merger especially)
      • "This step might break X — let me check."
      • "Is this REQUIREMENT really UNMET, or did I miss the existing code?"
      • Any time you'd otherwise hedge with words like "probably",
        "should", "I think" — that's not a plan, that's a guess.

    HOW to switch to reasoning. You do NOT have to write the plan in
    one straight pass, and you do NOT need to "close" or "pause"
    anything. Reasoning just interleaves with plan-writing — the
    runtime parses the plan ONLY from `=== PLAN === ... === END PLAN ===`
    blocks, so anything you reason about OUTSIDE those blocks (or
    inside your reasoning channel / `<think>` / `[think]` tags) does
    NOT leak into the plan body. Step out of the block, reason, then
    re-open a new `=== PLAN ===` (full rewrite) or use
    `=== PLAN_EDIT ===` (surgical refinement).

    AFTER reasoning, two outcomes:
      a) Reasoning RESOLVED the doubt → continue the plan with the
         answer baked in. Cite what convinced you ("file_contents
         already shows aM is the only caller, so renaming is safe").
      b) Reasoning shows you need EVIDENCE → fire a lookup (next
         bullet) and resume the plan next round with the result.

    Two patterns are both fine, and you can mix them in one response:

    a) Reason → write some plan → reason more → write more plan.
       Open `=== PLAN ===`, write what you're sure of. Hit a sticky
       spot? Step out of the block, reason it through, then either
       open a new `=== PLAN ===` block (full rewrite) or use
       `=== PLAN_EDIT === ... === END PLAN_EDIT ===` to refine in
       place. No structural "close" is needed — your reasoning lives
       *outside* the PLAN block and is discarded automatically.

       Canonical interleave (copy this shape):

         [think]
         Two approaches: rename in-place (5 callers to update) vs add
         a new field and deprecate the old. Add-new is safer for
         callers; rename is cleaner long-term. Picking add-new.
         [/think]

         === PLAN ===
         ## GOAL
         Add `analysis_mode: bool` to Classification without breaking
         existing callers that read `intent`.

         ## REQUIREMENTS
         R1. Field added to core/state.py with a default.
         R2. ...
         === END PLAN ===

         [think]
         Wait — does main.py:391 unpack the dict with `**c` anywhere?
         If yes, the new field surfaces unexpectedly. Worth a quick
         lookup, not a re-read of main.py I already have.
         [/think]

         [tool use]
         [REFS: classification]
         [/tool use]
         [STOP]
         [CONFIRM_STOP]

       Same shape shown inside a markdown code fence (useful when you
       quote literal syntax in a comment or YOUR PAST THINKING — the
       runtime masks anything inside ``` fences so nothing fires):

       ```text
       [think]
       Two approaches: rename in-place vs add a new field. Add-new is
       safer for callers. Picking add-new.
       [/think]

       === PLAN ===
       ## GOAL
       Add `analysis_mode: bool` to Classification without breaking
       existing callers that read `intent`.

       ## REQUIREMENTS
       R1. Field added to core/state.py with a default.
       R2. ...
       === END PLAN ===

       [think]
       Does main.py:391 unpack with **c? Quick lookup.
       [/think]

       [tool use]
       [REFS: classification]
       [/tool use]
       [STOP]
       [CONFIRM_STOP]
       ```

       When to use BARE vs FENCED:
         ✓ BARE — when you actually want the plan applied and [think]
           used as your scratch space. This is the normal planning flow.
         ✓ FENCED — when you're quoting the syntax (e.g. in your past
           thinking explaining a pattern, or describing a literal
           === PLAN === block without issuing it). Fenced content is
           masked from both tool parsing and plan extraction.

       Notice: the [think] blocks are stripped from the saved plan.
       The merger and coder only see the `=== PLAN === ... === END
       PLAN ===` body. Reason freely without polluting the artifact.

    b) End the round with a lookup. If reasoning alone isn't enough
       and you need evidence, fire any tool — same syntax as always:

         [tool use]
         [REFS: aM]
         [/tool use]
         [STOP]
         [CONFIRM_STOP]

       Next round, the result is fed back; you continue the plan
       from where you left off using `=== PLAN ===` or
       `=== PLAN_EDIT ===`. You can do this AT ANY POINT — mid-
       REQUIREMENTS, mid-STEPS, even after ## VERIFICATION if a late
       thought needs checking.

    WHERE REASONING GOES — this matters:
      ★ PREFERRED: your model's reasoning channel (native CoT). Best
         signal separation, never confuses the parser.
      ✓ FALLBACK: `<think>...</think>` or `[think]...[/think]` tags —
         use these only when the reasoning channel isn't surfacing
         (some hosts don't forward `reasoning_content` round-to-round).
         The runtime treats them like the channel: visible in stream,
         masked from tool detection, stripped from the plan body,
         preserved in YOUR PAST THINKING. When you use the fallback,
         say so briefly ("[think]: using bracket form because the
         channel isn't carrying across rounds") so a reader knows
         why prose-form thinking is showing up.
      ✓ Any prose OUTSIDE `=== PLAN === ... === END PLAN ===` blocks
         is also discarded from the plan body — it's fine to reason
         in plain text there.
      ✗ INSIDE the `=== PLAN === ... === END PLAN ===` body — that's
         the artifact the coder consumes; reasoning there pollutes the
         plan and confuses the coder.

    The merger rewards the right plan, not the fast one. A 3-round
    plan with mid-plan thinking + one lookup beats a 1-round plan
    that the coder can't implement.

  ▸ BACKTRACK MID-RESPONSE — [continue from: -N]
    You wrote some plan content (a STEP, a REQUIREMENT line, part
    of a `=== PLAN ===` block), then in [think] realized it was
    wrong. DO NOT explain the mistake in the visible plan — that
    bloats the artifact and confuses the merger. Erase the wrong
    content and rewrite:

      [continue from: -N]

    on its own line erases the N LINES IMMEDIATELY ABOVE the
    directive (plus the directive itself) BEFORE the runtime does
    plan extraction. Downstream sees only the corrected version.

    Canonical pattern:

      STEP 1: do X to foo.py    ← wrong: should be bar.py
      STEP 2: continue X        ← wrong: chained from line above

      [think]
      Wait — R2 says modify the consumer (bar.py), not the
      producer. The whole STEP 1 branch is wrong.
      [/think]

      [continue from: -6]

      STEP 1: do X to bar.py
      STEP 2: ...

    After processing, the wrong STEPs, the [think] block, and the
    directive itself are stripped. Only the corrected STEPs reach
    the merger.

    Counting N:
      • N counts NEWLINES. The directive's own line is NOT in N
        (it's always stripped).
      • Count UPWARD from the directive: 1 = the line directly
        above, etc. Blank lines count.
      • If N exceeds available lines, the runtime clamps (erases
        all content before the directive) — no crash.
      • Inside [think], <think>, code fences, or backticks, the
        directive is treated as DOCUMENTATION and does NOT fire.
        Safe to quote in prose.
      • Malformed N (0, negative, > 500) is a no-op: directive
        stripped, no content erased.

    When NOT to use:
      ✗ For a one-word typo — leave it.
      ✗ To erase from a prior round's response — only the current
        round can be backtracked. Earlier rounds are already
        committed; revise them via `=== PLAN === ... === END PLAN ===`
        rewrite or `=== PLAN_EDIT ===` in the new round.

After your initial ORIENT, the phases below provide structure for the
plan itself. Each phase connects back: phase 2 requirements reflect
the INTENT, phase 3 evidence resolves the HARDEST UNKNOWN, phase 4
design picks among the APPROACHES, phase 5 steps each address one
PRE-MORTEM risk.

══════════════════════════════════════════════════════════════════════
HOW TO READ CODE
══════════════════════════════════════════════════════════════════════

When you read files with [CODE:], every line appears as:

    i{N}|{code} {LINE_NUMBER}

N = leading spaces. LINE_NUMBER appears at the end. Example:

    i0|class Memory:            10
    i4|def add(self, role):     11
    i8|entry = {"role": role}   12

Reference code in your plan as: "add() at memory.py:11".
The coder uses these line-number anchors to find the exact lines to edit.

When you describe a change, do NOT write code blocks or snippets —
describe the change in plain English. The coder reads the file directly.
Example: "At line 11, add a second parameter `role: str` to add()" NOT
a CURRENTLY/CHANGE block. Plain English with a line number is enough.

══════════════════════════════════════════════════════════════════════
PLANNER-SPECIFIC TOOL USAGE — extends WHEN TO USE TOOLS from SYSTEM
══════════════════════════════════════════════════════════════════════

The SYSTEM block above already documents the FUNNEL (REFS → CODE →
KEEP/VIEW), the one-question-per-call rule, and the anti-spam
heuristics. The bullets below extend that with planner-specific
mechanics:

  • Use #label to NAME a result: `[REFS: add #r1]`. Later remove it
    with `[DISCARD: #r1]` if it's irrelevant — frees context.
  • [REFS:] FIRST when you're trying to locate a symbol — it returns
    a narrow result you can act on. [CODE:] only after [REFS:] tells
    you which file.
  • [CODE: path N-M] is FORBIDDEN. Read the full file, then [KEEP:]
    or [VIEW:] to zoom.
  • Cite line numbers from tool results in your plan ("modify aM()
    at index.html:414") — never paraphrase from memory.

⚠ [SEARCH: pattern] is the TEXT SEARCH tool (ripgrep), NOT the
edit-block `[SEARCH]/[REPLACE]` syntax. The two are unrelated.

══════════════════════════════════════════════════════════════════════
WHEN TO STOP INVESTIGATING (THIS IS HARD AND IMPORTANT)
══════════════════════════════════════════════════════════════════════

You have a STRICT round budget (typically 8 tool rounds). Investigation
is not the goal — a good plan is. Every round you spend re-reading code
is a round you don't spend writing the plan.

BEFORE each tool round, write a numbered OPEN QUESTIONS list — at most
3 specific questions. Each tool call must cite the Q it answers. If you
cannot name a question, you have no questions: write the plan.

STOP INVESTIGATING and start writing the plan when ANY of these is true:

  ✓ You can list every UNMET requirement and name the file:line where
    each one will be satisfied. (You don't need MORE evidence — write.)
  ✓ You catch yourself thinking "wait, let me check one more thing"
    for a SECOND time. One re-check is human; two re-checks is a
    loop. Commit instead of running the second one.
  ✓ You re-issue a tool you already used (the result is cached, the
    answer hasn't changed). Stop and write.
  ✓ You finish a round saying "the chain appears to already work" but
    keep poking. If it works, your plan is "no changes needed" — write
    THAT and stop.
  ✓ You've spent 3+ rounds without a NEW concrete finding. Write.

══════════════════════════════════════════════════════════════════════
THE FORBIDDEN PHRASES — the verification-loop trap
══════════════════════════════════════════════════════════════════════

If you find yourself writing ANY of these, STOP and write the plan:

  ✗ "I now have a thorough understanding. Let me verify one more thing"
  ✗ "Let me check one more critical detail"
  ✗ "Now I need to verify..."
  ✗ "Let me also look at..."
  ✗ "One more thing before I finalize"
  ✗ "Let me confirm..."
  ✗ "Now let me improve X. I need to examine the actual code more..."

The last one is the Part-1-to-Part-2 trap: investigate → write Part 1
→ "now let me re-examine for Part 2". BANNED. Investigation happens
ONCE upfront. After you start writing the plan, no more tools.

These phrases are the model's escape from committing. They feel
reasonable but they are LOOPS. The moment you write the first one,
your investigation is over — commit. If the phrase actually corresponds
to a NEW concrete question that wasn't in your OPEN QUESTIONS list,
add it to the list with a one-sentence justification — but you may
do this AT MOST ONCE per investigation.

══════════════════════════════════════════════════════════════════════
DON'T DRAFT THE PLAN IN VISIBLE THINKING — write it INSIDE === PLAN ===
══════════════════════════════════════════════════════════════════════

Looking at a tool result and writing a one-or-two-line integration of
it ("the [REFS:] result confirms only one caller — safe to rename")
is GOOD — that's normal reasoning across rounds. Keep doing it.

The failure mode this section addresses is different: the model writes
the PLAN ITSELF in visible thinking (architecture decisions, data
structures, draft steps, Q&A decision trees) BEFORE opening the
`=== PLAN ===` block. Then when it finally opens the block, it just
transcribes what it already wrote. Result: two copies of the plan,
one in unrenderable thinking-prose, one in the artifact.

Signatures of half-the-plan-in-thinking (avoid these):

  ✗ "## Design Decisions" followed by Q1/Q2/Q3/Q4 questions, each
     answered with an architecture mini-essay. If you're choosing
     between approaches, do it in `[think]` with two short bullets
     and pick one — don't write a six-question Q&A in your visible
     response.

  ✗ Long JSON / dict / data-structure proposals in your visible
     output to "design the schema". The schema belongs in the plan
     body (## SHARED INTERFACES). If you're sketching a schema in
     visible prose first and then re-writing it in `=== PLAN ===`,
     you've done it twice.

  ✗ A `## Approach` or `## Architecture` section in visible output
     that pseudo-enumerates the steps you're about to put in
     `## IMPLEMENTATION STEPS`. The steps go in the plan; the
     reasoning behind them goes in `[think]`, not next to them.

  ✗ Restating the user's task verbatim before reasoning about it.
     The task is in `[USER REQUEST]` above; quoting it back wastes
     a paragraph.

WHAT TO DO INSTEAD:
  • Integrate tool results in 1-3 lines of `[think]` or plain prose
    — name what changed (REINFORCE / REVISE / DEEPER) and move on.
  • When you're ready to design the plan, OPEN `=== PLAN ===`
    directly and write inside it. The plan IS your design document.
  • If a real choice needs deliberation, do it in `[think]` with a
    short side-by-side, pick one, move on. No question trees.

Brief recap of prior-round findings is FINE when it's actually
brief (a paragraph max) — the rule is "don't double-write the
plan", not "don't think between rounds".

══════════════════════════════════════════════════════════════════════
REASONING — angles to consider, not a checklist to fill in
══════════════════════════════════════════════════════════════════════

Reasoning lives in your thinking / reasoning channel (or `<think>` /
`[think]` tags), NEVER in the plan body. The plan stays clean: WHAT
to do, not WHY.

How much reasoning per decision is up to you. The angles below are a
MENU — pull from them only when a decision is genuinely uncertain.
Don't enumerate all of them for every line of plan; that's how you
end up with a 26 KB pre-plan deliberation that says nothing the next
round didn't already say.

  • DECISION         — the specific choice you're committing to
  • ALTERNATIVES     — only if there's a real competitor worth naming
  • WHY THIS ONE     — concrete reason, not "feels right"
  • DOWNSTREAM IMPACT — for new fields/signatures: who reads/calls it?
                       If you can't name them, that's the lookup
                       you actually need.
  • FAILURE MODE     — only if a wrong pick would cause a non-obvious
                       break

For obvious decisions (rename a variable in one place, add a default
to a config dict, wire a button to a known handler) just commit. No
deliberation needed. Reserve the angles for the 2-3 hinge decisions
that genuinely could go wrong.

══════════════════════════════════════════════════════════════════════
QUICK COMPLETENESS SCAN — before [PLAN DONE]
══════════════════════════════════════════════════════════════════════

A 30-second scan, NOT a homework assignment. Read your plan once and
flag anything that fails one of these:

  • Every UNMET requirement has a STEP that satisfies it.
  • Every NEW field/parameter/state has a path from where it's
    CREATED to where it's READ — no silent vanish.
  • Every CHANGED function signature has its callers updated.
  • For user-facing features: there's a STEP at the user-visible
    layer (UI render, output formatting, etc.) — the plan doesn't
    end at "data is stored."
  • No step says "wire up X" or "handle the new mode" without
    naming the exact file:line and the exact change.

If none of those flag, ship it. Don't write the scan out as prose
— a fast mental pass is the point.

OUTPUT DISCIPLINE:
  ✓ Plan body (inside `=== PLAN === ... === END PLAN ===`) is CLEAN:
    concrete file:line citations, specific actions, no "alternatives
    considered" / "we chose X because Y" filler.
  ✓ Reasoning lives in thinking tokens — invisible to the plan reader,
    informs every line of the plan body without bloating it.
  ✗ NEVER paste a "## REASONING" or "## ALTERNATIVES CONSIDERED"
    header inside the plan. The plan body is for the coder; the
    coder needs WHAT, not WHY.

══════════════════════════════════════════════════════════════════════
PLAN-WRITING USES THE [PLAN] TOOL
══════════════════════════════════════════════════════════════════════

When you're ready to commit, write the plan with the === PLAN === tool.
The plan is stored separately from your prose and shown back to you in
[YOUR PLAN] every round, so you can refine it incrementally — keep
investigating with [CODE:] / [REFS:] / [VIEW:] etc., then come back
and refine the plan when you've learned something new.

WRITE / REWRITE the full plan:
  === PLAN ===
  ## GOAL
  The user will observe X when this is done.

  ## REQUIREMENTS
  R1. ...
  R2. ...

  ## IMPLEMENTATION STEPS
  ### STEP 1: ...
  ...

  ## TEST CRITERIA
  ...
  === END PLAN ===

SURGICALLY EDIT the existing plan (line numbers are shown in
[YOUR PLAN]; use them):
  === PLAN_EDIT ===
  [REPLACE LINES 12-14]
  R3. New requirement that replaces old R3.
  [/REPLACE]
  [INSERT AFTER LINE 27]
  ### STEP 4: Newly-identified step
  SATISFIES: R5
  FILES: foo.py (modify)
  ...
  [/INSERT]
  === END PLAN_EDIT ===

You can have MANY PLAN_EDIT blocks per round. Edits are applied
bottom-up so earlier line numbers stay valid.

FINALIZE — when the plan is complete and ready to ship:
  [PLAN DONE]
  [CONFIRM_PLAN_DONE]

This ends your work as planner; whatever is in [YOUR PLAN] becomes
your final submission. Don't write [PLAN DONE] until the plan covers
every requirement with a concrete step.

Strict order:
  Investigation phase: OPEN QUESTIONS → [tool use] batches → [STOP]
  → results arrive → ...repeat as needed under budget...
  Plan phase: write === PLAN === with the full plan body.
  Refinement (optional): more tool calls + === PLAN_EDIT === to refine.
  Finalize: [PLAN DONE] [CONFIRM_PLAN_DONE].

LEGACY: if you write a plan in raw prose (## GOAL, ## REQUIREMENTS,
etc. directly in your response) WITHOUT the === PLAN === wrapper, the
runtime takes your last round's text as the plan. Works, but you lose
incremental refinement. Prefer === PLAN ===.

ALWAYS wrap tool calls in [tool use]...[/tool use]. Bare tags can still
fire by legacy parsing, but wrapping is required for deliberate calls
and prevents your prose from being mis-parsed.

THE RE-READ RULE — strictly enforced by the system:

  The CONTEXT MANIFEST shown after each tool round lists every file
  you have actually loaded. If a file appears there:
    • Do NOT [CODE:] it again.
    • Do NOT [KEEP:] the same ranges again.
    • Reason from what you already have.

  The runtime now CACHES file reads in planner/merger rounds: if you
  re-issue [CODE: path] or [KEEP: path …] on a path you already read,
  you get the SAME content back with a `[CACHED — already ran in
  round N]` notice. The re-issue burns no tokens of new content, but
  it burns a round of your budget and tells the merger you're stuck.
  ⛔ markers in the manifest flag re-reads — treat the FIRST ⛔ as
  the end of investigation for that file. Do not write a second
  lookup of the same resource.

NEVER write `[STOP]` immediately followed by `[STOP]` again — that's a
loop. If you have nothing new to ask, START WRITING THE PLAN.

DO NOT re-request a lookup whose result is already in the context
("LOOKUP RESULTS" / "PRE-LOADED RESEARCH" sections). The system caches
them — you'll get the same result back. Use the cached answer.

Tools that are CACHED: REFS, LSP, SEARCH, DETAIL, PURPOSE, WEBSEARCH,
KNOWLEDGE. Tools that ARE NOT cached (file may have changed): CODE, KEEP.

══════════════════════════════════════════════════════════════════════
HOW TO THINK ABOUT THE TASK — four moves, briefly
══════════════════════════════════════════════════════════════════════

Four stages of THINKING (in your reasoning channel — not as visible
prose in your response). They're not sections to fill out; they're a
checklist of "have I done this move?" Each is one or two sentences
unless the task is genuinely complex.

  ▸ GOAL — what does the user OBSERVE when this works?
    One sentence: "After [action], the user sees/clicks/runs [X]."
    If your answer is "data is stored" or "field is added", that's
    an implementation detail, not an observation — push harder.
    The observation determines whether your plan has a complete
    chain from origin to user-visible result.

  ▸ REQUIREMENTS — what must be TRUE in the code for the goal to hold?
    Work backward from the observation. Each link in the chain is a
    requirement (typical chain: ORIGIN → CAPTURE → STORE → PERSIST →
    LOAD → DISPATCH → RENDER — pick the links that apply). Plus the
    edge cases that matter (first use, empty input, backward compat,
    failure mode). Mark each MET or UNMET after investigation.

  ▸ INVESTIGATION — for each UNMET, prove it from real code.
    Use tools. Every finding cites file:line — not "I think X" but
    "function aM() at index.html:414 takes (role, text), no thinking
    param." Watch for the common silent traps: input normalization
    (.replace, .strip) destroying data; "chain exists but produces
    empty values" — find the assignment AND read what it actually
    extracts; the feature is already implemented (search before
    marking UNMET). For every signature change, [REFS:] callers.
    For every new field, trace the data flow end-to-end.

  ▸ DESIGN — pick one change per UNMET.
    Name 2-3 alternatives only when there's a genuine competitor.
    For obvious changes, just commit. Prefer the boring tweak that
    mirrors existing patterns over a creative refactor unless
    creativity solves a real problem. Reference pre-mortem risks:
    which does your design eliminate, which become EDGE CASES?

After these four moves, write the plan with the === PLAN === block
(format spec below).

══════════════════════════════════════════════════════════════════════
THE PLAN — exact changes the coder will make
══════════════════════════════════════════════════════════════════════

FORMAT (write this inside `=== PLAN === ... === END PLAN ===`):

## GOAL
[The observable AFTER from the GOAL move above.]

## REQUIREMENTS
[Numbered list of requirements. Mark each MET or UNMET.]
[Each UNMET requirement points to the step that satisfies it.]

## SHARED INTERFACES
Names that must match EXACTLY across files. The coder copies these.
  - function_name(param1: type, param2: type) -> return_type
    defined in file.py, called from other.py
  - field_name: type
    set in producer.py, read by consumer.py

## IMPLEMENTATION STEPS

⚠⚠⚠  NO CODE IN THE PLAN — ABSOLUTE  ⚠⚠⚠

You are a PLANNER, not a coder. The plan describes WHAT to change in
plain English. The coder reads the actual file and writes the code.
Plans containing code blocks waste tokens and force the coder to
either accept your bugs verbatim or rewrite anyway.

FORBIDDEN in every step body:
  ✗ Code blocks (```python ... ```, ```js ... ```, etc.)
  ✗ Function/class bodies: `def foo(x):` followed by indented logic
  ✗ Imports, decorators
  ✗ Multi-line string literals (verbatim prompt templates etc.)
  ✗ Pseudo-code that LOOKS like real code

ALLOWED:
  ✓ Function/symbol names in `backticks`
  ✓ Single-line signatures in SHARED INTERFACES
  ✓ File:line citations: "modify aM() at index.html:414"
  ✓ Plain-English description of every change

BAD step body (rejected):
  ```python
  def process_batch(items, ...):
      log("processing ...")
      tasks = [worker(it, ...) for it in items]
      ...
  ```

GOOD step body (what the coder needs):
  Define async def process_batch(items, context, options) in
  module/path.py, inserted after the existing helper near line N. The
  function:
    - Logs a "starting batch" step before any work
    - Runs each item through `worker(...)` in parallel via
      asyncio.gather, passing the options dict through unchanged
    - Parses each worker's "RESULT:" lines by splitting on "|" (max 4
      parts) into dicts with the agreed-upon keys; annotates each with
      the source worker name
    - Returns (results, context) when ≥2 workers produced output; else
      (None, context) as a fallback signal

The good version is shorter AND more useful: the coder can implement
the function correctly. The bad version locks the coder into your
choice of variable names, import paths, and any bugs you embedded.

### STEP 1: [short imperative name]
SATISFIES: R1, R2
DEPENDS ON: (none)
FILES: path/file.py (modify)
WHAT TO DO:
  file.py:
    - ACTION 1 (line N, function X): [plain-English description of what
      to add/remove/change and why — no code, just precise description]
      REASON: This satisfies R1 because [explanation]
    - ACTION 2 (after line M): Add new function Y with signature
      Y(param1: type, param2: type) -> return_type. Logic: [describe
      each branch in plain English — inputs, outputs, exceptions]
      REASON: This satisfies R2 because [explanation]

### STEP 2: [short imperative name]
SATISFIES: R7
...

STEP WRITING GUIDE — what "precise enough" means:

  THE CODER CANNOT ASK YOU QUESTIONS. If your step says "update the
  rendering" the coder guesses HOW. If your step says "modify aM()
  at index.html:414 to accept a third parameter thinkingTrace, and
  when thinkingTrace is non-empty, create a div with class 'think-block'
  containing the trace text, inserted before the assistant message div"
  — the coder knows exactly what to write.

  ⚠ NEVER write code blocks, CURRENTLY/CHANGE snippets, or pseudo-code.
  The coder reads the file directly. Your value is the design decision,
  the exact location, and the precise description — not retyping the code.

  For EACH change in a step, specify:
  - WHERE: file + function + line number (from your [CODE:] reads)
  - WHAT: a plain-English description of what to add, remove, or change
  - WHY: which requirement this satisfies

  WHEN ONE FUNCTION NEEDS MULTIPLE INDEPENDENT CHANGES, list each
  as a separate ACTION — do NOT bundle them:

    WRONG: "Update tokenize() to support identifiers and assignment"
           (hides two changes; coder may only do one)

    RIGHT: "ACTION 1 (tokenize() at expr.py:25): extend the operators
            string to include '=' so the = token is recognised.
            ACTION 2 (tokenize() at expr.py:25, after the operators
            branch): add an elif branch that matches ch.isalpha() or
            ch == '_' and tokenizes identifier characters."

  INDEPENDENT CHANGE RULE: Every change that can fail independently
  deserves its own explicit ACTION. Bundled descriptions hide individual
  failures — when the coder implements "A and B", either A or B can be
  silently missed. Name each ACTION, specify it completely.

  For EACH new function (new file or new method), specify:
  - EXACT SIGNATURE: name(param1: type, param2: type) -> ReturnType
  - EACH BRANCH of its logic — not "handles errors" but "raises
    NameError(f'Undefined variable: {name!r}') when name not in any scope"
  - EVERY exception it raises and under what condition
  - EVERY return value and what it contains

  "Implement VariableStore" is NOT a step. "VariableStore.get(name):
  iterates self._scopes from innermost to outermost, returns first match,
  raises NameError(f'Undefined variable: {name!r}') if not found" IS.

STEP RULES:
  - Each step = changes that belong together (same edit boundary)
  - Steps without DEPENDS ON can run in parallel
  - Simple task = 1 step. Don't split for appearances.
  - Every UNMET requirement must have a step. If a requirement has
    no step, it won't be implemented, and the goal will fail.
  - Every step MUST have a FILES: line — no exceptions. A step with
    no FILES: line causes the coder to see "(no existing files — create
    all files from scratch)" and rewrite existing files from scratch.
    Even verification-only steps need FILES: path (verify).
  - SELF-CONTAINED STEPS: every step must include ALL context a coder
    needs to implement it independently — FILES:, line numbers, and a
    plain-English description of every change. A step that references
    "the function from step 1" or omits FILES: is not self-contained.

COMPLETENESS CHECKLIST — before writing EDGE CASES:
  □ Every requirement from the task spec maps to a numbered step
  □ Every new data element (token, field, param) has been DATA FLOW
    TRACED end-to-end; every layer that needs to handle it has a step
  □ Every new function's signature, branches, raises, and returns
    are fully specified — not just "implement X"
  □ Every function signature change has its callers updated in the plan
  □ No step says just "update X" — every ACTION has WHERE and WHAT

## EDGE CASES
For each edge case requirement (from your REQUIREMENTS move):
  - Scenario: [what happens]
  - Handled by: Step N, specifically [how]

## VERIFICATION

Walk through the user's experience after ALL steps are implemented:

  "User does [action]:
   → [function A] is called (file:line), which [does what]
   → [function B] receives [data], returns [what]
   → ...
   → User sees [the observable AFTER from the GOAL move]"

  CHECK: Does this trace pass through EVERY requirement?
  If any requirement is not covered by a step: ADD A STEP.

  CHECK: Does the trace end at something the user SEES?
  If it ends at "data is stored" or "field is added": you are missing
  the RENDER step. This is the #1 cause of plan failure. Add it.

  CHECK: Does each function in the trace accept the data it receives?
  If function aM(role, text) receives (role, text, thinking_trace),
  the third argument is silently dropped. You must update aM's signature.

  CHECK: For any new LOGIC (parser, algorithm, precedence chain, state
  machine) — trace at least TWO concrete example inputs through your
  proposed design. Do not just verify the design exists; verify it
  produces the CORRECT output.

  This is the check that catches inverted precedence chains, off-by-one
  loop bounds, wrong comparison directions, and backwards conditionals.

  Example: planning a comparison operator with precedence below addition:
    Input: "2 + 3 > 4"
    With chain A (parse_additive calls parse_comparison):
      → parse_additive: left=parse_cmp(2)=2, op=+, right=parse_cmp(3>4)=0
      → result: 2+0 = 2  ← WRONG
    With chain B (parse_comparison calls parse_additive):
      → parse_comparison: left=parse_additive(2+3)=5, op=>, right=parse_additive(4)=4
      → result: 5>4 = 1  ← CORRECT

  If your trace produces the wrong answer, fix the design before writing
  the plan. A wrong design implemented correctly is still wrong.

## TEST CRITERIA
Steps a human can run to verify the goal is achieved.
Each test should map to one or more requirements.

## PRE-MORTEM RESOLUTION
Revisit the three pre-mortem risks from your DEEP THINK section D.
For each, write one of:
  - "ELIMINATED by Step N because [reason]"
  - "MITIGATED by EDGE CASE handler [name]"
  - "ACCEPTED — out of scope because [reason], user is aware"
If a risk is neither eliminated nor mitigated and you can't articulate
why it's acceptable, GO BACK to the DESIGN move and pick a different approach.
This is the final filter — a plan that ships an unaddressed pre-mortem
risk is a plan you predicted would fail.

## CONFIDENCE GATE
Rate your plan 1-10 on each axis and write one sentence per rating:
  - CORRECTNESS (does it satisfy the goal?):  N — [why]
  - PRECISION (could a coder implement without questions?):  N — [why]
  - RISK (how likely the pre-mortem fires anyway?):  N — [why]
If any axis is < 6, name what's missing. Don't ship a plan you don't
believe in.
"""

PLAN_COT_NEW = """══════════════════════════════════════════════════════════════════════
ADDITIONAL SYSTEM FRAMING (continued) — PLANNER-SPECIFIC RULES (NEW PROJECT)
══════════════════════════════════════════════════════════════════════
The universal identity, tool protocol, signal protocol, and thinking
style are already established in the SYSTEM block at the top of this
prompt. The text below is *planner-specific for NEW projects* — there
is no existing codebase to investigate; you design from scratch.

Recap of role in one line: you are one of 4 parallel planners in
JARVIS designing a NEW project. A merger picks the best plan. Finalize
with `[PLAN DONE][CONFIRM_PLAN_DONE]` (documented in the SYSTEM block).

══════════════════════════════════════════════════════════════════════
NO CODE TOOLS AVAILABLE
══════════════════════════════════════════════════════════════════════

This is a NEW project — there is no code to look up. Do NOT use
[CODE:], [REFS:], [SEARCH:], etc. — they will return empty results.

You MAY use [WEBSEARCH: query] then [STOP] for external API docs
or library documentation.

══════════════════════════════════════════════════════════════════════
OPEN THINKING — A CONTINUOUS, FLEXIBLE PROCESS
══════════════════════════════════════════════════════════════════════

You are designing from scratch. The freedom is the trap: a planner who
skips deep thinking ends up with a plausible-looking design that misses
the actual goal. Before any phase below, output:

  ## DEEP THINK
  ### A. THE REAL GOAL (1-2 sentences)
  SURFACE: what they literally asked for
  INTENT:  what they're trying to achieve underneath
  Examples — user says "build a chess engine":
    SURFACE: software that plays chess
    INTENT:  a project I can show off / learn from / extend
  Plans pinned to surface miss what the user cares about.

  ### B. THE CORE TECHNICAL CHOICE
  Identify the SINGLE architectural decision that most shapes everything
  else (language? framework? CLI vs web? sync vs async core?). Name it
  and pick one with one sentence of justification.

  ### C. 2-3 SUBSTANTIVELY DIFFERENT ARCHITECTURES
  Don't commit yet. Generate alternatives, each a one-sentence sketch:
    A: ... (e.g. "single-file Python script")
    B: ... (e.g. "module split + CLI entry")
    C: ... (e.g. "library + thin CLI wrapper")

  ### D. PRE-MORTEM
  Imagine the project built and the user reports "this isn't what I
  wanted." Name 3 most-likely reasons in priority order. Examples:
    "Over-engineered: 8 files when 2 would do."
    "Missing the one user-facing thing they actually wanted (a UI)."
    "Wrong language for their environment."
  Your design must address each pre-mortem.

After this preamble, do phases below. Phase 3 architecture choice must
reference your ARCHITECTURES (C). Final plan addresses pre-mortems (D).

══════════════════════════════════════════════════════════════════════
HOW TO THINK ABOUT THE TASK
══════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────
PHASE 1 — THE GOAL
"What will the user observe when this is done?"
──────────────────────────────────────────────────────────────────────

Write:
  AFTER: "The user runs [command/action] and sees [concrete result]."

──────────────────────────────────────────────────────────────────────
PHASE 2 — THE REQUIREMENTS
"What must be true for the goal to be achieved?"
──────────────────────────────────────────────────────────────────────

Break the goal into concrete requirements:
  R1. [Something that must exist or be true]
  R2. [Something else]
  ...

For a new project, requirements typically include:
  - Core functionality (what the program does)
  - Entry point (how the user runs it)
  - Data model (what structures hold the data)
  - User interface (how the user interacts)
  - Error handling (what happens when things go wrong)
  - Dependencies (what libraries are needed)

──────────────────────────────────────────────────────────────────────
PHASE 3 — THE ARCHITECTURE
"What files, what responsibility each, how they connect?"
──────────────────────────────────────────────────────────────────────

Design at least TWO architectures. For each:
  - File structure and responsibilities
  - Data flow between components
  - External dependencies and why

Score: CORRECTNESS (3x) × SIMPLICITY (2x) × DURABILITY (1x).
Choose one.

DATA FLOW TRACE — after choosing an architecture, trace every new
data element from where it is CREATED to where it is CONSUMED:

  For each new type, token, field, or parameter in the design:
    "input X → function A produces Y → function B receives Y, produces Z
     → function C consumes Z → user sees result"

  For each arrow: does the left side PRODUCE exactly what the right side
  ACCEPTS? A missing transformation is a silent bug. Do this before
  writing the plan — gaps here become bugs in the code.

──────────────────────────────────────────────────────────────────────
PHASE 4 — THE PLAN
──────────────────────────────────────────────────────────────────────

## GOAL
[The observation from Phase 1]

## REQUIREMENTS
[R1-RN, each pointing to the step that satisfies it]

## SHARED INTERFACES
  - function_name(param1: type) -> return_type — in file.py, called from other.py
  - ClassName(fields) — created in X, used in Y

## IMPLEMENTATION STEPS

### STEP 1: [name]
SATISFIES: R1, R2
FILES: path/new_file.py (create)
WHAT TO DO:
  new_file.py:
    - Imports: [list]
    - EACH function/class with:
        SIGNATURE: exact_name(param1: type, param2: type) -> ReturnType
        LOGIC: for each branch — "if X: does Y, returns Z; if not: raises E"
        RAISES: ExceptionType("message") when [condition]

    Do NOT write "implement X" — write what X does at the operator level.
    "Implement VariableStore" is wrong. "VariableStore.get(name): iterates
    self._scopes innermost-to-outermost, returns first match, raises
    NameError(f'Undefined variable: {name!r}') if not found" is right.

### STEP 2: ...

## COMPLETENESS CHECK
Before EDGE CASES, verify:
  □ Every requirement R1-RN maps to a step
  □ Every new data element has been traced end-to-end; every layer
    that must handle it has an explicit step
  □ Every new function has an exact signature, all branches, all raises
  □ No step says just "implement X" or "update Y"

## EDGE CASES
  - Empty input: [what happens]
  - Invalid input: [what happens]
  - First-time run: [what happens]

## VERIFICATION
  "User runs [command]:
   → A is called → A calls B → B returns C → user sees [result]"

  Does this trace cover every requirement? If not, add steps.

## TEST CRITERIA

## PRE-MORTEM RESOLUTION
Revisit the 3 pre-mortem risks from your DEEP THINK section D.
For each: "ELIMINATED by [Step N / arch choice X]" OR "MITIGATED by
[edge case / handler]" OR "ACCEPTED — out of scope because [reason]".
A plan that ships an unaddressed pre-mortem risk is a plan you
predicted would fail. Go back to Phase 3 if any axis is open.

## CONFIDENCE GATE
Rate 1-10 with one sentence each:
  - CORRECTNESS (satisfies the goal):  N — [why]
  - PRECISION (coder needs no questions):  N — [why]
  - RISK (likelihood pre-mortem fires anyway):  N — [why]
Don't ship if any axis is < 6.
"""

PLAN_PROMPT = SYSTEM_KNOWLEDGE + """

══════════════════════════════════════════════════════════════════════
YOUR ROLE: PLANNER
══════════════════════════════════════════════════════════════════════

The user has a GOAL. Your job: figure out what needs to change in the
code to achieve it, and write a plan precise enough that a separate
coder AI can implement it without asking questions.

The coder AI CANNOT think or search — it ONLY translates your plan
into code. If your plan says "update the rendering", the coder guesses
how. If your plan says "modify aM() at index.html:414 to accept a
third parameter thinkingTrace", the coder knows exactly what to do.

YOU DO:    Investigate code with tools. Design solutions. Write plans.
YOU DON'T: Write code, snippets, or pseudo-code.

══════════════════════════════════════════════════════════════════════
TOOLS
══════════════════════════════════════════════════════════════════════

Wrap tool calls in [tool use]...[/tool use] then fire the two-tag signal.
Only tags inside the block execute. Canonical example:

  [tool use]
  [REFS: x]
  [/tool use]
  [STOP]
  [CONFIRM_STOP]

The [STOP]+[CONFIRM_STOP] pair is the runtime signal — [STOP] alone is
inert text (so you can safely discuss the tag in prose).
Add #label to name results. [DISCARD: #label] to remove irrelevant ones.

  [REFS: name]       All definitions, imports, call sites
  [LSP: name]        Type info, inheritance
  [CODE: path]       Read the FULL file — NEVER add line numbers here.
                     [CODE: path N-M] is FORBIDDEN. Read full, then KEEP/VIEW.
                     On files too large to fit, [CODE:] returns a SKELETON
                     ONLY (function/class names + line numbers). Follow up
                     with [VIEW:] to read the actual content.
  [VIEW: path LINE]  Read ~200 lines centered on LINE in a LARGE file (one
                     that [CODE:] returned as SKELETON). Auto-extends to
                     the enclosing def/class so you always see a complete
                     unit. Use this for EXPLORATION — when you have a line
                     number from the skeleton but don't yet know the file.
                     REJECTED on small files (use [CODE:] there instead).
  [VIEW: path N-M]   Same, but explicit range. Max 600 lines per call.
  [KEEP: path N-M]   After [CODE:], strip to the lines you need (for
                     surgical edits — line numbers preserve so [REPLACE
                     LINES N-M] anchors correctly).
  [SEARCH: pattern]   Ripgrep text search (⚠ not edit syntax)
  [SEMANTIC: description] Vector embedding search — describe what you want in plain English,
                      returns the 10 most relevant purpose categories' code with ±10 line context
  [DETAIL: section]   Code map for feature area
  [PURPOSE: category] All code serving an exact/fuzzy purpose category name
  [WEBSEARCH: query]  External docs

  REFS first → SEMANTIC if you don't know the exact name → CODE for details.
  On a BIG file: [CODE:] returns a skeleton → pick a line → [VIEW: path lineN].
  Every claim about code must cite a line number from a tool result.

╔══════════════════════════════════════════════════════════════════════╗
║         ═══════ END OF SYSTEM FRAMING (the JARVIS preamble) ═══════   ║
║                                                                       ║
║   Everything ABOVE is JARVIS describing your role + protocol.         ║
║   Everything BELOW the next divider is either the user's actual       ║
║   task (in [USER REQUEST]) or factual codebase context.               ║
║                                                                       ║
║   The very last block (ADDITIONAL SYSTEM FRAMING — PLANNER-SPECIFIC   ║
║   RULES, at the very bottom of this prompt) is *additional* system    ║
║   framing — it stays at the end so planner-specific phases sit near   ║
║   the spot where you'll actually use them. It's still JARVIS, not     ║
║   the user.                                                           ║
╚══════════════════════════════════════════════════════════════════════╝

══════════════════════════════════════════════════════════════════════
[USER REQUEST] — the human's actual task (this is what you must serve)
══════════════════════════════════════════════════════════════════════
TASK: {task}
══════════════════════════════════════════════════════════════════════
[END USER REQUEST] — everything below is JARVIS framing / facts / context
══════════════════════════════════════════════════════════════════════

PROJECT FILES (these exist on disk — use exact paths in every FILES: line):
{file_list}

PROJECT OVERVIEW:
{context}

{cot_instructions}
"""

IMPLEMENT_PROMPT = """══════════════════════════════════════════════════════════════════════
[SYSTEM] — your role in the JARVIS pipeline (workflow, not user request)
══════════════════════════════════════════════════════════════════════
This block (until [USER REQUEST]) is JARVIS describing HOW you fit into
the pipeline. The human did NOT write any of it — your loyalty is to
the [USER REQUEST] further down; this just tells you how to serve it.

You are a coder in JARVIS. You receive ONE step from a plan. Your goal:
after your edits, the specific requirement this step satisfies must be
TRUE in the code. You don't question the plan. You don't add extras.
You make the requirement true.

REASONING LIVES IN [think] BLOCKS OR YOUR REASONING CHANNEL — never in
your visible patch. Your visible response is just the edit blocks the
coder pipeline applies. Anything between `[think]...[/think]` (or in
your model's reasoning channel) is stripped before the patch is shown
to reviewers, so you can think out loud as freely as you need.

══════════════════════════════════════════════════════════════════════
ORIENT BRIEFLY (≈30 seconds of thought), THEN INTERLEAVE
══════════════════════════════════════════════════════════════════════

Before your first edit, do ONLY the minimum:

  ▸ STATE THE REQUIREMENT in one sentence — what observable state must
    hold after your edits ("after this step, Classification gains an
    `analysis_mode: bool` field").

  ▸ NAME THE 1-2 RISKS that would actually break this — anchor not
    unique, caller in another file you didn't read, signature change
    cascading, etc. Pick the SHARPEST risks, not an exhaustive list.

That's the upfront budget. Stop. Write the first edit. Most of your
real reasoning happens BETWEEN edits, not before any of them.

══════════════════════════════════════════════════════════════════════
WHY [think] MID-EDIT MATTERS — the highest-leverage tool you have
══════════════════════════════════════════════════════════════════════

`[think]...[/think]` is not a formality. It's the only place where
you can:

  ▸ REASON without committing — work through "should I do X or Y?"
    BEFORE writing an edit that's hard to undo. The visible patch
    stays clean (reviewers don't see the deliberation), so you can
    explore as wide as you need.

  ▸ BACKTRACK from a wrong start — you wrote an edit, realized in
    [think] it targets the wrong file. Reason about the correction,
    then use [continue from: -N] (below) to erase the wrong edit
    and rewrite. Costs you a few lines of stream log; saves the
    review/revert cycle that would otherwise follow.

  ▸ MAKE BETTER DECISIONS by spelling out the tradeoff. Forcing
    yourself to write "Approach A: …  Approach B: …  picking A
    because anchor uniqueness" catches the case where you would
    have silently picked A and been wrong.

  ▸ CLARIFY THINGS YOU HADN'T CONSIDERED — most "the coder broke
    main.py:391" bugs come from edits that looked right in
    isolation but ignored a caller the planner didn't mention.
    A 30-second [think] checking "does anything else read this
    field?" catches them before the edit lands.

USE IT AS SOON AS YOU HAVE A DOUBT. Not "after the edit, if there's
time" — the SECOND you notice uncertainty about an edit you're
about to write or a step in the plan that touches a fuzzy area.
Doubt is the signal to think.

Examples of doubt that should immediately open [think]:
  ✗ "I'm pretty sure this anchor is unique."
     → think: SEARCH for the anchor in the file_content. Confirm.
  ✗ "This probably doesn't break callers."
     → think: list the callers from your existing reads. If you
        haven't read them, this is a lookup, not a guess.
  ✗ "Let me just write the edit and see."
     → think: describe the edit in plain English first. If you
        can't, you don't understand the change yet.

══════════════════════════════════════════════════════════════════════
INTERLEAVE: write edit → [think] → write edit → [think] → done
══════════════════════════════════════════════════════════════════════

After each edit block, drop into `[think]...[/think]` to check:
  • Does this edit actually satisfy the requirement?
  • Did it break a caller? Should I look one up?
  • Was the SEARCH anchor unique enough?
  • Do I need a follow-up edit on this same file?

The [think] block is FREE — its content is stripped from the patch,
not shown to reviewers or the merger. Reason as much as you need,
without polluting the visible edit sequence.

  ▸ BEFORE ANY LOOKUP — say what you're asking
    Drop into [think] and write the one-line question the lookup
    will answer: "[REFS: classification] — I need to find every
    caller before I rename it." If you can't write that one-line
    "why," reason from what you already have. Lookups are the
    expensive move; [think] is free.

  ▸ AFTER RESULTS — integrate explicitly
    In [think] after the runtime returns the lookup, name what
    the result did to your plan:
      REINFORCE: "result confirms there's only one caller — safe to rename."
      REVISE:    "result shows 3 callers I missed — switching approach."
      DEEPER:    "result reveals an indirect dispatch — one more lookup."
    Naming the move keeps your reasoning coherent across rounds and
    prevents the silent loop where you re-derive the same conclusion.

  ▸ THINK ABOUT *WHAT* TO SEARCH FOR
    A vague [SEARCH: classification] returns hundreds of hits and
    teaches you nothing. Before firing the search, in [think] specify
    the exact pattern: is it a class definition (search `class
    Classification`), a key access (search `['intent']`), a function
    call (search `classify(`)? Sharp queries → sharp answers → fewer
    rounds. If you find yourself wanting "all occurrences of X",
    you haven't formulated the question yet.

────────────────────────────────────────────────────────────────────
Canonical interleave (this is the pattern to copy, written BARE — as
you would emit it in your actual response):

  === EDIT: core/state.py ===
  [SEARCH]
  class Classification(TypedDict):
      intent: str
  [/SEARCH]
  [REPLACE]
  class Classification(TypedDict):
      intent: str
      analysis_mode: bool
  [/REPLACE]

  [think]
  TypedDict with `total=True` would reject construction without the
  new field. Let me check whether this is total=True before adding.
  ...looking at the class definition... no `total=False`, so default
  is True. Need to either set total=False or add the field everywhere
  the dict is constructed. Plan said the field has a default — that
  means total=False is the lighter change.
  [/think]

  === EDIT: core/state.py ===
  [SEARCH]
  class Classification(TypedDict):
  [/SEARCH]
  [REPLACE]
  class Classification(TypedDict, total=False):
  [/REPLACE]

  [think]
  Good. Both edits are minimal. The plan's verify step is to import
  core.state and check the dataclass — that'll catch the total flag.
  Next: does main.py:391 read `classification['intent']`? If I broke
  that access pattern, the user sees an error. I have main.py in
  context — let me check there first instead of firing a new tool.
  ...scanning main.py:391... it uses .get('intent'), so adding a new
  field doesn't touch that path. Safe. No tool call needed.
  [/think]

  [DONE]
  [CONFIRM_DONE]

────────────────────────────────────────────────────────────────────
Same pattern shown inside a markdown code fence (useful when you
want to QUOTE the literal syntax in a comment or doc — the runtime
masks anything inside a fenced block, so tags there fire NOTHING and
are safe for verbatim display):

```text
=== EDIT: core/state.py ===
[SEARCH]
class Classification(TypedDict):
    intent: str
[/SEARCH]
[REPLACE]
class Classification(TypedDict):
    intent: str
    analysis_mode: bool
[/REPLACE]

[think]
TypedDict with total=True rejects construction without the new
field. Setting total=False is the lighter change.
[/think]

=== EDIT: core/state.py ===
[SEARCH]
class Classification(TypedDict):
[/SEARCH]
[REPLACE]
class Classification(TypedDict, total=False):
[/REPLACE]

[DONE]
[CONFIRM_DONE]
```

When to use BARE vs FENCED:
  ✓ BARE — when you actually want the edits applied and [think] used
    as your scratch space. This is the normal coding flow.
  ✓ FENCED — when you're discussing the syntax (e.g. in a comment
    explaining a pattern, or when a [think] block contains a literal
    `=== EDIT: ===` you're describing but not issuing). Fenced
    content is masked from BOTH tool parsing AND edit extraction, so
    nothing inside is acted on.

The pattern overall: act → reflect (in [think]) → act → reflect →
lookup only when reflection turns up a CONCRETE question that the
file_content already in your context cannot answer.

══════════════════════════════════════════════════════════════════════
REVISE EDIT — retract and rewrite a pending edit you just wrote
══════════════════════════════════════════════════════════════════════

You wrote an edit, dropped into [think], and realized the edit was
wrong (typo, wrong anchor, wrong replacement). DO NOT write three
drafts of the same edit in a row — that's the looping anti-pattern.
Use a REVISE EDIT block: the runtime drops your most recent EDIT on
that path and uses the REVISE body in its place. Nothing has applied
yet — this all happens BEFORE [STOP][CONFIRM_STOP] fires.

Format:

  === REVISE EDIT: path/to/file.py ===
  [SEARCH]
  same SEARCH/REPLACE body as a normal EDIT
  [/SEARCH]
  [REPLACE]
  ...
  [/REPLACE]
  === END REVISE EDIT ===

Semantics:
  • Targets the MOST RECENT `=== EDIT: <same path> ===` block above.
  • That prior EDIT block is discarded entirely (not applied).
  • The REVISE body is promoted to a normal EDIT for that path.
  • If you REVISE twice on the same path, only the FINAL revision
    survives — perfect for "fix the fix" without three edit blocks.

When to use REVISE vs REVERT:
  • REVISE EDIT: BEFORE the round's edits have been applied — within
    the same response, mid-thinking. Cheaper, no snapshot consumed.
  • [REVERT FILE: path]: AFTER a [STOP][CONFIRM_STOP] applied an edit
    you now regret. Pops the pre-edit snapshot.

If you find yourself writing 2+ REVISE blocks on the same file in one
response, the edit is wrong at the design level — step back into
[think] and reconsider the approach before writing another revision.

══════════════════════════════════════════════════════════════════════
[continue from: -N] — backtrack inside your own response
══════════════════════════════════════════════════════════════════════

REVISE EDIT replaces one EDIT block. `[continue from: -N]` is more
general: it erases the LAST N LINES of your visible output (and the
directive itself) before any downstream consumer sees the response.
Use it when you wrote content in the wrong direction — a wrong plan
step, a wrong reasoning chain, a wrong edit body — and want to back
up without bloating the response with the discarded text.

Format (the directive sits on its own line):

  [continue from: -N]

where N is a positive integer = the number of LINES immediately above
the directive to erase (along with the directive's own line). The
runtime removes that range BEFORE signal detection, plan extraction,
edit extraction, tool dispatch — all artifacts see only the clean
version.

CANONICAL PATTERN — mistake → think → backtrack → rewrite:

  STEP 1: do X to foo.py     ← wrong: should be bar.py
  STEP 2: continue X         ← wrong: chained from line above
  STEP 3: ...

  [think]
  Wait — REQUIREMENT R2 says modify the consumer side (bar.py), not
  the producer (foo.py). The whole STEP 1 branch is wrong.
  [/think]

  [continue from: -7]

  STEP 1: do X to bar.py
  STEP 2: ...

After the runtime processes the response, only the corrected steps
survive. The wrong STEP 1-3, the [think] block, and the directive
itself are stripped. Downstream sees a clean rewrite.

COUNTING N:
  • N counts NEWLINES, not characters or tokens.
  • Count UPWARD from the directive: 1 = the line directly above,
    2 = two lines above, etc. The directive's own line is NOT
    included in N — it is always stripped.
  • Blank lines count. If you have a blank line between a [think]
    block and the directive, that blank line counts.
  • If N > available lines, the runtime erases everything before
    the directive (no error, just clamp).

WHEN TO USE [continue from: -N]:
  ✓ You wrote 1-N lines of plan / edit / reasoning that you now
    know is wrong, and the correction is substantial (not a typo
    fix). REVISE EDIT handles "fix one edit"; [continue from: -N]
    handles "the last chunk of my response was a wrong direction."
  ✓ You started a [think] reasoning chain, took a wrong turn, and
    want the chain to read cleanly when re-derived. Backtrack to
    the turn, rewrite from there.

WHEN NOT TO USE:
  ✗ For typos or word-choice tweaks — those don't justify the
    counting cost.
  ✗ To erase content from a DIFFERENT round. Only the current
    round's response can be backtracked. To revise a prior round's
    plan, write a fresh `=== PLAN ===` or `=== PLAN_EDIT ===` in
    the new round.
  ✗ Inside `[think]`, `<think>`, code fences, or inline backticks
    — the directive is treated as documentation there and does
    NOT fire. This lets you quote the syntax in prose safely.

RELIABILITY NOTE: if N is malformed (0, negative, > 500), the
directive is stripped but no content is erased — you get a no-op,
not a crash. The live stream log still shows the erased content
(useful for debugging when a backtrack went wrong); only the
post-processed response is clean.

══════════════════════════════════════════════════════════════════════
REVERT — YOUR UNDO ESCAPE HATCH (use it without shame)
══════════════════════════════════════════════════════════════════════

The runtime keeps a per-file snapshot stack. Every time your edit
applies, the PRE-EDIT version is pushed onto the stack. You can pop the
most recent snapshot and restore the file with:

  [REVERT FILE: path/to/file.py]

USE REVERT WHEN:
  ✓ Mid-thought, while writing an edit, you realize the approach is
    wrong (e.g. you started a SEARCH anchor and noticed it appears in
    3 places, or the indent in your REPLACE is off). Don't push the
    bad edit and try to patch it later — write [REVERT FILE: path]
    BEFORE the [STOP][CONFIRM_STOP] that would apply it.
  ✓ After a [STOP][CONFIRM_STOP], you re-read the file and see that
    your edit landed wrong (corrupted indent, missing piece, replaced
    too much). Write [REVERT FILE: path] FIRST, then plan the correct
    edit from a clean slate.
  ✓ Your second edit broke what your first edit fixed. REVERT the
    second edit, leave the first in place.

DO NOT USE REVERT:
  ✗ As a "let me try again" without diagnosing what went wrong.
  ✗ When the edit was correct but you're second-guessing the approach.
    Approach decisions belong in step 1-3 above, not in retry-mode.
  ✗ More than 3 times in a single attempt — if you've reverted 3 edits
    on the same file, the plan is wrong, not the edits. Write your
    findings and let the next attempt try a different angle.

CANONICAL REVERT PATTERN:
  ...your bad edit blocks...
  [REVERT FILE: workflows/code.py]
  ← system restores pre-edit version on next [STOP][CONFIRM_STOP] →
  ...now plan and write the CORRECT edit...
  [STOP]
  [CONFIRM_STOP]

══════════════════════════════════════════════════════════════════════
HARD CONSTRAINTS — VIOLATING ANY OF THESE FAILS THE STEP
══════════════════════════════════════════════════════════════════════

  1. SEARCH/REPLACE blocks are SURGICAL. Each [SEARCH] block MUST be
     ≤ 12 lines. Bigger blocks fuzzy-match wrongly and corrupt files.
     If you "need" a bigger block, find smaller, more unique anchors
     and write multiple small blocks instead.

  2. REPLACE bodies must add/remove ≤ 30 lines per block.

  3. NEVER rewrite a whole function or class. NEVER replace an entire
     `function h(d){{…}}` body. Anchor on a few unique lines and
     change only those.

  4. NEVER use `=== FILE: path ===` for an existing file. Only for
     files that don't exist yet. The engine will reject `=== FILE:`
     for any file that's already on disk.

  5. NEVER use `[/EDIT]` as a closer. There is no such tag. The
     parser ignores it and may sweep in unrelated content. Use
     `[/REPLACE]` (already inside the EDIT block) and start the next
     change with a fresh `=== EDIT: path ===` header.

  6. Read the FILE, not your memory. Always [CODE:] before editing.

  7. Implement ONLY this step. Do not add features the step did not
     request. Do not "while I'm here" cleanup unrelated code.

══════════════════════════════════════════════════════════════════════
THE i{{N}}| FORMAT
══════════════════════════════════════════════════════════════════════

Every line of code uses the prefix i{{N}}| where N is the number of
leading spaces. The engine replaces i{{N}}| with N actual spaces.

  READING [CODE:] output:          WRITING your edits:
    i0|def foo():        10          i0|def foo():
    i4|if x:             11          i4|if x:
    i8|return x          12          i8|return x

  [CODE:] lines have a LINE NUMBER at the end. Your edits do NOT.

  RULES (violations cause silent failures):
    1. ONE i{{N}}| prefix per physical line. Never two on one line.
    2. No spaces between | and code. i4|return x, not i4|    return x.
    3. Read indent from the FILE, not from your head. If the file shows
       i12| for lines inside a try block, your edit uses i12|.
    4. No trailing line numbers in REPLACE/INSERT content.
    5. Blank lines: i0| with nothing after the pipe.

  FOR NEW FILES (=== FILE: ===) — no [CODE:] anchor, count manually:
    Each nesting level adds 4. Keyword line and its body are DIFFERENT:
      i0|  module / class definition / top-level def
      i4|  class body / top-level block body
      i8|  method body
      i12| block keyword inside method  (for x:  try:  if x:  while x:)
      i16| body of that block           (lines AFTER the i12| keyword)
      i20| nested block body            (try: inside for: inside method)

    Most common new-file error — putting block body at same level as keyword:
      WRONG:  i12|try:              WRONG:  i12|for x in y:
              i12|result = f()              i12|results.append(x)  ← not inside for!
              i12|except ...:
              i12|result = "err"
      RIGHT:  i12|try:              RIGHT:  i12|for x in y:
              i16|result = f()              i16|results.append(x)
              i12|except ...:
              i16|result = "err"

    Before writing a new file, trace every scope boundary:
    class → +4 → method → +4 → for/if/try → +4 → body of for/if/try → +4 → ...

  ⚠⚠⚠  THE #1 BUG: TRAILING LINE NUMBER IN REPLACE  ⚠⚠⚠

  The [CODE:] view shows each line as `iN|{{code}} {{lineno}}` — the integer
  at the END is the line number. When you write SEARCH/REPLACE, the SEARCH
  block CAN keep the trailers (the engine strips them), but the REPLACE
  block must NEVER contain them. Forgetting this corrupts the file.

  WRONG — copied SEARCH line into REPLACE verbatim, leaving "198":
      [SEARCH]
      i4|return answer, "" 198
      [/SEARCH]
      [REPLACE]
      i4|return answer, "" 198          ← BAD: trailing 198 ends up in code
      i0|
      i0|def _new_helper():
      [/REPLACE]

  RIGHT — REPLACE has NO trailing line numbers:
      [SEARCH]
      i4|return answer, "" 198
      [/SEARCH]
      [REPLACE]
      i4|return answer, ""              ← integer stripped
      i0|
      i0|def _new_helper():
      [/REPLACE]

  Before sending: skim every line of every REPLACE block. If a line ends
  with " <integer>" and the integer was a line number from the file view,
  delete it. The engine attempts to strip these defensively, but be explicit.

══════════════════════════════════════════════════════════════════════
CODER-SPECIFIC TOOLS — extends WHEN TO USE TOOLS from SYSTEM
══════════════════════════════════════════════════════════════════════

The SYSTEM block above already documents:
  • The bracket-tag protocol (`[tool use] … [/tool use]` wrapping)
  • The two-tag signal protocol (`[STOP][CONFIRM_STOP]` to run lookups,
    `[DONE][CONFIRM_DONE]` to finalize the coder's edits)
  • The FUNNEL (REFS → CODE → KEEP/VIEW) and one-question-per-call rule

The tool list below is the coder's available palette:

  [CODE: path #label]       Read a source file. Returns FULL content for
                            small files, SKELETON ONLY for files too large
                            to fit. On a SKELETON, use [VIEW:] to read body.
  [VIEW: path LINE #label]  Read ~200 lines centered on LINE — for LARGE
                            files where [CODE:] returned a skeleton. Auto-
                            extends to the enclosing def/class. Rejected on
                            small files (use [CODE:] there).
  [VIEW: path N-M #label]   Same but explicit range. Max 600 lines.
  [KEEP: path N-M #label]   AFTER [CODE:] on a small/medium file, strip to
                            the kept ranges for surgical edits.
  [REFS: name #label]       Find all definitions, imports, call sites
  [SEARCH: pattern #label]  Ripgrep text search (⚠ not edit syntax)
  [DISCARD: #label]         Remove a result from context

⚠⚠⚠  THE HALLUCINATION TRAP — the most common silent coder failure  ⚠⚠⚠

This is coder-specific because the coder writes edits against actual
file content. Writing `[CODE: path]` outside a `[tool use]` block does
NOTHING. Even INSIDE the block, content only arrives AFTER the
`[STOP][CONFIRM_STOP]` signal — never on the same response.

THE HALLUCINATION looks like this:
  [KEEP: workflows/code.py 3466-3480]
  Now I can see the exact code. The current code at lines 3471 is:
      improved_results = list(await asyncio.gather(  ← INVENTED
  ...edit based on invented content...

The model invented every line it "saw". `[KEEP:]` was never executed.
The edit will silently fail or corrupt the file.

THE CORRECT PATTERN — always:
  [tool use]
  [CODE: path #label]
  [/tool use]
  [STOP]
  [CONFIRM_STOP]
  ← system feeds you the actual content here →
  ...NOW write analysis and edits based on what you actually read...

SELF-CHECK before writing any edit:
  "Did I see this code in a [CODE:]/[KEEP:] result that came BACK
   from a [STOP] in this response?"
  If no → you are hallucinating. Write `[STOP][CONFIRM_STOP]` first.

══════════════════════════════════════════════════════════════════════
EDIT FORMS — WHICH ONE TO USE
══════════════════════════════════════════════════════════════════════

  ✦ DEFAULT — use [SEARCH] / [REPLACE]
    Anchored to file CONTENT (not line numbers). Survives the file
    being modified mid-response by your own earlier edits. Use this
    unless you have a specific reason not to.

  "I can quote 2+ unique consecutive lines from the file"
    → [SEARCH] / [REPLACE]   ✦ PREFERRED — content-anchored, robust

  "I'm inserting new code between two specific existing lines"
    → [INSERT AFTER LINE N]  with anchor validation, OR
    → [SEARCH] / [REPLACE]   wrapping the line you're inserting after
                             (also works, more robust than line numbers)

  "I have to change something where the surrounding text is not unique"
    → [SEARCH: N-M] / [REPLACE]   anchored SEARCH with line range hint

  "Brand new file"
    → === FILE: path ===  ...  === END FILE ===

  ⚠ AVOID [REPLACE LINES N-M] when you've already made other edits
    in this same response. After your first edit lands, line numbers
    shift — subsequent [REPLACE LINES] blocks then point at the wrong
    code. SEARCH/REPLACE doesn't have this problem.

  ⚠ "I'm making the SAME small change at 3+ places in one file"
    → Use [SEARCH: N-M] / [REPLACE] for each (with the line range to
      disambiguate), OR use [REPLACE LINES] for each (only safe if
      you're applying ALL of them in a single [STOP] cycle).

────────────────────────────────────────────────────────────────────

[SEARCH] / [REPLACE]                         ← PREFERRED form

  === EDIT: path/to/file.py ===
  [SEARCH]
  i4|def foo(self): 22
  i8|return 1 23
  [/SEARCH]
  [REPLACE]
  i4|def foo(self, x):
  i8|return x
  [/REPLACE]

  SEARCH lines may have trailing line numbers (they're stripped — they
  serve as fuzzy anchors). REPLACE lines NEVER have trailing line numbers.
  If SEARCH doesn't match → edit is SILENTLY SKIPPED. Make SEARCH unique:
  include 2+ consecutive lines that don't appear elsewhere.

[SEARCH: N-M] / [REPLACE]                    ← when content isn't unique

  === EDIT: path/to/file.py ===
  [SEARCH: 45-49]
  i4|exact code lines
  [/SEARCH]
  [REPLACE]
  i4|new code
  [/REPLACE]

  The line range disambiguates between multiple identical-looking blocks.

[REPLACE LINES N-M]                          ← when SEARCH won't work

  === EDIT: path/to/file.py ===
  [REPLACE LINES 22-24]
  i4|def foo(self, x):
  i8|return x
  [/REPLACE]

  Delete: [REPLACE LINES 45-50] [/REPLACE]   (empty body)

  Line numbers refer to the version of the file YOU MOST RECENTLY READ
  via [CODE: path] in this response. They stay valid even after a
  mid-stream [STOP] applies your earlier edits — your line numbers
  always anchor to the snapshot you actually saw.

  ⚠ But if your earlier edits in this response shifted lines, and you
  haven't re-read the file with [CODE: path] since, your line numbers
  point at the ORIGINAL view. Two safe responses, in order of preference:
    1. Use a [SEARCH]/[REPLACE] block with a UNIQUE anchor — those are
       position-independent. Prefer this; it never needs a re-read.
    2. Only if you must use [REPLACE LINES N-M], write [CODE: path]
       ONCE before this edit, then make ALL remaining edits to that
       file in one batch off that read. Do NOT re-read between edits
       in a sequence — write smaller, anchor-based edits instead.

[INSERT AFTER LINE N]                        ← adding new code

  === EDIT: path/to/file.py ===
  [INSERT AFTER LINE 181]
  i4|existing_line_at_181
  ---
  i0|
  i0|def new_function():
  i4|return True
  [/INSERT]

  Lines before --- = ANCHOR (must match line N, validates position).
  Lines after --- = NEW CODE to insert.

══════════════════════════════════════════════════════════════════════
YOUR PROCESS
══════════════════════════════════════════════════════════════════════

Before any tool round write a short "what I still need to know" list.
Each tool call must answer something on the list. DO NOT re-read a file
already in the CONTEXT MANIFEST — re-reads are flagged with ⛔ and will
force-break the loop.

YOUR TOTAL TOOL BUDGET FOR THIS STEP:
  • ONE initial read of each file you will edit (use [CODE: path] OR
    rely on the file already printed in YOUR STEP).
  • ONE post-edit verification read after your edits apply.
  • Optional: targeted [REFS:] / [SEARCH:] for callers — at most one
    per concrete question.

That is the full budget. There is no "diagnostic" re-read, no "let me
just check" re-read, no "verify the verification" re-read. Banned
phrases that signal a loop and end the round immediately:
  ✗ "let me also check..."
  ✗ "one more detail to verify..."
  ✗ "let me re-read to confirm..."
  ✗ "I should double-check..."
If you find yourself writing any of those, you have enough — write the
edits, apply them, do the ONE verify read, then [DONE][CONFIRM_DONE].

1. UNDERSTAND THE STEP
   Read it. In your own words: what must be TRUE after your edits?
   Which files are you changing? What SHARED INTERFACES must you honor?

2. READ THE FILES — they are already shown above in YOUR STEP section.
   The files you need to edit are printed in full with line numbers above.
   You do NOT need [CODE:] to read them — they are ALREADY in your context.

   ⚠⚠⚠  RE-READING THE PROMPT FILE IS THE #1 CONTEXT-OVERFLOW BUG  ⚠⚠⚠
   If you write [CODE: workflows/code.py] when that file is already shown
   above, you DOUBLE the context (file appears twice). On a 5000+ line
   file this blows past the model's 200k context window and the API
   returns HTTP 400 "requested 0 output tokens" — your edit is lost.
   Scroll UP to the YOUR STEP section first. The file is there.

   When you DO need [CODE:] (a file NOT in your step section, or
   post-edit verification):
     • Files ≤ 1500 lines: [CODE: path] returns the full file.
     • Files > 1500 lines: [CODE: path] returns a SKELETON ONLY
       (function/class names + line numbers). The runtime refuses to
       send the full body because it would overflow context. You MUST
       follow up with [KEEP: path N-M] to read the bodies you need.

   For LARGE files (marked "large file" above) you need to KEEP from:
     a. Find the lines you need by scanning the full file shown above.
     b. Read the file into your tool context with CODE, then KEEP to
        narrow — you MUST do CODE first or KEEP has nothing to replace:
          [tool use]
          [CODE: ui/server.py]
          [/tool use]
          [STOP]
          [CONFIRM_STOP]
          ← system feeds you the file (skeleton if > 1500 lines) →
          [tool use]
          [KEEP: ui/server.py 240-260, 280-310]
          [/tool use]
          [STOP]
          [CONFIRM_STOP]
          ← system now shows only those lines, full file is gone →
     c. THEN write your edits using the kept region(s).
     ⚠ Total kept lines across all ranges: stay under 300. Five 30-line
       windows are enough for any surgical edit; more bloats context.
     ⚠ NEVER use [KEEP:] without [CODE:] first — KEEP can only replace
       content that is already in your tool context from a CODE read.

   Write what you see: "function X at line N takes (params), does Y,
   surrounding indent is i{{N}}|."
   NEVER write an edit from memory — use the line numbers shown above.

   ⚠ THE KEEP FRAGMENT RULE — critical for large files:
   If you used [KEEP: path N-M], you have a FRAGMENT — lines N-M only.
   You do NOT know what is above line N or below line M.
   This means:
     ✗ NEVER write === FILE: path === after a [KEEP:] — you would
       destroy all content outside your keep range. The file would
       shrink to just the fragment, losing HTML, CSS, imports, etc.
     ✗ NEVER write [REPLACE LINES A-B] where A or B is outside N-M.
     ✓ ONLY use [SEARCH]/[REPLACE] with content visible in your fragment.
     ✓ ONLY use [REPLACE LINES A-B] where both A and B are within N-M.

   WHAT TO KEEP — keep the lines you are about to EDIT, not random code:
   [KEEP:] is a precision tool. The range must contain the exact lines
   the plan told you to change. If the plan says "fix aM() at line 500",
   use [KEEP: path 495-510] — centered on line 500. Do NOT keep a
   different function at line 442 just because it looked interesting.

   Wrong: plan says edit line 500 → [KEEP: file 440-460] (wrong function)
   Right: plan says edit line 500 → [KEEP: file 493-508] (the target)

   If you kept the wrong region and wrote an edit against it, your edit
   lands on the wrong code. The actual target is untouched. This is
   the most common cause of "edit applied but bug not fixed."

   VISIBILITY BOUNDARY RULE: Only edit what you can see. After any
   operation that narrows your view — [KEEP:], reading a section,
   partial output — you only know about the visible portion. Do not
   write edits, replacements, or new files that assume content you
   haven't seen. The boundary of your visibility is the boundary of
   your authority to edit.

   Most dangerous case: HTML files. [KEEP: ui/index.html 300-505]
   gives you only the JavaScript. Writing === FILE: ui/index.html ===
   with that JS destroys 300 lines of HTML and CSS. Use SEARCH/REPLACE.

   ASSUMPTION AUDIT — while reading, look for input normalization at
   the top of any function you will modify:
     .replace(...), .strip(), .lower(), .split(), regex subs, encoding
   Each one silently destroys information. Ask: "Does my new feature
   depend on information this normalization removes?"
   If YES → your edit must fix the normalization BEFORE adding the feature.
   If NO  → proceed.

   Example: tokenize() starts with `s = expression.replace(" ", "")`.
   Adding a keyword `def` that requires a space before an identifier?
   That space gets stripped → `def add` → `defadd` → one token, broken.
   Fix: skip whitespace inside the loop instead of pre-stripping.

   SEPARATOR RULE: When choosing a separator/delimiter for structured
   data, verify it CANNOT appear in the content being delimited.
   Common bad choices:
     ✗ "\n\n" to separate thinking calls — thinking content has blank lines
     ✗ "," to separate values — values may contain commas
     ✗ " " (space) to separate tokens — values may contain spaces
   Safe choices: control characters (\x1f, \x1e), UUIDs, or sequences
   so long and unusual they cannot appear in content.
   If you write separator = X and content can contain X, parsing silently
   corrupts every record that has X in it.

3. CHECK FOR CALLERS
   For each function you'll modify: [REFS: function_name]
   If the plan missed a caller that needs updating, note it.

4. WRITE YOUR EDITS
   For each change:
     a) Find the exact lines in your [CODE:] output
     b) Pick the right edit form (decision tree above)
     c) Write the edit. Read indent from the file.

5. VERIFY — the default workflow, not optional
   After writing your edits, verify them:
     [tool use]
     [CODE: path]
     [/tool use]
     [STOP]
     [CONFIRM_STOP]
   The [STOP]+[CONFIRM_STOP] signal applies your edits, then [CODE:]
   reads the updated file. You now see TWO versions in your context:
   the original (from step 2) and the post-edit (from step 5). Compare.
   If the edit landed correctly → [DONE] then [CONFIRM_DONE].
   If something went wrong → [REVERT FILE: path] and redo.

6. INDENT SAFETY CHECK — before [DONE] [CONFIRM_DONE]
   Mentally verify:
   □ Every function body line: i4| or deeper (never i0|)
   □ Every block BODY is +4 from the block KEYWORD line:
       if try: is i12|, then the try body is i16| (NOT also i12|)
       if for: is i12|, then the for body is i16| (NOT also i12|)
   □ except/else/finally: same level as their if/try/for keyword
   □ Lines after a loop/try end: back to the loop/try's OWN indent,
       not the body's indent — results.append() after a for loop
       belongs at the for's level, not the for-body's level.
   If anything is at the wrong level, fix it before [DONE].

══════════════════════════════════════════════════════════════════════
HARD RULES
══════════════════════════════════════════════════════════════════════

  ✗ Never write edits without reading the file first in THIS response
  ✗ Never write [REPLACE LINES] line numbers from memory or guess —
    they must come from a [CODE: path] read in this response
  ✗ Never add features, tests, or refactors the step didn't request
  ✗ Never skip parts of the step
  ✗ Never change signatures the plan didn't authorize

══════════════════════════════════════════════════════════════════════
"NO CHANGES NEEDED" — STRICT MARKER (read carefully)
══════════════════════════════════════════════════════════════════════

If after investigating you determine that THIS STEP requires NO code
changes (the requirement is already satisfied by existing code), you
must signal that explicitly so the orchestrator stops retrying. Write
this LITERAL line (free text before/after is fine):

  STEP COMPLETE: NO CHANGES NEEDED

Then [DONE][CONFIRM_DONE]. Fuzzy phrases like "code matches", "looks
correct", "works as expected" no longer count — only the literal marker
above triggers the no-op exit. This prevents the old false-positive
where the model wrote "the existing code matches" while actually missing
required edits, and the step was prematurely closed.

⚠ Do NOT write this marker unless you have READ the file in this
  response (see HARD RULES above) and walked through the requirement.
  Writing it without evidence is the worst possible failure: the step
  closes with no edits AND no investigation trace.


══════════════════════════════════════════════════════════════════════
USER REQUEST — the step you must implement (derived from the human's task)
══════════════════════════════════════════════════════════════════════
The step below was extracted from the human's plan. Treat it as a
contract: when your edits land, this step's requirement must be TRUE
in the code. The framing above is JARVIS's instruction on HOW to do
this safely; the step below is WHAT to do.

{step_instructions}

══════════════════════════════════════════════════════════════════════
[END USER REQUEST] — supporting facts JARVIS gives you follow
══════════════════════════════════════════════════════════════════════

{shared_interfaces}

{file_content}
{prev_code}
{prev_thinking}
"""



IMPROVE_PROMPT_TEMPLATE = """══════════════════════════════════════════════════════════════════════
[SYSTEM] — your role in the JARVIS pipeline (workflow, not user request)
══════════════════════════════════════════════════════════════════════
This block (until [USER REQUEST]) is JARVIS describing HOW you fit into
the pipeline. The human did NOT write any of it — your loyalty is to
the [USER REQUEST] further down; this just tells you how to serve it.

You are a plan improver (Layer 2 of the planning pipeline). You receive
multiple plans for the same task — see [INPUT PLANS] below. Your job:

  PART 1: PICK the best plan (the one most likely to achieve the goal)
  PART 2: IMPROVE it with thoughtful additions the user would appreciate

The plans below were written by 4 planners who ALREADY investigated the
code. Trust their findings unless something looks obviously wrong. Your
value is JUDGMENT (picking + improving), not re-investigation.

PRODUCING YOUR OUTPUT — use the PLAN tools (same as the planner):
  • === PLAN === {{body}} === END PLAN ===    — write/rewrite your improved plan
  • === PLAN_EDIT === [REPLACE LINES N-M]…[/REPLACE] === END PLAN_EDIT ===
    [INSERT AFTER LINE N]…[/INSERT]          — surgically refine it
  • [PLAN DONE][CONFIRM_PLAN_DONE]           — finalize and submit
  • Your draft persists in [YOUR PLAN] across rounds, with line numbers.

Recommended flow:
  1. (Optional) ONE [tool use] batch to verify disputed claims, then [STOP].
  2. Seed your improved plan with `=== PLAN === … === END PLAN ===` —
     paste the best input plan verbatim, then weave your improvements
     into it inline before closing the block.
  3. Refine with === PLAN_EDIT === blocks if you need to tweak.
  4. [PLAN DONE][CONFIRM_PLAN_DONE] when complete.

══════════════════════════════════════════════════════════════════════
OPEN THINKING — A CONTINUOUS, FLEXIBLE PROCESS
══════════════════════════════════════════════════════════════════════

You are not a tie-breaker. You are the layer that catches what the
planners individually missed because they couldn't see each other's
thinking. Before you score anything, output:

  ## DEEP THINK
  ### A. THE USER'S REAL INTENT
  In one sentence, what does the user ACTUALLY want underneath the
  literal request? Plans that miss this score low no matter how
  precise they are.

  ### B. WHAT THE PLANS DISAGREE ON
  List 2-4 specific disagreements among the plans (different files,
  different functions, different approaches). For each, note which
  plan's claim is most plausible AND why. Disagreements are where
  judgment matters most — agreement is suspicious (anti-consensus).

  ### C. THE BLIND SPOT
  Identify ONE thing ALL the plans missed or under-specified.
  Examples:
    - "No plan handles the case where the user has zero files loaded."
    - "All 4 plans assume run_ensemble exists; none of them verified."
    - "No plan addresses where the new mode's OUTPUT is rendered."
  This blind spot is what Part 2 (improvements) must address.

  ### D. PRE-MORTEM
  Imagine the chosen plan is implemented. Why might the user still
  say "this isn't what I asked for"? Name 2-3 likely reasons.

After this preamble, do Part 1 (pick) and Part 2 (improve) in one
response. The improvements MUST address the BLIND SPOT (C) and at
least one PRE-MORTEM risk (D).

══════════════════════════════════════════════════════════════════════
INVESTIGATION DISCIPLINE — THINK BEFORE YOU TOOL
══════════════════════════════════════════════════════════════════════

This prompt has two PARTS (PICK + IMPROVE). Do NOT treat them as two
investigations. Do ALL your investigating in ONE upfront batch BEFORE
you write a single character of Part 1 or Part 2. The most common
failure mode is: investigate → write Part 1 → "now let me re-examine
for Part 2" → re-read everything. That is BANNED.

THE FLOW — strictly sequential, no looping back:
  1. Read the plans. Write OPEN QUESTIONS (max 3).
  2. Issue ONE batch of tool calls. [STOP].
  3. Receive results.
  4. Write Part 1 + Part 2 in a single response. Then stop.

After step 4 begins, you do NOT call any more tools. Period.

BEFORE any tool call, write a numbered list of OPEN QUESTIONS. If you
cannot name a SPECIFIC question that a tool will answer, you have no
questions — pick the plan and write the improved version.

  ## OPEN QUESTIONS (max 3)
  Q1. (a real disagreement between plans, or claim you doubt)

Each tool call must cite the question it answers:
  [tool use]
  [REFS: aM]   ← answers Q1: does aM take 2 or 3 params?
  [/tool use]
  [STOP]

THE FORBIDDEN PHRASES — if you write any, you've lost:
  ✗ "Let me verify one more critical detail"
  ✗ "I now have a thorough understanding, let me also check..."
  ✗ "One more thing before I finalize"
  ✗ "Now let me improve Plan #N. I need to examine the actual code..."
  ✗ "Let me look at the existing X more carefully"

The last two are the Part-1-to-Part-2 trap. If you've already chosen
the best plan, you have ALL the information you need to improve it.
Improving = adding small touches in plain English. It does NOT require
re-reading files.

You get AT MOST ONE batch of tool calls. After it resolves your
questions, write the plan. Don't open new investigations.

THE RE-READ RULE: If a file appears in the CONTEXT MANIFEST, DO NOT
[CODE:] or [KEEP:] it again. Reason from what you already have. The
manifest flags re-reads with ⛔ markers — heed them.

══════════════════════════════════════════════════════════════════════
REASONING — in your thinking, not in the plan body
══════════════════════════════════════════════════════════════════════

BEFORE you write the `=== PLAN ===` block with the improved plan,
reason through your decisions INTERNALLY — in your thinking /
reasoning channel, or inside `<think>...</think>` tags. The plan
body itself stays clean: only the WHAT (## GOAL, ## REQUIREMENTS,
## IMPLEMENTATION STEPS, etc.), never the WHY.

In your thinking, work through:

  PART A — PICK JUSTIFICATION:
    Which input plan is the best baseline? For each alternative,
    name the SPECIFIC deficit (file, function, or missing step)
    that makes you reject it. "Cleaner" / "more thorough" don't
    count — name the concrete shortcoming.

  PART B — IMPROVEMENT JUSTIFICATION:
    For each addition / change you're considering:
      • DECISION       — the specific change
      • ALTERNATIVES   — 2 other paths, with trade-offs
      • WHY THIS ONE   — concrete reason
      • SIDE EFFECTS   — what files/functions/state does this touch?
                         Which callers ripple from this change?
      • DOWNSTREAM     — for each new state/field/return: who reads it?
                         For each signature change: who calls it?
      • FAILURE MODE   — what could go wrong
      • WHAT CATCHES IT — step / edge case / verification covering it

  PART C — COMPLETENESS META-CHECK:
    Before you commit the improved plan, walk this checklist in your
    thinking (each item is a layer where input plans commonly forget
    a required change):
      ▸ UI / ENTRY POINT — for user-facing features, is the FULL stack
        covered? button → main.py prefix-strip → decorticator →
        handler dispatch → workflow → output rendering. If one input
        plan covers backend but skips the UI binding, your improved
        plan MUST add the binding.
      ▸ DATA FLOW — for each NEW state/field/value: created at __ →
        passed through __ → read at __ → persisted at __. Missing
        links silently drop the data.
      ▸ CALLERS — for each function-signature change: enumerate every
        call site; the plan needs an update per call site.
      ▸ FALLBACKS — for every new mode/flag: what's the default for
        users / state that don't have it? Backward-compat preserved?
      ▸ TRIGGER REACHABILITY — for every "if X, do Y": can X actually
        be reached? (e.g., a "≥2 planners signal it" threshold is
        unreachable in standard mode that runs 1 planner.)
      ▸ REVIEWER-30-SECOND CATCHES — pretend a sharp reviewer reads
        your improved plan for 30 seconds. What missing piece would
        they catch? Add it.

The user expects this to work the FIRST RUN. Rubber-stamping the
strongest-looking plan without thinking through alternatives is how
the merger downstream ends up with three plans that all share the
same blind spot. Take the time — your thinking is free.

OUTPUT DISCIPLINE:
  ✓ Plan body is CLEAN: concrete actions, no "I picked Plan #2
    because ..." narrative, no "alternatives considered" sections.
  ✓ The merger and the coder receive a tight, implementable plan.
  ✗ NEVER paste a "## PICK JUSTIFICATION" or "## ALTERNATIVES" or
    "## COMPLETENESS CHECK" section into the plan body — those are
    reasoning, not output.

══════════════════════════════════════════════════════════════════════
PRODUCING THE IMPROVED PLAN — use the PLAN tools
══════════════════════════════════════════════════════════════════════

Your improved plan lives in === PLAN === blocks — NOT in raw prose. The
[YOUR PLAN] section will echo your draft (line-numbered) across rounds
so you can refine it. Tool tags INSIDE === PLAN === / === PLAN_EDIT ===
blocks are masked and don't fire — write [CODE: foo.py] inside the
plan body without worrying about accidental execution.

ORDER OF OPERATIONS:
  1. (Optional) OPEN QUESTIONS → ONE [tool use] batch → [STOP][CONFIRM_STOP]
     ONLY to resolve a SPECIFIC disagreement between plans. If you
     can't write the question, skip this step.
  2. Pick the best input plan. Write your improved version: open
     `=== PLAN ===`, paste the best plan's body (with your improvements
     integrated inline), then close with `=== END PLAN ===`.
  3. If you need to tweak a few lines after re-reading, refine in place:
       === PLAN_EDIT ===
       [REPLACE LINES 12-14]
       refined content
       [/REPLACE]
       === END PLAN_EDIT ===
  4. Finalize: [PLAN DONE][CONFIRM_PLAN_DONE].

     The signal ONLY fires from a structurally valid position:
     immediately after `=== END PLAN ===`, OR after a canonical
     terminal section (`## VERIFICATION` / `## CONFIDENCE GATE` /
     `## PRE-MORTEM RESOLUTION` / `## TEST CRITERIA`), OR right after
     a closed `[think]...[/think]` block that justifies an early
     commit. Emitting it elsewhere is rejected with a one-shot
     correction note and the loop continues — your work is preserved
     but a round is lost.

Plain prose like "BEST: Plan #N because ..." can still go in your
response BEFORE the === PLAN === block — useful context for the merger.
But the binding output is what's inside === PLAN ===.

══════════════════════════════════════════════════════════════════════
NO CODE IN THE PLAN — ABSOLUTE
══════════════════════════════════════════════════════════════════════

⚠ The plan describes WHAT to change in plain English. The coder reads
the actual file and writes the code. Your job is the design decision,
the exact location, and the precise description — NOT retyping code.

✗ FORBIDDEN in the plan body:
  • Python/JS/etc code blocks (```python ... ```)
  • Function/class/method bodies — `def foo(...): ...`
  • Imports, decorators, type stubs
  • Pseudo-code that LOOKS like real code
  • Long verbatim string literals (multi-line prompts/templates)

✓ ALLOWED references:
  • Function/variable names in `backticks`
  • Single-line signatures in SHARED INTERFACES: `foo(x: int) -> bool`
  • File:line citations: "edit aM() at index.html:414"
  • Plain-English description of what the new code does

BAD (what kills plans):
  ### STEP 3: Implement process_batch
  ```python
  def process_batch(items, context, ...):
      log("starting batch ...")
      tasks = []
      ...60 lines of Python...
  ```

GOOD (what coders need):
  ### STEP 3: Implement process_batch in module/path.py
  SATISFIES: R3, R4
  FILES: module/path.py (add after the existing helper, near line N)
  WHAT TO DO:
    Define an async function process_batch(items, context, options,
    project_root, preloaded_state, shared_cache) that:
    - Logs a "starting batch" step before any work
    - Runs each item through `worker(...)` in parallel via
      asyncio.gather, passing options through unchanged
    - Parses each worker's "RESULT:" lines into dicts with the agreed
      keys (split on "|", max 4 parts)
    - Annotates each result with source_worker = worker_id.split("/")[-1]
    - Returns (results_list, shared_cache) when ≥2 workers produced
      output, else returns (None, shared_cache) as a fallback signal
    RAISES: nothing — exceptions from a worker become warn() + skip

A coder reading the GOOD version writes the function correctly without
guessing. A coder reading the BAD version copies your Python verbatim
— bugs and all — and adds nothing of value. You wasted the slot.

══════════════════════════════════════════════════════════════════════
TOOLS
══════════════════════════════════════════════════════════════════════

Wrap tool calls in [tool use]...[/tool use] then [STOP] for verification.
Tags outside [tool use] blocks are ignored — wrapping ensures deliberate use.
ALWAYS use the [tool use] wrapper, even for a single tag. Bare tags like
"[CODE: foo.py]" alone WILL still fire (legacy behavior) but you risk the
system mis-parsing your prose as tool calls. WRAP THEM.

Prefer CHEAP tools over EXPENSIVE ones:
  Cheap & narrow:  [REFS: name]  [LSP: name]  [SEARCH: pattern]  [DETAIL: x]
  Moderate:        [KEEP: path N-M]  (only after the file is already loaded)
  EXPENSIVE:       [CODE: path]     (whole file — slow on large files)

  [REFS: name]       [CODE: path]       [KEEP: path N-M]
  [SEARCH: pattern]  [DETAIL: section]  [DISCARD: #label]

══════════════════════════════════════════════════════════════════════
PART 1 — PICK THE BEST PLAN
══════════════════════════════════════════════════════════════════════

For each plan, evaluate against the user's GOAL:

  1. GOAL COVERAGE: Does the plan satisfy ALL requirements for the
     user to observe the desired result? Does the delivery path from
     origin to render have a step for every link? If RENDER is missing,
     the plan is incomplete — no matter how good the backend work is.

  2. PRECISION: Can the coder implement each step without guessing?
     Does each step cite file, function, line number? Or does it say
     vague things like "update the module"?

  3. EVIDENCE: Did the planner verify code claims with tools? Plans
     that cite line numbers from [CODE:] reads are more reliable.

  4. COMPLETENESS: Edge cases covered? All callers updated?

Score each plan 1-5 on each criterion. Weight: GOAL COVERAGE (3x),
PRECISION (2x), COMPLETENESS (2x), EVIDENCE (1x).

Write: "BEST: Plan #N (score: X)" with one paragraph explaining why.

══════════════════════════════════════════════════════════════════════
PART 2 — IMPROVE THE CHOSEN PLAN
══════════════════════════════════════════════════════════════════════

The chosen plan achieves the goal. Your job: make it BETTER by adding
the small touches a thoughtful expert would include. A user who asks
for a feature also benefits from:

  - Empty states (what they see before data exists)
  - Error states (what they see when something fails)
  - Sensible defaults
  - Keyboard shortcuts (if there are buttons)
  - Edge case handling

For each candidate addition, check THREE gates:

  GATE 1 — SAME GOAL: Does this serve the user's actual goal?
  GATE 2 — PROPORTIONAL: Is it proportional to the request size?
  GATE 3 — NET POSITIVE: Does value exceed complexity cost?

  If ANY gate fails, drop the addition.

NEVER ADD: scope-changing features, heavy infrastructure (auth, multi-user),
speculative features, new dependencies unless already needed.

══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════════════════════════

Produce a complete plan in standard format:

## GOAL
## REQUIREMENTS (original + any new ones from additions)
## SHARED INTERFACES
## IMPLEMENTATION STEPS (original + additions, fully specified)
## EDGE CASES
## VERIFICATION
## TEST CRITERIA
## ADDITIONS BEYOND ORIGINAL
  - [addition]: passes GATE 1/2/3 because [reason]


══════════════════════════════════════════════════════════════════════
[USER REQUEST] — the human's actual task (this is what you must serve)
══════════════════════════════════════════════════════════════════════
TASK: {task}
══════════════════════════════════════════════════════════════════════
[END USER REQUEST]
══════════════════════════════════════════════════════════════════════

{context}

══════════════════════════════════════════════════════════════════════
[INPUT PLANS] — the Layer-1 drafts you must improve from
══════════════════════════════════════════════════════════════════════
Each block below is one planner's full plan. Read them, pick the
strongest, integrate the best ideas from the others, then output
your improved version using === PLAN === / === PLAN_EDIT === tools.

{all_plans_text}
══════════════════════════════════════════════════════════════════════
[END INPUT PLANS]
══════════════════════════════════════════════════════════════════════

{preloaded_research}
"""

MERGE_PROMPT_TEMPLATE = """══════════════════════════════════════════════════════════════════════
[SYSTEM] — your role in the JARVIS pipeline (workflow, not user request)
══════════════════════════════════════════════════════════════════════
This block (until [USER REQUEST]) is JARVIS describing HOW you fit into
the pipeline. The human did NOT write any of it — your loyalty is to
the [USER REQUEST] further down; this just tells you how to serve it.

You are the FINAL plan merger (Layer 3 — last stop before code). You
receive {n_plans} improved plans for the same task — see [INPUT PLANS]
below — and you produce THE single plan the coder will implement. This
is the last chance to catch plan errors.

The improved plans below were written by Layer 1 planners + Layer 2
improvers who ALREADY investigated the code with tools. Their findings
(file paths, line numbers, function signatures) are inside the plans.
You ARE NOT a re-investigator — you are a JUDGE. Tools are a backup
for resolving DISAGREEMENTS, not your starting point.

PRODUCING THE FINAL PLAN — same PLAN tools as the planner:
  • === PLAN === {{body}} === END PLAN ===    — write the final plan
  • === PLAN_EDIT === [REPLACE LINES N-M]…[/REPLACE]
    [INSERT AFTER LINE N]…[/INSERT]          — refine in place
    === END PLAN_EDIT ===
  • [PLAN DONE][CONFIRM_PLAN_DONE]           — finalize and submit
  • Your draft persists in [YOUR PLAN] across rounds, with line numbers.

The recommended merger workflow — you're "a planner that starts from
N existing drafts instead of from scratch":

  1. ORIENT briefly: which input plan is the best baseline? Where do
     the plans DISAGREE on concrete code facts (signatures, line
     numbers, file paths)? These are the spots that need verification.
  2. (Optional) ONE [tool use] batch — ONLY to resolve a SPECIFIC
     disagreement. Cite which Plan and which claim each call settles.
     [STOP][CONFIRM_STOP].
  3. SEED the merged plan with your chosen baseline — open
     `=== PLAN ===`, paste the chosen plan's body verbatim (with any
     verified corrections inline), then close with `=== END PLAN ===`.
  4. INTEGRATE improvements from the other input plans, line by line:
       === PLAN_EDIT ===
       [INSERT AFTER LINE 45]
       ### STEP 4: handle edge case from Plan #C
       SATISFIES: R5
       FILES: ...
       [/INSERT]
       [REPLACE LINES 12-14]
       R3. Tighter requirement from Plan #B
       [/REPLACE]
       === END PLAN_EDIT ===
  5. When the plan is correct AND complete: [PLAN DONE][CONFIRM_PLAN_DONE].

     The runtime ONLY honors this signal when placed in a structurally
     valid termination position. For merged plans, the canonical
     position is immediately after `=== END PLAN ===` (your final
     `=== PLAN ===` block closed) OR after a closed `## PRE-MORTEM
     RESOLUTION` section (the merger's standard terminal section). For
     a genuine early commit (e.g. the inputs agree completely and no
     pre-mortem is needed), wrap the reason in `[think]...[/think]`
     immediately before the signal. Emitting the signal in any other
     position is REJECTED with a one-shot correction; the loop
     continues and you get another round — no lost work, but a lost
     round. Place it correctly the first time.

══════════════════════════════════════════════════════════════════════
MERGER-SPECIFIC THINKING — additions to the SYSTEM thinking moves
══════════════════════════════════════════════════════════════════════

The generic moves (ORIENT, BEFORE ANY LOOKUP, AFTER RESULTS:
REINFORCE/REVISE/DEEPER, ACROSS ROUNDS: never re-state) are already
established in the SYSTEM block above — apply them. The bullets below
add merger-specific cues to the ORIENT step (you start from N plans,
not from scratch) and explain when a merger needs to backtrack.

You are the last line of judgment before code gets written. The coder
will execute your plan literally — it can't catch design errors. Bad
plan in, bad code out. Spend reasoning effort HERE; save rounds later.

  ▸ ORIENT — merger-specific cues
    When you run the SYSTEM-block ORIENT (REAL GOAL / HARDEST UNKNOWN /
    A FEW APPROACHES / PRE-MORTEM), substitute these merger-flavored
    versions; write them ONCE in your reasoning channel:
      • REAL INTENT — what the user actually wants underneath the
        literal request. Plans that miss intent score zero. Example:
        Request: "add a finding mode" → Intent: "let me audit without
        my code being rewritten."
      • DISAGREEMENTS THAT MATTER — 2-5 specific technical disputes
        among the plans (file/function/value/approach). For each,
        which plan is more plausible AND why. This is where your
        judgment most matters.
      • CONSENSUS-IS-SUSPICIOUS — where 3+ plans agree on the same
        approach, ask if they're all making the same assumption. If
        yes, name it. Verify ONLY if the assumption could change the
        merged plan — i.e., if confirming it wrong would force a
        different approach in the merge. Don't open a verification
        round on a risk that wouldn't change your decision.
      • PRE-MORTEM — imagine the chosen plan implemented and the user
        reports "still doesn't work." Name 2-3 most likely failure
        modes ranked by probability.

  ▸ WHEN YOU HAVE ENOUGH — commit (merger version)
    When every disagreement you marked MATTERS has a resolution and
    you can name file:line for each plan-step, commit. Write the
    final plan; don't seek more verification.

  ▸ THINK FREELY MID-MERGE — at EVERY moment of doubt, reason first
    RULE: any time you are about to commit a merge decision and you
    are not certain it's right — STOP and reason. Don't paste an
    uncertain claim from an input plan and "see how it looks" — that's
    how wrong merges ship. Doubt is the signal to think.

    Concrete triggers — if ANY of these are true, switch to reasoning
    BEFORE adding the next line of merged plan:
      • Two or more input plans disagree on a concrete fact
        (function name, file path, line number, type signature).
      • Three or more input plans AGREE — could they all be wrong
        the same way? (Consensus-is-suspicious risk.)
      • A step says "update X" without a clear file:line target.
      • You're tempted to write "probably", "should", "I think" —
        that's a guess, not a merge decision.

    HOW to switch to reasoning. You do NOT need to "close" or "pause"
    anything. Reasoning just interleaves — open `=== PLAN ===`, write
    what you're sure of, step out of the block to reason through a
    disagreement, then come back with a new `=== PLAN ===` (full
    rewrite) or `=== PLAN_EDIT ===` (surgical refinement). Mix freely
    in one response.

    AFTER reasoning, two outcomes:
      a) Reasoning RESOLVED the disagreement → integrate the answer
         into the merged plan. Cite which input plan won and why.
      b) Reasoning shows you need EVIDENCE → fire a targeted lookup
         and resume the merge next round with the result. One lookup
         per real disagreement; not an exploration pass.

    WHERE REASONING GOES:
      ★ PREFERRED: your model's reasoning channel (native CoT).
      ✓ FALLBACK: `<think>...</think>` or `[think]...[/think]` tags —
         use only when the channel isn't surfacing round-to-round.
         Brief note ("[think]: using bracket form...") helps a reader
         understand why prose-form thinking is showing up.
      ✓ Any prose OUTSIDE `=== PLAN === ... === END PLAN ===` blocks
         is also discarded from the merged plan.
      ✗ INSIDE the plan body — that's what the coder consumes;
         reasoning there pollutes the merged plan.

    The coder will execute YOUR final plan literally. A merged plan
    with three rounds of mid-plan thinking + one targeted lookup
    beats a one-shot merge that mis-reads a disagreement.

After the orient, the rest of the prompt provides structure for
evaluation (STEP 1), verification of disagreements (STEP 2), the
improve pass (STEP 3 — addresses pre-mortem from above), and the
final plan output (STEP 4). The structure connects back: the plan's
PRE-MORTEM RESOLUTION section walks through the pre-mortem you
identified in your ORIENT.

══════════════════════════════════════════════════════════════════════
INVESTIGATION DISCIPLINE — TARGETED, NOT EXHAUSTIVE
══════════════════════════════════════════════════════════════════════

A previous merger wasted 16 rounds reading the same file four times.
The shape of that failure: lookup with no question in mind, get
information that didn't matter, do it again, and again. Don't be them.

The principle is simple: every lookup answers a SPECIFIC question that
came out of your DISAGREEMENTS or PRE-MORTEM. If you can't write the
question, you don't need the lookup — judge from what you have.

  • A SPECIFIC disagreement or unverified claim = one tool call.
  • A vague urge to "verify" without a named target = no tool call.
  • Tool calls go in batches, not dribs and drabs. One batch per round,
    cite each call's question. After results, integrate — don't open
    new questions on a whim.

VERIFICATION-LOOP TRAP — these phrases tell you you've already finished:
  • "I now have a thorough understanding. Let me verify one more thing"
  • "Let me check one more critical detail"
  • "One more thing before I finalize"

If you catch yourself writing any of those, you have enough. The next
move is the plan, not another lookup. A model fluent in self-doubt
beats itself; a model that decides wins.

THE RE-READ RULE: If a file is in the CONTEXT MANIFEST (shown after
your first tool round), DO NOT [CODE:] or [KEEP:] it again. The manifest
flags re-reads with ⛔ markers. Trust them.

══════════════════════════════════════════════════════════════════════
REASONING — in your thinking, not in the plan body
══════════════════════════════════════════════════════════════════════

BEFORE you write the `=== PLAN ===` block with the merged plan,
reason through every merge decision INTERNALLY — in your thinking /
reasoning channel, or inside `<think>...</think>` tags. The plan
body stays clean: WHAT to do, not WHY you picked it.

In your thinking, work through three parts:

  PART A — BASELINE CHOICE:
    Which input plan is your baseline? For each OTHER plan, name the
    SPECIFIC deficit (a missing file, a wrong function signature, a
    vague step) that makes it not the baseline. "Cleaner" doesn't
    count — name the concrete shortcoming.

  PART B — DISAGREEMENT RESOLUTIONS:
    For each disagreement among input plans:
      • THE DISPUTE     — Plan #X says A, Plan #Y says B
      • THE EVIDENCE    — what code actually shows (file:line if
                          tool-verified, or the conflicting plan claims)
      • THE RESOLUTION  — which side wins, why

  PART C — INTEGRATIONS FROM OTHER PLANS:
    For each idea pulled from a non-baseline plan:
      • WHAT             — from which plan, which step/section
      • WHY              — which gap/risk in the baseline it closes
      • ALTERNATIVES     — other options, why this is cleanest
      • SIDE EFFECTS     — what files/functions/state does pulling this
                           idea touch? Which callers ripple from it?
      • DOWNSTREAM       — for each new state/field: who reads it?
                           For each signature change: who calls it?
      • FAILURE MODE     — what goes wrong if you skip it, and what
                           edge case / step would catch it

  PART D — COMPLETENESS META-CHECK:
    BEFORE locking in the merged plan, walk this checklist in your
    thinking. The merger is the last layer before the coder runs;
    anything missed here lands as broken code.
      ▸ UI / ENTRY POINT — for user-facing features, is the FULL stack
        covered across all input plans? button / hotkey / `!!shortcut`
        → main.py prefix-strip → override block → decorticator →
        handler dispatch → workflow → output rendering → memory save.
        Did ALL input plans miss the SAME layer? If so, ADD it now.
      ▸ DATA FLOW — for every NEW state/field/value across input plans:
        created at __ → passed through __ → read at __ → persisted at __.
        Any missing link silently drops the data.
      ▸ CALLERS — for every function-signature change ANY input plan
        proposes: does at least ONE plan enumerate every call site?
        If not, the merged plan must include the call-site sweep.
      ▸ FALLBACKS — for every new mode/flag/state: what's the default
        for users / state that don't have it? Backward-compat preserved?
      ▸ TRIGGER REACHABILITY — for every "if X, do Y" the input plans
        propose: can X actually be reached? Example: "trigger finding
        mode if ≥2 planners signal it" — runs only in extended mode
        (4 planners); standard mode has 1 planner so the threshold is
        unreachable. Catch this here.
      ▸ CROSS-PLAN BLIND SPOTS — if 3 of 4 input plans share the same
        gap (e.g., all cover backend but skip UI binding), that's a
        consensus-blind-spot. CONSENSUS IS NOT EVIDENCE — verify and
        fix.
      ▸ REVIEWER-30-SECOND CATCHES — pretend a sharp reviewer reads
        your merged plan for 30 seconds. What missing piece would they
        catch? Common shapes:
          - "the !!audit button has no backend routing" (UI orphaned)
          - "the new field has no fallback for old state" (compat)
          - "the trigger never fires because the threshold can't be met"
          - "the tool tag gets dropped because === PLAN === masks it"
        If any of those land, the plan isn't done.

The user expects the final code to work the FIRST RUN. The merger is
the last layer that can catch contradictions, vague spots, and
plan-against-plan blind spots before the coder runs. Rubber-stamping
costs the user a re-run.

OUTPUT DISCIPLINE:
  ✓ Plan body is CLEAN: only the concrete final plan (## GOAL,
    ## REQUIREMENTS, ## IMPLEMENTATION STEPS, ...) — no "BEST: Plan #1
    because ..." narrative, no "alternatives considered" sections.
  ✗ NEVER paste a "## BASELINE JUSTIFICATION" or "## DISAGREEMENT
    RESOLUTIONS" or "## COMPLETENESS CHECK" header into the plan body.
    Those are reasoning, not output. The coder needs WHAT, not WHY.

══════════════════════════════════════════════════════════════════════
PRODUCING THE FINAL PLAN — use the PLAN tools
══════════════════════════════════════════════════════════════════════

Your final plan lives in === PLAN === / === PLAN_EDIT === blocks — NOT
in raw prose. [YOUR PLAN] echoes the current draft (line-numbered)
across rounds so you can refine it. Tool tags INSIDE plan blocks are
masked and don't fire — write [CODE: foo.py] inside the plan body
without worrying about accidental execution.

ORDER OF OPERATIONS:
  Investigation phase (optional, ≤1 round):
    OPEN QUESTIONS → ONE [tool use] batch → [STOP][CONFIRM_STOP]
    Use this ONLY for SPECIFIC disagreements among input plans.
  Merge phase:
    Open `=== PLAN ===`, seed it with the best input plan, integrate as
    much as you can in one pass, then close with `=== END PLAN ===`.
  Refinement phase (optional):
    === PLAN_EDIT === blocks to integrate further improvements from
      other input plans, using line numbers from [YOUR PLAN].
  Finalize:
    [PLAN DONE][CONFIRM_PLAN_DONE]

You can mix investigation and refinement across rounds — call a tool
to verify a claim, then a === PLAN_EDIT === to fix the plan based on
what you learned. The plan persists; only finalize when correct AND
complete.

══════════════════════════════════════════════════════════════════════
NO CODE IN THE PLAN — ABSOLUTE
══════════════════════════════════════════════════════════════════════

⚠ The plan describes WHAT to change in plain English. The coder reads
the actual file and writes the code. The plan must NEVER contain:

  ✗ Code blocks (```python ... ``` or ```js ... ```)
  ✗ Function/class bodies — `def foo(...):` followed by an implementation
  ✗ Imports, decorators, multi-line string literals (verbatim prompts)
  ✗ Pseudo-code that LOOKS like real code

  ✓ Backticked names: `process_batch`, `WORKER_POOL_SIZE`
  ✓ Single-line signatures in SHARED INTERFACES
  ✓ File:line citations: "modify aM() at index.html:414"
  ✓ Plain-English description of every change

BAD plan step (REJECTED — too much code):
  ### STEP 3: Implement process_batch
  ```python
  def process_batch(items, context, ...):
      log("...")
      tasks = []
      for worker_id in workers:
          tasks.append(worker(...))
      ...
  ```

GOOD plan step (what the coder needs):
  ### STEP 3: Implement process_batch in module/path.py
  SATISFIES: R3, R4
  FILES: module/path.py (add after the existing helper, near line N)
  WHAT TO DO:
    Define async def process_batch(items, context, options, project_root,
    preloaded_state, shared_cache):
    - Log a "starting batch" step before any work
    - asyncio.gather over the worker pool, each calling `worker(...)`
      with the agreed-upon prompt template
    - For each result, split "RESULT:" lines on "|" (max 4 parts) into
      dicts with the agreed keys; tag each with
      source_worker = worker_id.split("/")[-1]
    - Return (results, shared_cache) when ≥ 2 workers produced output;
      else return (None, shared_cache) as a fallback signal
    - Exceptions from individual workers become warn() + skip

Plans containing code blocks will be REJECTED and the planner above
them will win by default. Compress the code out.

══════════════════════════════════════════════════════════════════════
TOOLS — EXACT FORMAT REQUIRED
══════════════════════════════════════════════════════════════════════

⚠ CRITICAL: The two-tag signal [STOP] + [CONFIRM_STOP] is MANDATORY
after every tool block. Without BOTH halves the runtime does NOT execute
your tools — you'll get no results and your next response will have the
same empty context. A bare [STOP] alone is inert text.

⚠ CRITICAL: Tags outside [tool use]...[/tool use] are IGNORED.
Bare [CODE: file] lines do nothing. Always wrap.

Exact format — write your OPEN QUESTIONS list first, then ONE batch:

  ## OPEN QUESTIONS
  Q1. (the disagreement)
  Q2. (the unverified claim)

  [tool use]
  [REFS: function_name]    ← answers Q1
  [CODE: path/file.py]     ← answers Q2
  [/tool use]
  [STOP]
  [CONFIRM_STOP]

After the system runs your tools, you write the plan. NOT more tools.

Available tools:
  [CODE: path]          read the FULL file — NEVER add line numbers.
                        [CODE: path N-M] is FORBIDDEN and returns nothing.
  [KEEP: path N-M]      AFTER [CODE:] — strips the file to just the lines
                        you need; everything else leaves your context.
                        ⚠ Plan to KEEP narrowly the FIRST time — re-KEEPing
                        the same ranges is a LOOP and will be flagged.
  [REFS: name]          find all definitions, imports, usages of a symbol.
                        Prefer REFS over CODE — it's faster, narrower, and
                        cached across all 4 planners' previous work.
  [SEARCH: pattern]     ripgrep text search across the project
  [DETAIL: section]     look up a section of the code map

CHEAP vs EXPENSIVE — pick the cheapest tool that answers the question:
  REFS / LSP / DETAIL / SEARCH — cheap, cached, narrow.
  KEEP — moderate, but only if you already have the file loaded.
  CODE — EXPENSIVE on a large file (5000+ lines). Use REFS or SEARCH
         first; only fall back to CODE when you genuinely need a region
         REFS can't reach.

MANDATORY WORKFLOW FOR LARGE FILES:
  Step 1 — read the full file:
    [tool use] [CODE: workflows/code.py] [/tool use]
    [STOP]
    [CONFIRM_STOP]
  Step 2 — IMMEDIATELY in the next round, KEEP the lines you actually
    need (decide the ranges BEFORE you ask — no exploring):
    [tool use] [KEEP: workflows/code.py 40-80, 200-250] [/tool use]
    [STOP]
    [CONFIRM_STOP]
  → context now holds only those lines; the rest is gone

  NEVER do [CODE: file.py 100-200]. That is always wrong.
  NEVER re-CODE: a file after KEEPing it (you'd reset all your work).

Use tools only to RESOLVE DISAGREEMENTS between plans — if Plan A says
"function X takes 2 params" and Plan B says "3 params", read the actual
code. Don't re-investigate things the plans already agree on.

══════════════════════════════════════════════════════════════════════
YOUR PROCESS
══════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────
STEP 1 — EVALUATE EACH PLAN AGAINST THE GOAL
──────────────────────────────────────────────────────────────────────

For each plan, check:

  □ GOAL: Does the plan cover the FULL delivery path from origin to
    render? If render is missing, the user sees nothing — the plan
    is incomplete regardless of other qualities.

  □ PRECISION: Can the coder implement each step without guessing?
    Steps that say "update X" without file/function/line = vague.

  □ EVIDENCE: Did the planner verify claims with tools? Line numbers
    from [CODE:] reads are reliable. Claims from memory are not.

  □ CALLERS: For every changed function, did the planner check ALL
    callers with [REFS:]? Missing caller updates = broken code.

Score: GOAL (3x), PRECISION (2x), EVIDENCE (2x), CALLERS (1x).

THE ANTI-CONSENSUS RULE: If 3 plans propose the same approach, that
is NOT 3 confirmations — it's 3 planners making the same assumption.
Judge each plan independently against the actual code.

──────────────────────────────────────────────────────────────────────
STEP 2 — VERIFY DISPUTED CLAIMS
──────────────────────────────────────────────────────────────────────

If plans disagree on a code fact: [REFS:] or [CODE:] it yourself.
Trust the code, not the majority.

──────────────────────────────────────────────────────────────────────
STEP 3 — PICK AND IMPROVE
──────────────────────────────────────────────────────────────────────

Pick the best plan. Then fix:
  - Vague steps → add file/function/line numbers
  - Missing render step → add it
  - Missing caller updates → add them from other plans
  - Unverified claims → verify with tools, correct if wrong

Do NOT add new features. Make the plan CORRECT, not bigger.

──────────────────────────────────────────────────────────────────────
STEP 4 — OUTPUT THE FINAL PLAN
──────────────────────────────────────────────────────────────────────

## GOAL
## REQUIREMENTS
## SHARED INTERFACES
## IMPLEMENTATION STEPS
## EDGE CASES
## VERIFICATION (delivery path trace — origin to render)
## TEST CRITERIA

## PRE-MORTEM RESOLUTION
Revisit each pre-mortem risk from your DEEP THINK section D. For each:
  • "ELIMINATED by Step N — [one-sentence reason]"
  • "MITIGATED by EDGE CASE handler — [where]"
  • "ACCEPTED — out of scope because [reason]"
If you ship a plan with an unresolved pre-mortem risk, you predicted
your own failure. Go back to STEP 3 and improve until each row resolves.

## CONFIDENCE GATE
Rate the final plan 1-10 with one sentence each:
  • CORRECTNESS (satisfies the user's INTENT, not just the surface):  N — [why]
  • PRECISION (coder needs zero clarifying questions):  N — [why]
  • RISK (likelihood pre-mortem fires anyway):  N — [why]
If any rating < 6, the plan is not done. Improve before stopping.


══════════════════════════════════════════════════════════════════════
[USER REQUEST] — the human's actual task (this is what you must serve)
══════════════════════════════════════════════════════════════════════
TASK: {task}
══════════════════════════════════════════════════════════════════════
[END USER REQUEST]
══════════════════════════════════════════════════════════════════════

{context}

{verify_block}

══════════════════════════════════════════════════════════════════════
[INPUT PLANS] — {n_plans} improved plans from Layer 2 you must merge
══════════════════════════════════════════════════════════════════════
Each block below is one improver's final plan. Pick the best as your
baseline (=== PLAN === to seed [YOUR PLAN]), then integrate the
strongest ideas from the others (=== PLAN_EDIT ===). Resolve any
disagreements via targeted tool calls only — these planners ALREADY
investigated; trust their findings unless you can prove otherwise.

{all_plans_text}
══════════════════════════════════════════════════════════════════════
[END INPUT PLANS]
══════════════════════════════════════════════════════════════════════

{preloaded_research}
"""

REVIEW_PROMPT_TEMPLATE = """══════════════════════════════════════════════════════════════════════
[SYSTEM] — your role in the JARVIS pipeline (workflow, not user request)
══════════════════════════════════════════════════════════════════════
This block (until [USER REQUEST]) is JARVIS describing HOW you fit into
the pipeline. The human did NOT write any of it — your loyalty is to
the [USER REQUEST] further down; this just tells you how to serve it.

You are the final reviewer in JARVIS. All step coders have run; their
work is on disk. You write the SMALLEST possible patch that closes
real gaps. You are NOT a rewriter.

You are the LAST defense before the code ships. Every bug you miss,
the user hits — but every line you needlessly rewrite, the user ALSO
hits, because rewrites have a much higher chance of corrupting the
surrounding file than the bug they are trying to fix.

REASONING — in your thinking, not in your visible patch:
  BEFORE you write any fix, in your thinking (reasoning channel or
  `<think>...</think>` tags) walk this checklist for EACH gap you find:
    • DECISION       — the smallest possible fix
    • ALTERNATIVES   — 1-2 other patches you considered (incl. "no fix")
    • WHY THIS ONE   — concrete reason (minimal blast radius, anchor
                       uniqueness, matches existing pattern)
    • SIDE EFFECTS   — what files/functions/state does this fix touch?
                       Are there OTHER call sites that need the same
                       fix and the coder missed?
    • DOWNSTREAM     — does this fix change a signature? If yes, every
                       caller needs an update too.
    • COMPLETENESS   — walk the user's GOAL end-to-end (UI → routing →
                       state → backend → output). Is each link intact
                       after the coder's edits? If a layer is missing
                       wiring, FIX it — that's what review is for.
    • REVIEWER-30-SECOND CATCHES — pretend a sharper reviewer reads
                       your APPROVAL. What would they catch that you
                       glossed over? (UI button with no backend?
                       Trigger that can't fire? Default for old state?)
  Visible output stays clean: APPROVED / fix blocks + brief rationale.

══════════════════════════════════════════════════════════════════════
THINK BEFORE ACTING — STREAMLINED, FLEXIBLE
══════════════════════════════════════════════════════════════════════

You are reading code that ALREADY passed a coder + a self-check. Most
real problems at this stage are CROSS-FILE — callers wired wrong,
shared interfaces drifted, a render step missing. Resist the urge to
re-verify what's already been verified.

Before you call any tool or write any fix, output:

  ## 1. INTEGRATION CHECKLIST (max 5 items, cross-file only)
  Each item is a specific cross-file invariant the goal depends on.
  EXAMPLES:
    "Caller in main.py:391 passes the new field analysis_mode through
     state['classification'] to code_agent."
    "Frontend index.html aM() accepts the new param thinkingTrace."
  Items that only check WITHIN a single function belong to the coder's
  self-check, not here. Don't duplicate that work.

  ## 2. EVIDENCE PLAN
  For each item, name the tool call that proves it. Prefer [REFS:] over
  [CODE:] — you usually need the call-site, not the whole file.

  ## 3. PASS / FAIL CRITERIA
  Write the snippet of text you expect to see for PASS. Anything else
  is FAIL.

If your checklist is empty (no cross-file concerns), the review is
trivially APPROVED. Skip phases B-D and write the decision.

══════════════════════════════════════════════════════════════════════
THE "PARTIAL VIEW" HALLUCINATION TRAP
══════════════════════════════════════════════════════════════════════

[CODE:] always includes a header naming the total line count:

  === Code: core/state.py (66 lines) ===

That number IS AUTHORITATIVE. If the header says 66 lines and you see
66 numbered lines, the file is COMPLETE. Truncations declare themselves
("SKELETON ONLY", "KEPT N/M lines"). Short files are short, not partial.

FORBIDDEN phrases (signatures of this hallucination):
  ✗ "appears to be a partial view"
  ✗ "this can't be the whole file"
  ✗ "the output seems filtered/truncated"
  ✗ "only N lines were returned" (when N matches the header)

A previous run wasted 5 rounds re-reading a 66-line file claiming
"the output only showed 2 lines". Do not be that reviewer.

══════════════════════════════════════════════════════════════════════
REVERT — UNDO A BAD FIX (use without shame)
══════════════════════════════════════════════════════════════════════

If your fix lands and the post-read shows visible corruption (wrong
indent, replaced the wrong block, broke a caller), write:

  [REVERT FILE: path/to/file.py]

before your next [STOP][CONFIRM_STOP]. The runtime restores the
pre-fix snapshot. Then plan the correct fix from the clean state.

Don't layer a second patch on top of a broken first patch. That's how
files get permanently corrupted. REVERT, replan, retry.

══════════════════════════════════════════════════════════════════════
HARD CONSTRAINTS — VIOLATING ANY OF THESE FAILS THE REVIEW
══════════════════════════════════════════════════════════════════════

  1. SEARCH/REPLACE blocks are SURGICAL. Each [SEARCH] block MUST be
     ≤ 12 lines. If you think you need a bigger block, you are wrong:
     find a smaller, MORE UNIQUE anchor inside the region instead.

  2. REPLACE bodies must add/remove ≤ 30 lines per block. If the change
     is bigger, split into multiple small SEARCH/REPLACE blocks each
     touching a separate anchor.

  3. Each fix changes ONE thing. No "while I'm here" cleanups.

  4. A fix touches ONE file per block. Cross-file fixes = multiple blocks.

  5. NEVER rewrite a whole function. NEVER replace an entire `function h(d){{…}}`
     or class body. If a function needs many small changes, write many
     small SEARCH/REPLACE blocks, each anchored on 2-4 unique lines.

  6. NEVER use `=== FILE: path ===` for an existing file. Only for files
     that don't exist yet.

  7. NEVER replace lines you have not READ in THIS round via [CODE:].
     If your last read was 2 rounds ago and another edit landed since,
     re-read before writing the fix.

  8. STOP after at most TWO fix-and-verify rounds. If your fix didn't
     land in 2 rounds, consider [REVERT FILE: path] to restore the
     pre-fix state, then write APPROVED [DONE][CONFIRM_DONE] and let
     the user inspect. The runtime restores the snapshot before final
     review approval, so the user sees the working pre-fix code rather
     than a half-applied corruption.

══════════════════════════════════════════════════════════════════════
CODE FORMAT
══════════════════════════════════════════════════════════════════════

Lines: i{{N}}|{{code}} {{LINE_NUMBER}}. N = leading spaces.
Edits: same i{{N}}| prefix, no trailing line number.

⚠ NEVER carry the trailing integer from the [CODE:] view into your
  REPLACE content. `i4|return x 198` in REPLACE leaves `198` in the
  file and breaks parsing.

⚠ ORPHAN EDIT BLOCKS: every [REPLACE LINES N-M] / [INSERT AFTER LINE N]
  / [DELETE LINE N] block MUST live inside `=== EDIT: <path> === …
  [/REPLACE]`. Wrap explicitly. NEVER use `[/EDIT]` — that closer
  doesn't exist and the parser will keep eating until the next
  `=== EDIT:` boundary, sweeping in unrelated content.

══════════════════════════════════════════════════════════════════════
TOOLS
══════════════════════════════════════════════════════════════════════

  [CODE: path #label]       Read source file (skeleton if too large)
  [VIEW: path LINE #label]  Read ~200 lines centered on LINE in a large
                            file. Auto-extends to enclosing def/class.
                            Use after [CODE:] returns a skeleton.
  [VIEW: path N-M #label]   Same, explicit range. Max 600 lines.
  [KEEP: path N-M #label]   Strip to kept ranges (small/medium files)
  [REFS: name #label]       Definitions, imports, call sites
  [SEARCH: pattern #label]  Ripgrep text search (⚠ NOT edit syntax)

THE TWO-TAG SIGNAL PROTOCOL — write tags inside [tool use]...[/tool use],
then fire the signal on adjacent lines:

  [tool use]
  [CODE: path/file.py]
  [/tool use]
  [STOP]
  [CONFIRM_STOP]

A bare [STOP] alone fires NOTHING — both halves are required. After your
fix lands, [CODE:] the file again to verify, then write:
  [DONE]
  [CONFIRM_DONE]
on adjacent lines to apply and finish.

══════════════════════════════════════════════════════════════════════
YOUR REVIEW — DETERMINISTIC PROCESS
══════════════════════════════════════════════════════════════════════

DISCIPLINE — before EVERY tool round, write a numbered checklist of
what you still don't know. Each tool call cites a checklist item.
DO NOT re-read files in the CONTEXT MANIFEST — re-reads are flagged
with ⛔ and will force-break the loop. If you wrote a fix and want to
verify it landed, ONE re-read of that file is allowed (post-edit), not
more. Banned phrases: "let me check one more thing", "I should verify
one more detail" — these are the verification-loop trap.

────────── PHASE A: READ ──────────

  A1. [CODE:] every changed file ONCE at the start of round 1.
      Use [KEEP:] only for files >400 lines, with the changed regions
      AND 20 lines above + below them. NEVER [KEEP:] only the changed
      lines; you will miss adjacent breakage.

  A2. State the goal as ONE observation:
      "When user does X, they should see Y."

  A3. Make a numbered LIST of things to verify. Pick at most 5.
      Examples:
        1. msg_counter is saved to JSON
        2. msg_counter is loaded from JSON on restart
        3. _on_message uses captured conv_id, not _active_conv
        4. Frontend filters thinking broadcasts by conv_id

────────── PHASE B: VERIFY EACH ITEM ──────────

  For each item N from your list:

    B1. State the EXPECTED code shape (1 sentence).
    B2. Cite the EXACT line you saw in [CODE:] output that proves
        the item is MET or UNMET.
    B3. Mark ✅ MET or ❌ UNMET. No partial credits, no maybes.

  If you cannot cite a line, the item is UNMET.

────────── PHASE C: FIX ONLY UNMET ITEMS ──────────

  For each ❌ UNMET item, write ONE [SEARCH]/[REPLACE] block:

    • [SEARCH] = 2-8 lines that uniquely identify the spot. Include
      a function name or distinctive comment if possible.
    • [REPLACE] = the corrected version. ≤ 30 lines total.
    • Different files = separate `=== EDIT:` headers.

  AFTER all fixes: write the TWO-TAG signal on adjacent lines:
    [STOP]
    [CONFIRM_STOP]
  The runtime applies the edits and gives you the post-edit file via
  the next [CODE:] you request. A bare [STOP] alone fires nothing.

────────── PHASE D: VERIFY THE FIX LANDED ──────────

  Read the post-edit file. For each fix you wrote, QUOTE the line
  where the new code now lives (don't just say "✅" — write the line).
  If it's there → ✅. If it's wrong (visible corruption) → write
  [REVERT FILE: path] before your next signal, then plan again.
  If it's missing → ONE more attempt with a different SEARCH anchor.

  If after 2 attempts a fix still hasn't landed, write
  "REVIEWER UNABLE TO LAND FIX FOR <item>" and proceed.

══════════════════════════════════════════════════════════════════════
DECISION
══════════════════════════════════════════════════════════════════════

  All items ✅ MET (or fixes landed) → APPROVED
                                         [DONE]
                                         [CONFIRM_DONE]
  Any item still UNMET after 2 attempts → write your findings, then:
                                         [DONE]
                                         [CONFIRM_DONE]
  The user can decide whether to ship.

YOU CAN fix: data not flowing through the chain, missing field passes,
broken signature wiring, missing imports, off-by-one, indent
corruption, leftover line-number trailers.

YOU CANNOT: refactor functions for style, rename variables, restructure
control flow, replace whole functions or classes, add features
the user didn't ask for.

══════════════════════════════════════════════════════════════════════
WHAT NOT TO DO — CONCRETE EXAMPLES OF FAILURES TO AVOID
══════════════════════════════════════════════════════════════════════

❌ BAD — replaces 50 lines to add 1 conditional:

    === EDIT: ui/index.html ===
    [SEARCH]
    function h(d){{
    switch(d.type){{
    case'init':
    ...50 lines of unchanged code...
    }}break;
    [/SEARCH]
    [REPLACE]
    function h(d){{
    const cvid=d.conv_id||'';
    switch(d.type){{
    case'init':
    ...50 lines, mostly the same, with one new line per case...
    }}break;
    [/REPLACE]

✅ GOOD — adds the conditional with a 4-line surgical anchor:

    === EDIT: ui/index.html ===
    [SEARCH]
    function h(d){{
    switch(d.type){{
    case'init':
    [/SEARCH]
    [REPLACE]
    function h(d){{
    const cvid=d.conv_id||'';
    switch(d.type){{
    case'init':
    [/REPLACE]

    === EDIT: ui/index.html ===
    [SEARCH]
    case'thinking_start':{{
    thinkId++;
    [/SEARCH]
    [REPLACE]
    case'thinking_start':
    if(cvid&&cvid!==activeConvId)break;
    {{
    thinkId++;
    [/REPLACE]

The good version: 4 small blocks, each ≤ 8 lines, each does ONE thing.
The bad version: 1 huge block that the fuzzy matcher can mis-locate
and the file's HTML scaffolding can get ripped out.


═══════════════════════════════════════════════════════════════════════
CONTEXT
═══════════════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════════════
[USER REQUEST] — the human's actual task (this is what you must serve)
══════════════════════════════════════════════════════════════════════
TASK: {task}
══════════════════════════════════════════════════════════════════════
[END USER REQUEST] — everything below is JARVIS framing / facts / context
══════════════════════════════════════════════════════════════════════

PLAN: {plan}

CHANGED FILES: {all_files_block}

PROJECT: {context}

{preloaded_research}
"""
SUMMARY_PROMPT = """You implemented changes to achieve a goal. Summarize for the user.

══════════════════════════════════════════════════════════════════════
[USER REQUEST] — the human's actual task (this is what you must serve)
══════════════════════════════════════════════════════════════════════
TASK: {task}
══════════════════════════════════════════════════════════════════════
[END USER REQUEST] — everything below is JARVIS framing / facts / context
══════════════════════════════════════════════════════════════════════

FILES CHANGED:
{files_changed}

DIFF:
{diff}

Write a clear summary:
1. What the user can now do that they couldn't before (the goal achieved)
2. What files were created or modified (brief, not line-by-line)
3. Anything the user needs to know (new dependencies, config changes, etc.)

Keep it short. The user wants to understand what changed, not read the code.
No code in the summary.
"""

MAP_UPDATE_PROMPT = """You implemented code changes. Update the project's code maps.

DO NOT rewrite the maps. Output ONLY edit blocks for the parts that changed.

══════════════════════════════════════════════════════════════════════
[USER REQUEST] — the human's actual task (this is what you must serve)
══════════════════════════════════════════════════════════════════════
TASK: {task}
══════════════════════════════════════════════════════════════════════
[END USER REQUEST] — everything below is JARVIS framing / facts / context
══════════════════════════════════════════════════════════════════════

FILES CHANGED:
{files_changed}

DIFF:
{diff}

CURRENT GENERAL MAP:
{general_map}

CURRENT DETAILED MAP:
{detailed_map}

OUTPUT FORMAT:

=== GENERAL MAP EDITS ===
[SEARCH]
exact text from current general map
[/SEARCH]
[REPLACE]
updated text
[/REPLACE]

[ADD_SECTION]
## New Feature Name
description
[/ADD_SECTION]

=== DETAILED MAP EDITS ===
(same format)

RULES:
- SEARCH text must match the current map EXACTLY
- Only edit what the diff actually changed
- Empty REPLACE = delete the matched text
- ADD_SECTION = append to end of that map
- If a map doesn't need changes: "GENERAL: no changes" or "DETAILED: no changes"
"""


def _apply_map_edits(original_map: str, response_text: str) -> str:
    """Parse [SEARCH]/[REPLACE] and [ADD_SECTION] blocks from response and
    apply them to the original map. Empty [REPLACE] deletes."""
    result = original_map

    # Find SEARCH/REPLACE pairs
    edit_pattern = re.compile(
        r'\[SEARCH\](.*?)\[/SEARCH\]\s*\[REPLACE\](.*?)\[/REPLACE\]',
        re.DOTALL,
    )
    for match in edit_pattern.finditer(response_text):
        find_text = match.group(1).strip()
        replace_text = match.group(2).strip()
        if not find_text:
            continue
        if find_text in result:
            # Refuse to apply if the exact match is ambiguous — silently
            # clobbering the first occurrence used to corrupt the map when
            # a section heading appeared more than once.
            if result.count(find_text) > 1:
                warn(
                    f"    Skipping ambiguous map edit — SEARCH matches "
                    f"{result.count(find_text)} locations"
                )
                continue
            result = result.replace(find_text, replace_text, 1)
        else:
            # Fuzzy: whitespace-normalized line match. Collect ALL matches
            # so we can detect ambiguity and refuse to clobber.
            find_lines = [l.strip() for l in find_text.split('\n')]
            result_lines = result.split('\n')
            matches = []
            for i in range(len(result_lines) - len(find_lines) + 1):
                window = [result_lines[i + j].strip() for j in range(len(find_lines))]
                if window == find_lines:
                    matches.append(i)
            if len(matches) == 1:
                i = matches[0]
                if replace_text:
                    result_lines[i:i + len(find_lines)] = replace_text.split('\n')
                else:
                    del result_lines[i:i + len(find_lines)]
                result = '\n'.join(result_lines)
            elif len(matches) > 1:
                warn(
                    f"    Skipping ambiguous fuzzy map edit — "
                    f"{len(matches)} locations match (normalized)"
                )

    # Find ADD_SECTION blocks — append to end
    add_pattern = re.compile(r'\[ADD_SECTION\](.*?)\[/ADD_SECTION\]', re.DOTALL)
    for match in add_pattern.finditer(response_text):
        addition = match.group(1).strip()
        if addition:
            result += "\n\n" + addition

    return result


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _format_research_cache(research_cache: dict | None, max_chars: int = 30000) -> str:
    """Format the shared research cache into a readable section for prompt injection.

    This lets downstream AIs (coders, reviewers) see everything that upstream
    AIs (planners) already looked up, so they don't need to re-search.
    Results are deduplicated by key — identical lookups only appear once.
    """
    if not research_cache:
        return ""

    parts = []
    total = 0
    # Iterate over a stable list so the truncated-count math is correct
    # (research_cache is shared and may be mutated by parallel runners).
    cache_items = list(research_cache.items())
    for idx, (key, value) in enumerate(cache_items):
        value = value.strip()
        if not value:
            continue
        # key format is "TAG_TYPE:query" e.g. "REFS:call_with_tools"
        entry = f"\n{value}"
        if total + len(entry) > max_chars:
            remaining = len(cache_items) - idx
            parts.append(
                f"\n... ({remaining} more cached lookup"
                f"{'s' if remaining != 1 else ''} truncated)"
            )
            break
        parts.append(entry)
        total += len(entry)

    if not parts:
        return ""

    return (
        "\n\n══════════════════════════════════════════════════════════════\n"
        "PRE-LOADED RESEARCH (from earlier pipeline stages — do NOT re-search these):\n"
        "The planning AIs already looked these up. Use this data directly.\n"
        "If you need something NOT listed here, you can still use tool tags.\n"
        "══════════════════════════════════════════════════════════════\n"
        + "\n".join(parts)
        + "\n══════════════════════════════════════════════════════════════\n"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  KEEP SYSTEM — Subtractive code selection + Auto-RAG
#
#  KEEP is a tool tag processed inside the tool call loop (tool_call.py).
#  When a model reads a large file with [CODE: path], it gets a hint to use
#  [KEEP: path X-Y, A-B] to strip irrelevant lines. The KEEP handler:
#    1. Parses the ranges
#    2. Builds a filtered view (line numbers preserved)
#    3. Runs auto-RAG on kept lines (REFS on all identifiers)
#    4. REPLACES the CODE entry in persistent_lookups — the full file is
#       literally gone from context, only the kept ranges remain.
#
#  The functions below (_parse_keep_ranges, _filter_by_ranges, _auto_rag)
#  are called by _run_keep() in tool_call.py.
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_keep_ranges(text: str, filepath: str) -> list[tuple[int, int]]:
    """Parse KEEP line ranges from model output. Returns sorted ranges.

    Accepts all of these forms (multiple ranges comma- or space-separated):
      [KEEP: path 50-80, 120-150]   →  [(50,80), (120,150)]
      [KEEP: path 50-80 120-150]    →  [(50,80), (120,150)]
      KEEP path 50-80, 120-150      →  [(50,80), (120,150)]
      50-80, 120-150                →  [(50,80), (120,150)]  (bare ranges)

    OVERLAP HANDLING: ranges that overlap (e.g. 50-80 and 75-100) ARE merged
    because keeping them separate would emit duplicate file lines. ADJACENT
    ranges (e.g. 50-80 and 81-100) are also merged because the gap is zero.
    BUT ranges with a gap (e.g. 50-80 and 85-100, 4-line gap) are NOT merged
    anymore — the model asked for two distinct windows, the runtime delivers
    two distinct windows. The previous +4 tolerance silently expanded the
    request, which then surprised the model when `_filter_by_ranges` returned
    code the model never asked to see and `_auto_rag` chased identifiers
    from lines outside the request.
    """
    ranges = []
    # Universal: find ALL N-M patterns anywhere in text (covers every format).
    # The filepath and KEEP keyword are stripped beforehand by _run_keep, so
    # the text passed here is often just the ranges portion already.
    bare_range = re.compile(r'(\d+)\s*-\s*(\d+)')
    for m in bare_range.finditer(text):
        start, end = int(m.group(1)), int(m.group(2))
        if start > 0 and end >= start:
            pair = (start, end)
            if pair not in ranges:
                ranges.append(pair)

    if not ranges:
        return []

    # Sort and merge ONLY truly overlapping/adjacent ranges (gap ≤ 0).
    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:  # overlap or adjacent — merge
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _extend_ranges_to_scope_anchor(
    ranges: list[tuple[int, int]], lines: list[str]
) -> list[tuple[int, int]]:
    """Extend each range upward to the nearest enclosing def/class at column 0.

    Without this, a model doing [KEEP: file 143-165] won't see the function
    definition that owns those lines. It then can't know the base indentation
    level, and writes fixes with wrong absolute indentation.

    We walk backward from range_start until we find a non-empty, non-decorator
    line at column 0 that starts a scope (def/class/async def). That line is
    included in the range so the model always has an indentation anchor.
    If no such line exists above (e.g. top-level code), the range is unchanged.
    """
    _SCOPE_RE = re.compile(r'^(def |class |async def )')
    extended = []
    for start, end in ranges:
        # Walk backward from start (0-based index = start-2)
        anchor = start  # 1-based; stays at start if nothing found
        for i in range(start - 2, -1, -1):  # 0-based, going up
            line = lines[i]
            if not line.strip():
                continue  # skip blank lines
            if line[0] != ' ' and line[0] != '\t':
                # Column-0 non-empty line
                if _SCOPE_RE.match(line) or line.startswith('@'):
                    # Decorator — keep going up to find the def/class
                    if line.startswith('@'):
                        continue
                    anchor = i + 1  # 1-based
                break
        extended.append((min(anchor, start), end))
    return extended


def _filter_by_ranges(content: str, ranges: list[tuple[int, int]], filepath: str) -> str:
    """Build a filtered view of a file showing only the kept ranges.

    Line numbers are PRESERVED — [REPLACE LINES] still works on the result.
    Hidden sections are marked with "(lines X-Y hidden)".

    Each range is automatically extended upward to the nearest enclosing
    def/class at column 0 so the model always has an indentation anchor.
    """
    lines = content.split('\n')
    total = len(lines)

    # Extend ranges to scope anchors, then re-sort and re-merge.
    # Use the same gap≤0 merge rule as _parse_keep_ranges so the runtime
    # only collapses ranges that would otherwise emit duplicate file lines.
    ranges = _extend_ranges_to_scope_anchor(ranges, lines)
    ranges.sort()
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    ranges = merged

    output_parts = []
    width = len(str(total))
    prev_end = 0  # 1-based, last line we showed

    for range_start, range_end in ranges:
        # Clamp to file bounds
        range_start = max(1, range_start)
        range_end = min(total, range_end)

        # Show hidden marker for gap
        if range_start > prev_end + 1:
            gap_start = prev_end + 1
            gap_end = range_start - 1
            hidden_count = gap_end - gap_start + 1
            output_parts.append(
                f"{'·' * 40} ({hidden_count} lines hidden: {gap_start}-{gap_end})"
            )

        # Show kept lines with explicit indent prefix and trailing line number.
        # New format: i{N}|{stripped_code} {lineno}
        #   i0|def foo(): 1
        #   i4|return 42 2
        #   i12|self.x = 0 6
        # The i{N}| prefix REPLACES the leading whitespace and tells the
        # model the exact indent depth as a number — no counting required.
        # The trailing space + line number is the cursor anchor for SEARCH.
        # Blank lines: i0| {lineno}  (just the prefix and number).
        # When the model writes an edit, it uses the same i{N}| prefix and
        # the engine re-emits N spaces. There is no possible off-by-N error
        # because the model emits a number, not characters.
        for i in range(range_start - 1, range_end):  # 0-based indexing
            line = lines[i]
            stripped_left = line.lstrip(' \t')
            lead = line[:len(line) - len(stripped_left)]
            indent_cols = 0
            for ch in lead:
                if ch == '\t':
                    indent_cols += 4                # match expandtabs(4)
                else:
                    indent_cols += 1
            # Unified format: i{N}|{code} {lineno}. Blank lines → "i0| {n}"
            output_parts.append(
                f"i{indent_cols}|{stripped_left} {i + 1}"
            )

        prev_end = range_end

    # Trailing hidden marker
    if prev_end < total:
        gap_start = prev_end + 1
        hidden_count = total - prev_end
        output_parts.append(
            f"{'·' * 40} ({hidden_count} lines hidden: {gap_start}-{total})"
        )

    return '\n'.join(output_parts)


async def _auto_rag(
    kept_content: str, filepath: str, project_root: str,
    research_cache: dict | None = None,
) -> str:
    """Extract identifiers from kept code and run REFS on each.

    Scans for function calls, class references, and local imports.
    Returns a dependency summary.
    """
    import ast

    ext = os.path.splitext(filepath)[1].lower()
    if ext != ".py":
        # For non-Python, do basic identifier extraction via regex
        # Match function calls: word( but not keywords
        calls = set(re.findall(r'(?<!\w)([a-zA-Z_]\w*)\s*\(', kept_content))
        calls -= {'if', 'for', 'while', 'return', 'print', 'def', 'class',
                  'with', 'async', 'await', 'import', 'from', 'try', 'except',
                  'raise', 'assert', 'not', 'and', 'or', 'in', 'is', 'True',
                  'False', 'None', 'len', 'str', 'int', 'float', 'list', 'dict',
                  'set', 'tuple', 'bool', 'type', 'isinstance', 'range', 'super',
                  'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
                  'any', 'all', 'min', 'max', 'sum', 'abs', 'open', 'hasattr',
                  'getattr', 'setattr'}
        identifiers = list(calls)[:15]  # cap at 15
    else:
        # Parse Python AST for precise extraction.
        # Strip line numbers from kept content for parsing. We must handle
        # THREE display formats because the [CODE:]/[KEEP:] view has
        # changed over time and old `research_cache` entries can still
        # use any of them:
        #   (new)     iN|{code} {lineno}           e.g. "i4|return 42 17"
        #             iN|{code}                    (no trailer)
        #   (legacy1) {N spaces}{lineno}\t{code}   e.g. "  42\treturn 42"
        #   (legacy2) {code}  │{lineno}            e.g. "    return 42  │42"
        # Skeleton/hidden-region markers (`·····` / `(... hidden ...)`)
        # are replaced with blank lines so ast.parse doesn't choke.
        _new_format = re.compile(r'^i(\d+)\|(.*?)(?:\s+\d+)?\s*$')
        _legacy_prefix = re.compile(r'^\s*\d+\t(.*)$')
        _legacy_suffix = re.compile(r'^(.*?)\s*│\s*\d+\s*$')
        clean_lines = []
        for line in kept_content.split('\n'):
            if not line.strip():
                clean_lines.append('')
                continue
            if line.startswith('·'):
                # hidden-region marker emitted by _filter_by_ranges
                clean_lines.append('')
                continue
            m = _new_format.match(line)
            if m:
                indent = int(m.group(1))
                code = m.group(2)
                clean_lines.append(' ' * indent + code)
                continue
            m = _legacy_prefix.match(line)
            if m:
                clean_lines.append(m.group(1))
                continue
            m = _legacy_suffix.match(line)
            if m:
                clean_lines.append(m.group(1).rstrip())
                continue
            clean_lines.append(line)
        clean_code = '\n'.join(clean_lines)

        identifiers = set()
        try:
            tree = ast.parse(clean_code, filename=filepath)
            for node in ast.walk(tree):
                # Function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        identifiers.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        identifiers.add(node.func.attr)
                # Imports (local only)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top_pkg = node.module.split('.')[0]
                        pkg_path = os.path.join(project_root, top_pkg)
                        if os.path.exists(pkg_path + '.py') or os.path.isdir(pkg_path):
                            for alias in (node.names or []):
                                identifiers.add(alias.name)
                # Class bases
                elif isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            identifiers.add(base.id)
        except SyntaxError:
            # Partial code won't parse — fall back to regex
            calls = set(re.findall(r'(?<!\w)([a-zA-Z_]\w*)\s*\(', clean_code))
            identifiers = calls

        # Filter out builtins and common names
        identifiers -= {'if', 'for', 'while', 'return', 'print', 'def', 'class',
                       'self', 'cls', 'None', 'True', 'False', 'str', 'int',
                       'float', 'list', 'dict', 'set', 'tuple', 'bool', 'len',
                       'range', 'super', 'isinstance', 'type', 'enumerate',
                       'zip', 'map', 'filter', 'sorted', 'reversed', 'any',
                       'all', 'min', 'max', 'sum', 'abs', 'open', 'hasattr',
                       'getattr', 'setattr', 'Exception', 'ValueError',
                       'TypeError', 'KeyError', 'RuntimeError', 'StopIteration',
                       'property', 'staticmethod', 'classmethod', 'asyncio',
                       'os', 're', 'json', 'sys', 'pathlib', 'subprocess',
                       'append', 'extend', 'update', 'get', 'items', 'keys',
                       'values', 'strip', 'split', 'join', 'replace', 'format',
                       'startswith', 'endswith', 'lower', 'upper'}
        identifiers = list(identifiers)[:15]

    if not identifiers:
        return ""

    # Run REFS on each identifier
    from tools.codebase import search_refs
    dep_parts = [f"\nDependencies found in kept code of {filepath}:"]

    for name in sorted(identifiers):
        # Check cache first
        cache_key = f"REFS:{name.strip().lower()}"
        if research_cache and cache_key in research_cache:
            # Already have it — extract just the summary line
            cached = research_cache[cache_key]
            # Get the first meaningful line
            for line in cached.split('\n'):
                if name in line and ('defined' in line.lower() or ':' in line):
                    dep_parts.append(f"  {line.strip()}")
                    break
            continue

        result = search_refs(name, project_root, max_results=5)
        if research_cache is not None:
            research_cache[cache_key] = result

        # Condense to a one-line summary
        locations = []
        for line in result.split('\n'):
            # Look for "file.py:123:" patterns
            m = re.match(r'\s*(.+?):(\d+):', line)
            if m:
                loc_file = m.group(1)
                loc_line = m.group(2)
                # Skip self-references
                if loc_file != filepath:
                    locations.append(f"{loc_file}:{loc_line}")

        if locations:
            dep_parts.append(f"  {name} → {', '.join(locations[:5])}")

    if len(dep_parts) <= 1:
        return ""  # no external dependencies found

    return '\n'.join(dep_parts)



def _check_syntax(filepath: str, content: str) -> tuple[bool, str]:
    """Run a syntax check on code content based on file extension.

    Returns (passed: bool, error_message: str).
    If no checker is available for the file type, returns (True, "").
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".py":
        import tokenize, io

        def _make_error(lineno, col, msg, lines, kind="SyntaxError") -> str:
            """Format an error with 10 lines of context centred on the real line."""
            context_lines = []
            if isinstance(lineno, int) and lineno > 0:
                start = max(0, lineno - 6)
                end = min(len(lines), lineno + 4)
                for i in range(start, end):
                    marker = ">>>" if i == lineno - 1 else "   "
                    context_lines.append(f"  {marker} {i + 1}: {lines[i]}")
            col_str = f", col {col}" if col else ""
            return (
                f"Python {kind} at line {lineno}{col_str}: {msg}\n"
                + "\n".join(context_lines)
            )

        lines = content.split("\n")

        # ── Step 1: tokenize catches IndentationErrors at the REAL line.
        # compile() reports them at a later line where the parser gives up,
        # which sends the verifier to the wrong place. tokenize is accurate.
        try:
            list(tokenize.generate_tokens(io.StringIO(content).readline))
        except tokenize.TokenError:
            pass  # incomplete input — not a real error, let compile() decide
        except IndentationError as e:
            return False, _make_error(e.lineno, e.offset, e.msg, lines, "IndentationError")

        # ── Step 2: compile() catches all other grammar-level syntax errors.
        try:
            compile(content, filepath, "exec")
            return True, ""
        except SyntaxError as e:
            kind = type(e).__name__  # SyntaxError or IndentationError subclass
            return False, _make_error(e.lineno, e.offset, e.msg, lines, kind)

    elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
        # Use Node.js --check for JS/TS syntax validation
        import tempfile
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=ext, delete=False, encoding="utf-8"
            ) as f:
                f.write(content)
                tmp_path = f.name

            # Try node --check (works for .js/.mjs/.cjs)
            if ext in (".js", ".mjs", ".cjs"):
                result = subprocess.run(
                    ["node", "--check", tmp_path],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode != 0:
                    err = result.stderr.strip()
                    # Clean up the temp path from the error message
                    err = err.replace(tmp_path, filepath)
                    return False, f"JavaScript syntax error:\n{err}"

            return True, ""
        except FileNotFoundError:
            # node not installed — skip
            return True, ""
        except subprocess.TimeoutExpired:
            return True, ""
        except Exception:
            return True, ""
        finally:
            # Guard against tempfile creation failing before tmp_path is set.
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    elif ext in (".json",):
        import json
        try:
            json.loads(content)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"JSON syntax error at line {e.lineno}, col {e.colno}: {e.msg}"

    # HTML and CSS: skip — naive tag/brace counting produces too many
    # false positives on complex files (scripts inside HTML, template
    # literals, minified code) and wastes rounds in fix loops.

    # No checker for this file type
    return True, ""



async def _call(model: str, prompt: str, max_tokens: int = 16384, log_label: str = "") -> dict:
    result = await call_with_retry(model, prompt, max_tokens=max_tokens, log_label=log_label)
    return {"model": model, "answer": result}


# _call_with_tools imported from core.tool_call (shared with chat + research)


_REVISE_EDIT_RE = re.compile(
    r'===\s*REVISE\s+EDIT:\s*(\S+)[^\n=]*===?\s*\n?(.*?)\n?===\s*END\s+REVISE\s+EDIT\s*===',
    re.DOTALL | re.IGNORECASE,
)
_EDIT_OPEN_RE = re.compile(
    r'===\s*EDIT:\s*(\S+)[^\n=]*===?\s*\n', re.IGNORECASE,
)
_SECTION_BOUNDARY_RE = re.compile(
    r'===\s*(?:EDIT|FILE|REVISE\s+EDIT):|===\s*END\s+FILE\s*===',
    re.IGNORECASE,
)


def _apply_revise_edits(response: str) -> str:
    """Resolve `=== REVISE EDIT: path === ... === END REVISE EDIT ===` blocks.

    A REVISE block lets the coder retract+replace its most recent pending
    edit on `path` BEFORE [STOP][CONFIRM_STOP] applies anything. Semantics:

      1. Find the most recent `=== EDIT: path ===` block earlier in the
         response on the same path.
      2. Remove that prior EDIT block from the response (so it never
         reaches the edit extractor).
      3. Rewrite the REVISE block as a normal `=== EDIT: path ===` block
         so the existing extractor picks up its SEARCH/REPLACE body.

    If no prior EDIT on that path exists, the REVISE is still treated as a
    plain EDIT — the coder may have intended to write one and skipped the
    initial draft. Idempotent: passing already-normalized text is a no-op.
    """
    while True:
        m = _REVISE_EDIT_RE.search(response)
        if not m:
            return response
        path = m.group(1).strip()
        body = m.group(2)
        r_start, r_end = m.span()

        # Locate the most recent `=== EDIT: path ===` opener earlier in
        # the response on the same path.
        prior_openers = [
            em for em in _EDIT_OPEN_RE.finditer(response[:r_start])
            if em.group(1).strip() == path
        ]
        if prior_openers:
            opener = prior_openers[-1]
            edit_start = opener.start()
            # The prior EDIT's body ends at the next === section boundary
            # (or right before this REVISE).
            tail = response[opener.end():r_start]
            nxt = _SECTION_BOUNDARY_RE.search(tail)
            edit_end = opener.end() + nxt.start() if nxt else r_start
            # Remove the prior EDIT block; shift the REVISE span.
            removed = edit_end - edit_start
            response = response[:edit_start] + response[edit_end:]
            r_start -= removed
            r_end -= removed

        # Rewrite the REVISE block as a regular EDIT block.
        new_edit = f"\n=== EDIT: {path} ===\n{body}\n"
        response = response[:r_start] + new_edit + response[r_end:]


def _extract_code_blocks(response: str) -> dict:
    """
    Extract edits and new files from AI response.
    Primary format: [SEARCH]...[/SEARCH] [REPLACE]...[/REPLACE] (text matching)
    Fallback: [REPLACE LINES start-end]...[/REPLACE] (line-number based)

    Returns {
        "edits": {filepath: [(start, end, code), ...]},  # line-number edits
        "text_edits": {filepath: [(search, replace), ...]},  # fallback text edits
        "new_files": {filepath: content},
    }
    """
    # Resolve any `=== REVISE EDIT: path === ... === END REVISE EDIT ===`
    # blocks first. Each REVISE block discards the most recent prior EDIT
    # on the same path and is itself promoted to a regular EDIT block,
    # so the rest of the extraction sees a clean linear sequence.
    response = _apply_revise_edits(response)

    result = {"edits": {}, "text_edits": {}, "new_files": {}, "reverts": []}

    # ── Pre-compute "consumed" spans (inside === FILE: or === EDIT: bodies) ─
    # REVERT directives, orphan line-edits, and any other top-level extraction
    # must skip text that lives inside an edit/new-file body — those are file
    # CONTENT, not directives. Without this, a new file whose body contains
    # the literal string `[REVERT FILE: foo]` (e.g. JARVIS source rewriting
    # its own prompts) triggers a spurious revert on `foo`.
    _early_consumed_spans: list[tuple[int, int]] = []
    for m in re.finditer(
        r'===\s*EDIT:\s*(\S+).*?(?:\[/REPLACE\]|\[/INSERT\]|===\s*END\s+FILE\s*===)',
        response, re.DOTALL,
    ):
        _early_consumed_spans.append(m.span())
    for m in re.finditer(
        r'===\s*FILE:\s*(\S+).*?===\s*END\s+FILE\s*===',
        response, re.DOTALL,
    ):
        _early_consumed_spans.append(m.span())

    def _early_in_consumed(pos: int) -> bool:
        return any(s <= pos < e for s, e in _early_consumed_spans)

    # ── Extract REVERT directives ─────────────────────────────────────────
    # Single-line directive: `[REVERT FILE: path]` — restores filepath to its
    # state immediately before the most recent edit applied to it in this
    # session. The model can use this mid-response if it realizes its edits
    # are wrong, then write fresh edits in the same response.
    #
    # Skip occurrences inside `=== FILE: ===` and `=== EDIT: ===` bodies —
    # those are file CONTENT, not directives.
    for revert_match in re.finditer(
        r'\[REVERT\s+FILE:\s*([^\]\n]+?)\s*\]',
        response
    ):
        if _early_in_consumed(revert_match.start()):
            continue
        rpath = revert_match.group(1).strip()
        if rpath:
            result["reverts"].append(rpath)

    # ── Extract EDIT blocks ──────────────────────────────────────────────
    # Accepts both "=== EDIT: path ===" and "=== EDIT: path" (no closing ===).
    # An EDIT body ends at the next "=== EDIT:" or "=== FILE:" header — OR
    # at "=== END FILE ===" if a preceding FILE block forgot its EDIT close.
    # Without END FILE in the boundary, a malformed response can let an EDIT
    # block run past the next FILE terminator and consume unrelated text.
    edit_pattern = re.compile(
        r'===\s*EDIT:\s*(\S+).*?\n(.*?)'
        r'(?====\s*(?:EDIT|FILE):|===\s*END\s+FILE\s*===|$)',
        re.DOTALL
    )
    for edit_match in edit_pattern.finditer(response):
        filepath = edit_match.group(1).strip()
        edit_body = edit_match.group(2)

        # ── Format 1 (primary): [SEARCH]...[/SEARCH] [REPLACE]...[/REPLACE] ──
        pairs = []
        xml_pairs = re.findall(
            r'\[SEARCH\][ \t]*\r?\n?(.*?)[ \t]*\r?\n?\[/SEARCH\][ \t]*\r?\n?\s*\[REPLACE\][ \t]*\r?\n?(.*?)[ \t]*\r?\n?\[/REPLACE\]',
            edit_body, re.DOTALL
        )
        if xml_pairs:
            pairs.extend(xml_pairs)

        # ── Format 1a (anchored): [SEARCH: 45-49]...[/SEARCH] [REPLACE]...[/REPLACE] ──
        # The line range is embedded in the SEARCH tag. This eliminates ambiguity
        # completely: the matcher uses it as a location anchor instead of scanning
        # the whole file. Use this when the plain SEARCH block is not unique enough.
        #
        # Format:
        #   [SEARCH: 45-49]
        #   exact code lines
        #   [/SEARCH]
        #   [REPLACE]          ← plain, OR [REPLACE: 45-52] — range is ignored
        #   new code
        #   [/REPLACE]
        #
        # The line range is injected as a synthetic header comment so _strip_line_numbers
        # extracts it as hint_line, pointing strategy 2 at the right location.
        anchored_raw = re.findall(
            r'\[SEARCH:\s*(\d+)\s*-\s*(\d+)\s*\][ \t]*\r?\n?(.*?)[ \t]*\r?\n?\[/SEARCH\][ \t]*\r?\n?\s*\[REPLACE(?::\s*\d+\s*-\s*\d+\s*)?\][ \t]*\r?\n?(.*?)[ \t]*\r?\n?\[/REPLACE\]',
            edit_body, re.DOTALL
        )
        for start_s, _end_s, find_text, replace_text in anchored_raw:
            # Prepend a synthetic "@line N" marker that _strip_line_numbers will
            # extract as hint_line. We use the start of the range as the anchor.
            hint_prefix = f"@line {start_s}\n"
            pairs.append((hint_prefix + find_text, replace_text))

        # Alt anchored syntax: <<<SEARCH: 45-49>>> ... <<<REPLACE>>> ... <<<END>>>
        anchored_alt = re.findall(
            r'<<<SEARCH:\s*(\d+)\s*-\s*(\d+)>>>[ \t]*\r?\n?(.*?)[ \t]*\r?\n?<<<REPLACE(?::\s*\d+\s*-\s*\d+\s*)?>>>[ \t]*\r?\n?(.*?)[ \t]*\r?\n?<<<END>>>',
            edit_body, re.DOTALL
        )
        for start_s, _end_s, find_text, replace_text in anchored_alt:
            hint_prefix = f"@line {start_s}\n"
            pairs.append((hint_prefix + find_text, replace_text))

        # ── Format 1b (alternate): <<<SEARCH>>> ... <<<REPLACE>>> ... <<<END>>> ──
        # Use this when the file being edited contains literal [SEARCH] or [/SEARCH]
        # text — e.g. when fixing a broken edit that left delimiter tags in the file.
        # The two syntaxes are mutually exclusive by design so one is always usable.
        alt_pairs = re.findall(
            r'<<<SEARCH>>>[ \t]*\r?\n?(.*?)[ \t]*\r?\n?<<<REPLACE>>>[ \t]*\r?\n?(.*?)[ \t]*\r?\n?<<<END>>>',
            edit_body, re.DOTALL
        )
        if alt_pairs:
            pairs.extend(alt_pairs)

        # Also accept <<<< FIND ... >>>> <<<< REPLACE ... >>>>
        if not pairs:
            old_pairs = re.findall(
                r'<<<<\s*FIND\s*\n?(.*?)\s*>>>>\s*\n?\s*<<<<\s*REPLACE\s*\n?(.*?)\s*>>>>',
                edit_body, re.DOTALL
            )
            if old_pairs:
                pairs.extend(old_pairs)

        if pairs:
            result["text_edits"].setdefault(filepath, []).extend(pairs)
            # Fall through to also pick up [REPLACE LINES] / [INSERT AFTER]
            # blocks in the same EDIT — the model is allowed to mix formats.

        # ── Format 2 (fallback): [REPLACE LINES start-end]...[/REPLACE] ──
        # Pattern note: `[ \t]*\r?\n?` consumes only horizontal whitespace
        # plus the line terminator after `]` and before `[/REPLACE]`. Using
        # `\s*\n?` would greedily eat the FIRST content line's leading indent
        # (because `\s*` matches newlines), making `_reindent_replace` see
        # rep_indent=0 and shift the entire block by the wrong delta.
        line_edits = re.findall(
            r'\[REPLACE\s+LINES?\s+(\d+)\s*-\s*(\d+)\s*\][ \t]*\r?\n?(.*?)[ \t]*\r?\n?\[/REPLACE\]',
            edit_body, re.DOTALL
        )

        # ── Format 2b: [INSERT AFTER LINE X]...[/INSERT] ──
        insert_edits = re.findall(
            r'\[INSERT\s+AFTER\s+LINE\s+(\d+)\s*\][ \t]*\r?\n?(.*?)[ \t]*\r?\n?\[/INSERT\]',
            edit_body, re.DOTALL
        )

        # ── Format 2c: [DELETE LINES start-end] ──
        delete_edits = re.findall(
            r'\[DELETE\s+LINES?\s+(\d+)\s*-\s*(\d+)\s*\]',
            edit_body
        )
        delete_single = re.findall(
            r'\[DELETE\s+LINE\s+(\d+)\s*\]',
            edit_body
        )

        if line_edits or insert_edits or delete_edits or delete_single:
            parsed = []
            for start_s, end_s, code in line_edits:
                parsed.append((int(start_s), int(end_s), code))
            for line_s, code in insert_edits:
                parsed.append((0, int(line_s), code))
            for start_s, end_s in delete_edits:
                parsed.append((int(start_s), int(end_s), ""))
            for line_s in delete_single:
                parsed.append((int(line_s), int(line_s), ""))
            result["edits"].setdefault(filepath, []).extend(parsed)
            continue

    # ── Orphan [REPLACE LINES] / [INSERT AFTER LINE] / [DELETE LINE] ─────
    # The reviewer / coder sometimes writes a line-number edit WITHOUT the
    # `=== EDIT: <path> ===` wrapper. That used to silently drop the edit,
    # leaving the model to retry endlessly without progress.
    #
    # When we see an orphan block, attach it to the most recently mentioned
    # file in the response — in priority order:
    #   1. The most recent `=== EDIT: <path> ===` block (if the orphan
    #      appears after one — covers "I forgot to wrap").
    #   2. The most recent `[CODE: <path>]` tag (covers "I just read a file
    #      and want to fix it").
    # If neither exists, the orphan is dropped (no file context = unsafe).
    # Use the SAME regex as the main extraction above so consumed_spans
    # line up exactly with the EDIT blocks that produced text_edits / edits.
    # The old variant inserted an extra `.*?` which made the span slightly
    # shorter and let orphan-rescue mis-classify edits that genuinely
    # belonged to a wrapped block.
    consumed_spans: list[tuple[int, int]] = []
    for edit_match in re.finditer(
        r'===\s*EDIT:\s*(\S+).*?\n(?:.*?)'
        r'(?====\s*(?:EDIT|FILE):|===\s*END\s+FILE\s*===|$)',
        response, re.DOTALL,
    ):
        consumed_spans.append(edit_match.span())
    # Also exclude `=== FILE: path === ... === END FILE ===` regions — any
    # `[REPLACE LINES]`-shaped text inside a new-file body is literal content
    # being written, not an orphan edit. Without this, a new file whose body
    # contains the verbatim string `[REPLACE LINES N-M]` (e.g. JARVIS source
    # itself) gets a spurious orphan edit grafted onto the most recent
    # unrelated file in scope.
    for file_match in re.finditer(
        r'===\s*FILE:\s*(\S+).*?\n(?:.*?)'
        r'(?:===\s*END\s+FILE\s*===|(?====\s*(?:EDIT|FILE):)|$)',
        response, re.DOTALL,
    ):
        consumed_spans.append(file_match.span())

    def _in_consumed(pos: int) -> bool:
        return any(s <= pos < e for s, e in consumed_spans)

    def _last_file_before(pos: int) -> str | None:
        # 1. last === EDIT: <path> === before pos
        last_edit_path = None
        for m in re.finditer(r'===\s*EDIT:\s*(\S+)\s*===', response[:pos]):
            last_edit_path = m.group(1).strip()
        if last_edit_path:
            return last_edit_path
        # 2. last [CODE: <path>] before pos (strip any line range / #label)
        last_code_path = None
        for m in re.finditer(r'\[CODE:\s*([^\]\n]+?)\s*\]', response[:pos],
                              re.IGNORECASE):
            arg = m.group(1).strip()
            # drop trailing line ranges and #label
            arg = re.sub(r'\s+#\w+\s*$', '', arg)
            arg = re.sub(r'\s+(?:\d+\s*-\s*\d+)(?:\s*,\s*\d+\s*-\s*\d+)*\s*$',
                         '', arg)
            last_code_path = arg.strip()
        return last_code_path

    orphan_patterns = [
        (re.compile(
            r'\[REPLACE\s+LINES?\s+(\d+)\s*-\s*(\d+)\s*\][ \t]*\r?\n?(.*?)[ \t]*\r?\n?\[/REPLACE\]',
            re.DOTALL), 'replace_lines'),
        (re.compile(
            r'\[INSERT\s+AFTER\s+LINE\s+(\d+)\s*\][ \t]*\r?\n?(.*?)[ \t]*\r?\n?\[/INSERT\]',
            re.DOTALL), 'insert_after'),
        (re.compile(
            r'\[DELETE\s+LINES?\s+(\d+)\s*-\s*(\d+)\s*\]'), 'delete_range'),
        (re.compile(
            r'\[DELETE\s+LINE\s+(\d+)\s*\]'), 'delete_single'),
    ]
    rescued = 0
    for pat, kind in orphan_patterns:
        for m in pat.finditer(response):
            if _in_consumed(m.start()):
                continue  # already inside a properly-wrapped EDIT block
            target = _last_file_before(m.start())
            if not target:
                continue  # no file context — unsafe to apply
            if kind == 'replace_lines':
                start, end, code = int(m.group(1)), int(m.group(2)), m.group(3)
                result["edits"].setdefault(target, []).append((start, end, code))
            elif kind == 'insert_after':
                line_n, code = int(m.group(1)), m.group(2)
                result["edits"].setdefault(target, []).append((0, line_n, code))
            elif kind == 'delete_range':
                start, end = int(m.group(1)), int(m.group(2))
                result["edits"].setdefault(target, []).append((start, end, ""))
            elif kind == 'delete_single':
                ln = int(m.group(1))
                result["edits"].setdefault(target, []).append((ln, ln, ""))
            rescued += 1
    if rescued:
        warn(f"  Rescued {rescued} orphan line-edit block(s) — attached to most recent file in scope")

    # ── Extract FILE blocks (new files) ──────────────────────────────────
    # Two accepted forms:
    #   1. === FILE: path ===          (preferred — uses the documented terminator)
    #      <content lines>
    #      === END FILE ===
    #   2. === FILE: path ===          (legacy — content in a fenced block)
    #      ```optional-lang
    #      <content>
    #      ```
    # Form 1 is tried first. It's bounded by the terminator and can't
    # accidentally consume code from a later, unrelated section.
    file_pattern_terminated = re.compile(
        r'===\s*FILE:\s*(\S+).*?\n(.*?)\n===\s*END\s+FILE\s*===',
        re.DOTALL
    )
    matched_spans: list[tuple[int, int]] = []
    for file_match in file_pattern_terminated.finditer(response):
        filepath = file_match.group(1).strip()
        content = file_match.group(2).strip()
        result["new_files"][filepath] = content
        matched_spans.append(file_match.span())

    # Legacy backticks form — only scan regions NOT already consumed.
    # Stop the fence body at the FIRST ``` that comes BEFORE the next
    # section boundary (=== EDIT:, === FILE:, or === END FILE ===), so a
    # missing closing fence cannot let one FILE block swallow the next.
    # The previous variant was just `.*?```` which, if the writer dropped
    # a closing fence, would silently consume the entire rest of the
    # response — including unrelated files that came after.
    file_pattern_fenced = re.compile(
        r'===\s*FILE:\s*(\S+).*?```[^\n]*\n'
        r'(.*?)'
        r'(?:```|(?====\s*(?:EDIT|FILE):|===\s*END\s+FILE\s*===))',
        re.DOTALL
    )
    def _in_matched_span(pos: int) -> bool:
        return any(s <= pos < e for s, e in matched_spans)
    for file_match in file_pattern_fenced.finditer(response):
        if _in_matched_span(file_match.start()):
            continue
        filepath = file_match.group(1).strip()
        # Sanity cap. 500K chars is generous; pushed higher to allow huge
        # generated files but still finite so a truly runaway match stops.
        if file_match.end() - file_match.start() > 500_000:
            warn(
                f"    Skipping ``` FILE block for {filepath} — content "
                f"exceeds 500K chars (likely runaway match across sections)"
            )
            continue
        if filepath not in result["new_files"]:
            content = file_match.group(2).strip()
            result["new_files"][filepath] = content

    # ── Fallback: plain code blocks ──────────────────────────────────────
    # DISABLED — this used to grab the longest ``` block in the response and
    # write it to a file called "main", which silently destroyed real files
    # whenever the regex above didn't match. New files must use the proper
    # `=== FILE: path === ... === END FILE ===` form.
    # if not result["edits"] and not result["text_edits"] and not result["new_files"]:
    #     all_blocks = re.findall(r'```[^\n]*\n(.*?)```', response, re.DOTALL)
    #     if all_blocks:
    #         longest = max(all_blocks, key=len)
    #         result["new_files"]["main"] = longest.strip()

    return result


def _apply_line_edits(
    original: str, edits: list[tuple[int, int, str]],
    on_skip: "callable | None" = None,
) -> tuple[str, int, list[str]]:
    """Apply line-number based edits to file content.

    Each edit is (start_line, end_line, new_code) where lines are 1-based.
    ALL line numbers refer to the ORIGINAL file — they do NOT shift.
    This works because edits are applied in reverse order (bottom to top).

    Returns (new_content, applied_count, skip_messages). Callers that used
    the legacy single-return form (just `_apply_line_edits(orig, edits)`)
    must adapt — `_apply_extracted_code` does. `on_skip`, if provided, is
    called with each skip message at the moment of detection so callers
    can stream feedback.

    Semantics:
    - (34, 40, "code")  → REPLACE lines 34-40 (inclusive) with "code"
    - (34, 34, "code")  → REPLACE just line 34 with "code"
    - (34, 40, "")      → DELETE lines 34-40
    - (34, 34, "")      → DELETE just line 34
    - (0, 34, "code")   → INSERT "code" AFTER line 34 (start=0 is the signal)

    Content format: each line uses the `i{N}|{code}` indent-prefix format,
    where N is the absolute number of leading spaces. This eliminates the
    indent-counting failure mode entirely — the model writes a number, the
    engine emits the actual whitespace.

    INSERT AFTER may include an anchor before a `---` separator:
        i12|pass
        ---
        i0|def get_traces():
        i4|return ""
    The anchor is matched against file line `end` (with ±20 line fuzzy
    fallback). The new content is everything after the `---`. This catches
    off-by-N line counting errors before they corrupt the file.
    """
    TAB_WIDTH = 4
    # Normalize the whole file first so all indent comparisons are in spaces
    lines = original.expandtabs(TAB_WIDTH).split('\n')
    original_line_count = len(lines)
    skip_messages: list[str] = []
    applied_count = 0

    def _report_skip(msg: str) -> None:
        skip_messages.append(msg)
        if on_skip is not None:
            try:
                on_skip(msg)
            except Exception:
                pass

    # ── Pre-validation 1: drop out-of-bounds REPLACEs first ────────────
    # A range that doesn't exist in the file is a SKIP per-edit (not a
    # batch reject), and must not poison the shrink-tripwire's accounting
    # below. An OOB request would otherwise look like a "100-line delete"
    # for a file that's 50 lines long.
    keep_after_oob: list[tuple[int, int, str]] = []
    for s, e, code in edits:
        if s == 0:
            # INSERT AFTER: end can legitimately equal original_line_count;
            # we clamp at apply time. Out of bounds is end > total + 1.
            if e < 0 or e > original_line_count:
                _report_skip(
                    f"[INSERT AFTER LINE {e}]: anchor out of bounds "
                    f"(file has {original_line_count} lines) — skipped."
                )
                continue
        else:
            if s < 1 or s > original_line_count or e < s:
                _report_skip(
                    f"[REPLACE LINES {s}-{e}]: range out of bounds "
                    f"(file has {original_line_count} lines) — skipped."
                )
                continue
            # Clamp end to file bounds — better than rejecting outright
            # when the end exceeds the file by a small amount (off-by-one
            # in the model's line numbering).
            if e > original_line_count:
                e = original_line_count
        keep_after_oob.append((s, e, code))
    edits = keep_after_oob

    # ── Pre-validation 2: detect overlapping REPLACE ranges ─────────────
    # Two REPLACE LINES blocks that intersect each other cannot both be
    # applied cleanly bottom-to-top — the second one ends up writing on
    # top of (or partly inside) the first. Refuse and surface a skip.
    # INSERT AFTER is allowed to share its anchor line with a REPLACE.
    replace_intervals: list[tuple[int, int, int]] = []  # (start, end, idx)
    for idx, (s, e, _code) in enumerate(edits):
        if s == 0:
            continue  # INSERT — no range
        replace_intervals.append((s, e, idx))
    replace_intervals.sort()
    bad_indices: set[int] = set()
    for i in range(len(replace_intervals)):
        s1, e1, idx1 = replace_intervals[i]
        for j in range(i + 1, len(replace_intervals)):
            s2, e2, idx2 = replace_intervals[j]
            if s2 > e1:
                break
            # Overlap detected
            bad_indices.add(idx1)
            bad_indices.add(idx2)
            _report_skip(
                f"OVERLAPPING [REPLACE LINES] blocks: {s1}-{e1} and {s2}-{e2} "
                f"intersect. Pick one or combine into a single block."
            )
    if bad_indices:
        edits = [e for k, e in enumerate(edits) if k not in bad_indices]

    # ── Pre-validation 3: catastrophic-shrink tripwire for line edits ───
    # If applying every line edit as-given would shrink the file by more
    # than 50% (lines or bytes), the edits almost certainly target the
    # wrong ranges (off-by-N, plan referenced wrong file, etc.). Surface
    # and refuse — same protection the text-edit path already has.
    def _projected_new_line_count(code: str) -> int:
        """Count how many file-lines `code` will produce once applied.

        `code` is the RAW model output for a REPLACE/INSERT body — it may
        still carry `i{N}|` indent prefixes, `---` INSERT-AFTER separators,
        etc. Those constructs are 1:1 with file lines (no prefix splits
        across lines, no separator emits empty lines that survive), so a
        raw-line count is the same as the post-restore line count.

        Strips leading/trailing terminator newlines so 'abc\\n' counts as
        1 line, not 2 — without this, a single-line REPLACE that includes
        a trailing newline inflated projections and triggered false-positive
        catastrophic-shrink rejections on legitimate edits.

        INSERT-AFTER bodies with `---` anchor separators: only the lines
        AFTER the `---` end up in the file. Strip the anchor portion before
        counting so the projection reflects the actual growth.
        """
        if not code.strip():
            return 0
        sep_match = re.search(r'^[ \t]*---[ \t]*$', code, re.MULTILINE)
        if sep_match:
            code = code[sep_match.end():]
        if not code.strip():
            return 0
        return len(code.strip('\n').split('\n'))

    if original_line_count >= 50 and edits:
        projected_lines = original_line_count
        for s, e, code in edits:
            new_n = _projected_new_line_count(code)
            if s == 0:  # INSERT AFTER — grows
                projected_lines += new_n
            else:
                old_n = e - s + 1
                projected_lines += (new_n - old_n)
        if projected_lines < original_line_count * 0.5:
            msg = (
                f"REJECTING [REPLACE LINES] batch: would shrink file from "
                f"{original_line_count} to ~{projected_lines} lines (>50% loss). "
                f"This is almost certainly the wrong line range — split into "
                f"smaller surgical edits."
            )
            _report_skip(msg)
            warn(f"    {msg}")
            return original, 0, skip_messages

    # Sort edits by start (or end for inserts) DESCENDING — apply bottom to top
    # so each application can use the ORIGINAL line numbers without re-mapping.
    #
    # Tiebreakers (in order):
    #   1. ANCHOR position descending (bottom-to-top apply order).
    #   2. KIND — when an INSERT AFTER and a REPLACE LINES share the same
    #      anchor line, the REPLACE goes FIRST (i.e. sorts AFTER the INSERT
    #      in the reverse-sorted list so it pops first). Otherwise the
    #      INSERT could land inside a span the REPLACE then overwrites,
    #      silently dropping the inserted lines.
    #   3. ORIGINAL document order DESCENDING, so two INSERT AFTER at the
    #      same anchor are applied in reverse-write order. The model's
    #      writing order is preserved in the final file:
    #
    #        [INSERT AFTER LINE 5] A     ← written first
    #        [INSERT AFTER LINE 5] B     ← written second
    #
    #      Apply B first (lines[5:5] = [B]) then A (lines[5:5] = [A]) →
    #      final order is …,5, A, B, 6,… — matching the model's intent.
    def sort_key(e_pair):
        idx, (s, end, _) = e_pair
        anchor = end if s == 0 else s
        # kind_rank: 0 = INSERT, 1 = REPLACE. With reverse=True the larger
        # rank pops first → REPLACE applies before INSERT at the same anchor.
        kind_rank = 0 if s == 0 else 1
        return (anchor, kind_rank, idx)
    sorted_edits = [e for _, e in sorted(enumerate(edits), key=sort_key, reverse=True)]

    # Detect whether content uses the new i{N}| prefix format
    indent_prefix_re = re.compile(r'^i\d+\|')
    def _is_new_format(code: str) -> bool:
        for ln in code.split('\n'):
            if ln.strip():
                return bool(indent_prefix_re.match(ln))
        return False

    for start, end, new_code in sorted_edits:
        # ── INSERT AFTER with anchor (split on '---' separator) ──────────
        anchor_text = None
        adjusted_end = end
        if start == 0:
            sep_match = re.search(r'^[ \t]*---[ \t]*$', new_code, re.MULTILINE)
            if sep_match:
                anchor_text = new_code[:sep_match.start()].rstrip('\n')
                new_code = new_code[sep_match.end():].lstrip('\n')

        if not new_code.strip():
            new_lines = []
        else:
            # Convert i{N}| prefixes (and legacy markers) to real spaces
            restored = _restore_replace_whitespace(new_code)
            norm_code = restored.expandtabs(TAB_WIDTH).strip('\n')
            new_format = _is_new_format(new_code)

            if start > 0 and start <= len(lines):
                if new_format:
                    # Indent is explicit in i{N}|; trust it. No auto-reindent.
                    new_lines = norm_code.split('\n')
                else:
                    # Legacy format — apply auto-reindent based on the slice.
                    anchor = lines[start - 1:end]
                    new_lines = _reindent_replace(norm_code, anchor)
            else:
                # INSERT AFTER: trust the model's literal indent. The line
                # above the insertion is NOT a reliable indent reference.
                new_lines = norm_code.split('\n')

        # ── For INSERT AFTER with anchor: validate or fuzzy-relocate ────
        if start == 0 and anchor_text is not None and anchor_text.strip():
            anchor_restored = _restore_replace_whitespace(anchor_text)
            anchor_lines = [
                l for l in anchor_restored.expandtabs(TAB_WIDTH).split('\n')
                if l.strip()
            ]
            if anchor_lines:
                # Compare on stripped content (forgiving of indent drift)
                anchor_keys = [l.strip() for l in anchor_lines]
                # Try exact line `end` first
                file_idx = end - 1  # 0-based
                def _matches_at(idx: int) -> bool:
                    if idx < 0 or idx + len(anchor_keys) > len(lines):
                        return False
                    for k, want in enumerate(anchor_keys):
                        if lines[idx + k].strip() != want:
                            return False
                    return True
                if _matches_at(file_idx):
                    pass  # exact hit — adjusted_end already correct
                else:
                    # Fuzzy search ±20 lines around the claim
                    best = None
                    for delta in range(1, 21):
                        for cand in (file_idx - delta, file_idx + delta):
                            if _matches_at(cand):
                                best = cand
                                break
                        if best is not None:
                            break
                    if best is not None:
                        adjusted_end = best + len(anchor_keys)  # insert AFTER last anchor line
                        if adjusted_end != end:
                            warn(f"    Anchor relocated: insert after line {end} → after line {adjusted_end}")
                    else:
                        warn(f"    Anchor for INSERT AFTER LINE {end} not found in file (±20 lines); inserting at requested position anyway")

        if start == 0:
            # INSERT AFTER line 'adjusted_end'
            insert_idx = min(len(lines), adjusted_end)
            lines[insert_idx:insert_idx] = new_lines
            status(f"    Inserted {len(new_lines)} lines after line {adjusted_end}")
            applied_count += 1
        else:
            # REPLACE lines start through end (inclusive)
            start_idx = max(0, start - 1)
            end_idx = min(len(lines), end)
            # Validate the range was actually in the file. If end < start
            # after clamping, the edit asked for a range that doesn't exist.
            if start > original_line_count or end < start:
                _report_skip(
                    f"[REPLACE LINES {start}-{end}]: range out of bounds "
                    f"(file has {original_line_count} lines) — skipped."
                )
                continue

            if new_lines:
                lines[start_idx:end_idx] = new_lines
                if start == end:
                    status(f"    Replaced line {start} with {len(new_lines)} lines")
                else:
                    status(f"    Replaced lines {start}-{end} with {len(new_lines)} lines")
            else:
                # DELETE
                del lines[start_idx:end_idx]
                if start == end:
                    status(f"    Deleted line {start}")
                else:
                    status(f"    Deleted lines {start}-{end}")
            applied_count += 1

    return '\n'.join(lines), applied_count, skip_messages


def _strip_line_numbers(text: str) -> tuple[str, int | None]:
    """Strip line number suffixes/prefixes from text copied from a numbered file listing.

    New suffix format: '····code here  │42'  (· = space, T = tab, number at end)
    Legacy suffix format: 'code here  │42'   (no whitespace markers)
    Legacy prefix format: '  42\\tcode here'
    Anchored hint prefix: '@line 45\\n...' (injected by the [SEARCH: 45-49] parser)
    Returns (stripped_text, first_line_number or None).
    """
    # ── Anchored hint: @line N injected by the [SEARCH: N-M] parser ─────────
    anchor_match = re.match(r'^@line\s+(\d+)\n?', text)
    if anchor_match:
        hint = int(anchor_match.group(1))
        rest = text[anchor_match.end():]
        rest_stripped, _ = _strip_line_numbers(rest)
        return rest_stripped, hint

    def _restore_whitespace(line: str) -> str:
        """Reverse visible whitespace markers (leading only).
        ⁃ (U+2043) → space  |  → (U+2192) → tab
        Legacy: · (U+00B7) → space  |  T → tab
        """
        if not line:
            return line
        result = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '\u2043':      # ⁃ hyphen bullet → space (new)
                result.append(' ')
                i += 1
            elif ch == '\u2192':    # → rightwards arrow → tab (new)
                result.append('\t')
                i += 1
            elif ch == '\u00b7':    # · middle dot → space (legacy)
                result.append(' ')
                i += 1
            elif ch == 'T' and (i + 1 >= len(line) or line[i+1] in ('\u2043', '\u2192', '\u00b7', 'T', ' ', '\t')):
                result.append('\t')  # legacy T marker
                i += 1
            else:
                result.append(line[i:])
                break
        return ''.join(result)

    # PRE-PASS: split mid-line i{N}| segments. If the model packed multiple
    # i{N}|... segments onto one physical line, recover by splitting at every
    # ` i{digits}|` boundary that comes after the line's leading marker.
    # See _restore_replace_whitespace for the full rationale.
    pack_split_re = re.compile(r'\s+(i\d+\|)')
    raw_lines = text.split('\n')
    unpacked = []
    for line in raw_lines:
        if re.match(r'^i\d+\|', line) and pack_split_re.search(line):
            parts = pack_split_re.split(line)
            unpacked.append(parts[0])
            i = 1
            while i < len(parts):
                marker = parts[i]
                rest = parts[i + 1] if i + 1 < len(parts) else ''
                unpacked.append(marker + rest)
                i += 2
        else:
            unpacked.append(line)
    text = '\n'.join(unpacked)

    lines = text.split('\n')
    stripped = []
    first_num = None
    has_numbers = False

    # New unified format:  i{N}|{code} {lineno}
    # Blank line variant:  i0| {lineno}
    # The trailing line number is separated by exactly one space from the code.
    # We match it as the LAST whitespace-separated token at end of line.
    new_format = re.compile(r'^i(\d+)\|(.*?)\s+(\d+)\s*$')
    # Same prefix WITHOUT trailing line number — accepts model writing
    # SEARCH content without copying the trailer. Indent prefix is still
    # the source of truth; we just don't get a hint_line from this form.
    new_format_no_lineno = re.compile(r'^i(\d+)\|(.*)$')

    for line in lines:
        # Try new i{N}|{code} {lineno} format first
        m_new = new_format.match(line)
        if m_new:
            has_numbers = True
            indent = int(m_new.group(1))
            code = m_new.group(2)
            lineno = int(m_new.group(3))
            if first_num is None:
                first_num = lineno
            # Re-emit indent as spaces
            stripped.append(' ' * indent + code)
            continue

        # Try i{N}|{code} WITHOUT trailing line number (model omitted it)
        m_new_nl = new_format_no_lineno.match(line)
        if m_new_nl:
            # Mark as has_numbers so we treat this as numbered-format input
            # (subsequent fallback lines won't be returned untouched).
            has_numbers = True
            indent = int(m_new_nl.group(1))
            code = m_new_nl.group(2)
            # If code itself happens to end in " {digits}", keep it as code —
            # we have no way to tell line number from "version = 1" without
            # the explicit trailing form. Models that need to be safe should
            # use the form with trailing line number.
            stripped.append(' ' * indent + code)
            continue

        # Legacy suffix format: code  │42       (line number)
        # Extended:             code  │42·i16   (line number + indent count)
        m = re.match(r'^(.*?)\s*│\s*(\d+)(?:·i\d+)?\s*$', line)
        if m:
            has_numbers = True
            if first_num is None:
                first_num = int(m.group(2))
            stripped.append(_restore_whitespace(m.group(1)))
            continue

        # Legacy prefix format: optional whitespace + digits + tab + content
        m2 = re.match(r'^\s*(\d+)\t(.*)$', line)
        if m2:
            has_numbers = True
            if first_num is None:
                first_num = int(m2.group(1))
            stripped.append(m2.group(2))
            continue

        # Legacy: bare line number with no tab (blank line)
        m3 = re.match(r'^\s*(\d+)\s*$', line)
        if m3 and has_numbers:
            stripped.append('')
            continue

        stripped.append(_restore_whitespace(line) if has_numbers else line)

    if has_numbers:
        return '\n'.join(stripped), first_num
    return text, None


def _restore_replace_whitespace(text: str) -> str:
    """Convert visible whitespace markers AND i{N}| indent prefixes back to real
    whitespace on every line of REPLACE/INSERT content.

    New unified format: `i{N}|{code}` — the prefix is REPLACED by N actual
    spaces. The model emits a number, the engine emits the spaces. This
    eliminates the indent-counting failure mode entirely.

    DEFENSIVE BEHAVIOUR:
    - Leading spaces/tabs in {code} (after the `|`) are STRIPPED. If the
      model writes `i4|    def foo`, it gets 4 spaces total, not 8. The
      `i{N}|` prefix is the SOLE source of indent — never additive.
    - MID-LINE i{N}| sequences are SPLIT into separate lines. If the model
      writes `i4|def foo(): i8|return 1` on one physical line, we treat it
      as two lines and emit them stacked. There is no legitimate code
      pattern producing ` i{digits}|` mid-line (that would require a
      literal pipe with a digit prefix in the same word context), so the
      split is safe and recovers the model's likely intent.

    Note: trailing line numbers in REPLACE content (e.g. `i4|x = 5 23`)
    are NOT auto-stripped because we cannot distinguish a copied line
    number from legitimate trailing digits (`x = 99`, `n = 4`). The
    coder prompt explicitly tells the model not to include line numbers
    in REPLACE blocks. If a syntax error appears with a stray trailing
    integer, the self-check round will surface it.

    Legacy: visible markers (· or ⁃ for space, T or → for tab) are also
    converted back, in case the model copied directly from an old view.
    """
    # PRE-PASS: split mid-line i{N}| segments. The pattern is " i\d+|" with
    # at least one whitespace before the `i` (so we don't split on tokens
    # like `i0|` at the very start). This recovers when the model packs
    # multiple intended lines onto one physical line.
    # Be conservative: only split if the line ALREADY starts with i{N}| —
    # that's our signal the model meant to use the format, and any further
    # i{N}| on the same line is almost certainly a missed newline.
    split_re = re.compile(r'\s+(i\d+\|)')
    lines_in = text.split('\n')
    lines_out = []
    for line in lines_in:
        if re.match(r'^i\d+\|', line) and split_re.search(line):
            # Split on every " i{N}|" boundary, then re-prepend the marker
            # to each fragment after the first.
            parts = split_re.split(line)
            # parts is [pre, marker1, between1, marker2, between2, ...]
            # First fragment is pre as-is; subsequent are marker + between.
            lines_out.append(parts[0])
            i = 1
            while i < len(parts):
                marker = parts[i]
                rest = parts[i + 1] if i + 1 < len(parts) else ''
                lines_out.append(marker + rest)
                i += 2
        else:
            lines_out.append(line)
    text = '\n'.join(lines_out)

    # New format: i{N}|content  →  N spaces + content
    indent_re = re.compile(r'^i(\d+)\|(.*)$')
    # Trailing line-number tail. The [CODE:] / [KEEP:] view emits each line
    # as `iN|{code} {lineno}`. Three sub-cases need stripping:
    #
    #   (a) statement-end + space + digits + EOL  → strip
    #         `return x, "" 198` → preceded by `"` (statement-end char)
    #   (b) BOX-drawing decoration + space + digits + EOL  → strip
    #         `# ── Header ─── 201` → comments are valid Python so this
    #         won't crash, but the trailer is visual clutter and should go
    #   (c) BLANK-line trailer: line is purely `<whitespace><digits>` → strip
    #         empty source line emitted as `i0| 503` → REPLACE produces a
    #         line containing only `503`, which is a NameError at runtime
    #         (and an IndentationError in some contexts).
    #
    # The heuristic does NOT strip when the digit is an operator-preceded
    # operand (`x = 5`, `n = 4`) — those stay legitimate.
    _STATEMENT_END = r'[\w\)\]\}\:\"\'─-╿]'  # word, brackets, quote, colon, box-drawing
    _TRAILING_LINENO = re.compile(rf'(?<={_STATEMENT_END})\s+\d{{1,6}}\s*$')
    # Pure-trailer line: only whitespace + digits (the blank-line trailer case)
    _PURE_LINENO = re.compile(r'^\s*\d{1,6}\s*$')

    def _restore_line(line: str) -> str:
        # Try new indent-prefix format first
        m = indent_re.match(line)
        if m:
            indent = int(m.group(1))
            content = m.group(2)
            # DEFENSIVE: strip leading whitespace from content. The `i{N}|`
            # prefix is authoritative; any extra indent in the content is
            # almost certainly a model mistake (typed both prefix + spaces).
            content = content.lstrip(' \t')
            # (c) Pure trailer — line is just whitespace + digits.
            # That's a blank-line trailer (the source line was empty and
            # the [CODE:] view rendered it as "i0| 503"). Drop the digits;
            # the result is a real blank line.
            if _PURE_LINENO.match(content):
                content = ""
            # (a)/(b) statement-end / box-drawing followed by trailer.
            elif content.strip() and _TRAILING_LINENO.search(content):
                stripped_trail = _TRAILING_LINENO.sub('', content)
                if stripped_trail.strip():
                    content = stripped_trail
            return ' ' * indent + content
        # Legacy: visible whitespace markers
        result = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '\u2043':      # ⁃ → space (new)
                result.append(' ')
                i += 1
            elif ch == '\u2192':    # → → tab (new)
                result.append('\t')
                i += 1
            elif ch == '\u00b7':    # · → space (legacy)
                result.append(' ')
                i += 1
            elif ch == 'T' and result and all(c in (' ', '\t') for c in result):
                result.append('\t')  # legacy T marker
                i += 1
            else:
                result.append(line[i:])
                break
        return ''.join(result)

    return '\n'.join(_restore_line(line) for line in text.split('\n'))


# ────────────────────────────────────────────────────────────────────────────
# REVERT mechanism — per-file undo stack
# ────────────────────────────────────────────────────────────────────────────
# When the model sees mid-thought that an edit is going wrong, it can write
# `[REVERT FILE: path]` to restore the file to its state immediately BEFORE
# the most recent successful edit applied to it. The stack persists across
# rounds within a session so the model can roll back even multi-step
# cascades. It's a per-file LIFO — each push corresponds to one apply pass,
# each pop restores the prior state.
_REVERT_STACK: dict[str, list[str]] = {}


def _push_revert_state(filepath: str, content_before: str) -> None:
    """Snapshot a file's content before applying edits to it. Called
    immediately before _apply_edits/_apply_line_edits writes a new version."""
    _REVERT_STACK.setdefault(filepath, []).append(content_before)
    # Cap stack depth so a long session doesn't grow unbounded
    if len(_REVERT_STACK[filepath]) > 32:
        _REVERT_STACK[filepath] = _REVERT_STACK[filepath][-32:]


def _pop_revert_state(filepath: str) -> str | None:
    """Pop and return the most recent pre-edit snapshot for filepath, or
    None if no history exists."""
    stack = _REVERT_STACK.get(filepath)
    if not stack:
        return None
    return stack.pop()


def _clear_revert_history(filepath: str | None = None) -> None:
    """Clear undo history for one file, or all files if filepath is None.
    Called at session boundaries by external code."""
    if filepath is None:
        _REVERT_STACK.clear()
    else:
        _REVERT_STACK.pop(filepath, None)


def _reindent_replace(replace_text: str, matched_lines) -> list[str]:
    """Re-indent replace_text so its first non-blank line aligns with the first
    non-blank line of the matched window in the file.

    When a model writes a SEARCH/REPLACE without leading indentation (or with
    wrong indentation), strategies 1/2/3/4 still find the right location by
    whitespace-normalized comparison — but without this they splice in the
    replacement verbatim, producing a wrongly-indented block.

    Computes the delta between the actual file indentation at the match and
    the replacement's indentation, then shifts all lines by that delta so
    relative indentation within the block is preserved.

    `matched_lines` may be a single line (legacy) or a list of file lines
    that the SEARCH matched. Passing the full window lets us skip leading
    blank lines on either side, which would otherwise yield indent=0 and
    misalign the splice. We also independently verify that the model's
    *relative* indentation within the REPLACE actually mirrors the file's
    structure — if it doesn't (e.g. model wrote a flat block but file has
    nested), we still shift uniformly because that's the safest correction.
    """
    rep_lines = replace_text.split('\n')
    if not rep_lines:
        return rep_lines

    # Accept either str (legacy) or list[str] (new).
    if isinstance(matched_lines, str):
        matched_lines = [matched_lines]

    first_nonempty_rep = next((l for l in rep_lines if l.strip()), None)
    first_nonempty_file = next((l for l in matched_lines if l.strip()), None)

    if first_nonempty_rep is None or first_nonempty_file is None:
        return rep_lines  # nothing to align — all blank, leave as-is

    rep_indent = len(first_nonempty_rep) - len(first_nonempty_rep.lstrip())
    file_indent = len(first_nonempty_file) - len(first_nonempty_file.lstrip())

    delta = file_indent - rep_indent
    if delta == 0:
        return rep_lines

    out = []
    for line in rep_lines:
        if not line.strip():
            out.append(line)  # preserve blank lines as-is
            continue
        cur = len(line) - len(line.lstrip())
        out.append(' ' * max(0, cur + delta) + line.lstrip())
    return out


def _apply_edits(original: str, edits: list[tuple[str, str]]) -> tuple[str, int, int, list[str]]:
    """Apply SEARCH/REPLACE edits to file content.

    SEARCH text may contain line numbers (e.g. '  42\\tcode') from the
    numbered file listing. These are stripped before matching, and used
    as a hint to find the right location in the file.

    Returns (result, matched_count, total_count, ambiguous_skips).
    ambiguous_skips is a list of human-readable messages describing SEARCH
    blocks that were skipped because they matched multiple locations.
    """
    import difflib

    # Normalize the file to spaces before doing anything.
    # If the model inserted tab-indented lines into a space-indented file
    # (or vice-versa), Python's tokenizer rejects the mixed result even when
    # the visual indentation looks correct. expandtabs(4) converts all tabs
    # consistently, making every subsequent indent calculation reliable.
    TAB_WIDTH = 4
    result = original.expandtabs(TAB_WIDTH)

    matched = 0
    total = 0
    ambiguous_skips: list[str] = []

    # Track which line ranges have been edited by previous edits in this batch.
    # Prevents later fuzzy matches from piling onto already-modified regions.
    # Each entry is (start_line_idx, end_line_idx) inclusive.
    edited_ranges: list[tuple[int, int]] = []

    def _overlaps_edited(start: int, length: int) -> bool:
        """Check if a candidate range overlaps any already-edited region."""
        end = start + length - 1
        for ed_start, ed_end in edited_ranges:
            if start <= ed_end and end >= ed_start:
                return True
        return False

    def _record_edit(start: int, old_length: int, new_length: int):
        """Record that lines [start, start+new_length) were just modified.
        Also shift all previously-recorded ranges that come AFTER this edit
        by the delta (new_length - old_length) so they stay correct."""
        delta = new_length - old_length
        if delta != 0:
            shifted = []
            for ed_start, ed_end in edited_ranges:
                if ed_start >= start + old_length:
                    # This range is entirely after the edit — shift it
                    shifted.append((ed_start + delta, ed_end + delta))
                elif ed_end < start:
                    # This range is entirely before the edit — no change
                    shifted.append((ed_start, ed_end))
                else:
                    # Overlapping — expand to cover both (shouldn't happen
                    # because we exclude overlaps, but be defensive)
                    shifted.append((min(ed_start, start),
                                    max(ed_end + delta, start + new_length - 1)))
            edited_ranges.clear()
            edited_ranges.extend(shifted)
        # Record the new edit's range
        edited_ranges.append((start, start + new_length - 1))

    for find_text, replace_text in edits:
        find_raw = find_text.strip('\n')
        # Only strip surrounding newlines from REPLACE — NOT spaces.
        # .strip() would eat the leading indentation of the first replaced
        # line, producing a de-indented line and a Python indentation error.
        # Also restore visible whitespace markers (· → space, T → tab) that
        # the model may have copied literally from the [CODE:] display.
        replace_clean = _restore_replace_whitespace(
            replace_text.strip('\n')
        ).expandtabs(TAB_WIDTH)

        # Strip line numbers from SEARCH if present, then normalize tabs
        find_clean, hint_line = _strip_line_numbers(find_raw)
        find_clean = find_clean.strip('\n').expandtabs(TAB_WIDTH)

        if not find_clean:
            continue

        total += 1

        # ── Strategy 1: Exact match ──────────────────────────────────
        if find_clean in result:
            # Find the position to check for edited-region overlap
            result_lines = result.split('\n')
            find_first_line = find_clean.split('\n')[0]
            find_n_lines = len(find_clean.split('\n'))
            # Enumerate ALL non-overlapping exact matches first.
            exact_positions = []
            for i in range(len(result_lines) - find_n_lines + 1):
                if _overlaps_edited(i, find_n_lines):
                    continue
                if result_lines[i] != find_first_line and find_first_line not in result_lines[i]:
                    # Cheap skip when first line doesn't even appear
                    continue
                candidate = '\n'.join(result_lines[i:i + find_n_lines])
                if candidate == find_clean:
                    exact_positions.append(i)
            if len(exact_positions) == 1:
                i = exact_positions[0]
                replace_lines = replace_clean.split('\n') if replace_clean else []
                result_lines[i:i + find_n_lines] = replace_lines
                _record_edit(i, find_n_lines, len(replace_lines))
                result = '\n'.join(result_lines)
                matched += 1
                continue
            if len(exact_positions) > 1:
                # Multiple exact matches in non-edited regions. If we have
                # a line-number hint, pick the closest. Otherwise REFUSE —
                # the old blind `replace(..., 1)` silently clobbered.
                if hint_line is not None:
                    best = min(exact_positions, key=lambda p: abs(p - (hint_line - 1)))
                    replace_lines = replace_clean.split('\n') if replace_clean else []
                    result_lines[best:best + find_n_lines] = replace_lines
                    _record_edit(best, find_n_lines, len(replace_lines))
                    result = '\n'.join(result_lines)
                    matched += 1
                    continue
                msg = (
                    f"SKIPPING ambiguous SEARCH block — {len(exact_positions)} EXACT "
                    f"locations match. Add more context lines OR use [SEARCH: N-M] "
                    f"with a line range."
                )
                warn(msg)
                ambiguous_skips.append(
                    f"- SEARCH starting with {repr(find_clean[:60])} matched "
                    f"{len(exact_positions)} exact locations — widen the SEARCH "
                    f"block OR use the anchored [SEARCH: N-M] form."
                )
                continue
            # 0 non-overlapping matches found, but find_clean is still
            # SOMEWHERE in result — i.e. only inside an already-edited
            # region. Refuse to clobber and let strategies 2/3/4 try.
            # (Old behaviour: blind .replace(find, repl, 1) — silently
            # overwrote a region we already edited.)

        # ── Strategy 2: Line-number-guided match ─────────────────────
        find_lines = [l.strip() for l in find_clean.split('\n')]
        result_lines = result.split('\n')
        found = False

        if hint_line is not None:
            # Search in a window around the hinted line (±30 lines)
            hint_idx = max(0, hint_line - 1)  # 1-based to 0-based
            search_start = max(0, hint_idx - 30)
            search_end = min(len(result_lines), hint_idx + len(find_lines) + 30)

            for i in range(search_start, min(search_end, len(result_lines) - len(find_lines) + 1)):
                if _overlaps_edited(i, len(find_lines)):
                    continue  # skip already-edited regions
                window = [result_lines[i + j].strip() for j in range(len(find_lines))]
                if window == find_lines:
                    replace_lines_list = replace_clean.split('\n') if replace_clean.strip() else []
                    if not replace_clean.strip():
                        result_lines[i:i + len(find_lines)] = []
                        _record_edit(i, len(find_lines), 0)
                    else:
                        new_lines = _reindent_replace(
                            replace_clean, result_lines[i:i + len(find_lines)]
                        )
                        result_lines[i:i + len(find_lines)] = new_lines
                        _record_edit(i, len(find_lines), len(new_lines))
                    result = '\n'.join(result_lines)
                    found = True
                    break

        # ── Strategy 3: Full whitespace-normalized scan ───────────────
        if not found:
            # Count ALL locations where the normalized SEARCH matches,
            # EXCLUDING already-edited regions.
            all_matches = [
                i for i in range(len(result_lines) - len(find_lines) + 1)
                if not _overlaps_edited(i, len(find_lines))
                and [result_lines[i + j].strip() for j in range(len(find_lines))] == find_lines
            ]

            if hint_line is not None and len(all_matches) > 1:
                # The model gave us a line number — use it to pick the closest
                # match instead of refusing. Strategy 2's window may have been
                # too narrow if earlier edits in this round shifted lines.
                # The hint is the disambiguation signal; honour it.
                hint_idx = hint_line - 1  # 0-based
                best = min(all_matches, key=lambda i: abs(i - hint_idx))
                if not replace_clean.strip():
                    result_lines[best:best + len(find_lines)] = []
                    _record_edit(best, len(find_lines), 0)
                else:
                    new_lines = _reindent_replace(
                        replace_clean, result_lines[best:best + len(find_lines)]
                    )
                    result_lines[best:best + len(find_lines)] = new_lines
                    _record_edit(best, len(find_lines), len(new_lines))
                result = '\n'.join(result_lines)
                found = True

            elif len(all_matches) == 1:
                i = all_matches[0]
                if not replace_clean.strip():
                    result_lines[i:i + len(find_lines)] = []
                    _record_edit(i, len(find_lines), 0)
                else:
                    new_lines = _reindent_replace(
                        replace_clean, result_lines[i:i + len(find_lines)]
                    )
                    result_lines[i:i + len(find_lines)] = new_lines
                    _record_edit(i, len(find_lines), len(new_lines))
                result = '\n'.join(result_lines)
                found = True
            elif len(all_matches) > 1:
                msg = (
                    f"SKIPPING ambiguous SEARCH block — {len(all_matches)} locations match "
                    f"(normalized). Add more context lines to make the SEARCH unique."
                )
                warn(msg)
                ambiguous_skips.append(
                    f"- SEARCH starting with {repr(find_clean[:60])} matched "
                    f"{len(all_matches)} locations — widen the SEARCH block."
                )
                # The attempt counted toward `total` at the top of this loop;
                # leave it there so the caller's `matched/total` ratio reflects
                # the ambiguous miss as "attempted but didn't apply."
                continue

        if found:
            matched += 1
            continue

        # ── Strategy 4: Fuzzy match ──────────────────────────────────
        # Threshold scales with block size: a 60% match on a 50-line block
        # easily picks up the wrong window and can chew across structural
        # boundaries (e.g. replacing JS inside an HTML <script> tag with
        # text that ends up overwriting unrelated HTML around it). Big
        # blocks must match much more precisely; small blocks (1-3 lines)
        # can be loose because the consequence of a wrong match is small.
        n = len(find_lines)
        if n <= 3:
            min_score = 0.6
        elif n <= 10:
            min_score = 0.75
        elif n <= 25:
            min_score = 0.88
        else:
            min_score = 0.95  # huge SEARCH block — must be near-perfect
        find_joined = "\n".join(find_lines)
        candidates = []
        for wsize in [n, n - 1, n + 1]:
            if wsize < 1 or wsize > len(result_lines):
                continue
            for i in range(len(result_lines) - wsize + 1):
                if _overlaps_edited(i, wsize):
                    continue  # skip already-edited regions
                window = [result_lines[i + j].strip() for j in range(wsize)]
                score = difflib.SequenceMatcher(None, find_joined, "\n".join(window)).ratio()
                if score >= min_score:
                    candidates.append((score, i, wsize))

        if not candidates:
            pass  # no match at all — fall through
        elif len(candidates) > 1 and hint_line is not None:
            # Model gave a line number — prefer PROXIMITY to the hint,
            # not just similarity score. A 69% match at the right line
            # is better than an 85% match 300 lines away.
            # Score: proximity_weight (0-1) + similarity (0-1), where
            # proximity decays with distance from the hint.
            hint_idx = hint_line - 1
            PROXIMITY_RADIUS = 40  # lines — full proximity credit within this radius

            def _pick_score(c):
                score, idx, length = c
                distance = abs(idx - hint_idx)
                proximity = max(0.0, 1.0 - distance / PROXIMITY_RADIUS)
                # Proximity gets 60% weight, similarity gets 40% weight
                return 0.6 * proximity + 0.4 * score

            best = max(candidates, key=_pick_score)
            best_score, best_idx, best_length = best
            success(f"Fuzzy matched FIND block ({best_score:.0%} similarity, anchored to line {hint_line})")
            result_lines = result.split('\n')
            if not replace_clean.strip():
                result_lines[best_idx:best_idx + best_length] = []
                _record_edit(best_idx, best_length, 0)
            else:
                new_lines = _reindent_replace(
                    replace_clean, result_lines[best_idx:best_idx + best_length]
                )
                result_lines[best_idx:best_idx + best_length] = new_lines
                _record_edit(best_idx, best_length, len(new_lines))
            result = '\n'.join(result_lines)
            matched += 1
        elif len(candidates) > 1:
            # Multiple fuzzy matches, no hint — too ambiguous to pick one safely
            best_score = max(c[0] for c in candidates)
            msg = (
                f"SKIPPING ambiguous SEARCH block — {len(candidates)} fuzzy matches "
                f"(best {best_score:.0%}). Add more context lines to make the SEARCH unique."
            )
            warn(msg)
            ambiguous_skips.append(
                f"- SEARCH starting with {repr(find_clean[:60])} had "
                f"{len(candidates)} fuzzy matches (best {best_score:.0%}) — widen the SEARCH block."
            )
            continue
        else:
            best_score, best_idx, best_length = candidates[0]
            success(f"Fuzzy matched FIND block ({best_score:.0%} similarity)")
            result_lines = result.split('\n')
            if not replace_clean.strip():
                result_lines[best_idx:best_idx + best_length] = []
                _record_edit(best_idx, best_length, 0)
            else:
                new_lines = _reindent_replace(
                    replace_clean, result_lines[best_idx:best_idx + best_length]
                )
                result_lines[best_idx:best_idx + best_length] = new_lines
                _record_edit(best_idx, best_length, len(new_lines))
            result = '\n'.join(result_lines)
            matched += 1

        if not found and not candidates:
            preview = find_clean[:80].replace('\n', '\\n')
            error(f"FIND block not matched — SKIPPING edit")
            warn(f"  Tried to find: {preview}...")
            if hint_line:
                warn(f"  Line number hint was: {hint_line}")

    return result, matched, total, ambiguous_skips


def _smart_apply(original: str, extracted: dict, filepath: str) -> str | None:
    """Apply edits from extracted code blocks — handles both line-number and text formats."""
    # Try line-number edits first
    if filepath in extracted["edits"]:
        new, _, _ = _apply_line_edits(original, extracted["edits"][filepath])
        return new

    # Fuzzy filepath match for line edits — path-bounded suffix only
    # (bare endswith was the foo/bar.py ↔ qux/bar.py collision bug)
    def _suffix_with_sep(longer: str, shorter: str) -> bool:
        if longer == shorter:
            return True
        if not longer.endswith(shorter):
            return False
        cut = len(longer) - len(shorter)
        return cut == 0 or longer[cut - 1] in '/\\'
    for fp, edits in extracted["edits"].items():
        if _suffix_with_sep(fp, filepath) or _suffix_with_sep(filepath, fp):
            new, _, _ = _apply_line_edits(original, edits)
            return new

    # Try text-based edits (fallback) — same path-bounded suffix rule.
    # Bare endswith was the foo/bar.py ↔ qux/bar.py collision bug.
    if filepath in extracted["text_edits"]:
        result, _, _, _ = _apply_edits(original, extracted["text_edits"][filepath])
        return result

    for fp, edits in extracted["text_edits"].items():
        if _suffix_with_sep(fp, filepath) or _suffix_with_sep(filepath, fp):
            result, _, _, _ = _apply_edits(original, edits)
            return result

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — UNDERSTAND
# ═══════════════════════════════════════════════════════════════════════════════

async def phase_understand(task: str, project_root: str) -> dict:
    """
    3 AIs search codebase in parallel. Union ALL findings.
    For empty/new projects: skips searching, returns minimal context.
    Returns: {context: str, files: list[str], classification: str}
    """
    step("═══ Phase 1: UNDERSTAND ═══")

    # Scan project structure
    project_structure = scan_project(project_root)
    # Extract file count from last line: "(N files)"
    file_count_match = re.search(r'\((\d+) files?\)', project_structure)
    file_count = int(file_count_match.group(1)) if file_count_match else 0

    status(f"Project scanned: {file_count} files")

    # ── Empty/new project: skip heavy searching ──────────────────────────
    if file_count == 0:
        status("Empty project — creating from scratch")
        context = (
            f"PROJECT: {project_root}\n"
            f"STATUS: Empty directory — this is a NEW project.\n"
            f"TASK: {task}\n\n"
            f"There are no existing files. Create all files from scratch."
        )
        return {
            "context": context,
            "files": [],
            "project_structure": project_structure,
        }

    # ── Existing project: 3 AIs search in parallel ────────────────────
    step("3 AIs searching codebase...")
    prompt = UNDERSTAND_PROMPT.format(
        task=task,
        project_structure=project_structure[:8000],
    )

    results = list(await asyncio.gather(
        *[_call_with_tools(m, prompt, project_root, log_label="understanding codebase", max_rounds=20, stop_on_tool_block=True, cache_file_reads=True) for m in UNDERSTAND_MODELS],
        return_exceptions=True,
    ))
    results = [r for r in results if isinstance(r, dict) and r.get("answer")]

    if not results:
        raise RuntimeError("All 3 AIs failed in Phase 1")

    # Union ALL findings — if ANY AI says it's relevant, include it
    all_files = set()
    all_context_parts = []

    for r in results:
        ai_name = r["model"].split("/")[-1]
        all_context_parts.append(f"\n=== {ai_name}'s analysis ===\n{r['answer']}")

        # Extract file paths mentioned
        for line in r["answer"].split("\n"):
            # Match common path patterns
            path_match = re.findall(r'[\w./\-]+\.(?:py|js|ts|html|css|json|lean|c|cpp|h|rs|java|go|rb|toml|yaml|yml|md)', line)
            all_files.update(path_match)

    # Don't dump full files into context — the planner uses [CODE:]+[KEEP:]
    # in the tool loop to read files and focus on relevant sections.
    # Just include the AI analyses (which reference specific code via tools).
    full_context = "\n".join(all_context_parts)
    status(f"Phase 1: {len(results)} AIs, {len(all_files)} relevant files identified, {len(full_context)} chars context")
    success("Phase 1 complete — code understood")

    return {
        "context": full_context,
        "files": sorted(all_files),
        "project_structure": project_structure,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — PLAN (multi-model generate → single AI merge)
# ═══════════════════════════════════════════════════════════════════════════════

async def phase_plan(task: str, context: str, complexity: int, project_root: str,
                     plan_feedback: str = "", detailed_map: str = "",
                     purpose_map: str = "",
                     is_new_project: bool = False,
                     files: list | None = None) -> tuple[str, dict]:
    """
    Planning:
      Standard (complexity < 7):
        Layer 1: 4 AIs race, first 3 win, last is cancelled
        Layer 2: GLM-5.1 merges the 3 winning plans into final plan
      Deep (complexity >= 7):
        Layer 1: 4 AIs race, first 3 win, last is cancelled
        Layer 2: 4 AIs each read the 3 plans, find flaws/strengths,
                 write their own improved plan (parallel)
        Layer 3: GLM-5.1 reads all 4 improved plans, writes final plan

    Planners use [CODE:]+[KEEP:] inside the tool loop to read files and
    focus on relevant sections. No pre-pass needed.

    Returns (final_plan, research_cache).
    """
    extended = complexity >= 7
    mode_label = "EXTENDED 3-layer" if extended else "STANDARD"
    step(f"=== Phase 2: PLAN [{mode_label}] ===")
    _wlog.phase_start("plan", mode_label, complexity=complexity,
                      task_chars=len(task), is_new_project=is_new_project)

    # Shared research cache — accumulates lookups across all AIs.
    # The planner uses [CODE:] to read files and [KEEP:] to strip irrelevant
    # lines from context. Both happen inside the tool loop automatically.
    research_cache: dict[str, str] = {}

    PLAN_MODELS = [
        "nvidia/deepseek-v4-pro",
        "nvidia/deepseek-v4-flash",
        "nvidia/glm-5.1",
        "nvidia/kimi-k2.6",
    ]

    cot = PLAN_COT_NEW if is_new_project else PLAN_COT_EXISTING
    file_list_str = (
        "\n".join(f"  {f}" for f in sorted(files)) if files else "(none — new project)"
    )
    plan_prompt = PLAN_PROMPT.format(
        task=task,
        file_list=file_list_str,
        context=context[:30000],
        cot_instructions=cot,
    )

    if plan_feedback:
        plan_prompt += (
            f"\n\nPREVIOUS PLAN WAS REJECTED:\n{plan_feedback}\n"
            f"Fix the issues above. Do NOT repeat the same mistakes."
        )

    # == Layer 1: 4 AIs RACE — keep first 3, kill the last ==
    step(f"Layer 1: {len(PLAN_MODELS)} models racing (first 3 win)...")
    _wlog.phase_event("Layer 1 start", models=",".join(m.split("/")[-1] for m in PLAN_MODELS))

    plan_tasks = [
        asyncio.create_task(_call_with_tools(
            m, plan_prompt, project_root,
            detailed_map=detailed_map, purpose_map=purpose_map,
            research_cache=research_cache,
            log_label=f"planning (Layer 1)",
            max_rounds=8,
            stop_on_tool_block=True,
            cache_file_reads=True,
        ))
        for m in PLAN_MODELS
    ]

    # Wait for 3 to complete, cancel the rest
    done, pending = await asyncio.wait(
        plan_tasks, return_when=asyncio.ALL_COMPLETED, timeout=None
    ) if len(PLAN_MODELS) <= 3 else await asyncio.wait(
        plan_tasks, return_when=asyncio.FIRST_COMPLETED
    )

    # For >3 models: keep collecting until we have 3, then cancel the rest
    if len(PLAN_MODELS) > 3:
        completed: list = list(done)
        pending_set = set(pending)
        while len(completed) < 3 and pending_set:
            more_done, pending_set = await asyncio.wait(
                pending_set, return_when=asyncio.FIRST_COMPLETED
            )
            completed.extend(more_done)
        # Cancel any remaining (the slowest)
        for t in pending_set:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        status(f"Layer 1: {len(completed)} finished first, cancelled {len(pending_set)} stragglers")
        done_tasks = completed
    else:
        done_tasks = list(done)

    # Collect results from winners
    plans = []
    for t in done_tasks:
        try:
            r = t.result()
            if isinstance(r, dict) and r.get("answer"):
                plans.append(r)
                _wlog.phase_event(
                    "Layer 1 plan complete",
                    model=r["model"].split("/")[-1],
                    chars=len(r["answer"]),
                    done=bool(r.get("done")),
                )
            else:
                _wlog.phase_warn(
                    "Layer 1 plan EMPTY",
                    model=(r.get("model") if isinstance(r, dict) else "?"),
                )
        except Exception as ex:
            _wlog.phase_error("Layer 1 plan raised", error=str(ex)[:200])

    # Snapshot every collected plan to its own file under plans/
    for i, p in enumerate(plans):
        _wlog.save_plan(layer=1, index=i, model_id=p["model"], content=p["answer"])

    if not plans:
        _wlog.phase_error("Layer 1 produced zero plans")
        raise RuntimeError("All models failed in planning")

    status(f"Layer 1: got {len(plans)} winning plans (research_cache: {len(research_cache)} entries)")
    _wlog.phase_event(
        "Layer 1 done",
        n_plans=len(plans),
        research_cache_entries=len(research_cache),
    )

    # Each plan gets a clearly-labelled [INPUT PLAN #N] block so the
    # improver can refer to them by number ("Plan #2 is the best
    # baseline; pull STEP 4 from Plan #3"). Matches the unified labelled-
    # section vocabulary used elsewhere in the prompts.
    all_plans_text = "\n\n".join(
        f"──────────────────────────────────────────────────────────────────────\n"
        f"[INPUT PLAN #{i + 1}] — by {p['model'].split('/')[-1]}\n"
        f"──────────────────────────────────────────────────────────────────────\n"
        f"{p['answer']}"
        for i, p in enumerate(plans)
    )

    if extended:
        # == Layer 2: 4 AIs each pick the best plan and improve it ==
        step(f"Layer 2: 4 AIs picking best plan and improving...")

        # Pre-load Layer 1 research
        preloaded_research = _format_research_cache(research_cache)

        improve_prompt = SYSTEM_KNOWLEDGE + IMPROVE_PROMPT_TEMPLATE.format(
            task=task,
            context=context[:15000],
            all_plans_text=all_plans_text,
            preloaded_research=preloaded_research,
        )
        # Use same 4 diverse models for Layer 2 debate as Layer 1 planning
        # to ensure consistent capability levels throughout the planning pipeline
        improved_results = list(await asyncio.gather(
            *[_call_with_tools(m, improve_prompt, project_root,
                               detailed_map=detailed_map, purpose_map=purpose_map,
                               research_cache=research_cache,
                               log_label="improving plan (Layer 2)",
                               max_rounds=20,
                               stop_on_tool_block=True,
                               cache_file_reads=True)
              for m in PLAN_MODELS],
            return_exceptions=True,
        ))
        improved = [d for d in improved_results if isinstance(d, dict) and d.get("answer")]

        # Log every improver outcome — including the ones that raised so a
        # bug-hunt can tell "model never returned" apart from "model
        # returned empty answer."
        for raw in improved_results:
            if isinstance(raw, dict) and raw.get("answer"):
                _wlog.phase_event(
                    "Layer 2 improver complete",
                    model=raw["model"].split("/")[-1],
                    chars=len(raw["answer"]),
                    done=bool(raw.get("done")),
                )
            elif isinstance(raw, dict):
                _wlog.phase_warn(
                    "Layer 2 improver returned EMPTY",
                    model=raw.get("model", "?"),
                )
            else:
                _wlog.phase_error("Layer 2 improver raised", error=str(raw)[:200])

        for i, d in enumerate(improved):
            _wlog.save_plan(layer=2, index=i, model_id=d["model"], content=d["answer"])

        if not improved:
            warn("Layer 2: all failed, falling back to Layer 1 plans")
            _wlog.phase_warn("Layer 2 all failed — falling back to Layer 1")
            improved = plans

        status(f"Layer 2: got {len(improved)} improved plans (cache: {len(research_cache)} entries)")
        _wlog.phase_event("Layer 2 done", n_improved=len(improved))


        # Same labelled-block style as Layer 1 plans, marked as IMPROVED
        # so the merger knows it's getting Layer 2 outputs (already
        # picked + refined), not raw drafts.
        all_improved_text = "\n\n".join(
            f"──────────────────────────────────────────────────────────────────────\n"
            f"[INPUT PLAN #{i + 1}] — IMPROVED, by {d['model'].split('/')[-1]}\n"
            f"──────────────────────────────────────────────────────────────────────\n"
            f"{d['answer']}"
            for i, d in enumerate(improved)
        )

        # == Layer 3: GLM-5 reads all improved plans, finds flaws/strengths, writes final ==
        step("Layer 3: GLM-5 writing final plan...")

        # Update pre-loaded research (now includes Layer 2's lookups too)
        preloaded_research = _format_research_cache(research_cache)

        verify_block = ""
        if not is_new_project:
            verify_block = (
                "Verify claims against real code:\n"
                "  [REFS: name] / [LSP: name] / [DETAIL: feature] / [CODE: path]\n"
                "  [REFS: name] — find all definitions, imports, usages\n"
            )

        merge_prompt = SYSTEM_KNOWLEDGE + MERGE_PROMPT_TEMPLATE.format(
            n_plans=len(improved),
            task=task,
            context=context[:10000],
            verify_block=verify_block if not is_new_project else "",
            all_plans_text=all_improved_text[:30000],
            preloaded_research=preloaded_research,
        )
        merger_result = await _call_with_tools(
            "nvidia/glm-5.1", merge_prompt, project_root,
            detailed_map=detailed_map, purpose_map=purpose_map,
            research_cache=research_cache,
            log_label="merging plans (final)",
            max_rounds=20,
            stop_on_tool_block=True,
            cache_file_reads=True)

    else:
        # == Standard: GLM-5 merges plans directly (no debate) ==
        step("Layer 2: GLM-5 merging plans...")

        # Pre-load Layer 1 research for the merger
        preloaded_research = _format_research_cache(research_cache)

        verify_block = ""
        if not is_new_project:
            verify_block = (
                "Verify claims against real code:\n"
                "  [REFS: name] / [LSP: name] / [DETAIL: feature] / [CODE: path]\n"
                "  [REFS: name] — find all definitions, imports, usages\n"
                "Write tags, wait, then proceed.\n"
            )

        merge_prompt = SYSTEM_KNOWLEDGE + MERGE_PROMPT_TEMPLATE.format(
            n_plans=len(plans),
            task=task,
            context=context[:15000],
            verify_block=verify_block,
            all_plans_text=all_plans_text,
            preloaded_research=preloaded_research,
        )
        merger_result = await _call_with_tools(
            "nvidia/glm-5.1", merge_prompt, project_root,
            detailed_map=detailed_map, purpose_map=purpose_map,
            research_cache=research_cache,
            log_label="merging plans",
            max_rounds=20,
            stop_on_tool_block=True,
            cache_file_reads=True)

    if not merger_result.get("answer"):
        _wlog.phase_warn("Merger returned EMPTY — falling back to longest Layer-1 plan")
        best = max(plans, key=lambda p: len(p["answer"]))
        _wlog.save_final_plan(best["answer"], merger_model=f"FALLBACK:{best['model']}")
        _wlog.phase_end("plan", mode_label, fallback=True, chars=len(best["answer"]))
        return best["answer"], research_cache

    best_plan = merger_result["answer"]
    # Strip leftover tool-loop signal tags. The tool loop strips fully-formed
    # two-tag signals before returning, but a bare half (e.g. an isolated
    # `[CONFIRM_DONE]` produced after the model already stopped) can still
    # ride through. Drop both halves so plan content is signal-free.
    for _pat in (r'\[DONE\]', r'\[CONFIRM_DONE\]',
                 r'\[STOP\]', r'\[CONFIRM_STOP\]',
                 r'\[CONTINUE\]', r'\[CONFIRM_CONTINUE\]'):
        best_plan = re.sub(_pat, '', best_plan, flags=re.IGNORECASE)
    best_plan = best_plan.rstrip()
    status(f"Phase 2: final plan = {len(best_plan)} chars")
    success(f"Phase 2 complete ({mode_label}, {len(research_cache)} cached lookups)")
    _wlog.save_final_plan(best_plan, merger_model=merger_result.get("model", "nvidia/glm-5.1"))
    _wlog.phase_end("plan", mode_label, chars=len(best_plan),
                    research_cache_entries=len(research_cache))
    return best_plan, research_cache


# =====================================================================
#  PHASE 3 -- IMPLEMENT (step-based DAG: parallel when independent,
#             sequential when dependent, one coder per step)
# =====================================================================

def _extract_shared_interfaces(plan: str) -> str:
    """Extract the SHARED INTERFACES section from the plan."""
    match = re.search(
        r'##\s*SHARED\s+INTERFACES\s*\n(.*?)(?=\n##\s|\Z)',
        plan, re.DOTALL | re.IGNORECASE,
    )
    if match:
        text = match.group(1).strip()
        if text.lower() not in ("(none)", "none", "n/a", ""):
            return text
    return ""


def _extract_impl_steps(plan: str) -> list[dict]:
    """Parse IMPLEMENTATION STEPS from the plan.

    Returns a list of step dicts:
      [{"num": 1, "name": "...", "depends_on": [int], "files": [str],
        "details": "...", "done": False, "produced_files": {}}]

    Falls back to a single step containing all files if no steps found.

    Only the FIRST occurrence of each step number is kept. Layer-3 mergers
    sometimes splice plans together leaving a duplicate `### STEP 1: ...`
    block at the bottom (e.g. when the merger appends an "ADDITIONS" or
    "REVISED PLAN" section). Without dedup, the implement loop re-runs
    step 1 after step N — exactly the bug the user is reporting.

    Falls back to ONLY the IMPLEMENTATION STEPS section if present, so that
    examples in earlier sections (e.g. "STEP 1: do X" appearing in prose)
    don't pollute the step list.
    """
    # Restrict to the IMPLEMENTATION STEPS section if it exists. The merger
    # often re-drafts the plan within the same response (incomplete draft +
    # final). The FINAL plan is the one we want, so we pick the LAST
    # `## IMPLEMENTATION STEPS` heading, not the first. (An earlier draft
    # ending mid-step would otherwise overwrite the final plan, dropping
    # whichever steps appeared only in the final.)
    # Find the LAST ## IMPLEMENTATION STEPS heading and capture from there
    # to end of string. Do NOT stop at any ## heading — step bodies often
    # contain ## headers as template content (e.g. "## BUGS" inside a prompt
    # format spec), and stopping at those drops subsequent steps silently.
    # Step boundaries are identified by ### STEP N: headers, not ## sections.
    section_matches = list(re.finditer(
        r'##\s*IMPLEMENTATION\s+STEPS\s*\n(.*)',
        plan, re.DOTALL | re.IGNORECASE,
    ))
    if section_matches:
        # Use the last (latest, most complete) draft
        plan_scoped = section_matches[-1].group(1)
    else:
        plan_scoped = plan

    steps = []
    step_pattern = re.compile(
        r'###\s*STEP\s*(\d+)\s*[:\-—]\s*(.+?)(?=\n)',
        re.IGNORECASE,
    )
    matches = list(step_pattern.finditer(plan_scoped))

    if not matches:
        # No steps found — return empty, caller will use single-step fallback
        return []

    seen_nums: set[int] = set()
    for i, m in enumerate(matches):
        num = int(m.group(1))
        name = m.group(2).strip()

        # Get the body of this step (text until next ### STEP or end-of-plan)
        start = m.end()
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            # LAST step — terminate at one of the KNOWN trailing section
            # headers that come AFTER all `### STEP N:` blocks in the plan
            # template. We CANNOT use a generic `\n##\s+[A-Z]` matcher
            # because step bodies often quote those headers as prose/example
            # content (e.g. "## EDGE CASES" inside a description), which
            # would silently truncate the last step's body.
            terminators = [
                r'\n##\s+COMPLETENESS\b',
                r'\n##\s+EDGE\s+CASES\b',
                r'\n##\s+VERIFICATION\b',
                r'\n##\s+TEST\s+CRITERIA\b',
                r'\n##\s+PRE-?MORTEM\s+RESOLUTION\b',
                r'\n##\s+CONFIDENCE\s+GATE\b',
            ]
            term_pat = re.compile(
                "|".join(terminators), re.IGNORECASE,
            )
            next_term = term_pat.search(plan_scoped[start:])
            end = start + next_term.start() if next_term else len(plan_scoped)
        body = plan_scoped[start:end]

        # Skip duplicate step numbers — keep only the first occurrence.
        # Without this, "### STEP 1: foo" appearing twice in the merged plan
        # makes the implement loop re-run step 1 after step N.
        if num in seen_nums:
            warn(f"  Duplicate STEP {num} in plan — skipping repeat occurrence")
            continue
        seen_nums.add(num)

        # Parse DEPENDS ON
        deps = []
        dep_match = re.search(r'DEPENDS\s*ON\s*[:]\s*(.+)', body, re.IGNORECASE)
        if dep_match:
            dep_text = dep_match.group(1).strip()
            if dep_text.lower() not in ("(none)", "none", "-", "n/a", ""):
                dep_nums = re.findall(r'STEP\s*(\d+)', dep_text, re.IGNORECASE)
                deps = [int(d) for d in dep_nums]

        # Parse FILES
        files = []
        files_match = re.search(r'FILES\s*[:]\s*(.+)', body, re.IGNORECASE)
        if files_match:
            files_text = files_match.group(1).strip()
            file_paths = re.findall(
                r'([\w./\-]+\.(?:py|js|ts|jsx|tsx|html|css|json|lean|c|cpp|h|rs|java|go|rb|toml|yaml|yml|md|mjs|cjs|svelte|vue))',
                files_text,
            )
            files = list(dict.fromkeys(file_paths))  # dedup, preserve order

        # Parse step instructions (WHAT TO DO or DETAILS section)
        details = body
        # Remove the parsed header lines from details
        for pattern in [r'DEPENDS\s*ON\s*[:].+', r'FILES\s*[:].+']:
            details = re.sub(pattern, '', details, flags=re.IGNORECASE)
        # Also remove the "WHAT TO DO:" header if present
        details = re.sub(r'WHAT\s+TO\s+DO\s*:', '', details, flags=re.IGNORECASE)
        details = details.strip()

        steps.append({
            "num": num,
            "name": name,
            "depends_on": deps,
            "files": files,
            "details": details,
            "done": False,
            "produced_files": {},  # fp → content, filled after execution
        })

    return steps


def _dedup_against_seen(extracted: dict, seen_keys: set[str]) -> dict:
    """Filter `extracted` to remove edit blocks whose content is already in
    `seen_keys`, and add new block keys to `seen_keys` for next time.

    This is essential when the same response (response_so_far) is re-extracted
    on each [STOP]: every prior round's edit blocks are still present and would
    be re-applied. With line-number edits in particular, re-applying old blocks
    against a since-modified file produces deterministic corruption (line
    numbers point to wrong content).

    Block identity is the SHA1 of a NORMALISED form of the raw text — leading
    / trailing whitespace, tabs vs. spaces, and trailing line numbers are
    flattened out. Without normalisation a single trailing newline difference
    produced a fresh key, and the model's accidentally-re-emitted edit
    was applied twice (a real failure mode in the line-edit path, where
    `_apply_line_edits` happily applied `(34, 34, code)` a second time and
    corrupted the file).

    Mutates `extracted` in place. Returns it for convenience.
    """
    import hashlib

    def _norm(s: str) -> str:
        # Whitespace-normalise each line, drop blank lines, then concatenate.
        # This collapses cosmetic differences (trailing spaces, CR/LF) while
        # preserving the meaningful content the matcher will see.
        return "\n".join(
            ln.rstrip() for ln in (s or "").replace("\r\n", "\n").split("\n")
            if ln.strip()
        ).strip()

    def _norm_fp(fp: str) -> str:
        # Normalise filepath so semantically-equal paths hash to the same key.
        # Without this, the same edit written twice with `tools/foo.py` and
        # `./tools/foo.py` (or trailing whitespace) produced two distinct
        # dedup keys, and the second copy applied a second time — duplicating
        # the change on disk.
        if not fp:
            return ""
        s = fp.strip().replace("\\", "/")
        # Strip leading "./" segments
        while s.startswith("./"):
            s = s[2:]
        return s.lower()

    def _hash(*parts) -> str:
        h = hashlib.sha1()
        for p in parts:
            h.update(b"\x1f")
            h.update(str(p).encode("utf-8", "replace"))
        return h.hexdigest()

    # Text edits: keyed on (NORMALISED filepath, normalized search, normalized replace)
    new_text_edits: dict[str, list] = {}
    for fp, edits in extracted.get("text_edits", {}).items():
        kept = []
        for find_text, replace_text in edits:
            key = "text::" + _hash(_norm_fp(fp), _norm(find_text), _norm(replace_text))
            if key not in seen_keys:
                seen_keys.add(key)
                kept.append((find_text, replace_text))
        if kept:
            new_text_edits[fp] = kept
    extracted["text_edits"] = new_text_edits

    # Line edits: keyed on (NORMALISED filepath, start, end, normalized code)
    new_line_edits: dict[str, list] = {}
    for fp, edits in extracted.get("edits", {}).items():
        kept = []
        for start, end, code in edits:
            key = "line::" + _hash(_norm_fp(fp), start, end, _norm(code))
            if key not in seen_keys:
                seen_keys.add(key)
                kept.append((start, end, code))
        if kept:
            new_line_edits[fp] = kept
    extracted["edits"] = new_line_edits

    # New files: keyed on (NORMALISED filepath, normalized content)
    new_files: dict[str, str] = {}
    for fp, content in extracted.get("new_files", {}).items():
        key = "file::" + _hash(_norm_fp(fp), _norm(content))
        if key not in seen_keys:
            seen_keys.add(key)
            new_files[fp] = content
    extracted["new_files"] = new_files

    # Reverts: keyed on NORMALISED filepath + occurrence count (a model may
    # legitimately revert the same file twice in one response, but not 5x).
    seen_revert_count: dict[str, int] = {}
    new_reverts = []
    for rpath in extracted.get("reverts", []):
        nfp = _norm_fp(rpath)
        n = seen_revert_count.get(nfp, 0)
        key = f"revert::{nfp}::{n}"
        seen_revert_count[nfp] = n + 1
        if key not in seen_keys:
            seen_keys.add(key)
            new_reverts.append(rpath)
    extracted["reverts"] = new_reverts

    return extracted


def _apply_extracted_code(
    extracted: dict, file_contents: dict[str, str], sandbox: Sandbox,
    viewed_versions: "dict[str, str] | None" = None,
) -> tuple[dict[str, str], int, int, list[str]]:
    """Apply extracted edits and new files.

    Returns (result_dict, total_matched, total_attempted, ambiguous_skips).
    ambiguous_skips is a list of messages for SEARCH blocks that were skipped
    because they matched multiple locations — the caller should feed these back
    to the model so it widens those SEARCH blocks rather than retrying blind.

    `viewed_versions`, if provided, anchors [REPLACE LINES] edits to the
    version of each file the model most recently saw via [CODE: path],
    rather than the current sandbox state. This makes line numbers robust
    across mid-stream [STOP] applications: line numbers always refer to
    whatever the model was looking at when it wrote the edit. SEARCH/REPLACE
    edits are content-anchored and ignore this parameter.
    """
    result = {}
    total_matched = 0
    total_attempted = 0
    all_ambiguous_skips: list[str] = []

    def _suffix_with_sep(longer: str, shorter: str) -> bool:
        """Path-bounded suffix match. ``foo/bar.py`` is a suffix of
        ``project/foo/bar.py`` but NOT of ``project/myfoo/bar.py``.
        Bare endswith() collided ``mylib.py`` with ``lib.py`` and made
        edits land on the wrong file."""
        if longer == shorter:
            return True
        if not longer.endswith(shorter):
            return False
        cut = len(longer) - len(shorter)
        return cut == 0 or longer[cut - 1] in "/\\"

    def _match_fp(filepath: str) -> str:
        if filepath in file_contents:
            return filepath
        # Prefer the longer-side suffix match — covers "model wrote bare
        # basename, file lives at a/b/basename.py" AND the inverse.
        for known_fp in file_contents:
            if _suffix_with_sep(known_fp, filepath):
                return known_fp
        for known_fp in file_contents:
            if _suffix_with_sep(filepath, known_fp):
                return known_fp
        return filepath

    def _resolve_viewed(matched_fp: str, raw_fp: str) -> str | None:
        """Look up the version the model most recently saw, using the same
        path-bounded suffix rule. Without this, line edits anchored to
        ``[REPLACE LINES 22-24]`` based on a viewed copy keyed under
        ``a/b/foo.py`` were silently re-anchored to the at-apply file
        when the edit block wrote a bare ``foo.py``."""
        if viewed_versions is None:
            return None
        if matched_fp in viewed_versions:
            return viewed_versions[matched_fp]
        if raw_fp in viewed_versions:
            return viewed_versions[raw_fp]
        for key in viewed_versions:
            if _suffix_with_sep(key, matched_fp) or _suffix_with_sep(matched_fp, key):
                return viewed_versions[key]
            if _suffix_with_sep(key, raw_fp) or _suffix_with_sep(raw_fp, key):
                return viewed_versions[key]
        return None

    # ── Process REVERT directives FIRST ──────────────────────────────────
    # The model may write [REVERT FILE: path] then provide fresh edits in
    # the same response. Reverting before applying means those new edits
    # go on top of the restored state.
    #
    # We mutate the caller's `file_contents` dict in place so that:
    #   1. Subsequent edits in THIS function see the reverted state.
    #   2. The caller (which holds the same dict reference) also sees the
    #      revert without having to mirror what `result` says.
    # The previous local rebind (`file_contents = dict(file_contents)`)
    # hid the revert from the caller's view, so any check it did against
    # `file_contents` post-call was stale.
    for rpath in extracted.get("reverts", []):
        matched_fp = _match_fp(rpath)
        prior = _pop_revert_state(matched_fp)
        if prior is not None:
            result[matched_fp] = prior
            file_contents[matched_fp] = prior  # subsequent edits target reverted state
            success(f"    Reverted {matched_fp} to prior state")
        else:
            warn(f"    REVERT requested for {rpath} but no undo history exists")

    # Collect every filepath touched by either edit type so we can apply
    # text + line edits SEQUENTIALLY on the same file (the prompt allows
    # the model to mix formats — previously the line-edit pass silently
    # short-circuited when the file already had text edits in `result`).
    text_edits_by_fp = extracted.get("text_edits", {})
    line_edits_by_fp = extracted.get("edits", {})

    all_edit_fps: list[str] = []
    seen_fp_keys: set[str] = set()
    for fp in list(text_edits_by_fp.keys()) + list(line_edits_by_fp.keys()):
        matched_fp = _match_fp(fp)
        key = matched_fp
        if key in seen_fp_keys:
            continue
        seen_fp_keys.add(key)
        all_edit_fps.append((matched_fp, fp))

    for matched_fp, raw_fp in all_edit_fps:
        existing = file_contents.get(matched_fp, "")
        # Collect this file's text edits across whichever keys the parser
        # used (matched_fp OR the raw form the model wrote).
        text_edits = list(text_edits_by_fp.get(matched_fp, []))
        if raw_fp != matched_fp:
            text_edits.extend(text_edits_by_fp.get(raw_fp, []))
        line_edits = list(line_edits_by_fp.get(matched_fp, []))
        if raw_fp != matched_fp:
            line_edits.extend(line_edits_by_fp.get(raw_fp, []))

        # Deduplicate identical SEARCH blocks within this response so a
        # repeated edit doesn't fuzzy-match twice.
        seen_searches: set[str] = set()
        deduped_text_edits = []
        for find_text, replace_text in text_edits:
            key = find_text.strip()
            if key not in seen_searches:
                seen_searches.add(key)
                deduped_text_edits.append((find_text, replace_text))
        text_edits = deduped_text_edits

        # ── Step 1: apply SEARCH/REPLACE to current file content ───────
        working = existing
        pushed_revert = False
        if text_edits:
            if existing:
                _push_revert_state(matched_fp, existing)
                pushed_revert = True
                modified, m, t, skips = _apply_edits(existing, text_edits)
                all_ambiguous_skips.extend(skips)
                # Catastrophic-shrink tripwire — same as before, but now
                # only triggers if SEARCH/REPLACE caused the shrink. Line
                # edits below have their own tripwire inside _apply_line_edits.
                orig_lines = existing.count('\n') + 1
                mod_lines = modified.count('\n') + 1
                if (
                    m > 0
                    and orig_lines >= 50
                    and (mod_lines < orig_lines * 0.5
                         or len(modified) < len(existing) * 0.5)
                ):
                    warn(
                        f"    Rejected SEARCH/REPLACE on {matched_fp}: would shrink "
                        f"file from {orig_lines} to {mod_lines} lines (>50% loss). "
                        f"This is almost certainly a fuzzy mismatch. Reverting."
                    )
                    _pop_revert_state(matched_fp)
                    pushed_revert = False
                    all_ambiguous_skips.append(
                        f"- Edit on {matched_fp} REJECTED: would have shrunk the file "
                        f"by >50% ({orig_lines} → {mod_lines} lines). Your SEARCH block "
                        f"matched far more than intended — likely a fuzzy match on a "
                        f"50+ line block. Split into ≤8-line SEARCH anchors."
                    )
                else:
                    working = modified if m > 0 else existing
                    if m > 0:
                        result[matched_fp] = modified
                    elif not line_edits:
                        # Text edits matched ZERO times AND there are no line
                        # edits coming downstream to claim/discard the snapshot.
                        # Without popping here, the snapshot sits orphaned on
                        # the LIFO stack — a later legitimate [REVERT FILE:]
                        # would pop that wrong (much older) state, silently
                        # destroying correct intermediate edits.
                        _pop_revert_state(matched_fp)
                        pushed_revert = False
                total_matched += m
                total_attempted += t
            else:
                replace_parts = [rt.strip() for _, rt in text_edits if rt.strip()]
                if replace_parts:
                    new_content = "\n\n".join(replace_parts)
                    result[matched_fp] = new_content
                    working = new_content
                    total_matched += len(text_edits)
                    total_attempted += len(text_edits)

        # ── Step 2: apply [REPLACE LINES] / [INSERT AFTER] / [DELETE] ──
        # Line edits anchor to the version the model most recently saw
        # via [CODE:] (viewed_versions). When text edits also applied to
        # this file in the same response, the working buffer no longer
        # matches the viewed version — but the model wrote the line edit
        # based on the viewed numbers, so the viewed version is the safe
        # anchor. We apply the line edits to viewed, then graft the result
        # back over `working` via SEARCH-style content replacement only
        # if both modifications touched DIFFERENT line ranges; otherwise
        # we drop the line edits with a skip message (mixing same-range
        # text+line edits is ambiguous).
        if line_edits:
            viewed = _resolve_viewed(matched_fp, raw_fp)
            # GUARD: line edits MUST anchor to a viewed snapshot when one
            # is available. If `viewed_versions` is provided AND the file
            # has no entry there, the model wrote `[REPLACE LINES N-M]`
            # without ever having a full read of the file (typical when
            # only a SKELETON was returned for a large file). Applying
            # those line numbers to the at-apply state silently lands on
            # the wrong code. Reject + surface a skip message.
            if viewed is None and viewed_versions is not None and existing:
                all_ambiguous_skips.append(
                    f"- [REPLACE LINES] / [INSERT AFTER] / [DELETE] on "
                    f"{matched_fp} REJECTED: you have not read a full "
                    f"version of this file in this response. Line numbers "
                    f"would anchor to whatever the file looks like at "
                    f"apply time, which may differ from what you reasoned "
                    f"about. Either [CODE: {matched_fp}] (small files) or "
                    f"[CODE: {matched_fp}] + [KEEP: {matched_fp} N-M] "
                    f"(large files) first, then re-issue the line edit."
                )
                total_attempted += len(line_edits)
                # Drop any snapshot we pushed for text edits that didn't
                # produce changes (none, since text path either matched or
                # already cleaned up).
                if pushed_revert and matched_fp not in result:
                    _pop_revert_state(matched_fp)
                    pushed_revert = False
                continue
            basis = viewed if viewed is not None else working
            n_edits = len(line_edits)
            total_attempted += n_edits
            if basis:
                if not pushed_revert:
                    _push_revert_state(matched_fp, existing or basis)
                    pushed_revert = True
                modified, applied_n, skip_msgs = _apply_line_edits(basis, line_edits)
                if applied_n > 0:
                    # When text edits also applied to this file, the working
                    # state is already in `result`. Replacing it with the
                    # line-edit result discards the text edits — unsafe.
                    if matched_fp in result and text_edits:
                        all_ambiguous_skips.append(
                            f"- Mixed SEARCH/REPLACE + REPLACE LINES on "
                            f"{matched_fp}: line edits applied to a stale snapshot. "
                            f"Re-emit the line edits as SEARCH/REPLACE blocks so "
                            f"they compose with the rest of your changes."
                        )
                    else:
                        result[matched_fp] = modified
                        total_matched += applied_n
                else:
                    # No revert needed if we never recorded one for this file.
                    if pushed_revert and matched_fp not in result:
                        _pop_revert_state(matched_fp)
                        pushed_revert = False
                for s in skip_msgs:
                    all_ambiguous_skips.append(f"- {s}")
            else:
                code_parts = [c.strip() for _, _, c in line_edits if c.strip()]
                if code_parts:
                    result[matched_fp] = "\n\n".join(code_parts)
                    total_matched += n_edits

    # New files
    for filepath, content in extracted["new_files"].items():
        matched_fp = _match_fp(filepath)
        # `=== FILE:` is for brand-new files only. If the file already exists
        # in file_contents, the model is using the wrong form — typically
        # rewriting from memory. That overwrites everything else and is the
        # single most destructive failure mode. Reject it AND surface a skip
        # message so the model sees explicit feedback in the next round and
        # falls back to a surgical edit. (Previously this rejection was
        # silent — the model thought the file was written and stopped.)
        existing = file_contents.get(matched_fp, "")
        if existing.strip():
            warn(f"    Rejected `=== FILE:` for existing file {matched_fp} "
                 f"— use [SEARCH]/[REPLACE] or [REPLACE LINES] instead")
            all_ambiguous_skips.append(
                f"- `=== FILE: {matched_fp}` REJECTED: that form is for brand-new "
                f"files only; this file already exists ({existing.count(chr(10)) + 1} "
                f"lines). Use `[SEARCH]/[REPLACE]` inside `=== EDIT: {matched_fp} ===` "
                f"to modify it, or `[REPLACE LINES N-M]` for line-anchored edits."
            )
            continue
        result[matched_fp] = _restore_replace_whitespace(content)

    return result, total_matched, total_attempted, all_ambiguous_skips



SELF_CHECK_PROMPT = """══════════════════════════════════════════════════════════════════════
[SYSTEM] — your role in the JARVIS pipeline (workflow, not user request)
══════════════════════════════════════════════════════════════════════
This block (until [USER REQUEST]) is JARVIS describing HOW you fit into
the pipeline. The human did NOT write any of it — your loyalty is to
the [USER REQUEST] further down; this just tells you how to serve it.

You are a verifier in JARVIS. The coder just implemented one step.
The edits have been applied. Your job: confirm the step's requirement
is now TRUE in the code. If it isn't, fix it until it is.

You are the safety net. If you approve broken code, it ships.

REASONING — in your thinking, not in your visible verdict:
  BEFORE you APPROVE or write a fix, in your thinking (reasoning
  channel or `<think>...</think>` tags) walk through:
    • THE REQUIREMENT — restate what must be TRUE in the code in one
                        sentence. If you can't, you don't know what
                        you're verifying.
    • THE EVIDENCE    — what specific file:line will you read to
                        confirm? Cite it; if you can't, the verify
                        is a guess.
    • SIDE EFFECTS    — does the coder's edit affect anything OUTSIDE
                        the step's named files/functions? Callers,
                        imports, persisted state, tests.
    • DOWNSTREAM      — does this edit change a signature / return
                        shape / state field? If yes, every consumer
                        needs to still work. Are they all named in
                        this step or another?
    • FAILURE MODE    — what could be wrong (anchor not unique → wrong
                        target; indent drift; partial application;
                        caller still has old signature).
    • COMPLETENESS    — walk the GOAL from this step: does the
                        delivery chain (origin → ... → user-visible)
                        still pass through after the edit? Or did
                        the edit break a link?
  Visible output stays clean: APPROVED or fix blocks + brief rationale.

══════════════════════════════════════════════════════════════════════
THINK BEFORE ACTING — STREAMLINED, FLEXIBLE
══════════════════════════════════════════════════════════════════════

The biggest failure mode in verification is HALLUCINATING that an edit
landed when it didn't, OR hallucinating that a file is "incomplete"
when the runtime sent the full content. Both errors come from skipping
the explicit thinking step. Before you read any file or write any
fix, output:

  ## 1. WHAT MUST BE TRUE (verification checklist, max 5 items)
  Restate IN YOUR OWN WORDS the observable facts that prove the step
  succeeded. Be SPECIFIC. NOT "analysis_mode is added" — instead:
  "core/state.py defines `Classification` with a field named
  `analysis_mode` of type `bool`."
  Each item must be checkable by reading ONE specific line.

  ## 2. EVIDENCE PLAN — which read answers which checklist item
  For each item N from above, name the [CODE:] or [KEEP:] call that
  will produce the proof. Plan ONE batch upfront. Re-reads waste rounds.

  ## 3. PASS / FAIL CRITERIA
  For each item, write the EXACT TEXT you expect to see in the file
  (a snippet from the line). If you don't see that text, the item fails.

After this preamble, do your tool calls in ONE batch, then verify each
item by quoting the line you saw. Verification means QUOTING — not
asserting. "✅" without a quoted line is a hallucination.

══════════════════════════════════════════════════════════════════════
THE "PARTIAL VIEW" HALLUCINATION TRAP — read this carefully
══════════════════════════════════════════════════════════════════════

When [CODE:] returns the content of a small file, the runtime ALWAYS
includes a header that names the total line count:

  === Code: core/state.py (66 lines) ===
  ...file content...

The line count in the header IS AUTHORITATIVE. If the header says
66 lines and you see 66 numbered lines of content, the file is COMPLETE.
The runtime never sends partial content from [CODE:] without saying so —
truncations always declare themselves ("SKELETON ONLY", "KEPT N/M lines",
etc.). A short file is a short file, not a partial view.

THESE PHRASES ARE FORBIDDEN — they are the signature of the hallucination:
  ✗ "The [CODE:] output only showed N lines — appears to be a partial view"
  ✗ "This can't be the whole file"
  ✗ "The output seems filtered or truncated"
  ✗ "Let me read the full file" (when no truncation header was shown)
  ✗ "The view is incomplete"

If you catch yourself wanting to write one of those, STOP and check:
  1. Is there a "SKELETON ONLY" or "KEPT N/M lines" header? → genuinely truncated.
  2. Is there a "(N lines)" header and you see N lines? → file is COMPLETE.
  3. Is the file just SMALL? Files can be 10 lines. Accept it.

The previous run wasted 5 rounds on this exact hallucination — re-reading
a 66-line file repeatedly while claiming "the output only showed 2 lines".
Do not be that verifier.

══════════════════════════════════════════════════════════════════════
REVERT — UNDO A WRONG FIX (use without shame)
══════════════════════════════════════════════════════════════════════

If your fix lands but is wrong (corrupted indent, wrong anchor matched
in the wrong place, broke a caller), write:

  [REVERT FILE: path/to/file.py]

before your next [STOP][CONFIRM_STOP]. The runtime pops the pre-edit
snapshot and restores the file. Then plan the correct fix from a clean
state instead of layering another patch on top of broken code.

REVERT counts: max 2 reverts per file per round before you give up.
If 2 reverts haven't fixed it, write "VERIFIER UNABLE TO LAND FIX —
<one-sentence reason>" and write [DONE][CONFIRM_DONE]. The next pass
will try a different approach.

══════════════════════════════════════════════════════════════════════
CODE FORMAT
══════════════════════════════════════════════════════════════════════

  i{{N}}|{{code}} {{LINE_NUMBER}}     ← reading [CODE:] output
  i{{N}}|{{code}}                     ← writing fixes (no line number)

N = leading spaces. i4|return x → "    return x" (4 spaces).

⚠ TRAILING LINE NUMBERS: the [CODE:] view shows `iN|code 198`. In your
REPLACE blocks, the trailing integer must NOT appear. The engine strips
it defensively but the rule is yours to follow:
   WRONG: i4|return answer, "" 198
   RIGHT: i4|return answer, ""

══════════════════════════════════════════════════════════════════════
TOOLS
══════════════════════════════════════════════════════════════════════

Wrap ALL tool calls in [tool use]...[/tool use] then fire the two-tag signal.
Tags outside the block are ignored — only deliberate, wrapped calls execute.

  [tool use]
  [CODE: ui/server.py #srv]
  [/tool use]
  [STOP]
  [CONFIRM_STOP]
  ← content arrives here

Writing [CODE:] outside a [tool use] block, or omitting the
[STOP]+[CONFIRM_STOP] signal, is a hallucination — results never arrive.
A bare [STOP] alone fires NOTHING; you need both halves of the signal.
When you're done, write [DONE] then [CONFIRM_DONE] on adjacent lines.

  [CODE: path #label]       Read the post-edit file
  [KEEP: path N-M #label]   Strip to kept line ranges
  [REFS: name #label]       Find definitions, imports, call sites
  [SEARCH: pattern #label]  Ripgrep text search (⚠ not edit syntax)
  [DISCARD: #label]         Remove a result from context

WRITING FIXES:

  DEFAULT — use [SEARCH] / [REPLACE]:

  === EDIT: path/to/file.py ===
  [SEARCH]
  i4|existing_code_to_replace
  i4|second_line_for_uniqueness
  [/SEARCH]
  [REPLACE]
  i4|fixed_code
  i4|second_line
  [/REPLACE]

  FALLBACK — use [REPLACE LINES N-M] only when the code is so corrupted
  that SEARCH cannot find a unique anchor (e.g. indent corruption where
  the same garbled lines repeat many times):

  === EDIT: path/to/file.py ===
  [REPLACE LINES 22-25]
  i4|fixed_code
  i8|more_fixed_code
  [/REPLACE]

VERIFICATION WORKFLOW:

  [tool use]
  [CODE: file.py #read1]
  [/tool use]
  [STOP]
  [CONFIRM_STOP]
  ...find bug, quote the line...
  === EDIT: file.py ===
  [SEARCH]
  i4|buggy_line
  [/SEARCH]
  [REPLACE]
  i4|corrected_line
  [/REPLACE]
  [tool use]
  [CODE: file.py #verify1]
  [/tool use]
  [STOP]
  [CONFIRM_STOP]
  ...quote the post-edit line that proves fix landed...
  VERIFIED
  [DONE]
  [CONFIRM_DONE]

  ⚠ RULE: If you write ANY edit block in this response, you MUST write
    a [CODE: file] then [STOP][CONFIRM_STOP] AFTER the edit and BEFORE
    writing VERIFIED + [DONE][CONFIRM_DONE].
    An edit written after the last [STOP][CONFIRM_STOP] is NOT applied
    before you declare verification — the engine will detect the
    unapplied edit, discard your VERIFIED claim, and force another round.
    This is the #1 self-check loop cause. Pattern to follow WITHOUT
    EXCEPTION:
      edit block → [CODE: file] → [STOP][CONFIRM_STOP] → quote the line
      that proves it landed → VERIFIED → [DONE][CONFIRM_DONE]

══════════════════════════════════════════════════════════════════════
YOUR PROCESS — ORDERED BY PRIORITY
══════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────
PRIORITY 1 — CAN THE FILE EVEN PARSE? (syntax errors first)
──────────────────────────────────────────────────────────────────────

If the system reports a syntax error:

  1. [CODE:] the file. [KEEP:] the ENTIRE enclosing function — from its
     `def` or `class` line to the next function/class. NOT just the
     error line. You need the FULL context to see the indent structure.

  2. DIAGNOSE — what kind of syntax error?

     INDENT CORRUPTION (most common):
       You see function body lines at i0| instead of i4|. Or a block
       at i16| when everything around it is i4|. Or the same 3-6 lines
       repeated 3-8 times in a row with garbled indentation.
       FIX: Use [SEARCH]/[REPLACE] to replace the corrupted function.
       If the corruption is so severe that no unique SEARCH anchor exists,
       fall back to [REPLACE LINES start-end] for the whole function.

     MISSING KEYWORD:
       `except` without `try`, `else` without `if`, missing `:`.
       FIX: [SEARCH]/[REPLACE] targeting the broken line + one unique
       neighbor. Or [REPLACE LINES N-N] for a single unambiguous line.

     UNBALANCED BRACKETS:
       FIX: [SEARCH]/[REPLACE] on the affected expression.

  3. Write the fix. Prefer [SEARCH]/[REPLACE] — it is content-anchored
     and survives any line-number shifts from earlier edits.

  4. [STOP][CONFIRM_STOP] to apply the fix. [CODE:] the file again.
     Is the syntax error gone? Quote the post-edit line that proves it.
     If yes → continue to Priority 2.
     If no → consider [REVERT FILE: path] and rewrite the fix from
     scratch instead of layering more patches on broken indentation.

──────────────────────────────────────────────────────────────────────
PRIORITY 2 — IS THE REQUIREMENT MET? (the actual goal)
──────────────────────────────────────────────────────────────────────

Read the step description. It should say what requirement it satisfies
or what must be true after implementation.

[CODE:] every changed file. [KEEP:] the changed regions + 10 lines context.

For EACH change the step described, check:

  □ DID THE EDIT LAND?
    Look at the actual [CODE:] output, not the coder's prose.
    If the coder wrote "I added X at line 50" but line 50 doesn't
    show X → the edit was silently skipped. Write it yourself using
    [SEARCH]/[REPLACE] with enough context lines to be unique.

  □ IS IT CORRECT?
    Does the code match what the step described?
    Right variable names? Right function signatures? Right logic?

  □ IS THE INDENT RIGHT?
    Read the i{{N}}| on lines ABOVE and BELOW the change.
    The change must be at the same level or one deeper for new blocks.
    If you see function body lines at i0| → indent corruption.

  □ ARE SHARED INTERFACES HONORED?
    Names, types, signatures match the plan's SHARED INTERFACES exactly?

  □ ARE IMPORTS PRESENT?
    Every new name used has an import at file top?

  □ ARE CALLERS COMPATIBLE?
    If a function signature changed: [REFS: function_name]
    Check every caller still passes correct arguments.

  □ IS THE VALUE CORRECT? (not just "the field exists" but "for a
    typical input, does the field have a real, non-empty value?")
    A feature that stores "" is not working.

──────────────────────────────────────────────────────────────────────
PRIORITY 3 — LOGIC CHECK (mental execution)
──────────────────────────────────────────────────────────────────────

Trace the changed code with realistic input:

  "When function X is called with [args]:
    Line A: evaluates to [value]
    Line B: calls Y with [args], Y returns [type]
    Caller C: receives [type] — compatible? Yes/No"

Check:
  □ Types match at every call boundary
  □ Async calls have await, sync calls don't
  □ No mutable defaults ([], {{}}) as parameter defaults
  □ Dictionary keys exist before access (or use .get())
  □ `global` declared when reassigning module-level variables
  □ Exception types are correct for the errors being caught

──────────────────────────────────────────────────────────────────────
PRIORITY 4 — DECIDE
──────────────────────────────────────────────────────────────────────

CORRECT → write 2-3 sentences quoting the lines that prove each
  checklist item passed. Then: VERIFIED  [DONE]  [CONFIRM_DONE]

BUGGY → write the fix using [SEARCH]/[REPLACE]. Then:
  [CODE: file] → [STOP][CONFIRM_STOP] → quote the post-edit line that
  proves the fix landed → VERIFIED → [DONE][CONFIRM_DONE]

  If the fix lands WRONG (visible corruption / wrong location) →
  [REVERT FILE: path] before your next [STOP][CONFIRM_STOP], then plan
  the correct edit from the clean restored state.

  Fix ONE thing at a time. Verify between fixes.
  SYNTAX errors before LOGIC errors (file must parse first).
  Use [REPLACE LINES N-M] only when the code is too corrupted for
  a unique SEARCH anchor.

══════════════════════════════════════════════════════════════════════
VERIFIED REQUIREMENTS — you MUST NOT write VERIFIED unless:
══════════════════════════════════════════════════════════════════════

  □ You [CODE:] read the file in THIS round (not from earlier context)
  □ If a syntax error was reported: you confirmed the error is gone
    by reading the actual error line in [CODE:] output
  □ The specific changes the step describes are VISIBLE in your
    [CODE:] output (not just assumed from the coder's prose)
  □ You did not base your judgment on [KEEP:] ranges that skip
    the changed lines

  If ANY checkbox fails, you cannot write VERIFIED. Read more of
  the file, or fix the issue first.

══════════════════════════════════════════════════════════════════════
"NO CHANGES NEEDED" — STRICT MARKER
══════════════════════════════════════════════════════════════════════

If after verification you determine that NO fix is needed (the coder's
edits are correct and the step's requirement is already true), AND you
want to close the step as a no-op fix-pass, write this LITERAL line:

  STEP COMPLETE: NO CHANGES NEEDED

Then [DONE][CONFIRM_DONE]. Fuzzy phrases like "looks correct", "all
good", "works as expected" no longer trigger the no-op exit — only the
literal marker above does. The marker also requires you have read the
file in this round (see VERIFIED REQUIREMENTS above).

══════════════════════════════════════════════════════════════════════
HARD RULES
══════════════════════════════════════════════════════════════════════

  ✗ Never approve without reading the file
  ✗ Never approve a syntax error
  ✗ Never use SEARCH/REPLACE on corrupted code
  ✗ Never refactor or add features — only verify and fix
  ✗ Never trust the coder's prose over the [CODE:] output


═══════════════════════════════════════════════════════════════════════
CONTEXT
═══════════════════════════════════════════════════════════════════════

══════════════════════════════════════════════════════════════════════
[USER REQUEST] — the human's actual task (this is what you must serve)
══════════════════════════════════════════════════════════════════════
TASK: {task}
══════════════════════════════════════════════════════════════════════
[END USER REQUEST] — everything below is JARVIS framing / facts / context
══════════════════════════════════════════════════════════════════════
STEP: {step_name}
{step_details}

CODER'S REASONING (the code has been applied — use [CODE:] to read it):
{coder_thinking}

FILES CHANGED:
{changed_files_list}
"""

SMALL_FILE_THRESHOLD = 400  # lines — show inline if smaller. KEEP is for 400+ line files.

def _build_file_block(
    file_contents: dict[str, str],
    modify_files: set[str] | None = None,
) -> str:
    """Build the file listing for the coder prompt.

    - Small files (<SMALL_FILE_THRESHOLD lines): shown in full with line
      numbers — the coder writes [REPLACE LINES] directly.
    - Large files: listed with line count — the coder uses [CODE: path]
      to read, then [KEEP: path X-Y, A-B] to strip irrelevant lines
      from context before writing edits.
    - New files: marked as "(does not exist yet)".
    - Context files (not in modify_files): not included.
    """
    if modify_files is None:
        modify_files = set(file_contents.keys())

    parts = []

    for fp, content in file_contents.items():
        if fp not in modify_files:
            continue

        if not content:
            parts.append(
                f"\n== {fp} (NEW FILE — write complete file) ==\n"
                f"(does not exist yet)\n"
            )
            continue

        line_count = content.count('\n') + 1
        numbered = add_line_numbers(content)

        if line_count <= SMALL_FILE_THRESHOLD:
            parts.append(
                f"\n== {fp} ({line_count} lines) ==\n"
                f"{numbered}\n"
            )
        else:
            # Always show the full file — the model must see it to know which
            # lines to target. It cannot KEEP what it has never seen.
            # For large files, instruct the model to KEEP only the relevant
            # section immediately after reading, so the rest is dropped from
            # context before writing any edits.
            parts.append(
                f"\n== {fp} ({line_count} lines — large file) ==\n"
                f"{numbered}\n"
                f"⚠ This file is large. After identifying the lines you need to edit,\n"
                f"use [KEEP: {fp} N-M, A-B] [STOP] to keep only the lines you need.\n"
                f"Multiple ranges are supported: [KEEP: {fp} 50-80, 120-150] keeps\n"
                f"both sections and drops everything else from your context.\n"
                f"Only then write your edits — working from the kept region(s).\n"
            )

    if not parts:
        parts.append("(no existing files — create all files from scratch)")

    return "\n".join(parts)


async def _implement_one_step(
    step_info: dict,
    task: str,
    shared_interfaces: str,
    file_contents: dict[str, str],
    sandbox: Sandbox,
    project_root: str,
    plan: str,
    detailed_map: str = "",
    purpose_map: str = "",
    research_cache: dict | None = None,
) -> dict[str, str]:
    """Implement a single plan step with the edit→verify→fix loop.

    1. Coder writes edits (with tool access)
    2. Apply edits, count matches — retry on failures
    3. Syntax + import check — feed errors back for fix
    4. Self-check: coder traces logic on resulting file
    5. Returns updated file_contents dict with this step's changes applied
    """
    step_num = step_info["num"]
    step_name = step_info["name"]
    step_files = step_info["files"]
    step_details = step_info["details"]

    step(f"  Step {step_num}: {step_name}")
    status(f"    Files: {', '.join(step_files) or '(from plan)'}")

    iface_block = ""
    if shared_interfaces:
        iface_block = f"SHARED INTERFACES (use these EXACT names):\n{shared_interfaces}\n"

    step_instructions = (
        f"Implement ONLY this step:\n\n"
        f"STEP {step_num}: {step_name}\n"
        f"Files: {', '.join(step_files)}\n"
        f"{step_details}\n"
    )

    MAX_RETRIES = 5
    # Across-attempt state — carries forward what the model thought and
    # tried in earlier attempts so it doesn't start from scratch.
    prev_attempt_thinking = ""
    prev_attempt_summary = ""
    for attempt in range(1, MAX_RETRIES + 1):
        _wlog.phase_event(
            f"Step {step_num} attempt {attempt}/{MAX_RETRIES}",
        )
        # Only load files this step modifies
        step_file_contents = {}
        modify_set = set()

        # If the plan step had no FILES: line, try to infer from the step body.
        # A missing FILES: line causes _build_file_block to return "(no existing
        # files — create all files from scratch)", making the coder think every
        # file is new and triggering === FILE: === rewrites of existing files.
        effective_files = list(step_files)
        if not effective_files:
            _file_pat = re.compile(
                r'[\w./\-]+\.(?:py|js|ts|jsx|tsx|html|css|json|lean|c|cpp|h|rs|'
                r'java|go|rb|toml|yaml|yml|md|mjs|cjs|svelte|vue)'
            )
            found = _file_pat.findall(step_instructions + " " + step_details)
            # Only include files that are already known to the project
            known = set(file_contents.keys())
            effective_files = list(dict.fromkeys(
                f for f in found if f in known
            ))
            if effective_files:
                warn(f"    Step {step_num}: no FILES: line — inferred {effective_files} from step body")

        for fp in effective_files:
            if fp in file_contents:
                step_file_contents[fp] = file_contents[fp]
                modify_set.add(fp)
            else:
                # Try to find by basename — handles plans that wrote the wrong
                # path (e.g. "tool_call.py" when the file is "core/tool_call.py").
                basename = os.path.basename(fp)
                fuzzy = next(
                    (k for k in file_contents
                     if os.path.basename(k) == basename and file_contents[k]),
                    None,
                )
                if fuzzy:
                    warn(f"    Step {step_num}: '{fp}' not found — "
                         f"resolved to '{fuzzy}' by basename match")
                    step_file_contents[fuzzy] = file_contents[fuzzy]
                    modify_set.add(fuzzy)
                else:
                    full_path = os.path.join(project_root, fp)
                    content = sandbox.load_file(fp) or read_file(full_path) or ""
                    step_file_contents[fp] = content
                    file_contents[fp] = content
                    modify_set.add(fp)

        file_block = _build_file_block(step_file_contents, modify_files=modify_set)

        # Build prev_thinking block — only on attempt 2+
        if prev_attempt_thinking:
            prev_thinking_block = (
                f"\n══════════════════════════════════════════════════════════════════════\n"
                f"YOUR PREVIOUS ATTEMPT (attempt {attempt - 1})\n"
                f"══════════════════════════════════════════════════════════════════════\n"
                f"\n{prev_attempt_summary}\n\n"
                f"This is what you wrote last attempt. Use it to inform this attempt —\n"
                f"don't repeat the same mistakes, but DO reuse correct analysis you\n"
                f"already did. The file content above shows the CURRENT state, which\n"
                f"may differ from what you saw last attempt if some edits applied.\n\n"
                f"--- BEGIN PREVIOUS THINKING ---\n"
                f"{prev_attempt_thinking}\n"
                f"--- END PREVIOUS THINKING ---\n"
            )
        else:
            prev_thinking_block = ""

        impl_prompt = IMPLEMENT_PROMPT.format(
            step_instructions=step_instructions,
            shared_interfaces=iface_block,
            file_content=file_block,
            prev_code="",
            prev_thinking=prev_thinking_block,
        )

        # ── on_stop callback: apply edits mid-stream so [CODE:] sees them ──
        # _seen_edit_keys tracks edit BLOCKS (not file content) that have
        # already been applied this attempt. Each [STOP] re-extracts the
        # full response_so_far, which contains every prior block — without
        # block-level dedup, line-number edits get re-applied against an
        # already-modified file and silently corrupt it.
        # _viewed_versions records what the model saw via [CODE: path]; line
        # edits anchor to those snapshots so line numbers always refer to
        # the version the model was looking at.
        #
        # PRE-POPULATE: the prompt's file_block ALREADY shows the model the
        # current content of each modify-target with line numbers. If the
        # model writes a `[REPLACE LINES]` without ever calling `[CODE:]`,
        # the line numbers refer to THAT inline listing. Seeding
        # _viewed_versions with the same content here keeps the anchor
        # consistent across mid-stream [STOP]s — even after on_stop
        # mutates the file on disk, the line edit still anchors to what
        # the model originally saw.
        _seen_edit_keys: set[str] = set()
        _stop_applied: dict[str, str] = {}
        _viewed_versions: dict[str, str] = {
            fp: content for fp, content in step_file_contents.items() if content
        }

        def _on_stop_apply(response_so_far: str) -> "str | None":
            """Called when the model writes [STOP]. Applies any pending
            edit blocks to the sandbox so subsequent [CODE:] reads
            return the post-edit state.

            Returns a feedback string describing what happened to the
            edits (which applied, which skipped, why) so the runtime
            can surface it to the model in the next round. Returns
            None when there are no new edits to report on.
            """
            try:
                ext = _extract_code_blocks(response_so_far)
                _dedup_against_seen(ext, _seen_edit_keys)
                # If dedup removed everything, there's nothing new to apply.
                if not (ext["edits"] or ext["text_edits"]
                        or ext["new_files"] or ext["reverts"]):
                    return None

                # Snapshot pre-apply line counts so the feedback can
                # report "84 → 112 lines" — explicit signal that the
                # file changed, which the model otherwise has to infer.
                pre_lines = {}
                for fp in list(ext["text_edits"].keys()) + list(ext["edits"].keys()):
                    existing = file_contents.get(fp, "")
                    pre_lines[fp] = existing.count('\n') + 1 if existing else 0

                produced, matched, total, skips = _apply_extracted_code(
                    ext, file_contents, sandbox,
                    viewed_versions=_viewed_versions,
                )

                feedback_lines = []
                if produced:
                    for fp, content in produced.items():
                        sandbox.write_file(fp, content)   # ← persist to disk so [CODE:] sees it
                        file_contents[fp] = content
                        _stop_applied[fp] = content
                        post = content.count('\n') + 1
                        pre = pre_lines.get(fp, 0)
                        if pre == 0:
                            feedback_lines.append(
                                f"  ✓ CREATED  {fp}  ({post} lines written)"
                            )
                        elif pre == post:
                            feedback_lines.append(
                                f"  ✓ MODIFIED {fp}  (still {post} lines — in-place change)"
                            )
                        else:
                            feedback_lines.append(
                                f"  ✓ MODIFIED {fp}  ({pre} → {post} lines)"
                            )
                    status(f"    [STOP] applied {len(produced)} file(s) mid-stream")
                else:
                    status("    [STOP] no edits applied this round")

                if skips:
                    for s in skips:
                        # skips already start with "- " or similar — normalize
                        text = s.strip().lstrip("-").strip()
                        feedback_lines.append(f"  ✗ REJECTED  {text}")
                # Edits that the parser couldn't match at all (SEARCH not found)
                # show up as text_edits in `ext` but absent from `produced`.
                attempted_fps = (
                    set(ext.get("text_edits", {}).keys())
                    | set(ext.get("edits", {}).keys())
                )
                missed_fps = attempted_fps - set(produced.keys())
                for fp in missed_fps:
                    # Don't duplicate skips that already mention this file
                    if any(fp in s for s in skips):
                        continue
                    feedback_lines.append(
                        f"  ✗ REJECTED  edit on {fp}: SEARCH anchor did not "
                        f"match the file. Re-read the file with [CODE:] and "
                        f"copy the exact lines, OR use [REPLACE LINES N-M]."
                    )

                # Reverts and new files (purely informational)
                for rpath in ext.get("reverts", []):
                    feedback_lines.append(f"  ↺ REVERTED {rpath} to prior snapshot")

                if not feedback_lines:
                    return None
                return "\n".join(feedback_lines)
            except Exception as e:
                warn(f"    [STOP] edit apply failed: {e}")
                return f"  ✗ runtime error while applying edits: {e}"

        # ── 1. Coder writes edits ────────────────────────────────────
        impl_result = await _call_with_tools(
            IMPLEMENT_MODEL, impl_prompt, project_root,
            detailed_map=detailed_map, purpose_map=purpose_map,
            research_cache=research_cache,
            log_label=f"step {step_num}: {step_name} (attempt {attempt})",
            on_stop=_on_stop_apply,
            viewed_versions=_viewed_versions,
        )

        # If edits were already applied at [STOP] time, use those results.
        # Otherwise extract and apply from the final response as usual.
        if _stop_applied:
            produced = dict(_stop_applied)
            # Re-extract to catch any edits written AFTER the last [STOP].
            # Use the same _seen_edit_keys set so blocks already applied
            # at [STOP] time aren't applied a second time here.
            extracted = _extract_code_blocks(impl_result["answer"])
            _dedup_against_seen(extracted, _seen_edit_keys)
            late_produced, late_m, late_t, late_skips = _apply_extracted_code(
                extracted, file_contents, sandbox,
                viewed_versions=_viewed_versions,
            )
            if late_produced:
                produced.update(late_produced)
                for fp, content in late_produced.items():
                    sandbox.write_file(fp, content)
                    file_contents[fp] = content
            matched = len(produced)
            total = matched
            # Surface every late skip — the model needs to see why post-STOP
            # blocks failed, even when no new file was produced. Hiding skips
            # when `late_produced` was empty made the retry loop see "no code
            # produced" instead of "your SEARCH didn't match", and burnt the
            # MAX_RETRIES budget retrying the same broken edit.
            ambiguous_skips = list(late_skips)
        else:
            extracted = _extract_code_blocks(impl_result["answer"])
            produced, matched, total, ambiguous_skips = _apply_extracted_code(
                extracted, file_contents, sandbox,
                viewed_versions=_viewed_versions,
            )

        if not produced:
            # Fallback for new files: the model wrote a code block but didn't
            # use the `=== FILE: ===` form. Only accept this if the target
            # file doesn't already exist on disk — otherwise we'd silently
            # overwrite real code with a stray code listing.
            if len(step_files) == 1:
                target_fp = step_files[0]
                existing_content = file_contents.get(target_fp, "")
                if not existing_content.strip():
                    raw_blocks = re.findall(
                        r'```[^\n]*\n(.*?)```', impl_result["answer"], re.DOTALL
                    )
                    if raw_blocks:
                        produced[target_fp] = max(raw_blocks, key=len).strip()
                        matched, total = 1, 1

        if not produced:
            # Check if the model is saying no changes are needed (valid outcome).
            # If it wrote [DONE] and indicated the code is already correct,
            # treat this as a successful no-op rather than retrying forever.
            #
            # STRICT MARKER (replaces the fuzzy-phrase exit that used to
            # false-positive on phrases like "code matches" or "all correct"):
            # the model must write the LITERAL line `STEP COMPLETE: NO CHANGES
            # NEEDED` (case-sensitive on the marker, free text around it).
            # IMPLEMENT_PROMPT and SELF_CHECK_PROMPT teach this marker. Verify
            # steps still get the verb-based exit because their step body is
            # explicit about not producing code.
            answer = impl_result["answer"]
            answer_lower = answer.lower()
            # [DONE] is stripped from the answer text by _call_with_tools before
            # returning, so NEVER search for "[done]" in answer_lower — it will
            # never be there. Use the explicit flag instead.
            done_signaled = impl_result.get("done", False)

            # Verify steps legitimately produce no code — if the step name or
            # details says verify/confirm/no changes, accept a [DONE] response
            # without requiring specific signal phrases.
            is_verify_step = any(kw in step_name.lower() for kw in (
                "verify", "verif", "confirm", "no changes", "no additional",
                "check", "validate",
            )) or any(kw in step_details.lower() for kw in (
                "no additional changes needed", "verification only",
                "no code changes needed", "verify no",
                "no change needed", "no changes needed",
            ))
            if is_verify_step and done_signaled:
                status(f"    Step {step_num}: verified (no changes needed)")
                break

            _NO_CHANGES_MARKER = re.compile(
                r'STEP\s+COMPLETE:\s*NO\s+CHANGES\s+NEEDED', re.IGNORECASE,
            )
            if done_signaled and _NO_CHANGES_MARKER.search(answer):
                status(
                    f"    Step {step_num}: no changes needed "
                    f"(model wrote STEP COMPLETE: NO CHANGES NEEDED)"
                )
                break
            warn(f"    No code produced (attempt {attempt})")
            # Stash thinking so the next attempt knows what was tried
            attempt_thinking = impl_result.get("answer", "")
            if len(attempt_thinking) > 6000:
                attempt_thinking = (
                    "(...earlier portion of this attempt's thinking trimmed...)\n"
                    + attempt_thinking[-6000:]
                )
            prev_attempt_thinking = attempt_thinking
            prev_attempt_summary = (
                "OUTCOME: NO edits were produced. The model wrote a response but "
                "no edit blocks were extractable. Use [SEARCH]/[REPLACE] or "
                "[REPLACE LINES N-M] format wrapped in === EDIT: path === markers."
            )
            continue

        # ── 2. Check match rate ───────────────────────────────────────
        if total > 0 and matched < total:
            failed = total - matched
            warn(f"    {failed}/{total} edits FAILED to match")

            if attempt < MAX_RETRIES:
                # Stash this attempt's thinking so the next attempt can see it.
                # Trim aggressively — the file content blows context budget if
                # the model wrote big edit blocks. Keep the last ~6000 chars
                # (typically 2-4 model "rounds" of analysis + edits).
                attempt_thinking = impl_result.get("answer", "")
                if len(attempt_thinking) > 6000:
                    attempt_thinking = (
                        "(...earlier portion of this attempt's thinking trimmed...)\n"
                        + attempt_thinking[-6000:]
                    )
                prev_attempt_thinking = attempt_thinking
                # Build a structured summary of what failed
                if ambiguous_skips:
                    skip_details = "\n".join(ambiguous_skips)
                    prev_attempt_summary = (
                        f"OUTCOME: {failed} of {total} edits SKIPPED — SEARCH blocks "
                        f"matched multiple locations. Use anchored [SEARCH: N-M] form.\n"
                        f"Specific failures:\n{skip_details}"
                    )
                else:
                    prev_attempt_summary = (
                        f"OUTCOME: {failed} of {total} edits did NOT match. The file "
                        f"content shown above is the CURRENT state. Use line numbers "
                        f"from the file listing — they are accurate."
                    )
                status(f"    Retrying step {step_num} with fresh file state...")
                continue
            else:
                warn(f"    Proceeding with {matched}/{total} matched edits")

        status(f"    Edits applied: {matched}/{total} matched")

        # Write to sandbox + update file_contents
        # First snapshot which files already had syntax errors BEFORE this
        # step's edits. We only want to show the model errors it introduced —
        # asking it to fix pre-existing errors in unrelated parts of the file
        # sends it on a wild-goose chase that cascades into new errors.
        pre_edit_errors = {}
        for fp in produced:
            original_content = file_contents.get(fp, "")
            if original_content:
                ok, msg = _check_syntax(fp, original_content)
                if not ok:
                    pre_edit_errors[fp] = msg

        syntax_errors = {}
        for fp, content in produced.items():
            sandbox.write_file(fp, content)
            file_contents[fp] = content
            status(f"    {fp}: done ({content.count(chr(10)) + 1} lines)")

            # Only flag errors that are NEW — not ones that existed before
            passed, err_msg = _check_syntax(fp, content)
            if not passed and fp not in pre_edit_errors:
                syntax_errors[fp] = err_msg
                warn(f"    {fp}: syntax error detected")

        # ── 4. Syntax-fix-only loop ───────────────────────────────────
        # The general LLM self-review pass was removed: when files parse,
        # we trust the coder + Phase 3.5 code-review to catch logic bugs
        # without spending another ~25 min per step on a re-read pass.
        # We still keep the loop body below for the syntax-error case —
        # if `_check_syntax` flagged anything, the coder gets one focused
        # chance to fix only the broken file(s). When no syntax errors
        # were detected, skip the whole loop and return immediately.
        if not syntax_errors:
            return produced
        MAX_VERIFY_ROUNDS = 5
        coder_thinking = impl_result.get("answer", "")

        # Strip edit blocks from thinking — we only want the REASONING,
        # not the code. If the code has bad indent, showing it here would
        # cause the self-checker to repeat the same mistake.
        coder_thinking = re.sub(
            r'===\s*(?:EDIT|FILE):\s*\S+.*?(?:```|\[/REPLACE\]|\[/INSERT\]|\[DELETE\s|<<<END>>>)',
            '[... edit block removed ...]',
            coder_thinking, flags=re.DOTALL,
        )
        # Also strip standalone REPLACE/INSERT blocks not inside === EDIT:
        coder_thinking = re.sub(
            r'\[REPLACE\s+LINES?\s+\d+\s*-\s*\d+\s*\].*?\[/REPLACE\]',
            '[... edit block removed ...]',
            coder_thinking, flags=re.DOTALL,
        )
        coder_thinking = re.sub(
            r'\[INSERT\s+AFTER\s+LINE\s+\d+\s*\].*?\[/INSERT\]',
            '[... edit block removed ...]',
            coder_thinking, flags=re.DOTALL,
        )

        # Trim thinking to last 4000 chars to avoid bloat
        if len(coder_thinking) > 4000:
            coder_thinking = "(...earlier thinking trimmed...)\n" + coder_thinking[-4000:]

        prev_syntax_errors = {}
        repeat_count = 0
        for verify_round in range(1, MAX_VERIFY_ROUNDS + 1):
            # Build file list — names + line counts + syntax errors
            files_list_parts = []
            for fp, content in produced.items():
                line_count = content.count('\n') + 1
                entry = f"  {fp} — {line_count} lines (use [CODE: {fp}] to read)"
                if fp in syntax_errors:
                    entry += f"\n    ⚠ SYNTAX ERROR:\n{syntax_errors[fp]}"

                    if syntax_errors == prev_syntax_errors:
                        repeat_count += 1
                        # Extract error line number
                        err_line_match = re.search(r'line\s+(\d+)', syntax_errors[fp], re.IGNORECASE)
                        if err_line_match:
                            err_line = int(err_line_match.group(1))
                            file_lines = content.split('\n')
                            ctx_start = max(0, err_line - 15)
                            ctx_end = min(len(file_lines), err_line + 5)
                            ctx_lines = []
                            for i in range(ctx_start, ctx_end):
                                marker = " >>>" if i + 1 == err_line else "    "
                                ctx_lines.append(f"{marker} {i+1:4d} | {file_lines[i]}")
                            entry += (
                                f"\n\n    ⚠ SAME ERROR REPEATED {repeat_count}x — "
                                f"here is the actual code around the error:\n"
                                + "\n".join(ctx_lines)
                                + f"\n\n    Look at the indentation of the lines ABOVE "
                                f"line {err_line}. Your replacement must match that "
                                f"indent level. Include the enclosing def/class line "
                                f"in your [REPLACE LINES] block."
                            )
                    else:
                        repeat_count = 0

                files_list_parts.append(entry)

            prev_syntax_errors = dict(syntax_errors)

            # Build a LOUD banner for syntax errors so the model can't ignore
            # them. Without this, models often [KEEP:] only the edited regions
            # and miss broken lines OUTSIDE those ranges (e.g. a stray line-
            # number trailer on an adjacent unchanged line). The banner names
            # the file + error line and forbids VERIFIED until it's fixed.
            syntax_banner = ""
            if syntax_errors:
                lines_summary = []
                for fp, msg in syntax_errors.items():
                    err_line_match = re.search(r'line\s+(\d+)', msg, re.IGNORECASE)
                    err_line = err_line_match.group(1) if err_line_match else "?"
                    lines_summary.append(f"    • {fp} line {err_line}")
                syntax_banner = (
                    "\n══════════════════════════════════════════════════════════════════════\n"
                    "🚨 SYNTAX ERROR — FIX THIS BEFORE ANYTHING ELSE\n"
                    "══════════════════════════════════════════════════════════════════════\n"
                    f"The file(s) below DO NOT PARSE. The user cannot run this code.\n"
                    "  Broken file(s):\n"
                    + "\n".join(lines_summary)
                    + "\n\n"
                    "PROCESS:\n"
                    "  1. [CODE: <broken_file>] to read the WHOLE file (do NOT use [KEEP:]\n"
                    "     to focus on the edited region — the broken line might be OUTSIDE\n"
                    "     the edited range, e.g. a stray trailing integer on an adjacent line).\n"
                    "  2. Find the line cited in the error context above.\n"
                    "  3. Common cause: a stray trailing integer on a line — that's a\n"
                    "     line number from the [CODE:] view that got copied into REPLACE\n"
                    "     content. Strip it.\n"
                    "  4. Write a [SEARCH]/[REPLACE] fix wrapped in === EDIT: <path> ===.\n"
                    "  5. Do NOT write VERIFIED until the file PARSES.\n"
                    "══════════════════════════════════════════════════════════════════════\n"
                )

            check_prompt = SELF_CHECK_PROMPT.format(
                task=task,
                step_name=f"Step {step_num}: {step_name}",
                step_details=step_details,
                coder_thinking=syntax_banner + coder_thinking,
                changed_files_list="\n".join(files_list_parts),
            )

            # on_stop for self-check: apply fix edits mid-stream
            _sc_seen_edit_keys: set[str] = set()
            _sc_stop_applied: dict[str, str] = {}
            # Pre-seed with the post-coder file state. The self-check prompt
            # lists files by name + line count (not full content), but if
            # the model writes [REPLACE LINES] right after [CODE:] reads it,
            # the line numbers refer to that read. Seeding from `produced`
            # gives a sensible default basis when [REPLACE LINES] arrives
            # before any [CODE:] read in the same response.
            _sc_viewed_versions: dict[str, str] = {
                fp: content for fp, content in produced.items() if content
            }

            def _on_stop_selfcheck(response_so_far: str) -> "str | None":
                """Apply fix edits during self-check. Returns a feedback
                string describing what applied vs was rejected so the
                runtime can show the verifier explicit results next round
                (same fix as for the coder)."""
                try:
                    ext = _extract_code_blocks(response_so_far)
                    # Self-check may not create new files — only fix existing ones.
                    # Line edits ([REPLACE LINES]) are allowed: the prompt instructs
                    # the model to use them and they are anchored to _sc_viewed_versions.
                    ext["new_files"] = {}
                    _dedup_against_seen(ext, _sc_seen_edit_keys)
                    if not (ext["edits"] or ext["text_edits"] or ext["reverts"]):
                        return None
                    pre_lines = {}
                    for fp in list(ext["text_edits"].keys()) + list(ext["edits"].keys()):
                        existing = file_contents.get(fp, "")
                        pre_lines[fp] = existing.count('\n') + 1 if existing else 0
                    produced, matched, total, skips = _apply_extracted_code(
                        ext, file_contents, sandbox,
                        viewed_versions=_sc_viewed_versions,
                    )
                    feedback_lines = []
                    if produced:
                        for fp, content in produced.items():
                            sandbox.write_file(fp, content)
                            file_contents[fp] = content
                            _sc_stop_applied[fp] = content
                            post = content.count('\n') + 1
                            pre = pre_lines.get(fp, 0)
                            if pre == post:
                                feedback_lines.append(f"  ✓ FIX APPLIED {fp} (still {post} lines)")
                            else:
                                feedback_lines.append(f"  ✓ FIX APPLIED {fp} ({pre} → {post} lines)")
                        status(f"    [STOP] self-check applied {len(produced)} fix(es)")
                    for s in skips:
                        feedback_lines.append(f"  ✗ FIX REJECTED  {s.strip().lstrip('-').strip()}")
                    attempted = set(ext.get("text_edits", {}).keys()) | set(ext.get("edits", {}).keys())
                    for fp in attempted - set(produced.keys()):
                        if any(fp in s for s in skips):
                            continue
                        feedback_lines.append(
                            f"  ✗ FIX REJECTED  edit on {fp}: SEARCH anchor "
                            f"did not match. Re-read with [CODE:] and copy "
                            f"the exact lines, OR use [REPLACE LINES N-M]."
                        )
                    for rpath in ext.get("reverts", []):
                        feedback_lines.append(f"  ↺ REVERTED {rpath} to prior snapshot")
                    return "\n".join(feedback_lines) if feedback_lines else None
                except Exception as e:
                    warn(f"    [STOP] self-check apply failed: {e}")
                    return f"  ✗ runtime error during self-check apply: {e}"

            check_result = await _call_with_tools(
                IMPLEMENT_MODEL, check_prompt, project_root,
                detailed_map=detailed_map, purpose_map=purpose_map,
                research_cache=research_cache,
                log_label=f"self-check step {step_num} (round {verify_round})",
                on_stop=_on_stop_selfcheck,
                viewed_versions=_sc_viewed_versions,
            )

            check_answer = check_result.get("answer", "")

            # Verified: model declared VERIFIED AND there are no NEW (un-applied)
            # edit blocks left over after on_stop ran. We can't trust literal
            # "[REPLACE" detection — on_stop applies edits mid-stream but the
            # text remains in the answer. Instead, re-extract and dedup against
            # _sc_seen_edit_keys: anything left is genuinely unapplied.
            if "VERIFIED" in check_answer.upper():
                pending = _extract_code_blocks(check_answer)
                pending["new_files"] = {}
                _dedup_against_seen(pending, _sc_seen_edit_keys)
                has_unapplied = bool(
                    pending["edits"] or pending["text_edits"] or pending["reverts"]
                )
                if not has_unapplied and not syntax_errors:
                    success(f"    Step {step_num} verified (round {verify_round})")
                    break
                if has_unapplied and not syntax_errors:
                    # Verifier said VERIFIED but wrote an edit without a [STOP]
                    # before [DONE] — the edit wasn't applied by on_stop.
                    # Apply it now and break rather than forcing a whole extra round.
                    late, _, _, _ = _apply_extracted_code(
                        pending, file_contents, sandbox,
                        viewed_versions=_sc_viewed_versions,
                    )
                    if late:
                        for fp, content in late.items():
                            sandbox.write_file(fp, content)
                            file_contents[fp] = content
                            produced[fp] = content
                    success(f"    Step {step_num} verified (round {verify_round}, late edits applied)")
                    break
                if syntax_errors and not has_unapplied:
                    # Model said VERIFIED but the file still has a syntax error
                    # and no fix was written. Force another round.
                    warn(f"    Self-check round {verify_round}: VERIFIED claimed but syntax errors remain — forcing another round")
                    coder_thinking = (
                        f"[Self-check round {verify_round}: you wrote VERIFIED but the "
                        f"file STILL has a syntax error. Read the file fresh and write "
                        f"a real fix. Do NOT write VERIFIED until the syntax error is gone.]"
                    )

            # Extract and apply fixes. Self-check may use [REPLACE LINES N-M]
            # (as the prompt instructs) or [SEARCH]/[REPLACE]. New files are
            # still forbidden — the self-checker only fixes existing files.
            if _sc_stop_applied:
                fix_produced = dict(_sc_stop_applied)
                # Also catch any edits written after the last [STOP], using the
                # same seen-set so already-applied blocks aren't double-applied.
                fix_extracted = _extract_code_blocks(check_answer)
                fix_extracted["new_files"] = {}
                _dedup_against_seen(fix_extracted, _sc_seen_edit_keys)
                late_fix, _, _, v_skips = _apply_extracted_code(
                    fix_extracted, file_contents, sandbox,
                    viewed_versions=_sc_viewed_versions,
                )
                if late_fix:
                    fix_produced.update(late_fix)
                    for fp, content in late_fix.items():
                        file_contents[fp] = content
                v_matched = len(fix_produced)
                v_total = v_matched
            else:
                fix_extracted = _extract_code_blocks(check_answer)
                fix_extracted["new_files"] = {}
                fix_produced, v_matched, v_total, v_skips = _apply_extracted_code(
                    fix_extracted, file_contents, sandbox,
                    viewed_versions=_sc_viewed_versions,
                )

            if fix_produced:
                for fp, content in fix_produced.items():
                    sandbox.write_file(fp, content)
                    file_contents[fp] = content
                    produced[fp] = content

                # Re-check ALL produced files — not just the ones written this
                # round. A previous round may have shifted line numbers or left
                # a stale error in syntax_errors that no longer reflects the
                # file on disk. Re-checking everything keeps the dict accurate.
                syntax_errors = {}
                for fp, content in produced.items():
                    passed, err_msg = _check_syntax(fp, content)
                    if not passed:
                        syntax_errors[fp] = err_msg

                # Replace coder_thinking with a minimal summary.
                # Keeping the full analysis text is harmful: it describes
                # what the code looked like BEFORE this round's fix, which
                # leads the next round to write SEARCH blocks for a state
                # that no longer exists. Give only a one-line status so the
                # next round reads the file fresh rather than reasoning from
                # a stale picture.
                coder_thinking = (
                    f"[Self-check round {verify_round} applied {len(fix_produced)} fix(es). "
                    f"Read the file(s) fresh to see the current state.]"
                )

                status(f"    Self-check round {verify_round}: applied {len(fix_produced)} fixes")

            elif v_skips:
                # Every edit was skipped because its SEARCH block matched
                # multiple locations. This is NOT the same as "nothing to fix"
                # — the syntax error is still there; we just can't apply the
                # fix blindly. Feed the specific skip reasons back so the model
                # widens those SEARCH blocks in the next round rather than
                # silently exiting as verified.
                skip_details = "\n".join(v_skips)
                warn(f"    Self-check round {verify_round}: all edits skipped (ambiguous SEARCH blocks)")
                coder_thinking = (
                    f"[Self-check round {verify_round}: your edits were NOT applied because "
                    f"the following SEARCH blocks each matched multiple locations in the file:\n"
                    f"{skip_details}\n"
                    f"Use the ANCHORED form to pin each edit to the right location:\n"
                    f"  [SEARCH: start-end]  (e.g. [SEARCH: 45-49])\n"
                    f"  exact code\n"
                    f"  [/SEARCH]\n"
                    f"  [REPLACE]\n"
                    f"  fixed code\n"
                    f"  [/REPLACE]\n"
                    f"Use the line numbers from [CODE:] or [KEEP:] output. "
                    f"The syntax error is still present.]"
                )
                # Don't break — force another round

            else:
                # Nothing was applied this round and there were no skips.
                # Only declare verified if the file actually parses. If
                # syntax_errors is non-empty, the model gave up without
                # fixing — force another round (or break out at MAX_VERIFY).
                if not syntax_errors:
                    success(f"    Step {step_num} verified (no actionable fixes)")
                    break
                warn(f"    Self-check round {verify_round}: no fix applied but {len(syntax_errors)} syntax error(s) remain")
                coder_thinking = (
                    f"[Self-check round {verify_round}: NO fix was applied this round, "
                    f"but the file still has syntax errors. Read the file fresh with "
                    f"[CODE: file] and write an actual fix using [SEARCH]/[REPLACE] or "
                    f"[REPLACE LINES N-M]. Do NOT just describe the fix — write the edit block.]"
                )

        return produced

    warn(f"    Step {step_num}: giving up after {MAX_RETRIES} attempts")
    return {}


async def phase_implement(
    task: str, plan: str, context: str, sandbox: Sandbox,
    project_root: str, files_to_modify: list[str], detailed_map: str = "",
    purpose_map: str = "",
    research_cache: dict | None = None,
) -> tuple[str, Sandbox]:
    """
    Per-step implementation with edit→verify→fix loop.

    For each plan step:
      1. ONE coder writes edits (with tool access for [CODE:]/[REFS:])
      2. Edits applied, match rate checked — hard retry on failures
      3. Syntax + import validation — errors fed back for fix
      4. Self-check: coder sees resulting file, traces logic, fixes bugs
      5. File state updated with fresh line numbers for next step

    Returns: (plan, sandbox_with_changes)
    """
    step("=== Phase 3: IMPLEMENT (per-step loop) ===")
    _wlog.phase_start("implement", task_chars=len(task), plan_chars=len(plan))

    # Collect all target files
    files_to_create = _extract_new_files_from_plan(plan)
    all_files = list(set(files_to_modify + files_to_create))

    # Parse plan steps
    impl_steps = _extract_impl_steps(plan)
    _wlog.phase_event("Parsed plan", n_steps=len(impl_steps),
                      n_files=len(set(files_to_modify + files_to_create)))
    for s in impl_steps:
        all_files.extend(s["files"])
    all_files = list(dict.fromkeys(all_files))  # dedup, preserve order

    if not all_files:
        all_files = files_to_modify if files_to_modify else ["main"]
    all_files = [
        os.path.relpath(f, project_root) if os.path.isabs(f) else f
        for f in all_files
    ]

    status(f"Target files ({len(all_files)}): {', '.join(all_files)}")

    # Load file contents
    file_contents: dict[str, str] = {}
    for fp in all_files:
        full_path = os.path.join(project_root, fp)
        existing = sandbox.load_file(fp) or read_file(full_path) or ""
        file_contents[fp] = existing

    # Extract shared interfaces
    shared_interfaces = _extract_shared_interfaces(plan)

    # ── If plan has structured steps, implement per-step ──────────────
    if impl_steps:
        status(f"Plan has {len(impl_steps)} steps — implementing each separately")

        total_produced = {}
        # Last-line defense against re-running the same step number even if
        # _extract_impl_steps somehow returned a duplicate. Each step number
        # may run at MOST once per phase_implement call.
        executed_step_nums: set[int] = set()
        for step_info in impl_steps:
            num = step_info["num"]
            if num in executed_step_nums:
                warn(f"  STEP {num} already executed — skipping duplicate")
                _wlog.phase_warn("Duplicate step number — skipped", step=num)
                continue
            executed_step_nums.add(num)
            _wlog.phase_event(
                f"Step {num} START",
                name=step_info.get("name", "")[:60],
                files=",".join(step_info.get("files", []))[:120],
            )
            try:
                step_result = await _implement_one_step(
                    step_info=step_info,
                    task=task,
                    shared_interfaces=shared_interfaces,
                    file_contents=file_contents,
                    sandbox=sandbox,
                    project_root=project_root,
                    plan=plan,
                    detailed_map=detailed_map,
                    purpose_map=purpose_map,
                    research_cache=research_cache,
                )
            except Exception as ex:
                _wlog.phase_error(f"Step {num} raised",
                                  error=str(ex)[:200])
                raise
            total_produced.update(step_result)
            # Per-step snapshot: which files this step actually wrote.
            step_summary = "\n".join(
                f"- {fp} ({len(content):,} chars)"
                for fp, content in step_result.items()
            ) or "(no files produced)"
            _wlog.save_step(
                step_num=num,
                step_name=step_info.get("name", f"step_{num}")[:40],
                model_id="(see per-model logs)",
                content=(
                    f"# Step {num}: {step_info.get('name', '')}\n\n"
                    f"## Files produced\n{step_summary}\n\n"
                    f"## Files declared in step\n"
                    f"{', '.join(step_info.get('files', [])) or '(none)'}\n\n"
                    f"## Step body\n{step_info.get('body', '')}\n"
                ),
                extra={"files_produced": len(step_result)},
            )
            _wlog.phase_event(f"Step {num} END",
                              files_produced=len(step_result))

        if total_produced:
            success(f"Phase 3 complete — {len(total_produced)} files implemented across {len(executed_step_nums)} steps")
            _wlog.phase_end("implement", n_steps=len(executed_step_nums),
                            n_files=len(total_produced))
        else:
            warn("Phase 3: no files produced")
            _wlog.phase_end("implement", n_steps=len(executed_step_nums),
                            n_files=0, status="empty")

        return plan, sandbox

    # ── Fallback: no steps parsed — single-step implementation ────────
    status("No structured steps found — single-pass implementation")

    fallback_step = {
        "num": 1,
        "name": "implement all changes",
        "depends_on": [],
        "files": all_files,
        "details": plan[:12000],
        "done": False,
        "produced_files": {},
    }

    await _implement_one_step(
        step_info=fallback_step,
        task=task,
        shared_interfaces=shared_interfaces,
        file_contents=file_contents,
        sandbox=sandbox,
        project_root=project_root,
        plan=plan,
        detailed_map=detailed_map,
        purpose_map=purpose_map,
        research_cache=research_cache,
    )

    return plan, sandbox


# =====================================================================
#  PHASE 3.5 -- REVIEW (single AI reviews ALL changes together)
# =====================================================================

async def phase_review(
    task: str, plan: str, sandbox: Sandbox,
    project_root: str, detailed_map: str = "",
    purpose_map: str = "",
    context: str = "",
    research_cache: dict | None = None,
) -> tuple[bool, Sandbox]:
    """
    ONE reviewer AI sees ALL changed files together, assesses whether the
    implementation will actually work as a whole, and fixes issues.
    Uses [REFS:], [LSP:], etc. to verify cross-file dependencies.
    Returns: (had_fixes, sandbox)
    """
    step("=== Phase 3.5: CODE REVIEW (single reviewer, all files) ===")
    _wlog.phase_start("review")

    # Collect all changed files
    changed_files = {}
    for fp, content in sandbox.modified_files.items():
        changed_files[fp] = content
    for fp, content in sandbox.new_files.items():
        changed_files[fp] = content

    if not changed_files:
        status("No changed files to review")
        _wlog.phase_end("review", status="skipped — no changed files")
        return False, sandbox

    status(f"Reviewing {len(changed_files)} changed file(s) together: {', '.join(changed_files.keys())}")
    _wlog.phase_event("Reviewing files", n_files=len(changed_files),
                      files=",".join(changed_files.keys())[:200])

    # Build file block — small files inline, large files listed for [CODE:]+[KEEP:]
    all_files_block = ""
    for fp, content in changed_files.items():
        line_count = content.count('\n') + 1
        if line_count <= SMALL_FILE_THRESHOLD:
            numbered = add_line_numbers(content)
            all_files_block += f"\n{'═' * 60}\n== {fp} ({line_count} lines) ==\n{'═' * 60}\n{numbered}\n"
        else:
            all_files_block += (
                f"\n{'═' * 60}\n{fp} — {line_count} lines "
                f"(use [CODE: {fp}] to read, then [KEEP:] to focus)\n{'═' * 60}\n"
            )

    # Pre-load research cache so reviewer already has planner+coder lookups
    preloaded_research = _format_research_cache(research_cache)

    review_prompt = SYSTEM_KNOWLEDGE + REVIEW_PROMPT_TEMPLATE.format(
        task=task,
        plan=plan[:10000],
        all_files_block=all_files_block,
        context=context[:8000],
        preloaded_research=preloaded_research,
    )
    # on_stop for reviewer: apply fix edits mid-stream.
    # Pre-seed viewed_versions with every changed file's content so that
    # [REPLACE LINES] edits the reviewer writes anchor to the version
    # shown in the review prompt — even when the reviewer writes the
    # line edit without first calling [CODE:]. Without this, line
    # numbers reference the at-apply state which can differ from the
    # state the reviewer was reasoning about after a previous mid-stream
    # [STOP] already mutated the same file.
    _rev_seen_edit_keys: set[str] = set()
    _rev_stop_applied: dict[str, str] = {}
    _rev_viewed_versions: dict[str, str] = {
        fp: content for fp, content in changed_files.items() if content
    }

    def _on_stop_review(response_so_far: str) -> "str | None":
        """Apply reviewer fixes mid-stream. Returns a feedback string
        describing applied vs rejected edits so the runtime can surface
        explicit signal in the next round (same fix as for the coder)."""
        try:
            ext = _extract_code_blocks(response_so_far)
            _dedup_against_seen(ext, _rev_seen_edit_keys)
            if not (ext["edits"] or ext["text_edits"]
                    or ext["new_files"] or ext["reverts"]):
                return None
            pre_lines = {}
            for fp in list(ext["text_edits"].keys()) + list(ext["edits"].keys()):
                existing = changed_files.get(fp, "")
                pre_lines[fp] = existing.count('\n') + 1 if existing else 0
            produced, matched, total, skips = _apply_extracted_code(
                ext, changed_files, sandbox,
                viewed_versions=_rev_viewed_versions,
            )
            feedback_lines = []
            if produced:
                for fp, content in produced.items():
                    changed_files[fp] = content
                    sandbox.write_file(fp, content)
                    _rev_stop_applied[fp] = content
                    post = content.count('\n') + 1
                    pre = pre_lines.get(fp, 0)
                    if pre == post:
                        feedback_lines.append(f"  ✓ FIX APPLIED {fp} (still {post} lines)")
                    else:
                        feedback_lines.append(f"  ✓ FIX APPLIED {fp} ({pre} → {post} lines)")
                status(f"    [STOP] reviewer applied {len(produced)} fix(es)")
            for s in skips:
                feedback_lines.append(f"  ✗ FIX REJECTED  {s.strip().lstrip('-').strip()}")
            attempted = set(ext.get("text_edits", {}).keys()) | set(ext.get("edits", {}).keys())
            for fp in attempted - set(produced.keys()):
                if any(fp in s for s in skips):
                    continue
                feedback_lines.append(
                    f"  ✗ FIX REJECTED  edit on {fp}: SEARCH anchor did not "
                    f"match. Re-read with [CODE:] and copy the exact lines, "
                    f"OR use [REPLACE LINES N-M]."
                )
            for rpath in ext.get("reverts", []):
                feedback_lines.append(f"  ↺ REVERTED {rpath} to prior snapshot")
            return "\n".join(feedback_lines) if feedback_lines else None
        except Exception as e:
            warn(f"    [STOP] reviewer apply failed: {e}")
            return f"  ✗ runtime error during reviewer apply: {e}"

    # Cap reviewer at 10 tool rounds. The reviewer's job is to TRACE
    # the chain and write small fixes — not to investigate forever.
    # If 10 rounds of tools didn't surface the bug, more won't either,
    # and longer reviewer sessions correlate with destructive rewrites
    # (the model gets confused about what it already changed and starts
    # re-replacing already-replaced blocks).
    result = await _call_with_tools(
        "nvidia/glm-5.1", review_prompt, project_root,
        detailed_map=detailed_map, purpose_map=purpose_map,
        research_cache=research_cache,
        log_label="reviewing all changes",
        on_stop=_on_stop_review,
        viewed_versions=_rev_viewed_versions,
        max_rounds=20,
    )
    answer = result.get("answer", "")

    # APPROVED only counts if there are no NEW unapplied edit blocks left over
    # after the on_stop callback. The literal text "[REPLACE]" remains in the
    # answer even after edits are applied mid-stream, so a substring check is
    # wrong — use the same seen-keys dedup that the apply path uses.
    if "APPROVED" in answer.upper():
        pending = _extract_code_blocks(answer)
        _dedup_against_seen(pending, _rev_seen_edit_keys)
        has_unapplied = bool(
            pending["edits"] or pending["text_edits"]
            or pending["new_files"] or pending["reverts"]
        )
        if not has_unapplied:
            success(f"Code review: all {len(changed_files)} files APPROVED")
            _wlog.save_review(answer, model_id=result.get("model", ""))
            _wlog.phase_end("review", approved=True, fixes_applied=0,
                            n_files=len(changed_files))
            return False, sandbox

    # Extract and apply edits — reviewer has read the actual files and may use
    # any edit format including [REPLACE LINES]. Unlike the self-checker (which
    # operates on a potentially shifting sandbox), the reviewer reads real files
    # from disk and writes targeted line-number edits. Blocking those caused
    # reviewer fixes to be silently dropped while reporting "APPROVED".
    if _rev_stop_applied:
        produced = dict(_rev_stop_applied)
        # Catch any edits written after the last [STOP]. Dedup against the
        # mid-stream seen-keys so blocks already applied during the loop
        # aren't applied a second time here (which would have duplicated
        # changes on the file and silently invalidated `pre_lines` tracking).
        extracted = _extract_code_blocks(answer)
        _dedup_against_seen(extracted, _rev_seen_edit_keys)
        late_produced, late_m, late_t, late_skips = _apply_extracted_code(
            extracted, changed_files, sandbox,
            viewed_versions=_rev_viewed_versions,
        )
        if late_produced:
            produced.update(late_produced)
            for fp, content in late_produced.items():
                sandbox.write_file(fp, content)
                changed_files[fp] = content
        for s in late_skips:
            warn(f"  Review late-skip: {s}")
        rev_matched = len(produced)
        rev_total = rev_matched
    else:
        extracted = _extract_code_blocks(answer)
        produced, rev_matched, rev_total, _skips = _apply_extracted_code(
            extracted, changed_files, sandbox,
            viewed_versions=_rev_viewed_versions,
        )
        for s in _skips:
            warn(f"  Review skip: {s}")

    if rev_total > 0 and rev_matched < rev_total:
        warn(f"  Review edits: {rev_matched}/{rev_total} matched (some fixes didn't apply)")

    total_fixes = 0
    for matched_fp, modified in produced.items():
        sandbox.write_file(matched_fp, modified)
        total_fixes += 1
        status(f"  {matched_fp}: fixed")

    if total_fixes:
        success(f"Code review: applied {total_fixes} fixes across {len(changed_files)} files")
    else:
        success(f"Code review: all {len(changed_files)} files APPROVED (no actionable fixes)")

    _wlog.save_review(answer, model_id=result.get("model", ""))
    _wlog.phase_end("review", fixes_applied=total_fixes,
                    n_files=len(changed_files))
    return total_fixes > 0, sandbox


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — TEST (optional)
# ═══════════════════════════════════════════════════════════════════════════════

async def phase_test(test_command: str, project_root: str) -> dict:
    """Run tests. Only called if user asked for testing."""
    step("═══ Phase 4: TEST ═══")
    _wlog.phase_start("test", command=test_command)

    try:
        result = subprocess.run(
            test_command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=project_root,
        )
        passed = result.returncode == 0
        output = result.stdout + "\n" + result.stderr

        if passed:
            success("Tests PASSED")
        else:
            warn(f"Tests FAILED (exit {result.returncode})")

        _wlog.save_test(test_command, output, passed)
        _wlog.phase_end("test", passed=passed, exit_code=result.returncode)
        return {
            "passed": passed,
            "output": output[:10000],
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        warn("Tests timed out (120s)")
        _wlog.phase_error("Tests timed out (120s)")
        _wlog.save_test(test_command, "TIMEOUT after 120s", False)
        return {"passed": False, "output": "TIMEOUT after 120s", "exit_code": -1}
    except Exception as e:
        error(f"Test execution failed: {e}")
        _wlog.phase_error("Test execution failed", error=str(e)[:200])
        _wlog.save_test(test_command, str(e), False)
        return {"passed": False, "output": str(e), "exit_code": -1}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def code_agent(state: AgentState) -> AgentState:
    """
    Full coding agent workflow.
    Expects state to have: raw_input, classification (with complexity, domain).
    Optionally: project_root in state or defaults to cwd.
    """
    step("═══ CODING AGENT ═══")

    task = state.get("processed_input", state["raw_input"])
    _wlog.phase_event(
        "CODING AGENT START",
        task=task[:200].replace("\n", " "),
    )
    classification = state.get("classification", {})
    complexity = classification.get("complexity", 5)

    # Determine project root
    project_root = state.get("project_root", os.getcwd())
    status(f"Project: {project_root}")
    status(f"Complexity: {complexity}")

    # Check if user wants testing
    wants_test = any(kw in task.lower() for kw in ["test", "run test", "verify", "check if"])
    test_command = None

    # Extract test command if specified
    test_match = re.search(r'\[TEST:\s*(.+?)\]', task, re.IGNORECASE)
    if test_match:
        test_command = test_match.group(1)
        wants_test = True
        task = re.sub(r'\[TEST:\s*.+?\]', '', task).strip()

    sandbox = Sandbox(project_root)

    try:
        # ── Generate/load code maps ───────────────────────────────────
        from tools.code_index import generate_maps, list_sections, list_purposes
        maps = await generate_maps(project_root)
        general_map = maps["general"]
        detailed_map = maps["detailed"]
        purpose_map = maps.get("purpose", "")

        # Empty project: clear maps so AIs don't try to look up nothing
        is_new_project = (not detailed_map or detailed_map == "(empty project)")
        if is_new_project:
            detailed_map = ""
            general_map = ""
            purpose_map = ""

        files = []
        if detailed_map:
            sections = list_sections(detailed_map)
            status(f"Code map: {len(sections)} sections indexed")

        purposes = list_purposes(purpose_map) if purpose_map else []
        if purposes:
            status(f"Purpose map: {len(purposes)} categories indexed")

        # Build context from maps
        context_parts = []

        # ── Existing files list — always shown so planner knows what already exists ──
        import glob as _glob
        existing_files = sorted([
            os.path.relpath(p, project_root)
            for p in _glob.glob(os.path.join(project_root, "**", "*.py"), recursive=True)
            + _glob.glob(os.path.join(project_root, "**", "*.js"), recursive=True)
            + _glob.glob(os.path.join(project_root, "**", "*.ts"), recursive=True)
            + _glob.glob(os.path.join(project_root, "**", "*.html"), recursive=True)
            + _glob.glob(os.path.join(project_root, "**", "*.css"), recursive=True)
            + _glob.glob(os.path.join(project_root, "*.toml"), recursive=False)
            + _glob.glob(os.path.join(project_root, "*.json"), recursive=False)
            + _glob.glob(os.path.join(project_root, "*.txt"), recursive=False)
            if not any(skip in p for skip in [
                "/.git/", "/node_modules/", "/__pycache__/", "/.jarvis/",
                "/jarvis_thinking_logs/", "/.venv/", "/venv/", "/dist/",
                "/build/", "/.pytest_cache/",
            ])
        ])
        if existing_files:
            file_list = "\n".join(f"  {f}" for f in existing_files[:200])
            context_parts.append(
                f"FILES ALREADY IN THE PROJECT:\n{file_list}"
            )

        if general_map:
            # Only show section headings — not the full content.
            # A summary that's too detailed lets the planner skip reading actual code.
            # The planner must use [DETAIL:], [CODE:], [REFS:] to learn the real shape.
            heading_lines = [
                line for line in general_map.splitlines()
                if line.startswith("##") or line.startswith("===")
            ]
            heading_summary = "\n".join(heading_lines) if heading_lines else "(see tools below)"
            context_parts.append(
                f"PROJECT STRUCTURE (headings only — use tools to see details):\n"
                f"{heading_summary}\n\n"
                f"⚠️  Do NOT rely on this summary alone. Read the actual files with\n"
                f"[CODE:], [DETAIL:], or [REFS:] before planning changes to them."
            )

            # List available sections
            if detailed_map and sections:
                section_list = "\n".join(f"  - {s}" for s in sections)
                context_parts.append(
                    f"AVAILABLE CODE SECTIONS (use [DETAIL: name] to expand):\n"
                    f"{section_list}"
                )

            # List available purpose categories
            if purposes:
                purpose_list = "\n".join(f"  - {p}" for p in purposes)
                context_parts.append(
                    f"AVAILABLE PURPOSE CATEGORIES:\n"
                    f"{purpose_list}\n"
                    f"\n"
                    f"Two ways to search by purpose:\n"
                    f"  [PURPOSE: exact name]   — use a category name from the list above\n"
                    f"  [SEMANTIC: description] — describe what you want in plain English;\n"
                    f"                            returns the 3 best-matching categories\n"
                    f"                            (use when you don't know the exact category name)\n"
                    f"Each category returns ALL code snippets that serve that purpose,\n"
                    f"with 10 lines of context."
                )

            # Auto-inject relevant knowledge based on the task
            from knowledge import get_auto_inject, list_knowledge
            knowledge_text = get_auto_inject(task)
            if knowledge_text:
                context_parts.append(knowledge_text)

            # List available knowledge topics
            knowledge_topics = list_knowledge()
            if knowledge_topics:
                kl = ", ".join(knowledge_topics)
                context_parts.append(
                    f"AVAILABLE KNOWLEDGE (use [KNOWLEDGE: topic] to consult):\n"
                    f"  {kl}"
                )

            context_parts.append(
                "TOOLS — use in order, escalate only if you need more:\n"
                "  1. [REFS: name]          — definitions, imports, usages (fast)\n"
                "  2. [LSP: name]           — semantic deps, types (if REFS not enough)\n"
                "  3. [DETAIL: section]     — organized code map for a feature\n"
                "     [PURPOSE: category]   — all code for a purpose (e.g. 'UI colors')\n"
                "  4. [CODE: path/to/file]  — read actual source code (last resort)\n"
                "     [SEARCH: pattern]     — ripgrep search\n"
                "     [WEBSEARCH: query]    — web search for API docs\n"
                "  [KNOWLEDGE: topic]       — consult design/game/planning guidelines\n"
                "Wrap tool calls in [tool use]...[/tool use] then [STOP].\n"
                "Tags outside [tool use] blocks are ignored. Add #label to name results.\n"
                "Use [DISCARD: #label] to remove irrelevant results from context."
            )
        else:
            context_parts.append(
                f"PROJECT: {project_root}\n"
                f"STATUS: New project — no existing code. Create all files from scratch.\n"
                f"Do NOT use [DETAIL:] or [CODE:] tags — there is nothing to look up.\n"
                f"Just write the complete implementation directly."
            )

        # Wrap the project-context blob in the unified labelled-section
        # marker so the model can clearly identify what's project facts
        # (factual codebase info from JARVIS) vs. the actual USER REQUEST
        # vs. its own past responses. Matches the [SYSTEM]/[USER REQUEST]
        # /[YOUR ...] vocabulary used in the continuation prompt.
        _ctx_body = "\n\n".join(context_parts)
        context = (
            "══════════════════════════════════════════════════════════════════════\n"
            "[PROJECT CONTEXT] — factual info about this codebase (JARVIS, not the user)\n"
            "══════════════════════════════════════════════════════════════════════\n"
            f"{_ctx_body}\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "[END PROJECT CONTEXT]\n"
            "══════════════════════════════════════════════════════════════════════"
        )

        # Create sandbox
        sandbox.setup()

        # ── Phase 2: PLAN ────────────────────────────────────────────────
        plan, research_cache = await phase_plan(
            task, context, complexity, project_root, "", detailed_map,
            purpose_map=purpose_map, is_new_project=is_new_project,
            files=existing_files if not is_new_project else [],
        )

        # Extract files to modify from plan
        files_to_modify = _extract_files_from_plan(plan, files)
        if not files_to_modify:
            files_to_modify = []
        status(f"Files to modify: {', '.join(files_to_modify) or '(new files)'}")
        status(f"Sharing {len(research_cache)} cached lookups with coders + reviewers")

        # ── Phase 3: IMPLEMENT (parallel coders, shared research) ────────
        final_plan, sandbox = await phase_implement(
            task, plan, context, sandbox, project_root, files_to_modify, detailed_map,
            purpose_map=purpose_map, research_cache=research_cache,
        )

        # ── Phase 3.5: CODE REVIEW (GLM-5 checks code against plan) ─────
        had_fixes, sandbox = await phase_review(
            task, plan, sandbox, project_root, detailed_map, purpose_map, context,
            research_cache=research_cache,
        )

        # ── Phase 4: TEST (optional) ─────────────────────────────────────
        if wants_test and test_command:
            test_result = await phase_test(test_command, project_root)
            if not test_result["passed"]:
                warn("Tests failed — including failure info in output")
                context += f"\n\nTEST FAILURE:\n{test_result['output']}"

        # ── Phase 5: DELIVER ─────────────────────────────────────────────
        step("═══ Phase 5: DELIVER ═══")

        diff = sandbox.get_all_diffs()
        file_summary = sandbox.summary()

        # Ask DeepSeek to summarize AND update maps in parallel
        step("DeepSeek summarizing + updating maps...")

        summary_task = _call(IMPLEMENT_MODEL, SUMMARY_PROMPT.format(
            task=task,
            files_changed=file_summary,
            diff=diff[:15000],
        ), max_tokens=2048, log_label="summarizing changes")

        map_update_task = _call(IMPLEMENT_MODEL, MAP_UPDATE_PROMPT.format(
            task=task,
            files_changed=file_summary,
            diff=diff[:15000],
            general_map=general_map[:8000],
            detailed_map=detailed_map[:30000],
        ), log_label="updating code maps")

        summary_result, map_result = await asyncio.gather(
            summary_task, map_update_task
        )

        ai_summary = summary_result["answer"] if summary_result.get("answer") else file_summary

        # Parse map edits from map_result and apply them
        updated_general = general_map  # fallback: keep current
        updated_detailed = detailed_map
        if map_result.get("answer"):
            raw = map_result["answer"]

            # Split response into general edits section and detailed edits section
            gen_match = re.search(
                r'===\s*GENERAL\s*MAP\s*EDITS\s*===(.*?)(?====\s*DETAILED\s*MAP\s*EDITS\s*===|$)',
                raw, re.DOTALL | re.IGNORECASE
            )
            det_match = re.search(
                r'===\s*DETAILED\s*MAP\s*EDITS\s*===(.*)',
                raw, re.DOTALL | re.IGNORECASE
            )

            if gen_match and "no changes" not in gen_match.group(1).lower()[:100]:
                updated_general = _apply_map_edits(general_map, gen_match.group(1))
            if det_match and "no changes" not in det_match.group(1).lower()[:100]:
                updated_detailed = _apply_map_edits(detailed_map, det_match.group(1))

        output = f"""## Changes Ready

{ai_summary}

Apply these changes to {project_root}? (y/n)"""

        state["final_answer"] = output
        state["pending_sandbox"] = sandbox
        state["updated_maps"] = {
            "general": updated_general,
            "detailed": updated_detailed,
        }
        success("Coding agent complete — waiting for user approval")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        error(f"Coding agent failed: {e}")
        error(tb)
        state["final_answer"] = f"Coding agent error: {e}\n\nTraceback:\n{tb}\n\nPartial results may be in the sandbox."

    finally:
        # Don't cleanup sandbox — user might want to inspect
        pass

    return state


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_files_from_plan(plan: str, known_files: list[str]) -> list[str]:
    """Extract file paths from the plan text (files to MODIFY)."""
    files = set()

    for line in plan.split("\n"):
        matches = re.findall(r'[\w./\-]+\.(?:py|js|ts|html|css|json|lean|c|cpp|h|rs|java|go|rb|toml|yaml|yml|md)', line)
        files.update(matches)

    for f in known_files:
        if f in plan:
            files.add(f)

    return sorted(files)


def _extract_new_files_from_plan(plan: str) -> list[str]:
    """Extract files to CREATE from the plan (listed under FILES TO CREATE)."""
    files = []
    in_create_section = False

    for line in plan.split("\n"):
        line_stripped = line.strip()
        if re.match(r'#+\s*FILES?\s*TO\s*CREATE', line_stripped, re.IGNORECASE):
            in_create_section = True
            continue
        if in_create_section:
            if line_stripped.startswith("##") or (not line_stripped and files):
                break  # End of section
            matches = re.findall(r'[\w./\-]+\.(?:py|js|ts|html|css|json|lean|c|cpp|h|rs|java|go|rb|toml|yaml|yml|md)', line_stripped)
            files.extend(matches)

    return files


def _guess_filename(task: str, content: str) -> str:
    """Guess a filename from task description and file content."""
    task_lower = task.lower()
    content_start = content[:200].lower()

    # Detect from content
    if "<!doctype html" in content_start or "<html" in content_start:
        return "index.html"
    if content_start.strip().startswith("<!doctype"):
        return "index.html"
    if "import react" in content_start or "from react" in content_start:
        return "App.jsx"
    if "package main" in content_start:
        return "main.go"
    if "fn main" in content_start:
        return "main.rs"
    if "public class" in content_start or "public static void main" in content_start:
        return "Main.java"
    if "#include" in content_start:
        return "main.cpp" if "iostream" or "cstdio" in content_start else "main.c"
    if "theorem " in content_start or "import Mathlib" in content_start:
        return "proof.lean"
    if content_start.strip().startswith("{"):
        return "data.json"

    # Detect from task
    if any(kw in task_lower for kw in ["html", "webpage", "website"]):
        return "index.html"
    if any(kw in task_lower for kw in [" css", "stylesheet"]):
        return "style.css"
    if any(kw in task_lower for kw in ["javascript", " js ", ".js"]):
        return "script.js"
    if any(kw in task_lower for kw in ["react", "component", "jsx"]):
        return "App.jsx"
    if any(kw in task_lower for kw in ["lean", "formal proof", "theorem prover"]):
        return "proof.lean"
    if any(kw in task_lower for kw in ["rust", "cargo"]):
        return "main.rs"
    if any(kw in task_lower for kw in ["web app", "web game", "for chrome", "browser game", "in browser"]):
        return "index.html"

    # Default to Python
    return "main.py"
