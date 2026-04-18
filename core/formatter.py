"""
Formatter — two-stage pipeline:
  1. LLM rewrites raw AI output into a clean, coherent answer for the user
     (strips assumptions, internal reasoning, meta-commentary, multi-AI artifacts)
  2. Local ANSI renderer makes it beautiful in the terminal

CRITICAL: If the LLM step fails, fall back to local cleanup + render.
NEVER lose the answer.
"""

import re
from core.cli import step, warn


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 1 — LLM Cleanup (raw AI mess → clean coherent answer)
# ═══════════════════════════════════════════════════════════════════════════

FORMAT_PROMPT = """You are the final output formatter for an AI assistant called JARVIS.
You receive a RAW internal answer that contains useful information mixed with internal
reasoning artifacts. Your job: rewrite it into a CLEAN, NATURAL answer for the user.

## What to REMOVE:
- Lines like "Assumptions:", "Before answering, check:", assumption lists
- Internal meta-commentary: "I'm least confident about...", "What would change my mind..."
- Multi-approach analysis scaffolding: "Approach 1:", "Approach 2:", "Best approach:"
- Confidence notes, "PASS", "FAIL", scoring, self-evaluation markers
- [CONTEXT_NOTES], [ATTEMPT_LOG], [STATUS] sections and everything after them
- [SEARCH:...], [WEBSEARCH:...], [CODE:...] tool tags
- "STOP.", "Continue from where you left off", or other control instructions
- Phrases like "As an AI...", "Based on the context provided..."
- Duplicate information (if the same fact appears multiple times, keep it once)

## What to KEEP:
- ALL actual information, explanations, facts, code, examples
- The user asked a question — make sure it's FULLY answered
- Technical details, formulas, numbers, data
- Code blocks (keep them complete and correct)
- Warnings about important caveats the user should know

## How to WRITE the output:
- Write like a knowledgeable person naturally explaining something in conversation.
- Use flowing prose and paragraphs — NOT bullet-point lists, NOT numbered lists.
- Do NOT use markdown headers (no # or ##). Just write continuous text.
- When you move to a new topic, start a new paragraph. That's your structure.
- Only use bullet points or lists when the content is genuinely a list of items
  (like a list of files, a set of commands, steps to follow).
- Use **bold** sparingly — only for truly key terms, not every other word.
- Use `backticks` for inline code, function names, variable names, commands.
- Use ```language blocks for code — those must stay formatted.
- Be direct. Start with the answer. No preamble like "Great question!" or
  "Here's what you need to know:". Just answer.
- The tone is confident, warm, and concise. Like a smart friend explaining something.
- Avoid robotic patterns like "It's important to note that..." or
  "There are several key considerations:". Just say the thing.

## CRITICAL RULES:
- Do NOT shorten or summarize — keep ALL useful detail
- Do NOT add information that wasn't in the original
- Do NOT wrap your output in markdown code fences
- Your output replaces the original — it must be complete and self-contained
- If the original is already clean and natural, return it mostly unchanged
- Output ONLY the final answer — nothing else

RAW ANSWER TO CLEAN UP:
{answer}"""


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 2 — ANSI Terminal Renderer (markdown → beautiful terminal output)
# ═══════════════════════════════════════════════════════════════════════════

# ANSI codes
R   = "\033[0m"       # reset
B   = "\033[1m"       # bold
DIM = "\033[2m"       # dim
IT  = "\033[3m"       # italic
UL  = "\033[4m"       # underline

RED = "\033[91m"
GRN = "\033[92m"
YEL = "\033[93m"
BLU = "\033[94m"
MAG = "\033[95m"
CYN = "\033[96m"
WHT = "\033[97m"
GRY = "\033[90m"

# Artifacts to strip in basic (no-LLM) cleanup
ARTIFACTS = [
    "[CONTEXT_NOTES]", "[context_notes]", "[Context_Notes]", "[CONTEXT NOTES]",
    "[ATTEMPT_LOG]", "[attempt_log]",
    "[STATUS]", "[status]",
    "CONFIDENCE:", "SEARCH:",
]


def _strip_artifacts(text: str) -> str:
    """Fast local artifact removal — no LLM."""
    clean = text.strip()
    for tag in ARTIFACTS:
        idx = clean.find(tag)
        if idx != -1:
            clean = clean[:idx].strip()
    # Strip tool tags
    clean = re.sub(r'\[(SEARCH|WEBSEARCH|CODE|DETAIL):\s*.+?\]', '', clean)
    # Strip "STOP." control lines
    clean = re.sub(r'^\s*STOP\.\s*$', '', clean, flags=re.MULTILINE)
    # Fix excessive blank lines
    clean = re.sub(r'\n{4,}', '\n\n\n', clean)
    return clean.strip()


def _format_inline(line: str) -> str:
    """Apply inline markdown → ANSI: bold, italic, code, links."""
    # Inline code `...` → cyan (FIRST so bold/italic don't touch code)
    line = re.sub(r'`([^`]+)`', rf'{CYN}\1{R}', line)
    # Bold + italic ***text***
    line = re.sub(r'\*\*\*(.+?)\*\*\*', rf'{B}{IT}\1{R}', line)
    # Bold **text**
    line = re.sub(r'\*\*(.+?)\*\*', rf'{B}{WHT}\1{R}', line)
    # Italic *text* (not inside words)
    line = re.sub(r'(?<!\w)\*([^*]+?)\*(?!\w)', rf'{IT}\1{R}', line)
    # Links [text](url)
    line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', rf'{BLU}{UL}\1{R} {GRY}(\2){R}', line)
    return line


def _render_ansi(text: str) -> str:
    """Convert clean markdown to beautiful ANSI terminal output."""
    lines = text.split("\n")
    output: list[str] = []
    in_code_block = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Code Blocks ──────────────────────────────────────────
        if stripped.startswith("```"):
            if not in_code_block:
                code_lang = stripped[3:].strip()
                output.append(f"  {GRY}┌{'─' * 60}{R}")
                if code_lang:
                    output.append(f"  {GRY}│{R} {DIM}{CYN} {code_lang} {R}")
                    output.append(f"  {GRY}├{'─' * 60}{R}")
                in_code_block = True
            else:
                output.append(f"  {GRY}└{'─' * 60}{R}")
                in_code_block = False
            i += 1
            continue

        if in_code_block:
            output.append(f"  {GRY}│{R}  {WHT}{line}{R}")
            i += 1
            continue

        # ── Empty lines ──────────────────────────────────────────
        if not stripped:
            output.append("")
            i += 1
            continue

        # ── Horizontal rules ─────────────────────────────────────
        if re.match(r'^[-*_]{3,}\s*$', stripped):
            output.append(f"  {GRY}{'─' * 50}{R}")
            i += 1
            continue

        # ── Headers ──────────────────────────────────────────────
        h_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if h_match:
            level = len(h_match.group(1))
            title = re.sub(r'\*\*(.+?)\*\*', r'\1', h_match.group(2))
            title = re.sub(r'\*(.+?)\*', r'\1', title)

            if level == 1:
                output.append("")
                output.append(f"  {B}{CYN}{'━' * len(title)}{R}")
                output.append(f"  {B}{CYN}{title.upper()}{R}")
                output.append(f"  {B}{CYN}{'━' * len(title)}{R}")
            elif level == 2:
                output.append("")
                output.append(f"  {B}{YEL}■ {title}{R}")
                output.append(f"  {YEL}{'─' * (len(title) + 2)}{R}")
            elif level == 3:
                output.append(f"  {B}{WHT}▸ {title}{R}")
            else:
                output.append(f"  {B}{GRY}▪ {title}{R}")
            i += 1
            continue

        # ── Blockquotes ──────────────────────────────────────────
        if stripped.startswith(">"):
            qt = _format_inline(stripped.lstrip("> ").strip())
            output.append(f"  {CYN}┃{R} {DIM}{IT}{qt}{R}")
            i += 1
            continue

        # ── Unordered list items ─────────────────────────────────
        ul_match = re.match(r'^(\s*)[*\-+]\s+(.+)$', line)
        if ul_match:
            indent = len(ul_match.group(1))
            content = _format_inline(ul_match.group(2))
            pad = "  " + "  " * (indent // 2)
            bullet = f"{CYN}•{R}" if indent == 0 else f"{GRY}◦{R}"
            output.append(f"{pad}{bullet} {content}")
            i += 1
            continue

        # ── Ordered list items ───────────────────────────────────
        ol_match = re.match(r'^(\s*)(\d+)[.)]\s+(.+)$', line)
        if ol_match:
            indent = len(ol_match.group(1))
            num = ol_match.group(2)
            content = _format_inline(ol_match.group(3))
            pad = "  " + "  " * (indent // 2)
            output.append(f"{pad}{CYN}{num}.{R} {content}")
            i += 1
            continue

        # ── Regular paragraph ────────────────────────────────────
        output.append(f"  {_format_inline(stripped)}")
        i += 1

    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

async def format_output(answer: str) -> str:
    """
    Two-stage formatter:
      1. LLM rewrites raw answer → clean coherent response
      2. Local ANSI renderer → beautiful terminal output

    NEVER crashes. NEVER loses the answer.
    Falls back to local-only cleanup on any error.
    """
    step("Format output")

    # Very short answers — just render, skip LLM
    if len(answer) < 120:
        return _render_ansi(_strip_artifacts(answer))

    # ── Stage 1: LLM cleanup ────────────────────────────────────────────
    clean = answer
    try:
        from clients.gemini import call_flash
        result = await call_flash(
            FORMAT_PROMPT.format(answer=answer),
            max_tokens=16384,
        )

        formatted = result.strip()
        # Strip markdown code fences the LLM might wrap output in
        formatted = re.sub(r'^```\w*\n?', '', formatted)
        formatted = re.sub(r'\n?```$', '', formatted)
        formatted = formatted.strip()

        # Guard: if LLM shortened too much, use original
        if len(formatted) < len(answer) * 0.4:
            warn(f"Formatter shortened {len(answer)} → {len(formatted)} chars — using raw")
            clean = _strip_artifacts(answer)
        elif len(formatted) < 30:
            warn("Formatter returned empty — using raw")
            clean = _strip_artifacts(answer)
        else:
            clean = formatted

    except Exception as e:
        warn(f"LLM cleanup failed ({e}) — using local cleanup")
        clean = _strip_artifacts(answer)

    # ── Stage 2: ANSI rendering ─────────────────────────────────────────
    try:
        rendered = _render_ansi(clean)
        if not rendered.strip():
            warn("ANSI render produced empty output — using clean text")
            return clean
        return rendered
    except Exception as e:
        warn(f"ANSI render failed ({e}) — returning clean text")
        return clean
