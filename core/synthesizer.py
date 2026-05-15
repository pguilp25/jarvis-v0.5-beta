"""
Synthesizer — three outcomes:
  AGREE: same conclusion → use best answer
  COMPLEMENTARY: different but all valid (recipes, lists, recommendations) → merge them
  CONFLICT: actually contradicting each other → majority vote or tiebreaker

This prevents calling Gemini on questions with multiple valid answers.
"""

import re
from core.retry import call_with_retry
from core.cli import step, agree, disagree, status


VERIFIER_PROMPT = """You compare multiple AI answers to the same question and
classify the relationship between them in ONE word.

Question: {question}

{answers_block}

Classification rules:
  AGREE         — they reach the SAME general conclusion, even with
                  different wording, examples, structure, or detail.
  COMPLEMENTARY — they give DIFFERENT answers that are all VALID
                  (different recipes, recommendations, angles, examples).
  CONFLICT      — at least one answer seems WRONG or INCOMPATIBLE
                  with the others: contradictory facts/numbers, opposite
                  recommendations, or one claims something the others
                  contradict in a way that changes the bottom line.

Reminders:
  • Different wording of the same idea           → AGREE
  • Different examples for the same point        → AGREE
  • Different levels of detail                   → AGREE
  • Different valid options for an open question → COMPLEMENTARY
  • Bottom lines that contradict                 → CONFLICT

Reply with ONLY one word: AGREE, COMPLEMENTARY, or CONFLICT."""


MERGE_PROMPT = """Multiple AIs gave different but all valid answers to this question.
Merge them into ONE comprehensive answer that COMBINES the unique content
from each — do not summarize, do not invent.

Question: {question}

{answers_block}

Rules:
- Include the unique valuable content from EACH answer.
- Keep ALL calculations, numbers, specific data, and failure-mode analyses.
- Remove only exact duplicates — if two AIs make the same point with
  different wording, keep the clearer version.
- Organize logically with clear sections.
- Do NOT summarize or shorten the inputs.
- Do NOT add content that wasn't in any input — "expand" only by
  including content other answers had that this output is missing.

Write the complete merged answer:"""


SYNTHESIZER_PROMPT = """You are a vote counter. Multiple AIs gave CONFLICTING answers.
Your job: count which answer appears most often. You do NOT judge quality.

Question: {question}

{answers_block}

Rules:
- Group answers by their core conclusion (ignore wording differences)
- Count how many AIs reach the same conclusion
- If genuinely tied, say TIED

Output format:
MAJORITY (X/N): [the majority answer, synthesized clearly]
Where X is how many AIs agree and N is the total number of AIs.
or
TIED: [both positions summarized]"""


async def verify_agreement(question: str, answers: list[dict]) -> str:
    """
    Check how answers relate. Uses Gemini Flash Lite for fast judgment.
    Returns: "agree", "complementary", or "conflict"
    """
    block = _format_answers(answers)

    from clients.gemini import call_flash
    result = await call_flash(
        VERIFIER_PROMPT.format(question=question, answers_block=block[:60000]),
        max_tokens=64,
    )

    cleaned = result.strip().upper()

    if cleaned.startswith("AGREE"):
        agree()
        return "agree"
    elif cleaned.startswith("COMP"):
        status("Answers are complementary — merging")
        return "complementary"
    else:
        disagree()
        return "conflict"


async def merge_answers(question: str, answers: list[dict], fast: bool = False) -> dict:
    """
    Merge complementary answers.
    fast=True: Flash Lite (for intelligent tier)
    fast=False: deepseek-v4-flash (for very intelligent tier)
    """
    step("Merging complementary answers")

    block = _format_answers(answers)
    total = len(answers)

    if fast:
        from clients.gemini import call_flash
        merged = await call_flash(
            MERGE_PROMPT.format(question=question, answers_block=block[:60000]),
            max_tokens=16384,
        )
    else:
        merged = await call_with_retry(
            "nvidia/deepseek-v4-flash",
            MERGE_PROMPT.format(question=question, answers_block=block[:60000]),
            max_tokens=16384,
        )

    status(f"Merged {total} answers")
    return {
        "answer": merged.strip(),
        "vote_split": f"merged/{total}",
        "total": total,
        "tied": False,
    }


async def synthesize(question: str, answers: list[dict]) -> dict:
    """
    Majority vote for CONFLICTING answers.
    Returns: {"answer": str, "vote_split": str, "total": int, "tied": bool}
    """
    step("Synthesize — majority vote (conflict)")

    block = _format_answers(answers)
    total = len(answers)
    total_tokens = len(block) // 4

    if total_tokens < 8000:
        model = "nvidia/kimi-k2.6"
    else:
        model = "nvidia/glm-5"

    result = await call_with_retry(
        model,
        SYNTHESIZER_PROMPT.format(question=question, answers_block=block),
        max_tokens=16384,
    )

    cleaned = result.strip()

    if cleaned.upper().startswith("TIED"):
        status("Vote tied")
        answer = cleaned.split(":", 1)[-1].strip() if ":" in cleaned else cleaned
        return {"answer": answer, "vote_split": f"tied/{total}", "total": total, "tied": True}

    vote_split = f"?/{total}"
    answer = cleaned
    if "MAJORITY" in cleaned.upper() and ":" in cleaned:
        header, body = cleaned.split(":", 1)
        answer = body.strip()
        match = re.search(r'\((\d+)/(\d+)\)', header)
        if match:
            vote_split = f"{match.group(1)}/{match.group(2)}"

    status(f"Majority: {vote_split}")
    return {"answer": answer, "vote_split": vote_split, "total": total, "tied": False}


def _format_answers(answers: list[dict]) -> str:
    """Format answers for prompts.

    Uses `══` block markers rather than `---` because triple-dash is the
    same shape as a git diff hunk header — models occasionally interpret
    answer blocks as diff context and lose the surrounding instruction.
    """
    parts = []
    for i, a in enumerate(answers):
        model = a["model"].split("/")[-1]
        parts.append(
            f"══════════════════════════════════════════════════════════════════════\n"
            f"ANSWER {i+1} of {len(answers)} — model: {model}\n"
            f"══════════════════════════════════════════════════════════════════════\n"
            f"{a['answer']}\n"
        )
    return "\n".join(parts)
