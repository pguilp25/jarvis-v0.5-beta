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


VERIFIER_PROMPT = """You are comparing multiple AI answers to the same question.

Question: {question}

{answers_block}

Think step by step:
1. Do they reach the SAME general conclusion, even if they use different words, examples, or structure?
   → If yes: AGREE
2. Do they give DIFFERENT answers that are ALL VALID? (different recipes, different recommendations, 
   different angles on the same topic, different examples)
   → If yes: COMPLEMENTARY
3. Do they give DIFFERENT answers where at least one seems WRONG or INCOMPATIBLE with the others?
   This includes: one claims something the others don't mention at all and it changes the conclusion,
   one gives completely different facts/numbers, or they reach opposite recommendations.
   → If yes: CONFLICT

IMPORTANT:
- Different wording of the same idea = AGREE
- Different examples supporting the same point = AGREE
- Different levels of detail = AGREE
- Different valid recommendations = COMPLEMENTARY
- One says something completely different that changes the answer = CONFLICT

Reply with ONLY one word: AGREE, COMPLEMENTARY, or CONFLICT"""


MERGE_PROMPT = """Multiple AIs gave different but all valid answers to this question.
Merge them into ONE comprehensive answer that combines the best of ALL.

Question: {question}

{answers_block}

Rules:
- Include the unique valuable content from EACH answer
- Keep ALL calculations, numbers, specific data, and failure mode analyses
- Remove only exact duplicates — if two AIs mention the same thing differently, keep the better version
- Organize logically with clear sections
- The merged answer should be LONGER and MORE DETAILED than any single answer
- Do NOT summarize or shorten — COMBINE and EXPAND

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
    fast=False: qwen-3.5 (for very intelligent tier)
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
            "nvidia/qwen-3.5",
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
        model = "groq/kimi-k2"
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
    """Format answers for prompts."""
    parts = []
    for i, a in enumerate(answers):
        model = a["model"].split("/")[-1]
        parts.append(f"--- AI-{i+1} ({model}) ---\n{a['answer']}\n")
    return "\n".join(parts)
