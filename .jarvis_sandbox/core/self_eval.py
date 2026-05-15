"""
Self-eval — checks if the final answer is reasonable.
Has access to system knowledge AND search results from the pipeline,
so it can verify claims against actual Google data.
"""

from clients.gemini import call_flash
from core.cli import step, success, warn
from core.system_knowledge import SYSTEM_KNOWLEDGE


SELF_EVAL_PROMPT = """You are a quality checker.

""" + SYSTEM_KNOWLEDGE + """

QUESTION: {question}

ANSWER: {answer}

{search_context}

Think step by step:
1. What is the user actually asking?
2. Does the answer address THAT question, or something else entirely?
3. If GOOGLE SEARCH RESULTS are provided above, does the answer align with them?
   If the search results confirm something, the answer is NOT hallucinating.

Rules — be GENEROUS:
- Good answer to the right question = PASS
- Imperfect answer to the right question = PASS
- Answer consistent with the Google search results above = PASS
- Answer that mentions models/products from system knowledge = PASS (they are REAL)
- Great answer to the WRONG question = FAIL
- Answer that forces irrelevant context onto a simple message = FAIL
- Do NOT fail because you don't recognize a name. Check search results first.

Reply with ONLY:
PASS — if the answer addresses what the user actually meant
FAIL: [one sentence: what went wrong]"""


async def self_eval(
    question: str,
    answer: str,
    complexity: int = 5,
    context_tokens: int = 0,
    search_results: str = "",
) -> dict:
    """
    Run self-eval with Flash Lite + system knowledge + search results.
    Returns: {"passed": bool, "feedback": str or None}
    """
    step("Self-eval")

    search_context = ""
    if search_results:
        search_context = (
            f"GOOGLE SEARCH RESULTS (current facts from this session — trust these):\n"
            f"{search_results[:15000]}"
        )

    prompt = SELF_EVAL_PROMPT.format(
        question=question,
        answer=answer[:30000],
        search_context=search_context,
    )

    result = await call_flash(prompt, max_tokens=256)
    cleaned = result.strip().upper()

    if cleaned.startswith("PASS"):
        success("Self-eval passed")
        return {"passed": True, "feedback": None}

    feedback = result.strip()
    if "FAIL:" in feedback.upper():
        feedback = feedback.split(":", 1)[-1].strip()

    warn(f"Self-eval flagged: {feedback[:80]}")
    return {"passed": False, "feedback": feedback}
