"""
Dynamic model selector — picks the best model for context size, role, budget tier.
"""

from config import MODELS, RESERVES, PRIORITY_ORDER, BEST_PAIRS
from core.costs import cost_tracker


def model_fits(model_id: str, context_tokens: int, task_type: str) -> bool:
    """Check if a model can handle the given context with the required reserves."""
    model = MODELS[model_id]
    reserve = RESERVES[task_type]["total"]

    if model["tpm"] is not None:
        effective = min(model["window"], model["tpm"]) - reserve
    else:
        effective = model["window"] - reserve

    return context_tokens < effective


def select_model(
    role: str,
    context_tokens: int,
    task_type: str = "simple",
    domain: str = "general",
) -> str:
    """
    Select the best model for a role, given context size and budget tier.
    Returns model_id or 'COMPRESS_NEEDED'.
    """
    tier = cost_tracker.get_tier()
    candidates = PRIORITY_ORDER.get(role, PRIORITY_ORDER["fast_chat"])

    # Drop paid models in restricted/emergency
    if tier in ("restricted", "emergency"):
        candidates = [m for m in candidates if not m.startswith("gemini/")]

    for model_id in candidates:
        if model_fits(model_id, context_tokens, task_type):
            return model_id

    return "COMPRESS_NEEDED"


def select_domain_pair(domain: str) -> tuple[str, str]:
    """Get the best model pair for a domain."""
    return BEST_PAIRS.get(domain, BEST_PAIRS["general"])


def select_for_context(context_tokens: int, task_type: str = "hard") -> list[str]:
    """
    Get all NVIDIA models that fit the given context + task type.
    Useful for ensemble selection.
    """
    nvidia_models = [m for m in MODELS if m.startswith("nvidia/")]
    return [m for m in nvidia_models if model_fits(m, context_tokens, task_type)]
