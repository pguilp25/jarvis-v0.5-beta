"""
Abort handling — detect cancel signals between pipeline steps.
"""

from config import ABORT_SIGNALS


def is_abort(text: str) -> bool:
    """Check if user input is an abort signal."""
    cleaned = text.strip().lower()
    return cleaned in ABORT_SIGNALS


def check_override(text: str) -> tuple[str, int | None]:
    """
    Check for !!simple / !!medium / !!hard prefix.
    Returns (cleaned_text, forced_complexity_or_None).
    """
    from config import OVERRIDE_MAP

    stripped = text.strip()
    for prefix, complexity in OVERRIDE_MAP.items():
        if stripped.lower().startswith(prefix):
            cleaned = stripped[len(prefix):].strip()
            return cleaned, complexity

    return text, None
