"""
Token counting — tiktoken cl100k_base, with regex fallback if unavailable.
On your real machine tiktoken will work. This fallback is for environments
where the encoding file can't be downloaded.
"""

import re

_encoder = None

try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    pass

# Regex fallback: splits on word boundaries + punctuation, ~1.3 tokens/word
_WORD_RE = re.compile(r"""\S+|\s+""")


def count_tokens(text: str) -> int:
    """Count tokens. Uses tiktoken if available, else regex approximation."""
    if not text:
        return 0
    if _encoder is not None:
        return len(_encoder.encode(text))
    # Fallback: ~1.3 tokens per whitespace-separated word (empirically close for English)
    words = len(text.split())
    return int(words * 1.3) + 1


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-truncate text to fit within max_tokens."""
    if _encoder is not None:
        tokens = _encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return _encoder.decode(tokens[:max_tokens])
    # Fallback: estimate chars per token (~4) and truncate
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
