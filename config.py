"""
JARVIS Configuration — Models, reserves, pairs, fallbacks, budget.
"""

# ─── Token Reserves by Task Type ─────────────────────────────────────────────

RESERVES = {
    "simple":  {"think": 0,     "output": 8_000,  "total": 8_000},
    "medium":  {"think": 15_000, "output": 8_000,  "total": 23_000},
    "hard":    {"think": 40_000, "output": 16_000, "total": 56_000},
    "extreme": {"think": 60_000, "output": 16_000, "total": 76_000},
}

# ─── Model Definitions ───────────────────────────────────────────────────────

MODELS = {
    # Groq (free, TPM-limited)
    "groq/llama-3.1-8b":     {"window": 128_000, "tpm": 6_000,  "provider": "groq"},
    "groq/qwen3-32b":        {"window": 32_000,  "tpm": 6_000,  "provider": "groq"},
    "groq/gpt-oss-120b":     {"window": 131_000, "tpm": 8_000,  "provider": "groq"},
    "groq/llama-3.3-70b":    {"window": 128_000, "tpm": 12_000, "provider": "groq"},
    "groq/llama-4-scout":    {"window": 128_000, "tpm": 30_000, "provider": "groq"},

    # NVIDIA (free, 40 RPM shared, no TPM limit)
    "nvidia/deepseek-v4-pro":  {"window": 1_000_000, "tpm": None, "provider": "nvidia"},
    "nvidia/minimax-m2.7":   {"window": 128_000, "tpm": None, "provider": "nvidia"},
    "nvidia/kimi-k2.6":      {"window": 256_000, "tpm": None, "provider": "nvidia"},
    "nvidia/qwen-3.5":       {"window": 128_000, "tpm": None, "provider": "nvidia"},
    "nvidia/glm-5":          {"window": 200_000, "tpm": None, "provider": "nvidia"},
    "nvidia/glm-5.1":        {"window": 200_000, "tpm": None, "provider": "nvidia"},
    # nemotron-super kept as a fallback target only — active workflows now
    # route to nvidia/kimi-k2.6 instead. If it's deprecated server-side, the
    # 410 fast-fail in core/retry.py routes downstream automatically.
    "nvidia/nemotron-super": {"window": 1_000_000, "tpm": None, "provider": "nvidia"},
    "nvidia/ultralong-8b":   {"window": 4_000_000, "tpm": None, "provider": "nvidia"},

    # Gemini (free Flash Lite for utility, paid Pro for tiebreakers)
    "gemini/flash-lite":     {"window": 1_000_000, "tpm": None, "provider": "gemini"},
    "gemini/3.1-flash-lite": {"window": 1_000_000, "tpm": None, "provider": "gemini", "cost_per_1k_in": 0.0, "cost_per_1k_out": 0.0},
    "gemini/2.5-pro":        {"window": 1_000_000, "tpm": None, "provider": "gemini", "cost_per_1k_in": 0.00125, "cost_per_1k_out": 0.01},
    "gemini/3.1-pro":        {"window": 1_000_000, "tpm": None, "provider": "gemini", "cost_per_1k_in": 0.002, "cost_per_1k_out": 0.012},

    # OpenRouter (free tier)
    "openrouter/qwen3.6-plus": {"window": 1_000_000, "tpm": None, "provider": "openrouter"},
}

# ─── Groq Model ID Mapping (config name → API model string) ─────────────────

GROQ_MODEL_IDS = {
    "groq/llama-3.1-8b":  "llama-3.1-8b-instant",
    "groq/qwen3-32b":     "qwen/qwen-3-32b",
    "groq/gpt-oss-120b":  "openai/gpt-oss-120b",
    "groq/llama-3.3-70b": "llama-3.3-70b-versatile",
    "groq/llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
}

# ─── NVIDIA Model ID Mapping ────────────────────────────────────────────────

NVIDIA_MODEL_IDS = {
    "nvidia/deepseek-v4-pro":  "deepseek-ai/deepseek-v4-pro",
    "nvidia/minimax-m2.7":   "minimaxai/minimax-m2.7",
    "nvidia/kimi-k2.6":      "moonshotai/kimi-k2.6",
    "nvidia/qwen-3.5":       "qwen/qwen3.5-397b-a17b",
    "nvidia/glm-5":          "z-ai/glm5",
    "nvidia/glm-5.1":        "z-ai/glm5",
    "nvidia/nemotron-super": "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/ultralong-8b":   "nvidia/Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct",
}

# ─── Priority Order per Role ─────────────────────────────────────────────────

PRIORITY_ORDER = {
    "decorticator":  ["nvidia/deepseek-v4-pro", "nvidia/minimax-m2.7", "nvidia/glm-5.1"],
    "fast_chat":     ["nvidia/kimi-k2.6", "groq/llama-4-scout", "nvidia/minimax-m2.7"],
    "synthesizer":   ["nvidia/kimi-k2.6", "groq/llama-4-scout", "nvidia/minimax-m2.7", "nvidia/glm-5.1"],
    "verifier":      ["groq/gpt-oss-120b", "groq/llama-4-scout", "nvidia/qwen-3.5", "nvidia/glm-5.1"],
    "search_exec":   ["groq/llama-3.1-8b"],
    "self_eval":     ["groq/llama-3.1-8b", "groq/llama-4-scout", "nvidia/deepseek-v4-pro"],
    "plan_compare":  ["nvidia/glm-5.1"],
    "formatter":     ["nvidia/kimi-k2.6"],
}

# ─── Domain-Matched Pairs ────────────────────────────────────────────────────

BEST_PAIRS = {
    "math":    ("nvidia/deepseek-v4-pro", "nvidia/qwen-3.5"),
    "code":    ("nvidia/minimax-m2.7",  "nvidia/glm-5.1"),
    "science": ("nvidia/qwen-3.5",      "nvidia/deepseek-v4-pro"),
    "cfd":     ("nvidia/deepseek-v4-pro", "nvidia/glm-5.1"),
    "arduino": ("nvidia/minimax-m2.7",  "nvidia/glm-5.1"),
    "web":     ("nvidia/minimax-m2.7",  "nvidia/qwen-3.5"),
    "general": ("nvidia/deepseek-v4-pro", "nvidia/minimax-m2.7"),
}

# ─── Fallback Maps ───────────────────────────────────────────────────────────

NVIDIA_FALLBACKS = {
    "nvidia/deepseek-v4-pro": "nvidia/glm-5.1",
    "nvidia/glm-5":         "nvidia/glm-5.1",
    "nvidia/glm-5.1":       "nvidia/kimi-k2.6",
    "nvidia/minimax-m2.7":  "nvidia/qwen-3.5",
    "nvidia/qwen-3.5":      "nvidia/deepseek-v4-pro",
    "nvidia/kimi-k2.6":     "nvidia/deepseek-v4-pro",
    "nvidia/nemotron-super": "nvidia/kimi-k2.6",
}

GROQ_FALLBACKS = {
    "groq/llama-3.1-8b":  "nvidia/deepseek-v4-pro",
    "groq/qwen3-32b":     "nvidia/deepseek-v4-pro",
    "groq/gpt-oss-120b":  "nvidia/minimax-m2.7",
    "groq/llama-3.3-70b": "nvidia/deepseek-v4-pro",
    "groq/llama-4-scout": "nvidia/qwen-3.5",
}

# ─── Compression ─────────────────────────────────────────────────────────────

COMPRESS_THRESHOLD = 72_000
COMPRESS_TARGET = 50_000

# ─── Budget ──────────────────────────────────────────────────────────────────

MONTHLY_BUDGET = 45.0

# ─── NVIDIA Rate Limit ───────────────────────────────────────────────────────

NVIDIA_MAX_RPM = 40
NVIDIA_SLEEP_BETWEEN = 1.6  # seconds between sequential NVIDIA calls

# ─── Abort Signals ───────────────────────────────────────────────────────────

ABORT_SIGNALS = ["stop", "cancel", "abort", "nevermind", "start over", "scratch that"]

# ─── Override Prefixes ───────────────────────────────────────────────────────

OVERRIDE_MAP = {
    "!!simple":     2,
    "!!medium":     5,
    "!!hard":       10,
    "!!deep":       99,  # Special: routes to deep thinking mode
    "!!conjecture": 99,
    "!!compute":    99,
    "!!prove":      99,
}
