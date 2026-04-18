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
    "groq/kimi-k2":          {"window": 262_000, "tpm": 10_000, "provider": "groq"},
    "groq/llama-3.3-70b":    {"window": 128_000, "tpm": 12_000, "provider": "groq"},
    "groq/llama-4-scout":    {"window": 128_000, "tpm": 30_000, "provider": "groq"},

    # NVIDIA (free, 40 RPM shared, no TPM limit)
    "nvidia/deepseek-v3.2":  {"window": 128_000, "tpm": None, "provider": "nvidia"},
    "nvidia/minimax-m2.5":   {"window": 128_000, "tpm": None, "provider": "nvidia"},
    "nvidia/qwen-3.5":       {"window": 128_000, "tpm": None, "provider": "nvidia"},
    "nvidia/glm-5":          {"window": 200_000, "tpm": None, "provider": "nvidia"},
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
    "groq/kimi-k2":       "moonshotai/kimi-k2-instruct-0905",
    "groq/llama-3.3-70b": "llama-3.3-70b-versatile",
    "groq/llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
}

# ─── NVIDIA Model ID Mapping ────────────────────────────────────────────────

NVIDIA_MODEL_IDS = {
    "nvidia/deepseek-v3.2":  "deepseek-ai/deepseek-v3.2",
    "nvidia/minimax-m2.5":   "minimaxai/minimax-m2.5",
    "nvidia/qwen-3.5":       "qwen/qwen3.5-397b-a17b",
    "nvidia/glm-5":          "z-ai/glm5",
    "nvidia/nemotron-super": "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/ultralong-8b":   "nvidia/Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct",
}

# ─── Priority Order per Role ─────────────────────────────────────────────────

PRIORITY_ORDER = {
    "decorticator":  ["nvidia/deepseek-v3.2", "nvidia/minimax-m2.5", "nvidia/glm-5"],
    "fast_chat":     ["groq/kimi-k2", "groq/llama-4-scout", "nvidia/minimax-m2.5"],
    "synthesizer":   ["groq/kimi-k2", "groq/llama-4-scout", "nvidia/minimax-m2.5", "nvidia/glm-5"],
    "verifier":      ["groq/gpt-oss-120b", "groq/llama-4-scout", "nvidia/qwen-3.5", "nvidia/glm-5"],
    "search_exec":   ["groq/llama-3.1-8b"],
    "self_eval":     ["groq/llama-3.1-8b", "groq/llama-4-scout", "nvidia/deepseek-v3.2"],
    "plan_compare":  ["nvidia/glm-5"],
    "formatter":     ["groq/kimi-k2"],
}

# ─── Domain-Matched Pairs ────────────────────────────────────────────────────

BEST_PAIRS = {
    "math":    ("nvidia/deepseek-v3.2", "nvidia/qwen-3.5"),
    "code":    ("nvidia/minimax-m2.5",  "nvidia/glm-5"),
    "science": ("nvidia/qwen-3.5",      "nvidia/deepseek-v3.2"),
    "cfd":     ("nvidia/deepseek-v3.2", "nvidia/glm-5"),
    "arduino": ("nvidia/minimax-m2.5",  "nvidia/glm-5"),
    "web":     ("nvidia/minimax-m2.5",  "nvidia/qwen-3.5"),
    "general": ("nvidia/deepseek-v3.2", "nvidia/minimax-m2.5"),
}

# ─── Fallback Maps ───────────────────────────────────────────────────────────

NVIDIA_FALLBACKS = {
    "nvidia/deepseek-v3.2": "nvidia/glm-5",
    "nvidia/glm-5":         "nvidia/nemotron-super",
    "nvidia/minimax-m2.5":  "nvidia/qwen-3.5",
    "nvidia/qwen-3.5":      "nvidia/deepseek-v3.2",
    "nvidia/nemotron-super": "nvidia/glm-5",
}

GROQ_FALLBACKS = {
    "groq/llama-3.1-8b":  "nvidia/deepseek-v3.2",
    "groq/qwen3-32b":     "nvidia/deepseek-v3.2",
    "groq/gpt-oss-120b":  "nvidia/minimax-m2.5",
    "groq/kimi-k2":       "nvidia/minimax-m2.5",
    "groq/llama-3.3-70b": "nvidia/deepseek-v3.2",
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
