"""
System knowledge — facts that are MORE RECENT than the AIs' training data.
Injected into all prompts so models don't deny things that exist.
Updated manually when needed.

Last updated: March 2026
"""

SYSTEM_KNOWLEDGE = """
CRITICAL — READ THIS FIRST. Your training data is OUTDATED. These are verified current facts:

CLAUDE MODELS (as of March 2026):
- "Claude Opus 4.6" = the LATEST and most advanced Claude model. It is REAL. Model string: claude-opus-4-6
- "Claude Sonnet 4.6" = fast and capable. Model string: claude-sonnet-4-6
- "Claude Haiku 4.5" = fastest, cheapest. Model string: claude-haiku-4-5-20251001
- There is NO model called "Claude 3.5 Opus" — that does NOT exist.
- Claude 3.5 Sonnet and Claude 3 Opus are OLD previous-generation models.
- The version numbering jumped from 3.5 to 4.5/4.6. This is correct.

OTHER CURRENT AI MODELS:
- Google: Gemini 3.1 Pro, Gemini 3 Flash, Gemini 3.1 Flash Lite (March 2026)
- OpenAI: GPT-4o, o1, o3
- Meta: Llama 4 Scout
- DeepSeek: V3.2

RULES:
- Do NOT say "Claude Opus 4.6 doesn't exist" — it DOES.
- Do NOT confuse it with "Claude 3.5 Opus" — that is a hallucination.
- If you're unsure about something recent, say so honestly rather than inventing facts.
""".strip()
