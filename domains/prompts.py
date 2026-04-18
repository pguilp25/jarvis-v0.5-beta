"""
Domain-specific assumption prompts — injected into ensemble queries.
Forces models to check domain-relevant assumptions before answering.
"""

ASSUMPTIONS = {
    "math": """Before answering, check:
- List all mathematical assumptions explicitly.
- Check every formula for correctness.
- Verify units and dimensional consistency.
- Check every algebraic step.
- Consider edge cases (division by zero, negative values, limits).
""",

    "code": """Before answering, check:
- What about null/empty inputs?
- Off-by-one errors?
- Thread safety concerns?
- Memory usage for large inputs?
- What if a file doesn't exist or network fails?
- Error handling for all failure modes.
""",

    "cfd": """Before answering, check:
- Compressible or incompressible flow?
- Reynolds number and flow regime?
- Boundary conditions compatible with physics?
- Lattice units vs physical units (FluidX3D)?
- Courant number impact on stability?
- Mesh/resolution sufficient for the physics?
- Turbulence model appropriate for this Re?
""",

    "science": """Before answering, check:
- Is the model/theory valid for this regime?
- What approximations are being used and are they justified?
- Are sources peer-reviewed?
- Confounding variables or alternative explanations?
- Reproducibility concerns?
""",

    "arduino": """Before answering, check:
- Which board exactly? (Uno, Mega, ESP32, etc.)
- Pin conflicts with existing connections?
- Power supply adequate for all components?
- delay() vs millis() for timing?
- SRAM usage (especially with strings)?
- Baud rate matching between serial devices?
- Voltage levels (3.3V vs 5V logic)?
""",

    "web": """Before answering, check:
- Browser compatibility concerns?
- Accessibility (ARIA, keyboard navigation)?
- Performance (bundle size, render blocking)?
- Security (XSS, CSRF, injection)?
- Mobile responsiveness?
""",

    "general": """Before answering:
- List every assumption you're making.
- What changes if each assumption is wrong?
- What's the most common misconception about this topic?
""",
}

# Extra prompt appended to ALL domains for complexity 7+
DEEP_ANALYSIS = """
Additionally:
- Generate 3+ approaches including one unconventional.
- For each approach: what's the worst case? Most subtle failure mode?
- Pick the best approach and explain why.
- What would change your mind about this choice?
- What are you least confident about?
- Prioritize depth over speed. Full reasoning.
"""


def get_assumption_prompt(domain: str, complexity: int) -> str:
    """Get the domain-specific assumption prompt, with deep analysis for 7+."""
    base = ASSUMPTIONS.get(domain, ASSUMPTIONS["general"])
    if complexity >= 7:
        return base + DEEP_ANALYSIS
    return base
