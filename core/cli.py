"""
CLI status — colored output for every pipeline step.
"""

import sys

# ANSI color codes
_COLORS = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "red":     "\033[91m",
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "blue":    "\033[94m",
    "magenta": "\033[95m",
    "cyan":    "\033[96m",
    "white":   "\033[97m",
}

_ENABLED = True


def disable():
    global _ENABLED
    _ENABLED = False


def status(msg: str, color: str = "cyan"):
    """Print a status line: ▸ message"""
    if not _ENABLED:
        return
    c = _COLORS.get(color, "")
    r = _COLORS["reset"]
    print(f"  {c}▸ {msg}{r}", file=sys.stderr)


def step(name: str):
    """Print a pipeline step header."""
    status(name, "blue")


def success(msg: str):
    status(f"✓ {msg}", "green")


def warn(msg: str):
    status(f"⚠️  {msg}", "yellow")


def error(msg: str):
    status(f"✗ {msg}", "red")


def thinking(model: str):
    """Show which model is thinking."""
    short = model.split("/")[-1] if "/" in model else model
    status(f"Thinking with {short}...")


def agree():
    status("Models agree ✓", "green")


def disagree():
    status("Models disagree — resolving...", "yellow")


def debating():
    status("Debating...", "magenta")


def rate_limit_wait(seconds: float):
    status(f"NVIDIA rate limit — waiting {seconds:.1f}s...", "yellow")


def budget_warn(pct: float):
    warn(f"Budget {pct:.0%} spent — avoiding expensive models")
