# ANSI codes
R = "\033[0m"  # reset
B = "\033[1m"  # bold
DIM = "\033[2m"  # dim
IT = "\033[3m"  # italic
UL = "\033[4m"  # underline

# Grayscale palette (ANSI 256-color) — brighter, more gray aesthetic
GRAY_DARK = "\033[38;5;245m"   # subtle elements, blockquote markers
GRAY_MID = "\033[38;5;249m"    # borders, secondary text, inline code
GRAY_LIGHT = "\033[38;5;253m"  # headers, bullets, primary UI
GRAY_WHITE = "\033[38;5;231m"  # near-white for maximum emphasis

# Semantic colors — muted/desaturated variants
GRN = "\033[38;5;108m"  # muted sage green (success)
YEL = "\033[38;5;144m"  # muted gold (warning)
RED = "\033[38;5;138m"  # muted rose (error)
CYN = "\033[38;5;152m"  # muted slate
BLU = "\033[38;5;153m"  # muted blue-gray
MAG = "\033[38;5;139m"  # muted mauve

# Base grays
WHT = "\033[38;5;231m"  # pure white
GRY = "\033[38;5;246m"  # medium gray (brighter than old 90m)

def _format_inline(line: str) -> str:
    """Apply inline markdown → ANSI: bold, italic, code, links."""
    # Inline code `...` → gray (FIRST so bold/italic don't touch code)
    line = re.sub(r'`([^`]+)`', rf'{GRAY_MID}\1{R}', line)

    # Bold + italic ***text***
    line = re.sub(r'\*\*\*(.+?)\*\*\*', rf'{B}{IT}\1{R}', line)

    # Bold **text**
    line = re.sub(r'\*\*(.+?)\*\*', rf'{B}{WHT}\1{R}', line)

    # Italic *text* (not inside words)
    line = re.sub(r'(?<!\w)\*([^*]+?)\*(?!\w)', rf'{IT}\1{R}', line)

    # Links [text](url) — muted gray aesthetic
    line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', rf'{GRAY_LIGHT}{UL}\1{R} {GRAY_DARK}(\2){R}', line)

    return line

# ── Code Blocks ──────────────────────────────────────────
    if stripped.startswith("```"):
        if not in_code_block:
            code_lang = stripped[3:].strip()
            output.append(f" {GRAY_MID}┌{'─' * 60}{R}")
            if code_lang:
                output.append(f" {GRAY_MID}│{R} {DIM}{GRAY_LIGHT} {code_lang} {R}")
            output.append(f" {GRAY_MID}├{'─' * 60}{R}")
            in_code_block = True
        else:
            output.append(f" {GRAY_MID}└{'─' * 60}{R}")
            in_code_block = False
        i += 1
        continue

    if in_code_block:
        output.append(f" {GRAY_MID}│{R} {WHT}{line}{R}")
        i += 1
        continue

# ── Horizontal rules ─────────────────────────────────────
    if re.match(r'^[-*_]{3,}\s*$', stripped):
        output.append(f" {GRAY_MID}{'─' * 50}{R}")
        i += 1
        continue

# ── Headers ──────────────────────────────────────────────
    h_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
    if h_match:
        level = len(h_match.group(1))
        title = re.sub(r'\*\*(.+?)\*\*', r'\1', h_match.group(2))
        title = re.sub(r'\*(.+?)\*', r'\1', title)
        if level == 1:
            output.append("")
            output.append(f" {B}{GRAY_WHITE}{'━' * len(title)}{R}")
            output.append(f" {B}{GRAY_WHITE}{title.upper()}{R}")
            output.append(f" {B}{GRAY_WHITE}{'━' * len(title)}{R}")
        elif level == 2:
            output.append("")
            output.append(f" {B}{GRAY_LIGHT}■ {title}{R}")
            output.append(f" {GRAY_LIGHT}{'─' * (len(title) + 2)}{R}")
        elif level == 3:
            output.append(f" {B}{WHT}▸ {title}{R}")
        else:
            output.append(f" {B}{GRY}▪ {title}{R}")
        i += 1
        continue

# ── Blockquotes ──────────────────────────────────────────
    if stripped.startswith(">"):
        qt = _format_inline(stripped.lstrip("> ").strip())
        output.append(f" {GRAY_DARK}┃{R} {DIM}{IT}{qt}{R}")
        i += 1
        continue

# ── Unordered list items ─────────────────────────────────
 ul_match = re.match(r'^(\s*)[*\-+]\s+(.+)$', line)
 if ul_match:
  indent = len(ul_match.group(1))
  content = _format_inline(ul_match.group(2))
  pad = " " + "  " * (indent // 2)
  bullet = f"{GRAY_LIGHT}•{R}" if indent == 0 else f"{GRY}◦{R}"
  output.append(f"{pad}{bullet} {content}")
  i += 1
  continue

 # ── Ordered list items ───────────────────────────────────
 ol_match = re.match(r'^(\s*)(\d+)[.)]\s+(.+)$', line)
 if ol_match:
  indent = len(ol_match.group(1))
  num = ol_match.group(2)
  content = _format_inline(ol_match.group(3))
  pad = " " + "  " * (indent // 2)
  output.append(f"{pad}{GRAY_LIGHT}{num}.{R} {content}")
  i += 1
  continue