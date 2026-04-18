# JARVIS v0.5.1 — Multi-Brain AI Agent

### Opus-level performance. $0/month.

JARVIS coordinates 10+ free AI models to deliver results comparable to Claude Opus 4.6 or GPT-5 — just slower. Instead of one frontier model thinking alone, JARVIS has multiple models plan independently, debate each other's ideas, find flaws, and merge the best approaches. The result is frontier-quality output at zero cost.

**100% free.** Every model JARVIS uses is available on free API tiers. No paid subscriptions, no credits, no usage fees. You get your API keys for free and JARVIS handles the rest.

> **Beta** — JARVIS is functional and usable, but the context management and the coding agent prompts are not fully optimized yet. Even so, the multi-brain approach already produces results that compete with the best paid models. Contributions and feedback welcome.

### Why JARVIS?

| | Claude Opus 4.6 | GPT-5 | JARVIS |
|---|---|---|---|
| **Cost** | $15/$75 per 1M tokens | $10/$30 per 1M tokens | **$0** |
| **Subscription** | $20-200/month | $20-200/month | **Free forever** |
| **Coding quality** | Excellent | Excellent | **Comparable** (multi-brain debate) |
| **Speed** | ~30 sec | ~20 sec | ~2-5 min (tradeoff for free) |
| **How** | 1 frontier model | 1 frontier model | 10+ free models collaborating |

---

## Installation

### Requirements

- Python 3.10+
- An internet connection
- Free API keys (see below)

### Linux / macOS

```bash
git clone https://github.com/YOUR_USERNAME/jarvis.git
cd jarvis
python3 -m venv venv
source venv/bin/activate
pip install aiohttp
```

### Windows

```bash
git clone https://github.com/YOUR_USERNAME/jarvis.git
cd jarvis
python -m venv venv
venv\Scripts\activate
pip install aiohttp
```

### Getting API Keys (all free, no credit card needed)

You need at least the NVIDIA key. The others unlock more models and faster responses.

| Provider | Get your key at | What it unlocks |
|----------|----------------|-----------------|
| **NVIDIA** (required) | [build.nvidia.com](https://build.nvidia.com) | DeepSeek V3.2, Qwen 3.5, GLM-5, Nemotron Super, MiniMax M2.5, SD3 image gen |
| **Groq** | [console.groq.com](https://console.groq.com) | Kimi K2, Llama 4 Scout, GPT-OSS 120B (fast inference) |
| **Gemini** | [aistudio.google.com](https://aistudio.google.com) | Flash Lite (formatting), 2.5 Pro, 3.1 Pro |
| **OpenRouter** (optional) | [openrouter.ai](https://openrouter.ai) | Additional free models, future use |

### Optional: LSP Server (smarter code search)

JARVIS can use Language Server Protocol for semantic code search — finding dependencies, types, and indirect references that text search misses. Without LSP, JARVIS falls back to ripgrep (still works, just less precise).

```bash
# Python projects
pip install python-lsp-server

# JavaScript/TypeScript projects
npm install -g typescript-language-server typescript
```

JARVIS auto-detects your project language and starts the right server. No configuration needed.

---

## Usage

### Web UI (recommended)

```bash
# Linux/macOS
source venv/bin/activate
python ui_main.py

# Windows
venv\Scripts\activate
python ui_main.py
```

Open `http://localhost:3000` in your browser.

**First time setup:** Click the gear icon (⚙) in the sidebar to open Settings, go to API Keys, paste your keys, and click Save.

**Set a project** for the coding agent: Settings → Project → paste your project path → Set Project.

### Terminal CLI

```bash
python main.py
```

Type your message and press Enter. Use `/help` for available commands.

---

## How It Works

### The Multi-Brain Advantage

A single free model (like DeepSeek or Llama) is good but not frontier-level. JARVIS gets around this by making multiple free models collaborate: they each think independently, then critique each other's work, then the best ideas are merged. This debate-and-merge process produces output quality that matches models costing $15-20/million tokens — for free. The tradeoff is speed: where Opus gives you one answer in 30 seconds, JARVIS might take 2-5 minutes because it's running 4-8 models in parallel behind the scenes.

### Automatic Routing

You just type your message. JARVIS analyzes it and routes to the right workflow:

- **Simple questions** → fast single-model answer (Groq, ~1 second)
- **Complex questions** → multi-model ensemble with debate and synthesis
- **Coding tasks** → full coding pipeline (see below)
- **Research** → parallel web search + multi-source synthesis
- **Image generation** → prompt expansion + Stable Diffusion 3
- **Deep reasoning** → iterative thinking loop (100+ cycles)

### Mode Overrides

Prefix your message with a mode to force a specific workflow:

| Prefix | What it does |
|--------|-------------|
| `!!simple` | Force fast single-model response |
| `!!medium` | Force intelligent multi-model response |
| `!!hard` | Force full ensemble with debate |
| `!!deepcode` | Force deep coding mode (3-layer plan debate + parallel coders + review) |
| `!!deep` | Deep iterative thinking (for hard math/science) |
| `!!compute` | Code-execution research loop |
| `!!image` | Image generation |

The Web UI also has a mode dropdown in the input area.

---

## Coding Agent

The coding agent is where JARVIS's multi-brain approach shines most. A single free model writing code makes mistakes. JARVIS has 4 models plan independently, 4 more critique and improve those plans, one merge the best ideas, then the coder implements while a separate reviewer catches bugs. This pipeline produces code changes comparable to what you'd get from Opus or Cursor — it just takes longer.

### Standard Mode (auto-detected for code tasks)

1. **UNDERSTAND** — 3 AIs search the codebase in parallel to find relevant files and functions
2. **PLAN** — 4 AIs write independent implementation plans → GLM-5 merges the best ideas into one plan
3. **IMPLEMENT** — GLM-5 codes each file in parallel
4. **REVIEW** — GLM-5 reviews each file in parallel, finds flaws, writes fixes
5. **DELIVER** — Shows you the diff, you approve or reject

### Deep Code Mode (`!!deepcode`)

Same as above but with racing and 3-layer planning:

1. **Layer 1** — 4 AIs race to write plans. First 3 to finish win, the slowest is cancelled mid-flight.
2. **Layer 2** — 4 Nemotron Super instances each read all 3 winning plans, find flaws and strengths, write their own improved plan (parallel)
3. **Layer 3** — GLM-5 reads all 4 improved plans, finds remaining flaws, writes the final plan

All research (code lookups, reference searches) is cached and shared across every AI in the pipeline — the coder doesn't re-run searches the planner already did.

All edits are sandboxed — your original files are never touched until you approve.

### Code Indexing (3 Maps)

When you set a project, JARVIS indexes it into three maps:

- **General Map** — high-level overview by feature/component (shown to every AI)
- **Detailed Map** — function signatures, logic, variables, dependencies (queried on demand via `[DETAIL: section name]`)
- **Purpose Map** — code categorized by what it does (e.g. "API calls", "UI colors", "error handling"). Queried via `[PURPOSE: category]`, returns actual code snippets with 10 lines of context

Maps are cached and only regenerated when files change. Processed in ~100K token batches using Nemotron Super (128K context), so it works on large codebases.

### Mid-Thought Tools (Cascade)

Every AI in the pipeline can pause mid-thought to look things up. Tools are used in escalating order — start cheap, go deeper only if needed:

1. `[REFS: name]` — find all definitions, imports, and usages (ripgrep, fast)
2. `[LSP: name]` — semantic search: dependencies, types, indirect references (requires LSP server)
3. `[DETAIL: section]` / `[PURPOSE: category]` — query code maps
4. `[CODE: path/to/file]` / `[SEARCH: pattern]` — read raw source code (last resort)

The AI writes all the tags it needs, then writes STOP. All lookups run in parallel, results come back, and the AI continues.

---

## Other Features

### Chat

- **Fast mode** — single model, instant responses for simple questions
- **Intelligent mode** — multiple models generate answers, best one is selected
- **Very intelligent mode** — full ensemble: multiple models respond, then debate, then synthesize

### Research

- Parallel web searches across multiple queries
- Multi-source synthesis with citations
- Domain-aware prompting (science, math, history, etc.)

### Image Generation

- Prompt expansion: a separate AI turns your brief description into a detailed image prompt
- Stable Diffusion 3 Medium via NVIDIA NIM
- Supports modification of previous images ("make it darker", "add a sunset")

### Deep Thinking

- Iterative reasoning loop for hard math/science/logic problems
- 100+ thinking cycles with self-evaluation
- Optional code execution mode for computational verification
- Optional Lean 4 formal proof mode

### Web UI

- Multi-conversation support (create, switch, delete, auto-title)
- Thinking display — expandable accordion showing each API call with live streaming
- Stop button — cancel any in-progress thinking
- Settings page — API keys, project path, model info
- Image rendering inline in chat

---

## Project Structure

```
jarvis/
├── main.py                  # CLI entry point + main pipeline
├── ui_main.py               # Web UI launcher
├── config.py                # Model configs, API mappings
├── ui/
│   ├── server.py            # WebSocket server (aiohttp)
│   └── index.html           # Single-file web UI
├── core/
│   ├── decorticator.py      # Intent classifier / router
│   ├── ensemble.py          # Multi-model debate + synthesis
│   ├── tool_call.py         # Mid-thought tool use loop
│   ├── memory.py            # Conversation memory
│   ├── retry.py             # Retry + fallback logic
│   └── ...                  # formatter, compressor, self-eval, etc.
├── clients/
│   ├── api.py               # Unified API router
│   ├── nvidia.py            # NVIDIA NIM client (SSE streaming)
│   ├── groq.py              # Groq client (SSE streaming)
│   ├── gemini.py            # Gemini client
│   └── openrouter.py        # OpenRouter client
├── workflows/
│   ├── chat.py              # Chat (fast/intelligent/ensemble)
│   ├── code.py              # Coding agent (plan/implement/review)
│   ├── research.py          # Research agent
│   ├── image.py             # Image generation
│   └── deep_thinking_v5.py  # Iterative deep reasoning
└── tools/
    ├── code_index.py         # 3-map code indexer
    ├── codebase.py           # File scanning, search, refs
    ├── lsp.py                # LSP client for semantic code search
    ├── sandbox.py            # Sandboxed code editing
    └── search.py             # Web search
```

---

## Models Used (all free — $0 total cost)

Every model below is available on a free API tier. No credit card required for any of them.

| Model | Provider | Used for |
|-------|----------|----------|
| DeepSeek V3.2 | NVIDIA | Planning, understanding |
| Qwen 3.5 | NVIDIA | Planning |
| MiniMax M2.5 | NVIDIA | Planning |
| Nemotron Super | NVIDIA | Code indexing, planning |
| GLM-5 | NVIDIA | Plan merging, code writing, code review |
| Kimi K2 | Groq | Fast chat, ensemble |
| Llama 4 Scout | Groq | Fast chat, ensemble |
| GPT-OSS 120B | Groq | Intelligent chat |
| Flash Lite | Gemini | Output formatting |
| SD3 Medium | NVIDIA | Image generation |

---

## Beta Status

This is a **beta release**. The multi-brain architecture already produces results comparable to frontier paid models — but it's not perfect yet.

What works:

- ✅ All chat modes (fast, intelligent, very intelligent)
- ✅ Coding agent with multi-AI planning, parallel implementation, and review
- ✅ Deep code mode with 3-layer debate — competitive with Opus/Cursor for complex changes
- ✅ Racing in planning (first 3 of 4 AIs win, slowest cancelled)
- ✅ Shared research cache across entire pipeline (plan → code → review)
- ✅ LSP integration for semantic code search (auto-detects, falls back to ripgrep)
- ✅ Code indexing with 3 maps (general, detailed, purpose)
- ✅ Image generation
- ✅ Web UI with multi-conversation, thinking display, stop button
- ✅ Research with web search
- ✅ Deep thinking mode
- ✅ Windows path support (both / and \ work everywhere)

What's not optimized (it works, but could be better):

- ⚠️ Context management — the system sometimes includes too much or too little context
- ⚠️ Coding agent prompts — the AIs occasionally explore unrelated code or miss parts of the task
- ⚠️ Speed — the multi-brain approach is slower than a single frontier model (2-5 min vs 30 sec for complex tasks)
- ⚠️ Shell agent — stub only, not implemented
- ⚠️ Error handling — some edge cases aren't gracefully handled

The goal: **frontier-level AI for everyone, for free.** We're not there 100% yet, but it's close enough to be useful today.

---

## License

MIT
