# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Python Agents
```sh
# Run any chapter agent (all self-contained, no monorepo build needed)
# Use the virtual environment Python
~/.venv/bin/python3 agents/s01_agent_loop.py
~/.venv/bin/python3 agents/s_full.py

# Run all Python tests
~/.venv/bin/python3 -m pytest tests/ -q

# Run a single test
~/.venv/bin/python3 -m pytest tests/test_agents_smoke.py -q
```

### Web Platform (Next.js)
```sh
cd web
npm install        # install dependencies
npm run dev        # start dev server (runs extract-content.ts via predev)
npm run build      # typecheck (tsc --noEmit) + production build
```

## Project Structure

This is a **teaching repository** for building a coding-agent harness. The structure mirrors the learning path:

### `agents/` — Python Reference Implementations
- **Self-contained**, runnable Python files (s01–s19, s_full.py)
- Each chapter builds on previous concepts but is independently executable
- `s_full.py` integrates all mechanisms; read last
- The model is the agent; these files are the **harness**
- Dependency: `anthropic`, `python-dotenv`, `pyyaml`

### Learning Path (4 Stages)
1. **s01–s06**: Single-agent core (loop, tools, planning, subagent, skills, context)
2. **s07–s11**: Safety & extensions (permissions, hooks, memory, prompts, recovery)
3. **s12–s14**: Durable work (tasks, background tasks, cron scheduler)
4. **s15–s19**: Multi-agent platform (teams, protocols, autonomous agents, worktree isolation, MCP)

### `docs/` — Multi-Language Teaching Docs
- `docs/zh/` — canonical, most complete
- `docs/en/` — main chapters + bridge docs available
- `docs/ja/` — main chapters + bridge docs available
- Bridge docs (s00a, s00b, s00d, s00f, s02a, s02b, s10a, s13a, s19a, entity-map, etc.) provide cross-chapter context

### `web/` — Next.js Teaching Platform
- Next.js 16 with Tailwind CSS 4, TypeScript
- i18n: en, zh, ja (Next.js route-based with `[locale]` param)
- Key routes: `/en/timeline`, `/en/layers`, `/en/compare`, `/en/docs/[slug]`
- Components under `src/components/` organized by domain (architecture, code, diff, docs, layout, simulator, timeline, ui, visualizations)
- Content data in `src/data/` (annotations, generated docs, scenarios)
- Pre-build step: `tsx scripts/extract-content.ts` extracts docs into JSON

### `skills/` — Skill Plugins
- Loaded dynamically by s05+ agents
- Includes: code-review, pdf, agent-builder, mcp-builder
- Each has a `SKILL.md` and optional scripts/references

### `tests/` — Smoke Tests
- `test_agents_smoke.py`: parametrized pytest that verifies each agent script compiles
- `test_s_full_background.py`: integration test for the full agent

## Key Architecture Concepts

- **Agent Loop**: `ask model → run tools → append results → continue` is the central pattern
- **Tool Dispatch**: tools are dispatched via a map, not hardcoded conditionals
- **Subagent**: fresh context per delegated subtask
- **Context Compaction**: keeps the active window small by summarizing/trimming
- **Permission System**: safety gate before tool execution
- **Hooks**: extension points around the loop
- **Memory**: durable cross-session knowledge store
- **Task System**: persistent dependency graph for multi-step work
- **Worktree Isolation**: separate execution lanes via git worktrees

## Reference Source

The production Claude Code source lives at `/Users/liurui/workspace/claude-code` — this learning repo teaches the design backbone, not a 1:1 mirror. Cross-reference the two when studying implementation details.
