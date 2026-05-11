#!/usr/bin/env python3
"""
ch06_planning.py - Planning strategies with raw Anthropic API

Planning elevates an agent from reactive responder to strategic executor.
Instead of answering a single query in one shot, the agent:

  1. DECOMPOSES a high-level goal into a structured, multi-step plan
  2. EXECUTES each step sequentially
  3. EVALUATES results after each step — adapts remaining plan if needed
  4. SYNTHESIZES all step outcomes into a coherent final answer

This chapter demonstrates THREE planning strategies:

  A) SEQUENTIAL planning — decompose first, then execute all steps in order
  B) ADAPTIVE planning  — plan → step → re-plan remaining → step → ...
  C) COMPARE both strategies on the same goal to see trade-offs

  high-level goal
        |
        v
  [Planner LLM] ─── decomposes goal into ordered steps
        |
        v
  +------+------+------+------+
  | step 1 | step 2 | ... | step N |
  +--------+--------+------+------+
        |
        v
  [Synthesizer LLM] ─── combines step outputs + original goal
        |
        v
  final structured answer

Key ideas:
  - Planning = goal decomposition + ordered execution + adaptation
  - The plan is a STARTING POINT, not a rigid script — adapt as new info arrives
  - "What" vs "How": user defines the goal, agent discovers the path
  - Trade-off: sequential (stable, predictable) vs adaptive (flexible, overhead)
  - Planning vs Tool Use (ch05): tools are individual capabilities;
    planning orchestrates multiple steps toward a complex goal

Usage:
    ~/.venv/bin/python3 agents_new/ch06_planning.py

--------------------------------------------------------------------------
PRODUCTION INSIGHTS (from Claude Code source analysis at
/Users/liurui/workspace/claude-code/src/)

Claude Code has FIVE layers of planning, from the most explicit to fully implicit:

1. PLAN MODE — Permission-layer transition (src/tools/EnterPlanModeTool/)
   The most explicit planning path. EnterPlanMode is a tool that switches the
   session from execution mode to plan mode. In plan mode, the model can ONLY
   read/explore — any write/edit tools are blocked. The model produces a .md
   plan file at ~/.claude/plans/{slug}.md. On exit (ExitPlanModeV2Tool), the
   plan is presented to the user for approval, then the pre-plan permission
   mode is restored. If the session was in auto mode, it re-enables auto mode.
   This is the purest "plan-first" pattern: explore → plan → approve → execute.

2. TASK SYSTEM — Persistent structured plan (src/tools/TaskCreateTool/)
   Two generations exist. V1 (TodoWriteTool) stores a flat todo array in memory.
   V2 (TaskCreate/List/Get/Update/Stop/Output) persists tasks as individual JSON
   files under ~/.claude/tasks/, with proper-lockfile for multi-process safety
   (swarm mode). Key features:
     - Dependency tracking: blocks / blockedBy fields
     - Atomic claim: TOCTOU-safe locking when multiple agents compete for tasks
     - Auto-assignment: completing a task triggers assignment of the next pending
     - Verification nudge: closing 3+ tasks without a "verify" step appends a hint
     - Tasks survive session restart (file-based persistence)

3. COORDINATOR MODE — Orchestrator pattern (src/coordinator/coordinatorMode.ts)
   Set via CLAUDE_CODE_COORDINATOR_MODE env var. The model becomes an orchestrator
   with ONLY four tools: AgentTool, TaskStopTool, SendMessageTool, SyntheticOutputTool.
   It cannot read/write files directly — all work is delegated to worker subagents.
   The system prompt teaches a formal workflow: Research → Synthesis → Implementation
   → Verification. This is hierarchical planning in production: the coordinator
   plans at the goal level, workers plan at the task level.

4. SUBAGENT-BASED PLANNING — Plan Agent (src/tools/AgentTool/built-in/planAgent.ts)
   A built-in read-only subagent with system prompt "You are a software architect
   and planning specialist for Claude Code." It cannot write code — only explore
   and produce architecture plans. When the main model encounters a complex task,
   it spawns this agent for parallel exploration and planning, then consumes the
   results. This mirrors this file's hierarchical decomposition pattern.

5. IMPLICIT RE-PLANNING — Agent loop error recovery (src/query.ts)
   The most common re-planning mechanism is invisible to users. When errors occur,
   the loop re-enters the model with synthetic continuation messages:
     - max_output_tokens cut: injects "Pick up mid-thought if that is where the
       cut happened. Break remaining work into smaller pieces." (up to 3 retries)
     - prompt_too_long: compacts (summarizes) old context, retries with summary
     - model fallback: clears tool results, retries from scratch on backup model
     - stop-hook blocking: post-tool hooks append errors → model self-corrects
   There is no explicit replan() function — re-planning emerges from retry logic.

COMPARISON TABLE:
| Aspect                  | Book ch06 (this file)   | Claude Code production       |
|-------------------------|-------------------------|------------------------------|
| Plan visibility         | Internal to agent       | User-visible (tasks, /plan) |
| Plan granularity        | Paragraph-level steps   | Per-tool-call or per-task   |
| Planning trigger        | Always before execution | Explicit (/plan) or implicit |
| Plan format             | LLM-generated text      | Structured task list (JSON)  |
| Adaptation              | Context accumulates     | Error recovery + task update |
| User in loop            | No                      | Plan approval, task reorder  |
| Persistence             | In-memory only          | File-based (survives restart)|
| Hierarchical planning   | Single level            | Nested (sub-agents re-plan)  |
| Plan ↔ Execution mix    | Depends on strategy     | Interleaved (plan→do→adjust) |

--------------------------------------------------------------------------
LEARNING PATH

  ch04_reflection.py  → evaluate past output (backward-looking)
  ch05_tool_use.py    → interact with external world (capabilities)
  ch06_planning.py    → orchestrate steps toward a goal ← YOU ARE HERE
  ch07_...            → composition patterns
"""

import logging
import os
import re
from dataclasses import dataclass, field

from anthropic import Anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ.get("MODEL_ID", "claude-sonnet-4-20250514")

# ---------------------------------------------------------------------------
# Tools — minimal for this chapter (planning, not tools)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "calculate",
        "description": "执行数学计算，支持四则运算、幂运算、三角函数、对数等。使用 Python math 模块。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如: '4 * (7 + 3)', 'math.sqrt(144)'",
                }
            },
            "required": ["expression"],
        },
    },
]


def _exec_calculate(expression: str) -> str:
    import ast
    import math

    allowed: dict = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed["math"] = math
    allowed["__builtins__"] = {"abs": abs, "round": round, "float": float, "int": int}
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        code = compile(tree, "<string>", "eval")
        result = eval(code, allowed)
        return f"结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"


TOOL_DISPATCH = {
    "calculate": lambda **kw: _exec_calculate(**kw),
}

# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Outcome of one planned step."""
    index: int
    description: str
    raw_output: str = ""
    adapted: bool = False


@dataclass
class PlanResult:
    """Full result of a planning run."""
    final_answer: str
    steps: list[StepResult] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    plan_text: str = ""
    strategy: str = "sequential"


# ---------------------------------------------------------------------------
# Single LLM call
# ---------------------------------------------------------------------------


def _call_llm(
    system: str,
    messages: list,
    step_name: str,
    tools: list | None = None,
    max_tokens: int = 4096,
) -> tuple[str, list, dict]:
    """Single LLM call, returns (response_text, full_messages, usage)."""
    kwargs = dict(model=MODEL, system=system, max_tokens=max_tokens, messages=messages)
    if tools:
        kwargs["tools"] = tools

    response = client.messages.create(**kwargs)
    usage = response.usage
    logger.info("[%s] tokens=%d+%d", step_name, usage.input_tokens, usage.output_tokens)

    text_parts = []
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append((block.name, block.id, block.input))

    messages.append({"role": "assistant", "content": response.content})

    if tool_calls:
        tool_results = []
        for name, tool_id, args in tool_calls:
            logger.info("  -- tool: %s(%s)", name, args)
            executor = TOOL_DISPATCH.get(name)
            result = executor(**args) if executor else f"未知工具: {name}"
            tool_results.append({"type": "tool_result", "tool_use_id": tool_id, "content": result})
        messages.append({"role": "user", "content": tool_results})
        return _call_llm(system, messages, step_name, tools, max_tokens)

    return "\n".join(text_parts), messages, {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


# ---------------------------------------------------------------------------
# Helper: parse numbered steps from plan text
# ---------------------------------------------------------------------------


def _parse_steps(text: str) -> list[str]:
    """Extract step descriptions from the plan output."""
    steps = []
    lines = text.split("\n")
    current_step_lines: list[str] = []
    in_step = False

    for line in lines:
        stripped = line.strip()
        if re.match(r"^(?:#+\s*)?(?:Step\s+)?(\d+)[.)]\s+", stripped):
            if current_step_lines:
                steps.append(" ".join(current_step_lines))
            current_step_lines = [stripped]
            in_step = True
        elif in_step and stripped and not stripped.startswith("##"):
            current_step_lines.append(stripped)
        elif stripped.startswith("##") and not re.match(r"##\s+Step", stripped):
            if current_step_lines:
                steps.append(" ".join(current_step_lines))
                current_step_lines = []
            in_step = False

    if current_step_lines:
        steps.append(" ".join(current_step_lines))

    if not steps and text.strip():
        sections = re.split(r"\n##+\s+", text)
        steps = [s.strip() for s in sections if s.strip() and len(s.strip()) > 20]

    if not steps:
        steps = [text.strip()]

    return steps


# ---------------------------------------------------------------------------
# Strategy A: Sequential Planning  —  plan → execute → synthesize
# ---------------------------------------------------------------------------


def _generate_plan(goal: str, verbose: bool) -> tuple[str, list[str], dict]:
    """Phase 1: Decompose goal into ordered steps."""
    system = (
        "你是一个擅长任务分解的规划专家。分析用户的目标，将其拆解为有序的、"
        "可执行的步骤。每个步骤应该是一个可以独立完成的具体任务。\n\n"
        "输出格式（严格按以下结构）：\n"
        "## 分析\n"
        "[对目标的简要分析，说明需要做什么]\n\n"
        "## 计划\n"
        "1. [步骤1标题]\n"
        "   [步骤1描述]\n"
        "2. [步骤2标题]\n"
        "   [步骤2描述]\n"
        "..."
    )

    if verbose:
        print("\n--- Phase 1: Planning ---")
        print("  Generating plan...")

    plan_text, _, plan_usage = _call_llm(
        system=system,
        messages=[{"role": "user", "content": goal}],
        step_name="plan",
    )
    steps = _parse_steps(plan_text)
    if not steps:
        steps = [plan_text]

    if verbose:
        print(f"\n  Plan generated with {len(steps)} steps:")
        for line in plan_text.strip().split("\n"):
            print(f"    {line}")
        print()

    return plan_text, steps, {
        "input_tokens": plan_usage["input_tokens"],
        "output_tokens": plan_usage["output_tokens"],
    }


def _execute_steps(
    goal: str,
    plan_text: str,
    steps: list[str],
    verbose: bool,
    max_steps: int,
    allow_replan: bool = False,
) -> tuple[list[StepResult], dict]:
    """Phase 2: Execute each step. If allow_replan, re-plan remaining after each step."""
    usage = {"input_tokens": 0, "output_tokens": 0}
    step_results: list[StepResult] = []

    execute_system = (
        "你正在按计划逐步执行一个多步骤任务。当前步骤的描述如下。\n\n"
        "根据步骤描述来完成任务。如果需要计算，可以使用 calculate 工具。\n"
        "完成当前步骤后，输出该步骤的结果摘要。\n\n"
        "要点：\n"
        "1. 专注于当前步骤，不要跳步\n"
        "2. 提供清晰的结果说明"
    )

    step_context = f"原始目标：{goal}\n\n完整计划：\n{plan_text}\n\n"

    for i, step_desc in enumerate(steps):
        if i >= max_steps:
            break

        if verbose:
            print(f"\n  -- Executing Step {i+1}/{len(steps)} --")
            print(f"     {step_desc[:150]}")

        step_prompt = (
            f"{step_context}"
            f"当前正在执行的步骤 {i+1}/{len(steps)}：\n{step_desc}\n\n"
            f"请执行此步骤。完成后，提供该步骤的结果摘要。"
        )

        step_text, _, step_usage = _call_llm(
            system=execute_system,
            messages=[{"role": "user", "content": step_prompt}],
            step_name=f"step{i+1}",
            tools=TOOLS,
        )
        usage["input_tokens"] += step_usage["input_tokens"]
        usage["output_tokens"] += step_usage["output_tokens"]

        if verbose:
            print(f"     Result: {step_text[:200]}...")

        step_results.append(StepResult(index=i + 1, description=step_desc, raw_output=step_text))
        step_context += f"\n步骤 {i+1} 结果：\n{step_text}\n"

        # --- Adaptive re-planning hook ---
        if allow_replan and i < len(steps) - 1:
            remaining = steps[i + 1 :]
            revised = _replan_remaining(goal, plan_text, step_results, remaining, verbose)
            if revised:
                if verbose:
                    print(f"\n  ~~ Re-planning: {len(remaining)} remaining steps → {len(revised)} revised steps ~~")
                # Replace remaining steps with revised ones
                steps[i + 1 :] = revised
                step_context = _rebuild_context(goal, plan_text, step_results, steps[i + 1 :])
                # Mark subsequent results as adapted
                for sr in step_results:
                    if sr.index > i + 1:
                        sr.adapted = True

    return step_results, usage


def _replan_remaining(
    goal: str,
    original_plan: str,
    completed: list[StepResult],
    remaining: list[str],
    verbose: bool,
) -> list[str] | None:
    """Given completed steps, ask the LLM to revise remaining steps."""
    completed_text = "\n".join(
        f"步骤 {sr.index}: {sr.description}\n→ {sr.raw_output[:200]}" for sr in completed
    )
    remaining_text = "\n".join(f"- {s}" for s in remaining)

    prompt = (
        f"原始目标：{goal}\n\n"
        f"原始计划：\n{original_plan}\n\n"
        f"已完成的步骤：\n{completed_text}\n\n"
        f"剩余的计划步骤：\n{remaining_text}\n\n"
        "基于已完成步骤的结果，评估剩余步骤是否需要调整。\n"
        "如果不需要调整，回复「无需调整」。\n"
        "如果需要调整，请给出新的步骤列表（格式：1. ... 2. ...）。"
    )

    system = (
        "你是一个规划调整专家。分析已完成步骤的结果，判断剩余步骤是否需要修改。\n"
        "考虑：结果是否超出预期？是否有新信息需要后续步骤覆盖？\n"
        "只在必要时调整，不要为调整而调整。"
    )

    revised_text, _, _ = _call_llm(
        system=system,
        messages=[{"role": "user", "content": prompt}],
        step_name="replan",
    )

    if "无需调整" in revised_text:
        return None

    new_steps = _parse_steps(revised_text)
    return new_steps if new_steps else None


def _rebuild_context(
    goal: str, plan_text: str, completed: list[StepResult], remaining: list[str]
) -> str:
    """Rebuild step context after re-planning."""
    ctx = f"原始目标：{goal}\n\n原始计划：\n{plan_text}\n\n"
    for sr in completed:
        ctx += f"\n步骤 {sr.index} 结果：\n{sr.raw_output}\n"
    ctx += "\n调整后的剩余计划：\n"
    for i, s in enumerate(remaining, start=len(completed) + 1):
        ctx += f"步骤 {i}: {s}\n"
    return ctx


def _synthesize(
    goal: str, plan_text: str, step_results: list[StepResult], verbose: bool
) -> tuple[str, dict]:
    """Phase 3: Combine all step results into final answer."""
    if verbose:
        print(f"\n--- Phase 3: Synthesis ---")

    system = (
        "你是一名综合分析专家。你的任务是基于多步骤执行的结果，"
        "整合成一份完整的最终答案。\n\n"
        "要求：\n"
        "1. 回到用户的原始目标，确保完全覆盖\n"
        "2. 引用各步骤的执行结果作为依据\n"
        "3. 结构清晰，逻辑连贯\n"
        "4. 如果某些步骤的结果不完整，诚实说明"
    )

    prompt = f"原始目标：{goal}\n\n原始计划：\n{plan_text}\n\n各步骤执行结果：\n"
    for sr in step_results:
        prompt += f"\n--- 步骤 {sr.index} ---\n{sr.description}\n结果：{sr.raw_output}\n"

    prompt += "\n请基于以上信息，生成一份完整的最终答案。"

    final_text, _, final_usage = _call_llm(
        system=system,
        messages=[{"role": "user", "content": prompt}],
        step_name="synthesize",
    )
    return final_text, {"input_tokens": final_usage["input_tokens"], "output_tokens": final_usage["output_tokens"]}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_and_execute(
    goal: str,
    *,
    strategy: str = "sequential",
    verbose: bool = True,
    max_steps: int = 6,
) -> PlanResult:
    """Run a planning session with the chosen strategy.

    Strategies:
      - "sequential": classic plan-then-execute (stable, predictable)
      - "adaptive":   plan → step → re-plan → step → ... (flexible, overhead)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"[{strategy.upper()}] Goal: {goal}")
        print(f"{'='*60}")

    total_usage = {"input_tokens": 0, "output_tokens": 0}

    # Phase 1: Plan
    plan_text, steps, plan_usage = _generate_plan(goal, verbose)
    total_usage["input_tokens"] += plan_usage["input_tokens"]
    total_usage["output_tokens"] += plan_usage["output_tokens"]

    # Phase 2: Execute
    allow_replan = strategy == "adaptive"
    if verbose:
        print(f"\n--- Phase 2: Execution ({'adaptive' if allow_replan else 'sequential'}) ---")

    step_results, exec_usage = _execute_steps(
        goal, plan_text, steps, verbose, max_steps, allow_replan=allow_replan
    )
    total_usage["input_tokens"] += exec_usage["input_tokens"]
    total_usage["output_tokens"] += exec_usage["output_tokens"]

    # Phase 3: Synthesize
    final_text, syn_usage = _synthesize(goal, plan_text, step_results, verbose)
    total_usage["input_tokens"] += syn_usage["input_tokens"]
    total_usage["output_tokens"] += syn_usage["output_tokens"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"[{strategy.upper()}] Complete")
        print(f"Total usage: input={total_usage['input_tokens']} output={total_usage['output_tokens']}")
        print(f"{'='*60}")

    return PlanResult(
        final_answer=final_text,
        steps=step_results,
        usage=total_usage,
        plan_text=plan_text,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_sequential() -> None:
    """Strategy A: Sequential planning — stable, predictable."""
    print(f"\n\n{'#'*60}")
    print("# Demo 1: 顺序规划 — 先计划后执行，步骤有序推进")
    print(f"{'#'*60}")

    result = plan_and_execute(
        "我计划装修书房，需要购买以下材料：\n"
        "1. 地板：每平米 120 元，需要铺 25 平米\n"
        "2. 书架：每个 680 元，需要 3 个\n"
        "3. 书桌：每张 1500 元，需要 1 张\n"
        "4. 椅子：每把 450 元，需要 2 把\n"
        "请帮我做预算规划，计算总费用。\n"
        "然后告诉我如果我有 10000 元预算，是否够用，还差多少或剩多少。",
        strategy="sequential",
    )

    print(f"\n>>> 最终回答:\n{result.final_answer}")


def demo_adaptive() -> None:
    """Strategy B: Adaptive planning — re-plan after each step."""
    print(f"\n\n{'#'*60}")
    print("# Demo 2: 自适应规划 — 每执行一步重新评估剩余计划")
    print(f"{'#'*60}")

    result = plan_and_execute(
        "我正在组织一个团队建设活动，预算 8000 元。\n"
        "方案 1：户外拓展，每人费用 350 元，预计 15 人参加\n"
        "方案 2：聚餐+桌游，每人费用 180 元，预计 20 人参加\n"
        "方案 3：短途旅行，每人费用 500 元，预计 10 人参加\n\n"
        "请帮我计算每个方案的总费用，然后推荐最合适的方案并说明理由。",
        strategy="adaptive",
    )

    print(f"\n>>> 最终回答:\n{result.final_answer}")


def demo_compare_strategies() -> None:
    """Compare sequential vs adaptive on the same goal."""
    print(f"\n\n{'#'*60}")
    print("# Demo 3: 策略对比 — 对同一目标使用不同规划策略")
    print(f"{'#'*60}")

    goal = (
        "我计划开始一个健身计划，持续 12 周。\n"
        "第一阶段的 4 周：每周 3 次有氧运动，每次消耗 300 卡路里\n"
        "第二阶段的 4 周：每周 4 次有氧运动，每次消耗 350 卡路里\n"
        "第三阶段的 4 周：每周 5 次有氧运动，每次消耗 400 卡路里\n\n"
        "请帮我计算整个计划总共会消耗多少卡路里。"
    )

    # Run sequential
    seq_result = plan_and_execute(goal, strategy="sequential")
    print(f"\n>>> [SEQUENTIAL] 最终回答:\n{seq_result.final_answer}")

    # Blank line between demos
    print()

    # Run adaptive
    adp_result = plan_and_execute(goal, strategy="adaptive")
    print(f"\n>>> [ADAPTIVE] 最终回答:\n{adp_result.final_answer}")

    # Compare
    print(f"\n\n{'='*60}")
    print("策略对比总结")
    print(f"{'='*60}")
    print(f"  顺序规划: {len(seq_result.steps)} 步, "
          f"总 token: {seq_result.usage['input_tokens'] + seq_result.usage['output_tokens']}")
    print(f"  自适应规划: {len(adp_result.steps)} 步, "
          f"总 token: {adp_result.usage['input_tokens'] + adp_result.usage['output_tokens']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_sequential()
    demo_adaptive()
    demo_compare_strategies()
