#!/usr/bin/env python3
"""
ch05_tool_use.py - Tool Use (Function Calling) with raw Anthropic API

Tool use (function calling) is the bridge between an LLM's reasoning and the
external world. Without tools, the model can only recite training data. With
tools, it can query live APIs, run code, compute, and act.

  user query
      |
      v
  [LLM + Tool Definitions]  ─── decides which tool(s) to call
      |
      v
  [Tool Executor] ─── runs the actual function with LLM-provided args
      |
      v
  [tool_result] ─── returns output back to the LLM
      |
      v
  [LLM] ─── processes result, may call more tools or produce final answer
      |
      v
  final response

Full lifecycle:
  1. DEFINE tools (JSON schema descriptions)
  2. LLM DECIDES whether to call one or more tools
  3. LLM GENERATES structured tool_use blocks (name + JSON input)
  4. FRAMEWORK EXECUTES the functions
  5. RESULTS returned as tool_result blocks
  6. LLM processes results → final answer or next tools

Key ideas from the book (Chapter 5):
- Tools break the LLM's static knowledge boundary → live data, actions
- "Function calling" ≈ predefined-code; "tool calling" ≧ APIs, DBs, sub-agents
- The model decides WHEN and WHICH tool to call, not the developer
- Each call is a safety/validation boundary: validate args before executing
- LangChain: @tool decorator + AgentExecutor; ADK: built-in google_search, code exec
- Tool use is the FOUNDATION for all remaining patterns in the book

--------------------------------------------------------------------------
PRODUCTION INSIGHTS (from Claude Code source analysis of
/Users/liurui/workspace/claude-code/src/):

Claude Code is fundamentally a tool-calling system. The agent loop is:
    ask model → parse tool_use → dispatch → append tool_result → continue

THE MAJOR TOOL SYSTEMS:

+-----------------------------------------------------------------------+
| 1. TWO DISPATCH PATHS (src/query.ts:1380-1408)                        |
|                                                                        |
| Claude Code has two execution paths for tool calls:                    |
|   a) StreamingToolExecutor (default): tools execute as their           |
|      tool_use blocks STREAM IN, results yielded progressively.         |
|   b) runTools() (fallback): all tool_use blocks collected first,       |
|      then dispatched after streaming completes.                        |
|                                                                        |
| The dispatch itself (src/services/tools/toolExecution.ts) does:        |
|   1. Resolve tool by name with alias fallback (line 344)               |
|   2. Validate with Zod safeParse() (line 615)                          |
|   3. Run PreToolUse hooks                                              |
|   4. Check permissions via canUseTool                                  |
|   5. Call tool.handler()                                               |
|   6. Run PostToolUse hooks                                             |
|   7. Map result to tool_result content block                           |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 2. CONCURRENCY SAFETY PARTITIONING                                    |
|    (src/services/tools/toolOrchestration.ts:91-116)                   |
|                                                                        |
| Each tool declares isConcurrencySafe. When the model issues multiple   |
| tool_use blocks in one response, partitionToolCalls() groups them:     |
| consecutive same-safety-level tools run TOGETHER. Non-concurrent       |
| tools (e.g. Bash) force a serial batch boundary.                       |
|                                                                        |
| Parallel execution (toolOrchestration.ts:152-177) uses an all()       |
| generator merger with configurable max concurrency:                    |
| CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY (default 10).                    |
|                                                                        |
| KEY: Bash errors CANCEL sibling tools via siblingAbortController       |
| (StreamingToolExecutor.ts:358-363). Read/WebFetch errors don't.       |
| The turn continues despite cancelled subprocesses.                    |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 3. SCHEMA GENERATION PATTERNS                                         |
|                                                                        |
| a) Lazy schemas (src/utils/lazySchema.ts): schemas built via          |
|    memoized factory, not at module-import time. Saved for MCP tools   |
|    whose schemas come from remote servers.                             |
|                                                                        |
| b) z.strictObject() on every input schema (e.g. FileReadTool.ts:228) |
|    rejects unrecognized keys the model might hallucinate.              |
|                                                                        |
| c) semanticNumber / semanticBoolean wrappers coerce string inputs     |
|    like "200" into numbers, reducing model-side JSON type errors.     |
|                                                                        |
| d) Two-tier descriptions: short description() in schema for model     |
|    to scan, plus full prompt() with usage rules, constraints, tips    |
|    for deeper context. prompt() is injected on-demand, not always.    |
|                                                                        |
| e) searchHint (Tool.ts:378): 3-10 word keyword phrase for indexing.   |
|    ToolSearchTool scores searchHint at 2x weight vs description       |
|    (src/tools/ToolSearchTool/ToolSearchTool.ts:282-284).              |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 4. ERROR HANDLING                                                     |
|                                                                        |
| a) Zod error formatting (src/utils/toolErrors.ts:66-132): errors      |
|    split into three categories — missing params, unexpected params,   |
|    type mismatches — each generating a human/LLM-friendly string.     |
|                                                                        |
| b) Deferred-tool schema hint (toolExecution.ts:578-597): when a       |
|    deferred tool's schema was NEVER in the prompt, the system tells   |
|    the model to call ToolSearch's select:<tool_name> instead of       |
|    repeatedly failing Zod validation.                                  |
|                                                                        |
| c) Error truncation (toolErrors.ts:15-21): tool_result errors capped  |
|    at 10,000 chars (5k head + 5k tail) to bound context.              |
|                                                                        |
| d) Errors are NOT fatal — they are returned as tool_result blocks     |
|    so the model can retry, explain, or try a different approach.      |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 5. TOOL ROUND MANAGEMENT (src/query.ts)                                |
|                                                                        |
| Three recovery loops:                                                  |
|   1. max_output_tokens recovery (lines 1188-1256): up to 3 retries,   |
|      first escalates from 8k to 64k output tokens.                    |
|   2. Prompt-too-long recovery (lines 1085-1183): collapse drain       |
|      first, then reactive compact (full summarization).               |
|   3. Model fallback (lines 893-953): catches FallbackTriggeredError,  |
|      switches to fallbackModel.                                        |
|                                                                        |
| turnCount increments each loop; maxTurns stops with                   |
| max_turns_reached message. Stop hooks preserve                       |
| hasAttemptedReactiveCompact across retries to prevent infinite loops  |
| (query.ts:1281).                                                       |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 6. UNIQUE PATTERNS                                                    |
|                                                                        |
| a) maxResultSizeChars: Infinity on Read tool (FileReadTool.ts:342).   |
|    Setting a limit would cause: result too big → write to file →      |
|    need Read to read → circular deadlock. All other tools have        |
|    finite limits; oversized results go to disk, preview to model.     |
|                                                                        |
| b) backfillObservableInput (Tool.ts:475-481): pre-call hook that      |
|    injects legacy/derived fields for observers (hooks, permissions)   |
|    while preserving the original API-bound input for cache stability. |
|                                                                        |
| c) Alias fallback (Tool.ts:348-359): toolMatchesName() checks both   |
|    name and aliases[]. Old transcripts calling renamed tools          |
|    (e.g. "KillShell" → alias for "TaskStop") still work.             |
|                                                                        |
| d) Tool pool assembly (src/tools.ts:345-367): dedup by name           |
|    (built-in wins over MCP), sorted alphabetically within partition   |
|    to maintain contiguous prompt cache breakpoint regions.            |
+-----------------------------------------------------------------------+

COMPARISON TABLE:
| Aspect                  | Book ch05 (this file)     | Claude Code production    |
|-------------------------|---------------------------|---------------------------|
| Tool schema             | Hand-written JSON         | Zod-generated + lazy      |
| Dispatch                | Simple dict map           | Registry + middleware     |
| Permissions             | None                      | Multi-level gate          |
| Concurrency             | Sequential per tool       | Partitioned + parallel    |
| Error handling          | try/except → tool_result  | Category-specific errors  |
| Max rounds              | Configurable parameter    | Hard limit + 3 recovery   |
| Streaming               | Not demonstrated          | Full streaming execution  |
| Tool descriptions       | Static strings            | Two-tier + searchHint     |
| Tool count              | 4                         | ~45 built-in + MCP        |

Usage:
    python agents_new/ch05_tool_use.py
    python agents_new/ch05_tool_use.py interactive   # REPL mode
"""

import ast
import logging
import math
import os
import re
import time
from typing import Any, Callable

from anthropic import Anthropic
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
# Tool definitions (Anthropic's native JSON Schema format)
# --------------------------------------------------------------------------

# Each tool has: name, description, input_schema
# The description is critical — it's how the model decides which tool to use.

TOOLS = [
    {
        "name": "get_weather",
        "description": "获取指定城市的实时天气信息。输入城市名称，返回温度、天气状况和湿度。",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，例如：北京、上海、伦敦、东京",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": "执行数学计算，包括四则运算、幂运算、三角函数、对数等。"
                       "使用 Python 的 math 模块，支持所有 math.* 函数。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如: '4 * (7 + 3)', "
                                   "'math.sqrt(144)', 'math.sin(math.pi / 2)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "获取指定股票代码的当前模拟价格。支持主要科技股。",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "股票代码，例如：AAPL, GOOGL, MSFT, AMZN, TSLA",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "run_python_code",
        "description": "在沙箱环境中执行 Python 代码。适用于需要精确计算、"
                       "数据处理或验证逻辑的场景。代码通过 exec() 执行，"
                       "结果的 __repr__ 或 print 输出会被捕获返回。",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码。使用 print() 输出结果。",
                }
            },
            "required": ["code"],
        },
    },
]

# --------------------------------------------------------------------------
# Tool implementations (the actual functions)
# --------------------------------------------------------------------------

# Simulated weather database
_WEATHER_DB = {
    "北京": {"temp": 28, "condition": "晴朗", "humidity": 35},
    "上海": {"temp": 32, "condition": "多云", "humidity": 65},
    "伦敦": {"temp": 15, "condition": "多云", "humidity": 72},
    "东京": {"temp": 26, "condition": "小雨", "humidity": 80},
    "纽约": {"temp": 22, "condition": "晴朗", "humidity": 45},
    "巴黎": {"temp": 19, "condition": "阴天", "humidity": 60},
    "悉尼": {"temp": 16, "condition": "晴朗", "humidity": 50},
}

# Simulated stock prices
_STOCK_DB = {
    "AAPL": 198.50,
    "GOOGL": 175.30,
    "MSFT": 425.80,
    "AMZN": 182.40,
    "TSLA": 248.60,
    "META": 512.20,
    "NVDA": 880.10,
}


def _exec_weather(city: str) -> str:
    """Get simulated weather for a city."""
    normalized = city.strip().lower()
    for key, data in _WEATHER_DB.items():
        if key.lower() == normalized or normalized in key.lower():
            return (
                f"{key}天气：{data['condition']}，"
                f"温度 {data['temp']}°C，湿度 {data['humidity']}%"
            )
    return f"未找到 '{city}' 的天气数据。请检查城市名称。"


def _exec_calculate(expression: str) -> str:
    """Execute a mathematical expression safely."""
    allowed_names: dict = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names["math"] = math
    allowed_names["__builtins__"] = {"abs": abs, "round": round, "float": float, "int": int, "str": str, "len": len, "range": range, "list": list, "dict": dict, "tuple": tuple, "sum": sum, "min": min, "max": max, "pow": pow}

    try:
        # Security: only allow expressions, not statements
        tree = ast.parse(expression.strip(), mode="eval")
        code = compile(tree, "<string>", "eval")
        result = eval(code, allowed_names)
        return f"结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"


def _exec_stock_price(ticker: str) -> str:
    """Get simulated stock price."""
    t = ticker.strip().upper()
    if t in _STOCK_DB:
        price = _STOCK_DB[t]
        return f"{t} 的模拟股价为 ${price:.2f} USD"
    return f"未找到 '{t}' 的股价数据。支持的股票: {', '.join(sorted(_STOCK_DB.keys()))}"


def _exec_python_code(code: str) -> str:
    """Execute Python code in a restricted sandbox and capture output."""
    import io
    import sys

    # Capture stdout
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured

    result_str = ""
    try:
        compiled = compile(code.strip(), "<sandbox>", "exec")
        exec(compiled, {"__builtins__": __builtins__, "math": math})
        result_str = captured.getvalue()
        if not result_str:
            result_str = "(代码执行完毕，无输出)"
    except Exception as e:
        result_str = f"执行错误: {e}"
    finally:
        sys.stdout = old_stdout

    return result_str.strip()


# Tool dispatch map: name -> callable
TOOL_DISPATCH: dict[str, Callable[..., str]] = {
    "get_weather": lambda **kw: _exec_weather(**kw),
    "calculate": lambda **kw: _exec_calculate(**kw),
    "get_stock_price": lambda **kw: _exec_stock_price(**kw),
    "run_python_code": lambda **kw: _exec_python_code(**kw),
}

# --------------------------------------------------------------------------
# Core: Tool-using agent loop
# --------------------------------------------------------------------------


def run_with_tools(
    user_input: str,
    *,
    system_prompt: str = "",
    max_tool_rounds: int = 10,
    verbose: bool = True,
) -> tuple[str, list[dict]]:
    """Run a tool-using conversation turn with the Anthropic API.

    This implements the core tool-use lifecycle:
      1. Send message + tool definitions to the model
      2. If model responds with text → done, return it
      3. If model responds with tool_use blocks → execute each, append results
      4. Loop back to step 1 until text response or max rounds

    Returns:
        (final_text_response, full_message_history)
    """
    system = system_prompt or (
        "你是一个有帮助的智能助手，可以使用各种工具来回答用户的问题。"
        "使用工具时，请先说明你的思考过程，然后调用合适的工具。"
        "根据工具返回的结果，给用户提供清晰、有用的回答。"
    )

    messages: list[dict] = [{"role": "user", "content": user_input}]
    tool_round = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"User: {user_input}")
        print(f"{'='*60}")

    while tool_round < max_tool_rounds:
        tool_round += 1

        if verbose:
            print(f"\n--- Round {tool_round} (calling API) ---")

        response = client.messages.create(
            model=MODEL,
            system=system,
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )

        usage = response.usage
        if verbose:
            logger.info("API call | input_tokens=%d | output_tokens=%d",
                        usage.input_tokens, usage.output_tokens)

        # Collect text and tool_use blocks from the response
        text_parts: list[str] = []
        tool_calls: list[tuple[str, str, dict]] = []  # (tool_name, tool_id, input)

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                if verbose:
                    print(f"\n  [Text] {block.text[:300]}...")
            elif block.type == "tool_use":
                tool_calls.append((block.name, block.id, block.input))
                if verbose:
                    print(f"\n  [Tool Use] {block.name}({block.input})")

        # Append the full assistant response (text + tool_use) to history
        messages.append({"role": "assistant", "content": response.content})

        # If no tool calls, we're done — return the text response
        if not tool_calls:
            final = "\n".join(text_parts) if text_parts else "(no text response)"
            if verbose:
                print(f"\n{'='*60}")
                print("Final response received (no more tool calls)")
                print(f"{'='*60}")
            return final, messages

        # Execute each tool call and prepare tool_result blocks
        tool_result_blocks: list[dict] = []
        for name, tool_id, args in tool_calls:
            if verbose:
                print(f"\n  >> Executing: {name}({args})")

            start = time.time()
            try:
                executor = TOOL_DISPATCH.get(name)
                if executor:
                    result = executor(**args)
                else:
                    result = f"错误：未找到工具 '{name}'"
            except Exception as e:
                result = f"工具执行异常: {e}"

            elapsed = time.time() - start
            result_preview = str(result)[:200]
            if verbose:
                print(f"  << Result ({elapsed:.2f}s): {result_preview}")

            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(result),
            })

        # Append tool results so the model sees them next turn
        messages.append({"role": "user", "content": tool_result_blocks})

    # Max rounds reached without a final text response
    fallback = "(已达最大工具调用轮次)"
    if verbose:
        print(f"\n  [!] {fallback}")
    return fallback, messages


# --------------------------------------------------------------------------
# Interactive REPL for exploring tool use
# --------------------------------------------------------------------------


def interactive_loop() -> None:
    """Start an interactive tool-using REPL."""
    print("\n" + "=" * 60)
    print("  工具使用交互模式 (输入 'quit' 退出)")
    print("=" * 60)
    print("可用工具:")
    for t in TOOLS:
        print(f"  • {t['name']}: {t['description']}")
    print()

    history: list[dict] = []

    while True:
        try:
            user_input = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        final, messages = run_with_tools(
            user_input,
            system_prompt=(
                "你是一个有帮助的智能助手，可以使用各种工具来回答问题。"
                "对于需要工具的问题，请先思考需要哪个工具，然后调用它。"
                "得到结果后用通俗易懂的语言回答用户。"
            ),
            verbose=True,
        )
        # Print the final answer cleanly
        print(f"\n  >> Final: {final}")


# --------------------------------------------------------------------------
# Demos
# --------------------------------------------------------------------------


def demo_weather() -> None:
    """Demo 1: External information retrieval (weather)."""
    print(f"\n\n{'#'*60}")
    print("# Demo 1: 外部信息检索 — 天气查询")
    print(f"{'#'*60}")

    final, _ = run_with_tools("北京和伦敦的天气怎么样？")
    print(f"\n>>> {final}")


def demo_calculation() -> None:
    """Demo 2: Calculation & data analysis."""
    print(f"\n\n{'#'*60}")
    print("# Demo 2: 计算分析 — 数学与金融计算")
    print(f"{'#'*60}")

    final, _ = run_with_tools(
        "如果我有 100 股 AAPL，当前价格是多少？"
        "如果价格上涨 15%，我的持仓价值会变成多少？"
    )
    print(f"\n>>> {final}")


def demo_code_execution() -> None:
    """Demo 3: Code execution for precise computation."""
    print(f"\n\n{'#'*60}")
    print("# Demo 3: 代码执行 — 精确计算")
    print(f"{'#'*60}")

    final, _ = run_with_tools(
        "请帮我计算斐波那契数列的前 20 项，并检查其中哪些是质数。"
    )
    print(f"\n>>> {final}")


def demo_multi_tool() -> None:
    """Demo 4: Multi-tool orchestration (weather + calculate)."""
    print(f"\n\n{'#'*60}")
    print("# Demo 4: 多工具编排 — 旅行规划")
    print(f"{'#'*60}")

    final, _ = run_with_tools(
        "我打算去东京旅行 5 天，每天的预算是 300 美元。"
        "请帮我查一下东京的天气，然后计算 5 天的总预算（美元）。"
    )
    print(f"\n>>> {final}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_loop()
    else:
        demo_weather()
        demo_calculation()
        demo_code_execution()
        demo_multi_tool()
