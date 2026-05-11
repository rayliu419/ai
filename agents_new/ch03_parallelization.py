#!/usr/bin/env python3
"""
ch03_parallelization.py - Parallelization with raw Anthropic API

Parallelization executes multiple independent sub-tasks concurrently, reducing total
wall-clock time compared to sequential processing. Instead of waiting for step A to
finish before starting step B, independent tasks run simultaneously.

                     +--> [task_1] --+
                     |               |
  input --> [split]--+--> [task_2] --+--> [synthesize] --> output
                     |               |
                     +--> [task_3] --+

Key ideas from the book (Chapter 3):
- Identify sub-tasks that have no data dependency on each other
- Run them concurrently to minimize latency (especially valuable when calling external APIs)
- Synthesis step waits for ALL parallel results before proceeding
- Critical for: multi-source research, multi-perspective analysis, multi-API orchestration
- Can be combined with prompt chaining: parallel sections feed into a serial merge

--------------------------------------------------------------------------
PRODUCTION INSIGHTS (from Claude Code source analysis of
/Users/liurui/workspace/claude-code/src/):

Claude Code uses parallelization extensively. Analysis of the real codebase
reveals 6 major parallel subsystems and 15+ additional Promise.all / pMap
sites across the codebase.

THE MAJOR PARALLEL SYSTEMS:

+-----------------------------------------------------------------------+
| 1. TOOL-LEVEL PARALLELISM: Multiple tool_use blocks in one message     |
|    Prompt: src/tools/AgentTool/prompt.ts:271                           |
|                                                                        |
| The system prompt instructs the model to emit MULTIPLE tool_use blocks |
| in a single message for independent work:                              |
|                                                                        |
|   "If the user specifies that they want you to run agents 'in          |
|    parallel', you MUST send a single message with multiple Agent       |
|    tool use content blocks."                                           |
|                                                                        |
| The runtime takes the array of tool_use blocks and dispatches ALL      |
| of them concurrently. This is the most direct parallelization model    |
| matching chapter 3's pattern.                                          |
|                                                                        |
| Research pattern (prompt.ts:86):                                       |
|   "Research: fork open-ended questions. If research can be broken      |
|    into independent questions, launch parallel forks in one message."  |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 2. SUB-AGENT FORKING (Prompt-Cache-Optimized Parallelism)              |
|    src/tools/AgentTool/forkSubagent.ts                                 |
|                                                                        |
| When subagent_type is omitted (fork mode), the child inherits the      |
| parent's full conversation context. CRITICAL OPTIMIZATION: all forked  |
| children share a BIT-IDENTICAL API request prefix (same history, same  |
| placeholder tool_results), differing only in the final directive text  |
| block. This maximizes prompt cache hits across parallel children:      |
|                                                                        |
|   buildForkedMessages(directive, assistantMessage):                    |
|     - keep full assistant message (all tool_use blocks)                |
|     - build ONE user message: placeholder tool_results + directive     |
|     - only the final text block differs per child                      |
|                                                                        |
|   Guard: isInForkChild() detects recursive forks by marker in content  |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 3. BACKGROUND TASK SYSTEM (run_in_background + Promise.race)           |
|    src/tools/AgentTool/AgentTool.tsx:87                                |
|                                                                        |
| The Agent tool accepts a run_in_background flag:                       |
|                                                                        |
|   run_in_background: z.boolean().optional().describe(                  |
|     'Set to true to run this agent in the background.'                 |
|   )                                                                    |
|                                                                        |
| Auto-background trigger (line 70): 120s timeout via env var            |
| CLAUDE_AUTO_BACKGROUND_TASKS or feature flag.                          |
|                                                                        |
| Background transition uses Promise.race (line 886):                    |
|   Promise.race([nextMessagePromise, backgroundPromise])                |
|                                                                        |
| The parent continues working while the child completes asynchronously. |
| Result is injected back when done.                                     |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 4. CONCURRENCY-CONTROLLED BATCHING (pMap for MCP Server Connections)   |
|    src/services/mcp/client.ts:2218-2402                                |
|                                                                        |
| MCP server connections use pMap for controlled parallelism:            |
|                                                                        |
|   async function processBatched<T>(items, concurrency, processor) {    |
|     await pMap(items, processor, { concurrency })                      |
|   }                                                                    |
|                                                                        |
| Dual-concurrency strategy (line 2388):                                 |
|   await Promise.all([                                                  |
|     processBatched(localServers,  concurrency=3,  processServer),      |
|     processBatched(remoteServers, concurrency=20, processServer),      |
|   ])                                                                   |
|                                                                        |
| Local (stdio/SDK) servers limit to 3 to avoid process-spawning         |
| contention. Remote servers allow 20 since they are network-only.       |
|                                                                        |
| Additional parallel fetching per client (line 2171):                   |
|   const [tools, mcpCommands, mcpSkills, resources] =                  |
|     await Promise.all([fetchToolsForClient(client), ...])             |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 5. ERROR-ISOLATED PARALLELISM (Promise.allSettled across subsystems)   |
|                                                                        |
| 5a. Async Hooks (src/utils/hooks/AsyncHookRegistry.ts:144)            |
|     Periodic polling of running lifecycle hooks (BeforeCommand,        |
|     AfterToolUse). Promise.allSettled ensures one failing hook         |
|     doesn't orphan others:                                             |
|       const settled = await Promise.allSettled(                        |
|         hooks.map(async hook => { ... }),                              |
|       )                                                                |
|                                                                        |
| 5b. Plugin Loading (src/utils/plugins/pluginLoader.ts)                 |
|     13+ Promise.all/allSettled sites. Parallel filesystem checks       |
|     (pathExists) and parallel marketplace plugin catalog loading:      |
|       const results = await Promise.allSettled(                        |
|         marketplacePluginEntries.map(async ([id, val]) => { ... }),   |
|       )                                                                |
|                                                                        |
| 5c. Memory Scan (src/memdir/memoryScan.ts:45)                         |
|     Parallel frontmatter parsing of all memory files:                  |
|       const headerResults = await Promise.allSettled(                  |
|         mdFiles.map(async relativePath => { ... }),                   |
|       )                                                                |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 6. STARTUP PARALLELISM (Fire-and-Forget + Promise.all)                 |
|    src/main.tsx                                                        |
|                                                                        |
| 6a. Deferred prefetches (line 388) - fire-and-forget after first       |
|     render, results cached before user's first message:                |
|       void initUser();                                                 |
|       void getUserContext();                                           |
|       void prefetchAwsCredentialsAndBedRockInfoIfSafe();               |
|       void prefetchGcpCredentialsIfSafe();                             |
|       void refreshModelCapabilities();                                 |
|       // ... 15+ additional void calls                                 |
|                                                                        |
| 6b. Explicit parallelism (line 914):                                   |
|       await Promise.all([ensureMdmSettingsLoaded(),                    |
|                          ensureKeychainPrefetchCompleted()])           |
|                                                                        |
| 6c. Startup telemetry & setup (line 309, 1928):                       |
|       const [isGit, worktreeCount, ghAuthStatus] =                    |
|         await Promise.all([getIsGit(), getWorktreeCount(),             |
|                            getGhAuthStatus()])                        |
|       const [commands, agentDefinitions] = await Promise.all([        |
|         commandsPromise, agentDefsPromise                              |
|       ])                                                               |
+-----------------------------------------------------------------------+

KEY ARCHITECTURAL INSIGHTS (from real code):

1. PARALLELISM IS PERVASIVE BUT MOSTLY PROMISE-BASED: The JS/TypeScript
   runtime uses Promise.all / Promise.allSettled / pMap / Promise.race.
   There is NO thread pool or multiprocessing - it is all async concurrency
   on a single thread (event loop).

2. ERROR ISOLATION: The codebase consistently prefers Promise.allSettled
   over Promise.all for long-running parallel tasks. A failure in one
   hook/plugin/server should not crash its siblings. This differs from the
   book's "all must complete" assumption.

3. CONCURRENCY THROTTLING IS EXPLICIT: pMap with configurable concurrency
   limits prevents resource exhaustion. Different ceilings for local (3) vs
   remote (20) reflect real resource constraints - a pattern the book does
   discuss.

4. CACHE-AWARE PARALLELISM: The fork subagent system is designed to maximize
   prompt cache hits: all parallel children share a byte-identical prefix.
   This is a novel insight beyond the book's basic parallelization model.

5. FIRE-AND-FORGET AT STARTUP: 15+ non-blocking void calls prefetch data
   after first render. The cache is warm by the time the user types their
   first message. This is a practical optimization technique for production.

6. PARALLELISM + RACE: The background task system uses Promise.race to
   decide when to transition to background mode. Combined with auto-
   background timeout (120s), this is a pragmatic latency-management pattern.

COMPARISON TABLE:
| Aspect                 | Book ch03 (this file)    | Claude Code production        |
|------------------------|--------------------------|-------------------------------|
| Execution model        | ThreadPoolExecutor       | Promise.all / allSettled      |
| Concurrency control    | max_workers=N            | pMap with configurable limits |
| Error handling         | All must complete        | allSettled (failures isolated)|
| Cache awareness        | None                     | Fork shares byte-identical TX |
| Fire-and-forget        | No                       | void calls at startup         |
| Parallel unit          | LLM calls (homogeneous)  | Tools, agents, I/O, plugins   |
| Merge strategy         | Synthesis LLM prompt     | Context injection, var merge  |
| Backpressure           | None                     | 3-20 concurrent slots         |
| Shutdown parallelism   | N/A                      | Promise.race with 500ms cap   |

Usage:
    python agents_new/ch03_parallelization.py
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional, Dict

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
MODEL = os.environ.get("MODEL_ID", "claude-3-5-sonnet-20240620")

# --------------------------------------------------------------------------
# Core: ParallelTask, ParallelPipeline, and runner
# --------------------------------------------------------------------------


@dataclass
class ParallelTask:
    """One independent task that runs in parallel with others.

    Attributes:
        name:     Task identifier (used as key in results dict).
        system:   System prompt for this task's LLM call.
        template: User prompt template. Use {input} for the original request.
        handler:  Optional Python function. If set, called instead of LLM.
                  Signature: (input_text: str) -> str.
    """
    name: str
    system: str = ""
    template: str = "{input}"
    handler: Optional[Callable[[str], str]] = None


@dataclass
class ParallelPipeline:
    """A parallel-then-synthesis pipeline configuration.

    Attributes:
        name:                Human-readable pipeline name.
        tasks:               List of tasks to run concurrently.
        synthesizer_system:  System prompt for the merge step.
        synthesizer_template: User prompt for merge. Gets {input} and {results}
                              where {results} is JSON of task_name -> output.
    """
    name: str
    tasks: List[ParallelTask]
    synthesizer_system: str
    synthesizer_template: str


@dataclass
class ParallelResult:
    """Result from a parallel pipeline run."""
    input: str
    task_outputs: Dict[str, str] = field(default_factory=dict)
    synthesis: str = ""
    task_usages: Dict[str, dict] = field(default_factory=dict)
    synthesis_usage: dict = field(default_factory=dict)
    total_wall_time_ms: float = 0.0


def _call_llm(system: str, user_prompt: str, task_name: str) -> tuple[str, dict]:
    """Make one LLM call and return (text, usage_dict)."""
    logger.info("[%s] Calling model=%s", task_name, MODEL)

    response = client.messages.create(
        model=MODEL,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=4096,
    )

    usage = response.usage
    logger.info("[%s] Done | input_tokens=%d | output_tokens=%d",
                task_name, usage.input_tokens, usage.output_tokens)

    return response.content[0].text, {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


def _execute_task(task: ParallelTask, user_input: str) -> tuple[str, str, dict]:
    """Execute a single task (handler or LLM), return (name, output, usage)."""
    logger.info("[%s] Starting execution", task.name)

    if task.handler is not None:
        output = task.handler(user_input)
        usage = {}
    else:
        prompt = task.template.format(input=user_input)
        output, usage = _call_llm(task.system, prompt, task.name)

    preview = output[:120].replace("\n", " ")
    logger.info("[%s] Output: %s...", task.name, preview)
    return task.name, output, usage


def run_parallel(pipeline: ParallelPipeline, user_input: str) -> ParallelResult:
    """Execute the parallelization pattern: all tasks concurrently, then synthesize.

    Steps:
        1. Execute ALL tasks concurrently via ThreadPoolExecutor.
        2. Wait for all to complete.
        3. Format results as JSON and pass to synthesizer.
        4. Return combined result.
    """
    import time
    wall_start = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"Pipeline: {pipeline.name}")
    print(f"{'='*60}")
    print(f"Input: {user_input}")
    print(f"Parallel tasks ({len(pipeline.tasks)}): {[t.name for t in pipeline.tasks]}")
    print(f"{'─'*60}")

    task_outputs: Dict[str, str] = {}
    task_usages: Dict[str, dict] = {}

    # --- Step 1: Execute all tasks concurrently ---
    with ThreadPoolExecutor(max_workers=len(pipeline.tasks)) as executor:
        futures = {
            executor.submit(_execute_task, task, user_input): task.name
            for task in pipeline.tasks
        }

        for future in as_completed(futures):
            name, output, usage = future.result()
            task_outputs[name] = output
            task_usages[name] = usage

    # Log the parallel results
    print(f"\n{'─'*60}")
    print("All parallel tasks completed. Results summary:")
    for name, output in task_outputs.items():
        preview = output[:150].replace("\n", " ") + ("..." if len(output) > 150 else "")
        print(f"  [{name}] {preview}")
    print(f"{'─'*60}")

    # --- Step 2: Synthesize ---
    print(f"\n{'─'*60}")
    print("Running synthesis...")
    print(f"{'─'*60}")

    results_json = json.dumps(task_outputs, ensure_ascii=False, indent=2)
    synthesis_prompt = pipeline.synthesizer_template.format(
        input=user_input,
        results=results_json,
    )

    synthesis, synthesis_usage = _call_llm(
        pipeline.synthesizer_system, synthesis_prompt, f"{pipeline.name}/synthesis"
    )

    wall_end = time.perf_counter()
    total_time = (wall_end - wall_start) * 1000

    preview = synthesis[:300] + ("..." if len(synthesis) > 300 else "")
    print(f"\nSynthesis output:\n{preview}")

    return ParallelResult(
        input=user_input,
        task_outputs=task_outputs,
        synthesis=synthesis,
        task_usages=task_usages,
        synthesis_usage=synthesis_usage,
        total_wall_time_ms=total_time,
    )


# --------------------------------------------------------------------------
# Demo: Multi-perspective research brief
# --------------------------------------------------------------------------

DEMO_PIPELINE = ParallelPipeline(
    name="research-brief-generator",
    tasks=[
        ParallelTask(
            name="summary",
            system="你是一个专业摘要助手。用简洁的语言总结核心内容，突出关键信息。控制在200字以内。",
            template="请对以下主题进行简明扼要的总结：\n\n{input}",
        ),
        ParallelTask(
            name="questions",
            system="你是一个善于启发思考的提问专家。从不同角度提出有深度的问题。",
            template="针对以下主题，请从「技术挑战」「发展趋势」「社会影响」三个角度各提出一个有深度的问题：\n\n{input}",
        ),
        ParallelTask(
            name="key_terms",
            system="你是一个术语提取专家。提取最重要的关键词和核心概念，用逗号分隔。",
            template="请从以下主题中提取 8-12 个最关键的技术术语或核心概念，用逗号分隔：\n\n{input}",
        ),
    ],
    synthesizer_system="""你是一个研究简报汇编助手。
你将收到三份独立生成的素材：摘要、相关问题、关键术语。
请将它们整合为一份结构清晰、连贯的研究简报，格式如下：

## 研究简报：{input}

### 核心摘要
（整合 summary 的内容）

### 关键术语
（列出 key_terms，用粗体标注每个术语并加简短说明）

### 深入思考
（基于 questions 的内容，每个角度展开 1-2 句话的分析）

注意：仅基于提供的素材进行整合，不要添加外部知识。""",
    synthesizer_template="""请整合以下三份独立生成的研究素材，生成一份结构化的研究简报。

原始主题：{input}

--- 素材开始 ---
{results}
--- 素材结束 ---

请严格遵循你的系统指令进行整合。""",
)

DEMO_INPUTS = [
    "大语言模型在医疗诊断中的应用与挑战",
    "量子计算的现状与未来展望",
]


def _compare_time(user_input: str) -> None:
    """Demonstrate the time difference between sequential and parallel execution."""
    import time

    print(f"\n{'='*60}")
    print(f"Performance comparison for: {user_input[:40]}...")
    print(f"{'='*60}")

    # Sequential
    print("\n>> Sequential execution...")
    seq_start = time.perf_counter()
    for task in DEMO_PIPELINE.tasks:
        _execute_task(task, user_input)
    seq_end = time.perf_counter()
    seq_time = (seq_end - seq_start) * 1000

    # Parallel
    print("\n>> Parallel execution...")
    par_start = time.perf_counter()
    result = run_parallel(DEMO_PIPELINE, user_input)
    par_end = time.perf_counter()
    par_time = (par_end - par_start) * 1000

    # Comparison
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"\n{'='*60}")
    print(f"TIME COMPARISON:")
    print(f"  Sequential: {seq_time:.0f} ms")
    print(f"  Parallel:   {par_time:.0f} ms")
    print(f"  Speedup:    {speedup:.1f}x")
    print(f"{'='*60}")

    print(f"\n{'*'*60}")
    print(f"Final Synthesis:")
    print(f"{result.synthesis}")
    print(f"{'*'*60}")


if __name__ == "__main__":
    for user_input in DEMO_INPUTS:
        result = run_parallel(DEMO_PIPELINE, user_input)

        print(f"\n{'*'*60}")
        print(f"Pipeline: {DEMO_PIPELINE.name}")
        print(f"Input: {result.input}")
        print(f"Tasks: {list(result.task_outputs.keys())}")
        for name, output in result.task_outputs.items():
            print(f"\n  [{name}] ({len(output)} chars)")
            print(f"  {output[:200]}{'...' if len(output) > 200 else ''}")
        print(f"\n  Synthesis ({len(result.synthesis)} chars):")
        print(f"  {result.synthesis[:400]}{'...' if len(result.synthesis) > 400 else ''}")
        print(f"\n  Wall time: {result.total_wall_time_ms:.0f} ms")
        total_tokens = (
            sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in result.task_usages.values())
            + result.synthesis_usage.get("input_tokens", 0)
            + result.synthesis_usage.get("output_tokens", 0)
        )
        print(f"  Total tokens: {total_tokens}")
        print(f"{'*'*60}")
