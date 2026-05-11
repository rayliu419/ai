#!/usr/bin/env python3
"""
ch04_reflection.py - Reflection (self-evaluation) with raw Anthropic API

Reflection introduces a feedback loop into agentic workflows. Instead of a single
generate-and-done pass, the agent evaluates its own output and iteratively improves it.

  input
    |
    v
  [Producer] --(generates)--> draft
    |                            |
    |   +--(refine)-------------+  (loop back with feedback)
    |   |
    |   v
    |  [Critic] --(evaluates)--> feedback / score
    |                            |
    |                  +---------+---------+
    |                  v                   v
    |             quality OK          needs improvement
    |                  |                   |
    |                  v                   v
    |              final output      producer + feedback -> refine
    |
    +---<---<---<---<---<---<---<---< (iterate)

Key ideas from the book (Chapter 4):
- Producers generate content; Critics evaluate it with fresh perspective
- Two roles with DIFFERENT system prompts avoid self-review bias
- The iteration loop continues until quality threshold or max iterations
- Works well for: code generation, creative writing, complex reasoning, planning
- Combining with Memory (ch8) enables cumulative improvement across sessions
- Combining with Goals (ch11) gives the critic objective standards to measure against

Core pattern: (producer -> critic -> {accepted: output, rejected: producer + feedback})^

--------------------------------------------------------------------------
PRODUCTION INSIGHTS (from Claude Code source analysis of
/Users/liurui/workspace/claude-code/src/):

Claude Code implements the Producer-Critic pattern pervasively, with at least 10
distinct self-evaluation and self-correction subsystems.

THE MAJOR REFLECTION SYSTEMS:

+-----------------------------------------------------------------------+
| 1. VERIFICATION AGENT (Canonical Producer-Critic)                      |
|    src/tools/AgentTool/built-in/verificationAgent.ts                   |
|                                                                        |
| An independent "verification" agent is spawned to adversarially         |
| evaluate the main agent's work. It receives the original task, files   |
| changed, and approach, then runs builds, tests, and adversarial probes.|
| Produces a structured VERDICT: PASS / FAIL / PARTIAL.                  |
|                                                                        |
|   const VERIFICATION_SYSTEM_PROMPT =                                    |
|     "You are a verification specialist. Your job is not to confirm     |
|      the implementation works - it is to try to break it."              |
|                                                                        |
| The main agent is NUDGED to spawn verification when it closes out 3+   |
| tasks without a verification step (TaskUpdateTool.ts:326,              |
| TodoWriteTool.ts:73):                                                  |
|                                                                        |
|   "NOTE: You just closed out 3+ tasks and none of them was a           |
|    verification step. Before writing your final summary, spawn the     |
|    verification agent..."                                              |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 2. TWO-STAGE AUTO-MODE CLASSIFIER (Escalating Self-Evaluation)         |
|    src/utils/permissions/yoloClassifier.ts                             |
|                                                                        |
| Before EVERY tool call in auto mode, a classifier evaluates the        |
| action. TWO stages with escalating rigor:                              |
|                                                                        |
|   Stage 1 ("fast"): max_tokens=64, stop_sequences, quick allow/block   |
|     decision. Biased toward blocking ("Err on the side of blocking").  |
|                                                                        |
|   Stage 2 ("thinking", only if stage 1 blocked): max_tokens=4096,      |
|     chain-of-thought re-evaluation. Less biased, reduces false pos.    |
|     ("Review the classification process and follow it carefully.")     |
|                                                                        |
| This is the most frequent reflection in production - runs on every     |
| single tool call the model makes in auto mode.                         |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 3. HANDOFF CLASSIFICATION (Sub-Agent Output Review)                    |
|    src/tools/AgentTool/agentToolUtils.ts:389                           |
|                                                                        |
| After a sub-agent completes, its ENTIRE transcript is classified.      |
| If the classifier flags dangerous actions, a SECURITY WARNING is       |
| prepended instead of the raw result:                                   |
|                                                                        |
|   const handoffDecision = classifierResult.shouldBlock                 |
|     ? "blocked"                                                        |
|     : "allowed"                                                        |
|   if (classifierResult.shouldBlock) {                                  |
|     return `SECURITY WARNING: This sub-agent performed actions that    |
|              may violate security policy...`                           |
|   }                                                                    |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 4. STRUCTURED OUTPUT ENFORCEMENT (Hard Constraint + Retry)             |
|    src/utils/hooks/hookHelpers.ts:70 + QueryEngine.ts:1004             |
|                                                                        |
| A Stop hook checks whether the model called the required               |
| SyntheticOutputTool. If not, it blocks and tells the model it MUST:    |
|                                                                        |
|   addFunctionHook(setAppState, sessionId, "Stop", "",                  |
|     messages => hasSuccessfulToolCall(messages, SYNTHETIC_OUTPUT),     |
|     "You MUST call the SyntheticOutputTool to complete this request.") |
|                                                                        |
| After MAX_STRUCTURED_OUTPUT_RETRIES (default 5) consecutive            |
| failures, the system gives up with a clear error rather than           |
| looping forever - a circuit breaker pattern.                           |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 5. MULTI-STAGE RECOVERY CHAIN (Error-Driven Self-Correction)           |
|    src/query.ts:893-1252                                               |
|                                                                        |
| When the API returns errors, the system tries increasingly aggressive  |
| recovery strategies before surfacing the error:                        |
|                                                                        |
|   Recovery chain for prompt-too-long (413):                            |
|     1. Context collapse drain (cheapest, same context)                 |
|     2. Reactive compact (full summarization, loses detail)             |
|     3. Surface error to user                                           |
|                                                                        |
|   Recovery chain for max output tokens:                                |
|     1. Escalate from 8k to 64k output tokens (one retry)              |
|     2. Insert recovery message: "Resume directly - no apology"         |
|     3. Up to MAX_OUTPUT_TOKENS_RECOVERY_LIMIT attempts, then error    |
|                                                                        |
|   Model fallback:                                                      |
|     Primary model fails -> switch to fallback model, strip             |
|     incompatible thinking signatures, retry clean.                     |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 6. STOP HOOKS (Post-Turn Evaluation)                                   |
|    src/query/stopHooks.ts + src/query.ts:1267                          |
|                                                                        |
| After the model completes each turn, Stop hooks evaluate the           |
| output. They can produce blocking errors that force the model to       |
| retry:                                                                 |
|                                                                        |
|   const stopHookResult = yield* handleStopHooks(...)                   |
|   if (stopHookResult.blockingErrors.length > 0) {                     |
|     state.messages.push(...stopHookResult.blockingErrors)             |
|     continue  // re-enter the query loop                               |
|   }                                                                    |
|                                                                        |
| Also fires background reflection tasks: memory extraction,             |
| auto-dream consolidation, prompt suggestion.                           |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 7. AUTO-DREAM / MEMORY EXTRACTION (Background Self-Reflection)         |
|    src/services/autoDream/autoDream.ts + extractMemories.ts            |
|                                                                        |
| After a turn ends, background processes consolidate recent sessions    |
| into durable memory files:                                             |
|                                                                        |
|   auto-dream system prompt:                                            |
|     "You are performing a dream - a reflective pass over your          |
|      memory files. Synthesize what you have learned recently into      |
|      durable, well-organized memories."                                |
|                                                                        |
| Uses forked agents to avoid blocking the main loop and shares         |
| cache-critical params for prompt cache hits (forkedAgent.ts).          |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 8. AUTO-MODE RULES CRITIQUE (LLM Evaluates User Config)                |
|    src/cli/handlers/autoMode.ts:49                                     |
|                                                                        |
| Uses a separate LLM call (sideQuery) to critique user-written          |
| auto-mode rules for clarity, completeness, conflicts:                  |
|                                                                        |
|   const CRITIQUE_SYSTEM_PROMPT =                                       |
|     "You are an expert reviewer of auto mode classifier rules..."      |
|   response = await sideQuery({                                         |
|     system: CRITIQUE_SYSTEM_PROMPT,                                    |
|     messages: [{ role: "user", content: "Please critique rules." }],  |
|   })                                                                   |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 9. LSP PASSIVE FEEDBACK (External Diagnostics as Reflection Signal)    |
|    src/services/lsp/passiveFeedback.ts                                 |
|                                                                        |
| LSP diagnostics (compile errors, warnings) are formatted and fed       |
| back to the model as context. This gives the model a self-correction   |
| signal from an external compiler/language server.                      |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| 10. PLAN APPROVAL/REJECTION (User-in-the-Loop Self-Correction)         |
|    src/tools/SendMessageTool/SendMessageTool.ts:898                   |
|                                                                        |
| When a plan is rejected, feedback is passed back to the agent for      |
| revision in the next turn:                                             |
|                                                                        |
|   return handlePlanRejection(                                          |
|     input.to, input.message.request_id,                                |
|     input.message.feedback ?? "Plan needs revision", context)          |
+-----------------------------------------------------------------------+

KEY ARCHITECTURAL INSIGHTS (from real code):

1. PRODUCER-CRITIC IS A SPECTRUM, NOT A BINARY: The "critic" role ranges from
   a full independent agent (Verification Agent) to a lightweight LLM call
   (YOLO classifier) to passive diagnostics (LSP). The book's two-role model
   is correct but incomplete - production uses at least 5 critic variants.

2. REFLECTION HAPPENS AT EVERY LEVEL:
   - Per-tool: YOLO classifier evaluates each tool call before execution
   - Per-turn: Stop hooks evaluate the full model output after each turn
   - Per-task: Verification agent evaluates the complete implementation
   - Cross-session: Auto-dream / memory extraction reflects across sessions

3. CIRCUIT BREAKERS PREVENT INFINITE LOOPS: Every self-correction loop has
   a hard limit (max_structured_output_retries=5, MAX_CONSECUTIVE_AUTOCOMPACT
   _FAILURES=3, MAX_OUTPUT_TOKENS_RECOVERY_LIMIT). The book's model assumes
   "iterate until quality threshold" but production needs fallbacks.

4. ESCALATING RIGOR: The two-stage classifier (fast then thinking) is a
   pattern that appears throughout - cheap evaluation first, expensive
   evaluation only when needed. This mirrors the book's "iterative
   refinement" but with cost awareness.

5. EVALUATION IS ADVERSARIAL, NOT COLLABORATIVE: The verification agent is
   explicitly told "your job is to try to BREAK it." The critic is not a
   helpful editor but an adversarial tester. This is a stronger pattern
   than the book's "critic provides improvement suggestions."

6. BACKGROUND REFLECTION: Memory extraction and auto-dream run as FORKED
   sub-agents (non-blocking), not blocking the main loop. Reflection does
   not have to be synchronous - it can happen after the fact.

7. NUDGES ARE SOFTER THAN ENFORCEMENT: The verification nudge only reminds
   the model to verify - it does not force it. The structured output
   enforcement on the other hand BLOCKS continuation until satisfied.
   Production uses both soft and hard reflection triggers.

COMPARISON TABLE:
| Aspect                   | Book ch04 (this file)    | Claude Code production            |
|--------------------------|--------------------------|-----------------------------------|
| Critic role              | Single critic LLM        | Multiple: agent, LLM, or passive  |
| Evaluation rigor         | Fixed threshold          | Two-stage: fast then thinking     |
| Iteration control        | Quality threshold        | Circuit breakers + hard limits    |
| Critic objectivity       | Separate system prompt   | Adversarial "break it" mindset    |
| Reflection timing        | Synchronous loop         | Sync (per-tool) + async (dream)   |
| Error recovery           | None                     | Multi-stage recovery chain        |
| Output constraints       | Prompt-guided only       | Hard enforcement with retry limit |
| External signals         | Not included             | LSP diagnostics, test output      |
| Scaling                  | Single producer-critic   | Nested: per-tool, per-turn, tasks |

Usage:
    python agents_new/ch04_reflection.py
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

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
# Core: Reflection dataclasses and executor
# --------------------------------------------------------------------------


@dataclass
class ReflectiveAgent:
    """A producer-critic reflection configuration.

    Attributes:
        name:                   Agent name for logging.
        producer_system:        System prompt for the content generator.
        critic_system:          System prompt for the evaluator.
        user_template:          User message template. Use {input} for the user's
                                original request and {draft} for the current draft.
        max_iterations:         Maximum producer-critic rounds (default 3).
        quality_threshold:      Score threshold (0-10) above which iteration stops.
                                Set to 0 to skip critic-based stopping and always
                                run max_iterations rounds. Set to 11 to always stop
                                after one round (no iterative reflection).
    """
    name: str
    producer_system: str
    critic_system: str
    user_template: str = "{input}"
    max_iterations: int = 3
    quality_threshold: int = 8

    _iteration: int = field(default=0, repr=False)
    _history: list = field(default_factory=list, repr=False)


@dataclass
class ReflectionRound:
    """One iteration of the reflection loop."""
    iteration: int
    draft: str
    feedback: str
    score: int
    accepted: bool


@dataclass
class ReflectionResult:
    """Result from a full reflection run."""
    input: str
    rounds: list = field(default_factory=list)
    final_output: str = ""
    accepted_at_round: int = 0
    total_usage: dict = field(default_factory=dict)


def _call_llm(system: str, user_prompt: str, step_name: str) -> tuple[str, dict]:
    """Make one LLM call and return (text, usage_dict)."""
    logger.info("[%s] Calling model=%s", step_name, MODEL)

    response = client.messages.create(
        model=MODEL,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=4096,
    )

    usage = response.usage
    logger.info("[%s] Done | input_tokens=%d | output_tokens=%d",
                step_name, usage.input_tokens, usage.output_tokens)

    return response.content[0].text, {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


def _parse_score(feedback: str) -> int:
    """Extract a numeric score (0-10) from critic feedback text.

    The critic is instructed to output a score line like 'Score: 7'.
    This parser finds the first integer that follows 'Score:' or falls
    back to extracting the first integer found.
    """
    for line in feedback.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("score"):
            for token in stripped.replace(":", " ").split():
                try:
                    return max(0, min(10, int(token)))
                except ValueError:
                    continue
    # Fallback: scan for any integer in the text
    for token in feedback.replace(":", " ").split():
        try:
            val = int(token)
            if 0 <= val <= 10:
                return val
        except ValueError:
            continue
    return 0


def reflect(agent: ReflectiveAgent, user_input: str) -> ReflectionResult:
    """Run the producer-critic reflection loop.

    For each iteration:
      1. PRODUCER: generates or refines the draft.
      2. CRITIC:   evaluates the draft, returns feedback + numeric score.
      3. DECIDE:   if score >= threshold or max iterations reached -> accept.
                   otherwise, feed feedback back to producer and repeat.
    """
    result = ReflectionResult(input=user_input)
    total_input_tokens = 0
    total_output_tokens = 0

    print(f"\n{'='*60}")
    print(f"Reflective Agent: {agent.name}")
    print(f"Max iterations: {agent.max_iterations}, Quality threshold: {agent.quality_threshold}")
    print(f"{'='*60}")
    print(f"Input: {user_input}")

    draft = ""
    accepted = False
    final_round = 0

    for iteration in range(1, agent.max_iterations + 1):
        print(f"\n{'─'*60}")
        print(f"Iteration {iteration}/{agent.max_iterations}")
        print(f"{'─'*60}")

        # --- Step 1: Producer generates/refines ---
        if iteration == 1:
            # First pass: generate from scratch
            producer_prompt = agent.user_template.format(input=user_input)
            producer_step = f"{agent.name}/producer/v{iteration}"
        else:
            # Subsequent passes: refine based on critic feedback
            producer_prompt = (
                f"Original request: {user_input}\n\n"
                f"Current draft (iteration {iteration - 1}):\n{draft}\n\n"
                f"Critic feedback from last round:\n{feedback}\n\n"
                f"Please revise the draft above to address ALL of the critic's "
                f"concerns. Improve the quality based on the feedback provided."
            )
            producer_step = f"{agent.name}/producer/refine/v{iteration}"

        print(f"  >> PRODUCER (generating)...")
        draft, p_usage = _call_llm(agent.producer_system, producer_prompt, producer_step)
        total_input_tokens += p_usage.get("input_tokens", 0)
        total_output_tokens += p_usage.get("output_tokens", 0)

        preview = draft[:200].replace("\n", " ") + ("..." if len(draft) > 200 else "")
        print(f"     Draft: {preview}")

        # --- Step 2: Critic evaluates ---
        critic_prompt = (
            f"Original request: {user_input}\n\n"
            f"Draft to evaluate (iteration {iteration}):\n{draft}\n\n"
            f"Please evaluate the draft above. Provide a quality score and "
            f"specific, actionable feedback for improvement."
        )
        critic_step = f"{agent.name}/critic/v{iteration}"

        print(f"  >> CRITIC (evaluating)...")
        feedback, c_usage = _call_llm(agent.critic_system, critic_prompt, critic_step)
        total_input_tokens += c_usage.get("input_tokens", 0)
        total_output_tokens += c_usage.get("output_tokens", 0)

        score = _parse_score(feedback)

        # Show score prominently
        score_bar = "█" * score + "░" * (10 - score)
        print(f"     Score: {score}/10 {score_bar}")

        feedback_preview = feedback[:250].replace("\n", " ") + ("..." if len(feedback) > 250 else "")
        print(f"     Feedback: {feedback_preview}")

        # --- Step 3: Check acceptance criteria ---
        accepted = score >= agent.quality_threshold

        round_result = ReflectionRound(
            iteration=iteration,
            draft=draft,
            feedback=feedback,
            score=score,
            accepted=accepted,
        )
        result.rounds.append(round_result)

        if accepted:
            print(f"\n  >> ACCEPTED (score {score} >= threshold {agent.quality_threshold})")
            final_round = iteration
            break
        else:
            print(f"  >> Needs improvement (score {score} < {agent.quality_threshold}), refining...")

    # --- Final output ---
    if not accepted and result.rounds:
        # Use the last draft even if threshold not met
        final_round = agent.max_iterations
        print(f"\n  >> Max iterations ({agent.max_iterations}) reached, using last draft.")

    result.final_output = result.rounds[-1].draft
    result.accepted_at_round = final_round
    result.total_usage = {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}

    print(f"\n{'='*60}")
    print(f"Reflection complete: accepted at round {final_round}")
    print(f"Total tokens: {total_input_tokens + total_output_tokens}")
    print(f"{'='*60}")

    return result


# --------------------------------------------------------------------------
# Demo 1: Code quality reflection
# --------------------------------------------------------------------------

CODE_AGENT = ReflectiveAgent(
    name="code-refiner",
    producer_system=(
        "你是一名经验丰富的 Python 工程师。根据用户需求编写清晰、可运行、"
        "符合 PEP 8 标准的 Python 代码。包含必要的注释和类型提示。"
        "优先考虑可读性和正确性。"
    ),
    critic_system=(
        "你是一名严格的代码审查员（Code Reviewer）。评估代码的质量，检查：\n"
        "1. 正确性：代码是否能正确运行？是否有逻辑错误？\n"
        "2. 安全性：是否存在注入风险或其他安全问题？\n"
        "3. 性能：是否存在明显的性能问题或不必要的复杂度？\n"
        "4. 可读性：代码是否清晰？命名是否合理？是否需要更多注释？\n"
        "5. 风格：是否符合 PEP 8？\n\n"
        "最后一行必须输出 'Score: N'，其中 N 是 0-10 的整数。\n"
        "分数含义：0-3=严重问题 4-6=需要改进 7-8=少量问题 9-10=优秀。\n"
        "在评分之前，提供具体的改进建议。"
    ),
    user_template="请编写 Python 代码实现以下功能：\n\n{input}",
    max_iterations=3,
    quality_threshold=8,
)

CODE_DEMO_INPUTS = [
    "一个函数，接收 URL 列表，并发下载所有 URL 的内容，返回响应文本的字典。",
]


# --------------------------------------------------------------------------
# Demo 2: Creative writing reflection
# --------------------------------------------------------------------------

WRITING_AGENT = ReflectiveAgent(
    name="writing-refiner",
    producer_system=(
        "你是一名专业的中文文案撰稿人。根据要求创作高质量、有感染力、"
        "语言精炼的文案。注意节奏感和画面感。"
    ),
    critic_system=(
        "你是一名资深编辑。评估文案的质量，重点检查：\n"
        "1. 清晰度：信息是否明确传达？是否有歧义？\n"
        "2. 感染力：语言是否有力度？是否能打动目标读者？\n"
        "3. 结构：逻辑是否清晰？段落过渡是否自然？\n"
        "4. 简洁性：是否有冗余表达？能否更精简？\n"
        "5. 语气：是否符合要求的语气和风格？\n\n"
        "最后一行必须输出 'Score: N'，其中 N 是 0-10 的整数。\n"
        "在评分之前，提供具体的修改建议和示例。"
    ),
    user_template="{input}",
    max_iterations=3,
    quality_threshold=8,
)

WRITING_DEMO_INPUTS = [
    "写一段 100 字以内的产品宣传文案，推广一个名为「思镜」的 AI 写作助手产品。"
    "目标用户是内容创作者，核心卖点是「保持个人风格的同时提升效率」。",
]


# --------------------------------------------------------------------------
# Demo runner
# --------------------------------------------------------------------------

def run_demo(agent: ReflectiveAgent, inputs: list[str], title: str) -> None:
    """Run reflection demo for a given agent and set of inputs."""
    print(f"\n\n{'#'*60}")
    print(f"# {title}")
    print(f"{'#'*60}")

    for user_input in inputs:
        result = reflect(agent, user_input)

        print(f"\n{'*'*60}")
        print(f"Reflection Summary:")
        print(f"  Agent: {agent.name}")
        print(f"  Input: {result.input[:60]}...")
        print(f"  Rounds: {len(result.rounds)}")
        print(f"  Accepted at: round {result.accepted_at_round}")
        print(f"  Total tokens: "
              f"{result.total_usage['input_tokens'] + result.total_usage['output_tokens']}")
        for i, round_data in enumerate(result.rounds, 1):
            status = "ACCEPTED" if round_data.accepted else "REFINED"
            print(f"  Round {i}: score={round_data.score}/10 [{status}]")
        print(f"\nFinal output:\n{result.final_output[:600]}")
        if len(result.final_output) > 600:
            print("...")
        print(f"{'*'*60}")


if __name__ == "__main__":
    run_demo(CODE_AGENT, CODE_DEMO_INPUTS, "Demo 1: Code Quality Reflection")
    run_demo(WRITING_AGENT, WRITING_DEMO_INPUTS, "Demo 2: Creative Writing Reflection")
