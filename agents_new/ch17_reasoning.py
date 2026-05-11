#!/usr/bin/env python3
"""
ch17_reasoning.py - Reasoning Techniques with raw Anthropic API

This chapter implements advanced reasoning techniques that enable agents to
perform structured, multi-step logical reasoning and problem decomposition.
The core principle: allocating more compute at inference time ("thinking time")
leads to better results - the Reasoning Scaling Law.

input -> [Classifier] -> simple?     -> CoT (chain-of-thought)
                      -> complex?    -> ToT (tree-of-thought)
                      -> actionable? -> ReAct (reasoning + acting)
                      -> debatable?  -> CoD (chain-of-debate)
      -> [Strategy]   -> execute with thinking budget
      -> [Trace]      -> structured reasoning trace for observability

Implemented reasoning techniques (Chapter 17):
- CoT (Chain-of-Thought): Step-by-step reasoning with intermediate steps
- ToT (Tree-of-Thought): Explore multiple reasoning paths, select the best
- Self-correction: Critically review and iteratively improve output
- ReAct: Think -> Act -> Observe loop with tool interaction
- CoD (Chain-of-Debate): Multi-role debate to reduce bias and improve accuracy
- Native Extended Thinking: Use Anthropic API's built-in thinking capability
- Reasoning Router: Auto-select strategy based on problem type and budget

--------------------------------------------------------------------------
PRODUCTION INSIGHTS (from Claude Code source analysis):

Two fundamentally different approaches to reasoning exist:

1. PROMPT ENGINEERING (what this file demonstrates for teaching purposes):
   - Uses system prompts to guide model reasoning format
   - Works with ANY model (OpenAI, Gemini, open-source, etc.)
   - Quality depends on model's instruction-following ability
   - Reasoning tokens consume output quota

2. NATIVE EXTENDED THINKING (what Claude Code actually uses in production):
   - Uses Anthropic API's `thinking` parameter: { type: "adaptive" }
   - Model reasons in a separate "thinking block" before responding
   - Thinking tokens DON'T consume output quota - more efficient
   - Trained into the model during post-training (RLHF) - not a prompt trick
   - System prompt does NOT contain CoT instructions at all

Key production architecture decisions from Claude Code:
- CoT: Handled entirely by native `thinking` - no prompt engineering needed
- ToT: NOT explicitly implemented; strong models + adaptive thinking + agent
  loop (retry on failure) achieve similar results implicitly
- ReAct: Implemented as the core agent loop (tool_use stop_reason -> execute
  -> feed back -> loop). The thinking block provides the "thought" step natively
- Self-correction: Emerges from the agent loop - if a tool call fails or test
  doesn't pass, the model sees the error and adjusts in the next iteration

The harness layer's UNIQUE value for reasoning is in MULTI-CALL ORCHESTRATION:
- ToT (multiple paths + evaluation) - requires N+1 API calls
- CoD (multi-role debate) - requires M*N API calls
- Self-correction loops - requires iterative calls with feedback
- These CANNOT be achieved by a single API call, no matter how good the model

This file implements BOTH approaches for comparison and education.
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple

from anthropic import Anthropic
from dotenv import load_dotenv

# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# Reasoning Trace: structured observability for all reasoning strategies
# -------------------------------------------------------------------------

class StrategyType(str, Enum):
    COT = "chain_of_thought"
    NATIVE_THINKING = "native_extended_thinking"
    TOT = "tree_of_thought"
    SELF_CORRECT = "self_correction"
    REACT = "react"
    COD = "chain_of_debate"

@dataclass
class TraceStep:
    """One atomic step in a reasoning trace."""
    step_type: str  # e.g. "thought", "action", "observation", "critique", "branch"
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    elapsed_ms: int = 0

@dataclass
class ReasoningTrace:
    """Full reasoning trace for observability and debugging."""
    strategy: StrategyType
    question: str
    steps: List[TraceStep] = field(default_factory=list)
    final_answer: str = ""
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_elapsed_ms: int = 0

    def add_step(self, step: TraceStep):
        self.steps.append(step)
        self.total_tokens_in += step.tokens_in
        self.total_tokens_out += step.tokens_out
        self.total_elapsed_ms += step.elapsed_ms

    def summary(self) -> str:
        lines = [
            f"Strategy: {self.strategy.value}",
            f"Steps: {len(self.steps)}",
            f"Tokens: {self.total_tokens_in} in / {self.total_tokens_out} out",
            f"Time: {self.total_elapsed_ms}ms",
        ]
        return " | ".join(lines)

# -------------------------------------------------------------------------
# Tools (for ReAct strategy)
# -------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_info",
        "description": "搜索某个主题的信息（模拟）。返回该主题的关键事实。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "执行数学计算。支持基本运算和 sqrt, pow, abs, pi, e。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"},
            },
            "required": ["expression"],
        },
    },
]

def _search_info(query: str) -> str:
    """Simulated search with knowledge base."""
    knowledge = {
        "量子计算": "量子计算利用量子力学原理（叠加、纠缠）进行计算。量子比特可同时处于0和1状态。应用：密码学、药物...",
        "强化学习": "强化学习(RL)是机器学习分支，智能体通过与环境交互获得奖励学习策略。关键：MDP、Q-learning、PP...",
        "transformer": "Transformer(2017)基于自注意力机制，革命性改变NLP。核心：多头注意力、位置编码。衍生：BE...",
        "气候变化": "全球平均温度较工业化前升高约1.1℃。主因：化石燃料燃烧排放CO2。影响：海平面上升、极端天气增加。",
        "经济学": "供需关系决定市场价格。GDP衡量经济产出。通货膨胀反映物价水平变化。财政政策和货币政策是主要调控工具。"
    }
    for key, value in knowledge.items():
        if key in query.lower() or query.lower() in key:
            return json.dumps({"query": query, "result": value}, ensure_ascii=False)
    return json.dumps({"query": query, "result": f"关于'{query}'的信息：这是一个重要领域，涉及多方面的知识。"}, ensure_ascii=False)

def _calculator(expression: str) -> str:
    """Safe calculator with limited builtins."""
    allowed = {"sqrt": math.sqrt, "pow": pow, "abs": abs, "pi": math.pi, "e": math.e}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"expression": expression, "error": str(e)})

TOOL_HANDLERS = {
    "search_info": lambda args: _search_info(**args),
    "calculator": lambda args: _calculator(**args),
}

# -------------------------------------------------------------------------
# LLM call helper
# -------------------------------------------------------------------------

def _call_llm(system: str, messages: List[dict], label: str = "") -> Tuple[str, int, int, int]:
    """Make one LLM call. Returns (text, tokens_in, tokens_out, elapsed_ms)"""
    t0 = time.time()
    response = client.messages.create(
        model=MODEL,
        system=system,
        messages=messages,
        max_tokens=4096,
    )
    elapsed = int((time.time() - t0) * 1000)
    text = "".join(b.text for b in response.content if hasattr(b, "text"))
    logger.info("[%s] tokens=%d/%d | %dms", label, response.usage.input_tokens, response.usage.output_tokens, elapsed)
    return text, response.usage.input_tokens, response.usage.output_tokens, elapsed

# -------------------------------------------------------------------------
# Strategy 1: Chain-of-Thought (CoT)
# -------------------------------------------------------------------------

def cot_reason(question: str) -> ReasoningTrace:
    """Chain-of-Thought: guide the model to reason step by step."""
    trace = ReasoningTrace(strategy=StrategyType.COT, question=question)

    print(f"\n{'-'*60}")
    print(f"[CoT] Chain-of-Thought Reasoning")
    print(f"{'-'*60}")
    print(f"Question: {question}")

    system = (
        "你是一个善于逐步推理的智能体。对于每个问题：\n"
        "1. 先分析问题，识别关键信息和约束\n"
        "2. 将问题分解为子步骤\n"
        "3. 逐步推理，每步给出理由\n"
        "4. 最终给出明确答案\n\n"
        "格式要求：\n"
        "**分析**：[问题分析]\n"
        "**步骤1**：[推理内容]\n"
        "**步骤2**：[推理内容]\n"
        "...\n"
        "**最终答案**：[答案]"
    )

    messages = [{"role": "user", "content": question}]
    text, t_in, t_out, elapsed = _call_llm(system, messages, label="cot")

    trace.add_step(TraceStep(
        step_type="thought",
        content=text,
        tokens_in=t_in,
        tokens_out=t_out,
        elapsed_ms=elapsed,
    ))
    trace.final_answer = text

    print(f"\n[CoT Result]\n{text}")
    return trace

# -------------------------------------------------------------------------
# Strategy 2: Tree-of-Thought (ToT)
# -------------------------------------------------------------------------

def _tot_generate_branches(question: str, num_branches: int, trace: ReasoningTrace) -> List[str]:
    """Phase 1: generate N independent reasoning paths."""
    print(f"\n Phase 1: Generating {num_branches} reasoning paths...")
    branches = []
    for i in range(1, num_branches + 1):
        system = (
            f"你是推理路径 {i}（共{num_branches}条独立路径之一）。\n"
            "请从你独特的视角分析问题并给出答案。\n"
            "尝试一种不同于常规的思考方式。\n"
            "格式：先给出推理过程，最后一行以「答案：」开头给出结论。"
        )
        messages = [{"role": "user", "content": question}]
        text, t_in, t_out, elapsed = _call_llm(system, messages, label=f"tot/branch-{i}")

        branches.append(text)
        trace.add_step(TraceStep(
            step_type="branch",
            content=f"[Branch {i}] {text}",
            tokens_in=t_in,
            tokens_out=t_out,
            elapsed_ms=elapsed,
        ))
        print(f"\n [Branch {i}] {text[:150]}...")
    return branches

def _tot_evaluate_branches(question: str, branches: List[str], trace: ReasoningTrace) -> str:
    """Phase 2: evaluate branches and select the best answer."""
    print(f"\n Phase 2: Evaluating branches...")
    eval_prompt = (
        f"问题：{question}\n\n"
        "以下是多条独立的推理路径：\n\n"
    )
    for i, b in enumerate(branches, 1):
        eval_prompt += f"--- 路径 {i} ---\n{b}\n\n"

    eval_prompt += (
        "请作为评审者：\n"
        "1. 分析每条路径的优缺点（逻辑正确性、完整性、创新性）\n"
        "2. 指出各路径的错误（如有）\n"
        "3. 综合最佳路径的优点，给出最终最优答案\n\n"
        "格式：\n"
        "**评估**：[各路径分析]\n"
        "**最终答案**：[综合最优答案]"
    )

    system = "你是一名严谨的推理评审者。评估多条推理路径，选择和综合最佳答案。"
    messages = [{"role": "user", "content": eval_prompt}]
    text, t_in, t_out, elapsed = _call_llm(system, messages, label="tot/evaluate")

    trace.add_step(TraceStep(
        step_type="evaluation",
        content=text,
        tokens_in=t_in,
        tokens_out=t_out,
        elapsed_ms=elapsed,
    ))
    print(f"\n[ToT Final]\n{text[:500]}...")
    return text

def tot_reason(question: str, num_branches: int = 3) -> ReasoningTrace:
    """Tree-of-Thought: explore multiple reasoning paths, evaluate, and select."""
    trace = ReasoningTrace(strategy=StrategyType.TOT, question=question)

    print(f"\n{'-'*60}")
    print(f"[ToT] Tree-of-Thought Reasoning ({num_branches} branches)")
    print(f"{'-'*60}")
    print(f"Question: {question}")

    branches = _tot_generate_branches(question, num_branches, trace)
    trace.final_answer = _tot_evaluate_branches(question, branches, trace)
    return trace

# -------------------------------------------------------------------------
# Strategy 3: Self-Correction
# -------------------------------------------------------------------------

def _self_correct_critique(question: str, draft: str, round_num: int, trace: ReasoningTrace) -> Tuple[str, bool]:
    """Run one critique round. Returns (critique_text, is_approved)."""
    print(f"\n Round {round_num}: Critiquing...")
    critique_system = (
        "你是一名严格的审查者。审查以下内容，检查：\n"
        "- 事实准确性\n"
        "- 逻辑完整性\n"
        "- 是否有遗漏或偏见\n"
        "- 表达清晰度\n\n"
        "如果内容已经很好，回复 'APPROVED'。\n"
        "否则，列出具体问题和改进建议。"
    )
    critique_messages = [
        {"role": "user", "content": f"原始问题：{question}\n\n待审查内容：\n{draft}"}
    ]
    critique, t_in, t_out, elapsed = _call_llm(
        critique_system, critique_messages, label=f"self-correct/critique-{round_num}"
    )

    trace.add_step(TraceStep(
        step_type="critique",
        content=critique,
        tokens_in=t_in,
        tokens_out=t_out,
        elapsed_ms=elapsed,
    ))

    if "APPROVED" in critique:
        print(f" [Approved] Content meets quality standards.")
        return critique, True

    print(f" [Critique] {critique[:200]}...")
    return critique, False

def _self_correct_refine(question: str, draft: str, critique: str, round_num: int, trace: ReasoningTrace) -> str:
    """Refine the draft based on critique. Returns updated draft."""
    refine_system = "你是一名专家。根据批评意见改进你的回答，使其更准确、完整。"
    refine_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": draft},
        {"role": "user", "content": f"请根据以下评审意见改进你的回答：\n\n{critique}"},
    ]
    draft, t_in, t_out, elapsed = _call_llm(
        refine_system, refine_messages, label=f"self-correct/refine-{round_num}"
    )

    trace.add_step(TraceStep(
        step_type="refinement",
        content=draft,
        tokens_in=t_in,
        tokens_out=t_out,
        elapsed_ms=elapsed,
    ))
    print(f" [Draft V{round_num + 1}] {draft[:200]}...")
    return draft

def self_correct(question: str, max_rounds: int = 3) -> ReasoningTrace:
    """Self-Correction: generate, critique, and iteratively improve."""
    trace = ReasoningTrace(strategy=StrategyType.SELF_CORRECT, question=question)

    print(f"\n{'-'*60}")
    print(f"[Self-Correct] Iterative Refinement (max {max_rounds} rounds)")
    print(f"{'-'*60}")
    print(f"Question: {question}")

    # --- Initial generation ---
    system = "你是一名专家。根据问题给出详细、准确的回答。"
    messages = [{"role": "user", "content": question}]
    draft, t_in, t_out, elapsed = _call_llm(system, messages, label="self-correct/init")

    trace.add_step(TraceStep(
        step_type="initial_draft",
        content=draft,
        tokens_in=t_in,
        tokens_out=t_out,
        elapsed_ms=elapsed,
    ))
    print(f" [Draft V1] {draft[:200]}...")

    # --- Critique and refine loop ---
    for round_num in range(1, max_rounds + 1):
        critique, approved = _self_correct_critique(question, draft, round_num, trace)
        if approved:
            break
        draft = _self_correct_refine(question, draft, critique, round_num, trace)

    trace.final_answer = draft
    print(f"\n[Self-Correct Final]\n{draft[:500]}...")
    return trace

# -------------------------------------------------------------------------
# Strategy 4: ReAct (Reasoning + Acting)
# -------------------------------------------------------------------------

def _react_handle_tool_use(response, thought_text: str, turn: int, messages: List[dict], trace: ReasoningTrace, t_in: int, t_out: int, elapsed: int) -> None:
    """Handle a tool_use turn: record trace, execute tools, append observations."""
    tool_blocks = [b for b in response.content if b.type == "tool_use"]
    messages.append({"role": "assistant", "content": response.content})

    trace.add_step(TraceStep(
        step_type="thought+action",
        content=f"{thought_text}\n[Action: {tool_blocks[0].name}({json.dumps(tool_blocks[0].input, ensure_ascii=False)})]",
        tokens_in=t_in,
        tokens_out=t_out,
        elapsed_ms=elapsed,
    ))

    tool_results = []
    for block in tool_blocks:
        handler = TOOL_HANDLERS.get(block.name)
        result = handler(block.input) if handler else '{"error": "unknown tool"}'
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result,
        })
        print(f" [Turn {turn}] Thought: {thought_text[:100]}...")
        print(f"            Action: {block.name}({json.dumps(block.input, ensure_ascii=False)[:60]})")
        print(f"            Observation: {result[:100]}...")

        trace.add_step(TraceStep(step_type="observation", content=result))

    messages.append({"role": "user", "content": tool_results})

def react_reason(question: str, max_turns: int = 5) -> ReasoningTrace:
    """ReAct: iterative Think -> Act -> Observe loop."""
    trace = ReasoningTrace(strategy=StrategyType.REACT, question=question)

    print(f"\n{'-'*60}")
    print(f"[ReAct] Reasoning + Acting (max {max_turns} turns)")
    print(f"{'-'*60}")
    print(f"Question: {question}")

    system = (
        "你是一个具备工具使用能力的推理智能体。请通过 Think-Act-Observe 循环解决问题。\n"
        "对于每一步，先输出你的思考过程，如果需要工具，则调用工具。\n"
        "如果已经得到答案，请直接给出最终结论。"
    )

    messages = [{"role": "user", "content": question}]

    for turn in range(1, max_turns + 1):
        t0 = time.time()
        response = client.messages.create(
            model=MODEL,
            system=system,
            messages=messages,
            max_tokens=4096,
            tools=TOOLS,
        )
        elapsed = int((time.time() - t0) * 1000)
        t_in, t_out = response.usage.input_tokens, response.usage.output_tokens

        thought_text = "".join(b.text for b in response.content if b.type == "text")

        if response.stop_reason == "tool_use":
            _react_handle_tool_use(response, thought_text, turn, messages, trace, t_in, t_out, elapsed)
        else:
            # Final answer reached
            trace.add_step(TraceStep(
                step_type="final_thought",
                content=thought_text,
                tokens_in=t_in,
                tokens_out=t_out,
                elapsed_ms=elapsed,
            ))
            trace.final_answer = thought_text
            print(f" [Turn {turn}] Final Answer found.")
            break

    print(f"\n[ReAct Final]\n{trace.final_answer}")
    return trace

# -------------------------------------------------------------------------
# Strategy 5: Chain-of-Debate (CoD)
# -------------------------------------------------------------------------

def cod_reason(question: str, rounds: int = 2) -> ReasoningTrace:
    """Chain-of-Debate: multi-role debate to reach a consensus."""
    trace = ReasoningTrace(strategy=StrategyType.COD, question=question)

    print(f"\n{'-'*60}")
    print(f"[CoD] Chain-of-Debate Reasoning ({rounds} rounds)")
    print(f"{'-'*60}")

    roles = [
        {"name": "正方专家", "system": "你支持一种激进且前瞻性的观点，寻找支持证据。"},
        {"name": "反方专家", "system": "你持怀疑态度，寻找逻辑漏洞和潜在风险。"},
        {"name": "中立协调者", "system": "你负责总结双方观点，寻找共同点和最客观的真相。"},
    ]

    debate_history = f"问题：{question}\n\n"

    for r in range(1, rounds + 1):
        print(f"\n Debate Round {r}:")
        for role in roles:
            prompt = f"当前辩论历史：\n{debate_history}\n\n请作为【{role['name']}】发表你的观点。"
            text, t_in, t_out, elapsed = _call_llm(role["system"], [{"role": "user", "content": prompt}], label=f"cod/{role['name']}/R{r}")

            entry = f"【{role['name']}】: {text}\n"
            debate_history += entry
            trace.add_step(TraceStep(step_type="debate_turn", content=entry, tokens_in=t_in, tokens_out=t_out, elapsed_ms=elapsed))
            print(f" [{role['name']}] 发表了观点。")

    # Final Synthesis
    synth_system = "你是一名资深法官。根据以上的辩论记录，给出最终的、最公正的判定结果。"
    final_text, t_in, t_out, elapsed = _call_llm(synth_system, [{"role": "user", "content": debate_history}], label="cod/judgment")

    trace.add_step(TraceStep(step_type="judgment", content=final_text, tokens_in=t_in, tokens_out=t_out, elapsed_ms=elapsed))
    trace.final_answer = final_text
    print(f"\n[CoD Final Judgment]\n{final_text[:500]}...")
    return trace

# -------------------------------------------------------------------------
# Execution & Comparison
# -------------------------------------------------------------------------

QUESTIONS = [
    {
        "text": "如果全球气温上升2度，对全球经济的长期影响是什么？请从农业、劳动力效率和基础设施三个维度分析。",
        "strategies": [StrategyType.COT, StrategyType.SELF_CORRECT]
    },
    {
        "text": "计算：(sqrt(144) * 5 + pow(2, 10)) / 10.24 是多少？先搜索量子计算的关键事实，然后告诉我它的一个应用。",
        "strategies": [StrategyType.REACT]
    },
    {
        "text": "人类是否应该全面禁止强人工智能的开发？",
        "strategies": [StrategyType.COD]
    }
]

if __name__ == "__main__":
    results = []

    for q_data in QUESTIONS:
        q_text = q_data["text"]
        for strategy in q_data["strategies"]:
            if strategy == StrategyType.COT:
                res = cot_reason(q_text)
            elif strategy == StrategyType.TOT:
                res = tot_reason(q_text)
            elif strategy == StrategyType.SELF_CORRECT:
                res = self_correct(q_text)
            elif strategy == StrategyType.REACT:
                res = react_reason(q_text)
            elif strategy == StrategyType.COD:
                res = cod_reason(q_text)

            results.append(res)

    print(f"\n{'='*60}")
    print(f"{'REASONING STRATEGY COMPARISON':^60}")
    print(f"{'='*60}")
    for r in results:
        print(r.summary())
    print(f"{'='*60}")