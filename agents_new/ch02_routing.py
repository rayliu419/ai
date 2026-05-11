#!/usr/bin/env python3
#!/usr/bin/env python3
"""
ch02_routing.py - Routing with raw Anthropic API

Routing introduces conditional branching into agentic workflows. Instead of a fixed linear chain, an LLM first classifies the user's intent, then the system dispatches the request to a specialized handler (prompt / function).

input -> [LLM Router] --(category A)--> handler_A(input) -> output
                    --(category B)--> handler_B(input) -> output
                    --(default)-----> fallback(input)  -> output

Key ideas from the book (Chapter 2):
- Linear chains can't adapt to diverse inputs; routing adds decision logic
- The LLM acts as a classifier: it outputs a category label, not a final answer
- Each route has its own specialized handler (prompt, function, or sub-agent)
- Routing can be LLM-based, rule-based, embedding-based, or ML-model-based

--------------------------------------------------------------------------
PRODUCTION INSIGHTS (from Claude Code source analysis):

Claude Code is ROUTING-HEAVY - it implements 5+ distinct routing subsystems, each combining LLM classification with rule-based dispatch. This is by far the most pervasive pattern in production (much more than simple prompt chaining).

THE MAJOR ROUTING SYSTEMS:

/------------------------------------------------------------------------\
| 1. AUTO-MODE CLASSIFIER (Permission Routing)                           |
|    src/utils/permissions/yoloClassifier.ts                             |
|                                                                        |
| This is the CLOSEST match to the book's routing pattern:               |
|                                                                        |
|   tool_call -> [Allowlist check] --(safe tool)--> ALLOW immediately    |
|                      |                                                 |
|                      +---(needs check)--> [LLM Classifier]             |
|                                             |                          |
|                                     /-------+-------\                  |
|                                     v               v                  |
|                                   ALLOW           DENY           ASK   |
|                                                                        |
| Two-stage XML classifier (when enabled):                               |
|   Stage 1 ("fast"): 64 max tokens -> quick allow/deny decision        |
|   Stage 2 ("thinking"): 4096 tokens -> reasoning-based if stage 1 unsure|
|                                                                        |
| The LLM is given: tool name, arguments, conversation history,          |
| user-configured allow/deny rules, then outputs a structured decision.  |
\------------------------------------------------------------------------/

/------------------------------------------------------------------------\
| 2. AGENT TYPE ROUTING (AgentTool)                                      |
|    src/tools/AgentTool/AgentTool.tsx                                   |
|                                                                        |
| spawn_agent(subagent_type, prompt)                                     |
|      |                                                                 |
|      +---(type=undefined, fork enabled)--> FORK_AGENT                  |
|      |    (reuse parent system prompt, full tool pool)                 |
|      |                                                                 |
|      +---(type="Explore")--> Explore agent (read-only, fast)           |
|      +---(type="Plan")--> Plan agent (no edit tools)                   |
|      +---(type="general-purpose")--> General agent (full tools)        |
|      +---(type=custom)--> User-defined agent (.md file)                |
|                                                                        |
| Each agent type defines: tools, system prompt, model, permission mode. |
| The model decides which subagent_type to use - this IS LLM routing.    |
\------------------------------------------------------------------------/

/------------------------------------------------------------------------\
| 3. SKILL DISPATCH ROUTING (SkillTool)                                  |
|    src/tools/SkillTool/SkillTool.ts                                    |
|                                                                        |
| /skill_name args                                                       |
|      |                                                                 |
|      +---(context="fork")--> Spawn sub-agent with skill prompt         |
|      +---(context="inline")--> Inject into current conversation        |
|      +---(remote canonical)--> Load from cloud + execute               |
|                                                                        |
| Permission routing within skills:                                      |
|   safe_properties only -> auto-allow (no prompt to user)               |
|   unsafe properties   -> ask user for confirmation                     |
\------------------------------------------------------------------------/

/------------------------------------------------------------------------\
| 4. COMMAND ROUTING (Slash Commands)                                    |
|    src/commands.ts + src/utils/processUserInput/processSlashCommand.tsx|
|                                                                        |
| "/commit -m fix" -> [Registry lookup] -> commit handler                |
| "/help"          -> [Registry lookup] -> help handler                  |
| "/clear"         -> [Registry lookup] -> clear handler                 |
|                                                                        |
| This is RULE-BASED routing (no LLM):                                   |
| - Static registry of command name -> handler function                  |
| - Feature-gated commands (only registered if flag is on)               |
| - Alias support (multiple names -> same handler)                       |
| - Frontmatter parsing (model override, effort, allowedTools)           |
\------------------------------------------------------------------------/

/------------------------------------------------------------------------\
| 5. PERMISSION DECISION ROUTING (Three-Path System)                     |
|    src/hooks/useCanUseTool.tsx                                         |
|                                                                        |
| tool_call -> [hasPermissionsToUseTool()]                               |
|      |             |              |                                    |
|      v             v              v                                    |
|    ALLOW         DENY            ASK                                   |
|  (execute)     (reject)           |                                    |
|                                   +--> [coordinator handler]           |
|                                   +--> [swarm worker handler]          |
|                                   +--> [speculative bash classifier]   |
|                                   +--> [interactive UI prompt]         |
|                                                                        |
| Multi-layer decision: CLI args -> session rules -> user rules -> classify|
\------------------------------------------------------------------------/

/------------------------------------------------------------------------\
| 6. MODEL ROUTING (Implicit)                                            |
|    src/utils/model/model.ts + agent.ts                                 |
|                                                                        |
| Different contexts route to different models:                          |
| - Agent param override -> specified model                              |
| - Agent definition -> agent's model                                    |
| - Parent session -> inherited model                                    |
| - Classifier -> separate model (may be smaller/cheaper)                |
| - Auto-mode config -> GrowthBook-configured model                      |
|                                                                        |
| This is RESOURCE-AWARE routing: expensive models for main work,        |
| cheap models for classification (exactly what ch16 of the book covers).|
\------------------------------------------------------------------------/

KEY ARCHITECTURAL INSIGHTS:

1. LAYERED ROUTING: Production routing isn't a single LLM call - it's multiple routing stages in sequence. A tool call may pass through: allowlist -> permission rules -> LLM classifier -> UI prompt. Each layer filters traffic so the expensive LLM classifier only handles edge cases.

2. LLM ROUTING FOR CLASSIFICATION ONLY: The production system uses LLM as classifier (exactly like the book), but only for AMBIGUOUS cases. Clear cases (known-safe tools, exact-match rules) use deterministic routing to avoid latency and cost.

3. THREE UNIVERSAL OUTCOMES: All routing in Claude Code converges to three behaviors: ALLOW, DENY, or ASK. This is simpler than arbitrary route handlers - it's a permission-focused routing pattern.

4. ROUTING IS THE MODEL'S JOB TOO: The model itself does routing when it selects which tool to call or which subagent_type to spawn. The system prompt's "Using your tools" section acts as the routing table.

5. MULTI-MODAL ROUTING: Unlike the book's single-classifier pattern, Claude Code combines:
   - Rule-based (allowlist, exact match)
   - LLM-based (yoloClassifier, bash semantic matching)
   - Configuration-based (feature flags, GrowthBook)
   - User-interactive (permission prompts)

COMPARISON TABLE:
| Aspect            | Book ch02 (this file)    | Claude Code production       |
|-------------------|--------------------------|------------------------------|
| Classifier        | Single LLM call          | Multi-stage (rules + LLM)    |
| Routes            | N named handlers         | 3 outcomes (allow/deny/ask)  |
| Handler types     | LLM prompt or function   | Agent, skill, command, UI    |
| Fallback          | Default route            | "ask" -> interactive prompt  |
| Cost optimization | None                     | Allowlist skips classifier   |
| Classification    | LLM outputs label        | XML 2-stage with reasoning   |
| Route discovery   | Hardcoded in Router      | Registry + feature flags     |

Usage:
    python agents_new/ch02_routing.py
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Tuple

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
# Core: Route definition and Router
# -------------------------------------------------------------------------

@dataclass
class Route:
    """One possible route in the routing table.

    Attributes:
        name:     Route identifier (the label the LLM classifier outputs).
        system:   System prompt for the handler that processes this route.
        template: User prompt template for the handler. Use {input} for the
                  original user request and {route} for the matched route name.
        handler:  Optional Python function handler. If provided, it is called
                  instead of making an LLM call. Signature: (str) -> str.
    """
    name: str
    system: str = ""
    template: str = "{input}"
    handler: Callable[[str], str] | None = None

@dataclass
class Router:
    """A routing configuration.

    Attributes:
        name:                Human-readable name for this router.
        classifier_system:   System prompt for the LLM classifier.
        classifier_template: User prompt template for classification.
                             Must contain {input}. The LLM should output
                             exactly one of the route names.
        routes:              List of possible routes.
        default_route:       Name of the fallback route if classification fails.
    """
    name: str
    classifier_system: str
    classifier_template: str
    routes: List[Route]
    default_route: str = ""

@dataclass
class RoutingResult:
    """Result from a routing run."""
    input: str
    classified_route: str
    handler_output: str
    classification_usage: dict = field(default_factory=dict)
    handler_usage: dict = field(default_factory=dict)

def _call_llm(system: str, user_prompt: str, step_name: str) -> Tuple[str, dict]:
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

def route(router: Router, user_input: str) -> RoutingResult:
    """Execute the routing pattern: classify then dispatch.

    Steps:
        1. LLM classifies the user input into one of the route names.
        2. The matching route's handler (LLM prompt or Python function) processes
           the original input.
        3. Returns the combined result.
    """
    route_names = [r.name for r in router.routes]
    route_map = {r.name: r for r in router.routes}

    # --- Step 1: Classification ---
    print(f"\n{'='*60}")
    print(f"Router: {router.name}")
    print(f"{'='*60}")
    print(f"Input: {user_input}")
    print(f"Available routes: {route_names}")

    classifier_prompt = router.classifier_template.format(input=user_input)
    classification, cls_usage = _call_llm(
        router.classifier_system, classifier_prompt, step_name=f"{router.name}/classify"
    )

    classified = classification.strip().lower()

    # Match against known routes (fuzzy: check if any route name is contained)
    matched_route_name = router.default_route
    for rname in route_names:
        if rname.lower() == classified:
            matched_route_name = rname
            break
    else:
        # Fuzzy fallback: check if the classification contains a route name
        for rname in route_names:
            if rname.lower() in classified:
                matched_route_name = rname
                break

    logger.info("Classification: '%s' -> matched route: '%s'", classification, matched_route_name)

    # --- Step 2: Dispatch to handler ---
    matched = route_map.get(matched_route_name)
    if matched is None:
        matched = router.routes[0]
        matched_route_name = matched.name

    print(f"\n{'-'*60}")
    print(f"Dispatching to: {matched_route_name}")
    print(f"{'-'*60}")

    if matched.handler is not None:
        handler_output = matched.handler(user_input)
        handler_usage = {}
    else:
        handler_prompt = matched.template.format(
            input=user_input, route=matched_route_name
        )
        handler_output, handler_usage = _call_llm(
            matched.system, handler_prompt, step_name=f"{router.name}/{matched_route_name}"
        )

    preview = handler_output[:300] + ("..." if len(handler_output) > 300 else "")
    print(f"\nHandler output:\n{preview}")

    return RoutingResult(
        input=user_input,
        classified_route=matched_route_name,
        handler_output=handler_output,
        classification_usage=cls_usage,
        handler_usage=handler_usage,
    )

# -------------------------------------------------------------------------
# Demo: Customer service routing (booking / info / unclear)
# -------------------------------------------------------------------------

def _mock_booking(request: str) -> str:
    """Simulate a booking system call."""
    return (
        f"[booking system] 已收到预订请求：'{request}'。\n"
        f"模拟结果：预订确认号 BK-2024-{hash(request) % 10000:04d}。请等待确认邮件。"
    )

DEMO_ROUTER = Router(
    name="customer-service-router",
    classifier_system=(
        "你是一个客服请求分类器。分析用户请求，判断应由哪个处理器处理。\n"
        "只输出一个词：'booking'、'info' 或 'unclear'。\n"
        "不要输出任何其他内容。"
    ),
    classifier_template=(
        "请分类以下用户请求：\n\n"
        "- 若涉及预订机票、酒店、餐厅等，输出 'booking'\n"
        "- 若是一般信息查询（天气、常识、百科等），输出 'info'\n"
        "- 若请求不明确或无法归类，输出 'unclear'\n\n"
        "用户请求：{input}"
    ),
    routes=[
        Route(
            name="booking",
            handler=_mock_booking,  # Python function, no LLM call needed
        ),
        Route(
            name="info",
            system="你是一个知识渊博的信息助手。简洁准确地回答用户问题，控制在 200 字以内。",
            template="请回答以下问题：\n\n{input}",
        ),
        Route(
            name="unclear",
            system="你是一个友好的客服助手。用户的请求不够明确，请礼貌地要求用户补充信息。",
            template="用户说了：'{input}'。请礼貌地要求用户提供更多信息以便协助处理。",
        ),
    ],
    default_route="unclear",
)

DEMO_INPUTS = [
    "帮我预订下周五飞往东京的机票",
    "世界上最高的山峰是哪座？",
    "嗯……我想想",
]

if __name__ == "__main__":
    for user_input in DEMO_INPUTS:
        result = route(DEMO_ROUTER, user_input)
        print(f"\n{'*'*60}")
        print(f"Route: {result.classified_route}")
        print(f"Output:\n{result.handler_output}")
        print(f"{'*'*60}")