"""Supervisor graph builder — ``Flow.route()`` over the five repair agents."""

from __future__ import annotations

from draf.flow import Flow, SubFlow

from src.core.deps import build_deps
from src.graphs.prompts import (
    DIRECT_PROMPT,
    ESTIMATOR_PROMPT,
    MATERIALS_PROMPT,
    PLANNER_PROMPT,
    QA_PROMPT,
)
from src.nodes.context import AppendAssistant, ContextBuilder
from src.nodes.extractor import Extractor
from src.nodes.supervisor import Supervisor
from src.tools import build_tools

MODEL_DEFAULT = "llama3.1:8b"

#: Shared state keys rendered into each agent's context.
AGENT_SECTIONS = {
    "project_info": "Проект",
    "plan": "План",
    "estimate": "Смета",
    "material_findings": "Материалы",
}

_ROOM_TOOLS = [
    "calculate_wall_area",
    "calculate_floor_area",
    "calculate_ceiling_area",
    "calculate_perimeter",
]
_MATERIAL_TOOLS = [
    "calculate_tiles",
    "calculate_paint",
    "calculate_laminate",
    "calculate_plaster",
    "calculate_putty",
]
_BUDGET_TOOLS = ["estimate_material_cost", "estimate_total"]
_RAG_TOOLS = ["search_materials", "find_similar_material"]


def agent_chain(
    system: str, output_key: str, use_tools, *, model: str, provider: str
) -> SubFlow:
    """One routed agent: context -> ReAct harness -> append to conversation.

    Wrapped as a ``SubFlow`` so ``route()`` sees it as a single node while
    the nested graph keeps its own supervisor-free tool loop.

    The agent's scratch conversation lives in a private ``_<key>_messages``
    state slot (reset by the context builder); only the final assistant
    reply is appended to the shared ``messages`` conversation.  ``stream``
    makes tokens flow as stream events, so a CLI/SSE client can render the
    answer live.
    """
    scratch_key = f"_{output_key}_messages"
    inner = Flow(f"agent-{output_key}")
    inner.step(
        ContextBuilder(
            sections=AGENT_SECTIONS,
            reset_keys=(output_key, "input", scratch_key),
        )
    )
    inner.harness(
        model=model,
        system=system,
        input_key="input",
        output_key=output_key,
        messages_key=scratch_key,
        use_tools=use_tools,
        provider=provider,
        stream=True,
    )
    inner.step(AppendAssistant(output_key=output_key))
    return SubFlow(inner.compile())


def build_flow(
    model: str = MODEL_DEFAULT,
    *,
    provider: str = "ollama",
    services=None,
    catalog=None,
):
    """Assemble the supervisor graph: ``route()`` over the five agents.

    Flow::

        supervisor ─ next_agent=planner ──► ContextBuilder → ReAct ──┐
            ▲                                                         │
            └────────────────────────── supervisor ◀──────────────────┘
        supervisor ─ next_agent=finish ──► Extractor (project_info)

    *provider* is threaded into every agent's harness config (per-node) so
    the graph never touches the framework's global defaults.
    """
    services = services or build_deps(provider=provider)[0]
    catalog = catalog if catalog is not None else build_deps(provider=provider)[1]
    tools = build_tools(services, catalog)

    flow = Flow("production-repair-ai")
    flow.step(Supervisor(model=model, provider=provider))
    flow.route(
        "next_agent",
        finish=Extractor(model=model, provider=provider),
        direct=agent_chain(
            DIRECT_PROMPT, "direct_reply", use_tools=False, model=model, provider=provider
        ),
        planner=agent_chain(
            PLANNER_PROMPT, "plan", use_tools=_ROOM_TOOLS, model=model, provider=provider
        ),
        estimator=agent_chain(
            ESTIMATOR_PROMPT,
            "estimate",
            use_tools=[*_ROOM_TOOLS, *_MATERIAL_TOOLS, *_BUDGET_TOOLS],
            model=model,
            provider=provider,
        ),
        materials=agent_chain(
            MATERIALS_PROMPT,
            "material_findings",
            use_tools=_RAG_TOOLS,
            model=model,
            provider=provider,
        ),
        qa=agent_chain(
            QA_PROMPT, "qa_feedback", use_tools=False, model=model, provider=provider
        ),
    )
    return flow, tools
