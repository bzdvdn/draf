"""Supervisor graph builder — ``Flow.route()`` over the five repair agents."""

from __future__ import annotations

from draf.flow import Flow, agent_step

from src.core.deps import build_deps
from src.graphs.prompts import (
    DIRECT_PROMPT,
    ESTIMATOR_PROMPT,
    MATERIALS_PROMPT,
    PLANNER_PROMPT,
    QA_PROMPT,
)
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
    flow.step(
        Supervisor(
            model=model,
            provider=provider,
            sections=AGENT_SECTIONS,
            route_keys={
                "direct": "direct_reply",
                "planner": "plan",
                "estimator": "estimate",
                "materials": "material_findings",
                "qa": "qa_feedback",
            },
            done_keys={
                "direct_reply",
                "plan",
                "estimate",
                "material_findings",
                "qa_feedback",
            },
            done_mode="any",
            fallback_agent="direct",
        )
    )
    flow.route(
        "next_agent",
        finish=Extractor(model=model, provider=provider),
        direct=agent_step(
            DIRECT_PROMPT,
            "direct_reply",
            model=model,
            provider=provider,
            sections=AGENT_SECTIONS,
        ),
        planner=agent_step(
            PLANNER_PROMPT,
            "plan",
            use_tools=_ROOM_TOOLS,
            model=model,
            provider=provider,
            sections=AGENT_SECTIONS,
        ),
        estimator=agent_step(
            ESTIMATOR_PROMPT,
            "estimate",
            use_tools=[*_ROOM_TOOLS, *_MATERIAL_TOOLS, *_BUDGET_TOOLS],
            model=model,
            provider=provider,
            sections=AGENT_SECTIONS,
        ),
        materials=agent_step(
            MATERIALS_PROMPT,
            "material_findings",
            use_tools=_RAG_TOOLS,
            model=model,
            provider=provider,
            sections=AGENT_SECTIONS,
        ),
        qa=agent_step(
            QA_PROMPT,
            "qa_feedback",
            model=model,
            provider=provider,
            sections=AGENT_SECTIONS,
        ),
    )
    return flow, tools
