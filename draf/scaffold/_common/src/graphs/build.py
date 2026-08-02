"""Supervisor graph builder — the minimal ``draf`` pattern.

This is the skeleton every real app in this repo is built from.  Two core
building blocks:

1. ``Flow.route("next_agent", ...)`` — a supervisor loop: the decider node
   (:class:`~src.nodes.supervisor.Supervisor`) reads the latest user message
   and picks an agent; that agent runs, control returns to the supervisor,
   which decides again — until it says ``finish`` and the loop exits.

2. :func:`draf.flow.agent_step` — the framework helper wrapping one agent
   as a ``SubFlow``: context builder -> ReAct harness (LLM + tools) ->
   append the reply to the shared ``messages`` conversation.

HOW TO EXTEND
    * Add an agent: write a prompt in ``graphs/prompts.py`` and register it
      here under a new keyword in ``route()``; then mention the new route
      value in the supervisor prompt.
    * Give an agent tools: pass ``use_tools=["tool_name", ...]`` (an explicit
      allowlist; ``"all"`` enables everything) and build the ``Tool``
      subclasses in ``tools/``.
    * Add state: extend ``AppState`` in ``graphs/state.py``.
"""

from __future__ import annotations

from draf.flow import Flow, agent_step

from src.graphs.prompts import PLANNER_PROMPT, REVIEWER_PROMPT, WRITER_PROMPT
from src.nodes.supervisor import Supervisor
from src.tools import build_tools

MODEL_DEFAULT = "llama3.1:8b"

#: Shared state keys rendered into each agent's context.
AGENT_SECTIONS = {
    "plan": "Plan",
    "draft": "Draft",
    "review": "Review",
}


def build_flow(model: str = MODEL_DEFAULT, *, provider: str = "ollama"):
    """Assemble the supervisor graph over the routed agents.

    Flow::

        supervisor ─ next_agent=planner ──► ContextBuilder → ReAct ─┐
            ▲                                                       │
            └─────────────────────── supervisor ◀───────────────────┘
        (next_agent=writer / reviewer) ...  (next_agent=finish → exits)

    *provider* is threaded into every agent's harness config (per-node) so
    the graph never touches the framework's global defaults.
    """
    tools = build_tools()

    flow = Flow("{{project_slug}}")
    flow.step(
        Supervisor(
            model=model,
            provider=provider,
            sections=AGENT_SECTIONS,
            route_keys={"planner": "plan", "writer": "draft", "reviewer": "review"},
            done_keys={"plan", "draft", "review"},
            done_mode="all",
            fallback_agent="planner",
        )
    )
    flow.route(
        "next_agent",
        planner=agent_step(
            PLANNER_PROMPT,
            "plan",
            model=model,
            provider=provider,
            sections=AGENT_SECTIONS,
        ),
        writer=agent_step(
            WRITER_PROMPT,
            "draft",
            use_tools=["current_date"],
            model=model,
            provider=provider,
            sections=AGENT_SECTIONS,
        ),
        reviewer=agent_step(
            REVIEWER_PROMPT,
            "review",
            model=model,
            provider=provider,
            sections=AGENT_SECTIONS,
        ),
    )
    return flow, tools
