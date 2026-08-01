"""Supervisor graph builder — the minimal ``draf`` pattern.

This is the skeleton every real app in this repo is built from.  Two core
building blocks:

1. ``Flow.route("next_agent", ...)`` — a supervisor loop: the decider node
   (:class:`~src.nodes.supervisor.Supervisor`) reads the latest user message
   and picks an agent; that agent runs, control returns to the supervisor,
   which decides again — until it says ``finish`` and the loop exits.

2. :func:`agent_chain` — a helper wrapping one agent as a ``SubFlow``:
   context builder -> ReAct harness (LLM + tools) -> append the reply to the
   shared ``messages`` conversation.

HOW TO EXTEND
    * Add an agent: write a prompt in ``graphs/prompts.py`` and register it
      here under a new keyword in ``route()``; then mention the new route
      value in the supervisor prompt.
    * Give an agent tools: pass ``use_tools=True`` (all tools) or a list of
      tool names, and build the ``Tool`` subclasses in ``tools/``.
    * Add state: extend ``AppState`` in ``graphs/state.py``.
"""

from __future__ import annotations

from draf.flow import Flow, SubFlow

from src.graphs.prompts import (
    PLANNER_PROMPT,
    REVIEWER_PROMPT,
    SUPERVISOR_PROMPT,
    WRITER_PROMPT,
)
from src.nodes.context import AppendAssistant, ContextBuilder
from src.nodes.supervisor import Supervisor
from src.tools import build_tools

MODEL_DEFAULT = "llama3.1:8b"

#: Shared state keys rendered into each agent's context.
AGENT_SECTIONS = {
    "plan": "Plan",
    "draft": "Draft",
    "review": "Review",
}


def agent_chain(
    system: str, output_key: str, use_tools, *, model: str, provider: str
) -> SubFlow:
    """One routed agent: context -> ReAct harness -> append to conversation.

    The agent's scratch conversation lives in a private ``_<key>_messages``
    state slot (reset by the context builder); only the final reply is
    appended to the shared ``messages`` conversation.  ``stream`` makes
    tokens flow as stream events, so a CLI/SSE client can render the answer
    live.
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
    flow.step(Supervisor(model=model, provider=provider))
    flow.route(
        "next_agent",
        planner=agent_chain(
            PLANNER_PROMPT, "plan", use_tools=False, model=model, provider=provider
        ),
        writer=agent_chain(
            WRITER_PROMPT, "draft", use_tools=True, model=model, provider=provider
        ),
        reviewer=agent_chain(
            REVIEWER_PROMPT, "review", use_tools=False, model=model, provider=provider
        ),
    )
    return flow, tools
