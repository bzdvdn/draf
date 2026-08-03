"""Supervisor graph builder — the minimal ``Flow.route()`` pattern.

This is the smallest example of the supervisor loop: one decider node
(:class:`draf.node.Supervisor`, added with :meth:`draf.flow.Flow.supervisor`)
routes to one of two agents, each wrapped as a ``SubFlow`` via the
framework's :func:`draf.flow.agent_step` (context builder -> ReAct harness
-> append the reply to the shared ``messages``).  When the decider says
``finish`` the loop exits.  Read this file top to bottom to see the whole
pattern.
"""

from __future__ import annotations

from draf.flow import Flow, agent_step
from src.graphs.prompts import CODER_PROMPT, SUPERVISOR_PROMPT, TALK_PROMPT

MODEL_DEFAULT = "llama3.1:8b"


def build_flow(model: str = MODEL_DEFAULT, *, provider: str = "ollama"):
    """Assemble the router: supervisor -> ``route()`` over two agents.

    Flow::

        supervisor ─ next_agent=coder ──► ContextBuilder → ReAct ──┐
            ▲                                                       │
            └────────────────────── supervisor ◀───────────────────┘
        (next_agent=talk)  ...  (next_agent=finish → exits)

    *provider* is threaded into every agent's harness config (per-node) so
    the graph never touches the framework's global defaults.
    """
    flow = Flow("simple_router")
    flow.supervisor(
        system=SUPERVISOR_PROMPT,
        model=model,
        provider=provider,
        route_keys={"coder": "code", "talk": "talk"},
        done_keys={"code", "talk"},
        done_mode="any",
        fallback_agent="talk",
    )
    flow.route(
        "next_agent",
        coder=agent_step(CODER_PROMPT, "code", model=model, provider=provider),
        talk=agent_step(TALK_PROMPT, "talk", model=model, provider=provider),
    )
    return flow
