"""Supervisor graph builder — the minimal ``Flow.route()`` pattern.

This is the smallest example of the supervisor loop: one decider node
(:class:`~src.nodes.supervisor.Supervisor`) routes to one of two agents,
each wrapped as a ``SubFlow`` (context builder -> ReAct harness -> append
the reply to the shared ``messages``).  When the decider says ``finish``
the loop exits.  Read this file top to bottom to see the whole pattern.
"""

from __future__ import annotations

from draf.flow import Flow, SubFlow

from src.graphs.prompts import CODER_PROMPT, SUPERVISOR_PROMPT, TALK_PROMPT
from src.nodes.context import AppendAssistant, ContextBuilder
from src.nodes.supervisor import Supervisor

MODEL_DEFAULT = "llama3.1:8b"


def agent_chain(system: str, output_key: str, *, model: str, provider: str) -> SubFlow:
    """One routed agent: context -> ReAct harness -> append to conversation.

    Wrapped as a ``SubFlow`` so ``route()`` sees it as a single node while
    the nested graph keeps its own private scratch conversation (reset by
    the context builder); only the final assistant reply is appended to
    the shared ``messages``.  ``stream`` makes tokens flow as stream events,
    so a CLI/SSE client can render the answer live.
    """
    scratch_key = f"_{output_key}_messages"
    inner = Flow(f"agent-{output_key}")
    inner.step(
        ContextBuilder(
            sections={output_key: output_key.capitalize()},
            reset_keys=(output_key, "input", scratch_key),
        )
    )
    inner.harness(
        model=model,
        system=system,
        input_key="input",
        output_key=output_key,
        messages_key=scratch_key,
        provider=provider,
        stream=True,
    )
    inner.step(AppendAssistant(output_key=output_key))
    return SubFlow(inner.compile())


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
    flow.step(Supervisor(model=model, provider=provider))
    flow.route(
        "next_agent",
        coder=agent_chain(CODER_PROMPT, "code", model=model, provider=provider),
        talk=agent_chain(TALK_PROMPT, "talk", model=model, provider=provider),
    )
    return flow
