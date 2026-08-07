"""Agentic graph builder — one coordinator agent drives the whole pipeline.

The graph is a single top-level ReAct agent (the coordinator) that
orchestrates the entire process through *sub-agent tools* instead of a
supervisor/router topology::

    ContextBuilder ─► coordinator/agent ⇄ coordinator/tool ─► AppendAssistant

The coordinator sees six tools — ``extract_project_info``, ``propose_plan``,
``select_materials``, ``prepare_estimate``, ``run_qa_check`` and
``ask_human``.  Each sub-agent
tool runs its own internal ReAct loop against the domain tools
(room/material/budget/rag) and writes its result back into the shared state
via the ``__state__`` runtime kwarg.  The ``ask_human`` tool pauses the run
as an interrupt; the operator's answer arrives back as the tool result, so
approval handling (re-plan, re-estimate, clarify) is the coordinator's to
decide — there are no hardcoded Ask-loops in the graph.
"""

from __future__ import annotations

from src.core.deps import build_deps
from src.graphs.prompts import COORDINATOR_PROMPT
from src.tools import COORDINATOR_TOOLS, build_tools
from teff.flow import Flow
from teff.node import AppendAssistant, ContextBuilder
from teff.provider import ProviderRegistry

MODEL_DEFAULT = "llama3.1:8b"

#: Shared state keys rendered into the coordinator's context each turn.
AGENT_SECTIONS = {
    "project_info": "Проект",
    "plan": "План",
    "estimate": "Смета",
    "material_findings": "Материалы",
    "qa_feedback": "Замечания QA",
}


def build_flow(
    model: str = MODEL_DEFAULT,
    *,
    provider: str = "ollama",
    services=None,
    catalog=None,
):
    """Assemble the agentic repair graph.

    ``ContextBuilder`` composes the coordinator's ``input`` from the shared
    sections + latest user message and resets its private ``input`` /
    ``output`` / ``_coordinator_messages`` scratch each turn.  The coordinator
    is a :class:`~teff.node.agent.ReActAgent` loop (``Flow.react``) whose
    conversation lives in ``_coordinator_messages``; only the final reply is
    copied into the shared ``messages`` by ``AppendAssistant``, so
    ``last_reply`` / the API return the clean answer.

    ``ask_human`` approvals pause the run as ordinary interrupts — the API
    surfaces ``prompt``/``waiting`` and the operator's answer resumes in the
    same session.
    """
    services = services or build_deps(provider=provider)[0]
    catalog = catalog if catalog is not None else build_deps(provider=provider)[1]
    tools = build_tools(services, catalog, model=model, provider=provider)

    flow = (
        Flow("repair-ai", providers=ProviderRegistry.from_presets(provider))
        .step(
            ContextBuilder(
                sections=AGENT_SECTIONS,
                reset_keys=("input", "output", "_coordinator_messages"),
            ),
            id="context",
        )
        .react(
            system=COORDINATOR_PROMPT,
            model=model,
            provider=provider,
            input_key="input",
            output_key="output",
            messages_key="_coordinator_messages",
            use_tools=COORDINATOR_TOOLS,
            max_tool_rounds=None,
            id="coordinator",
        )
        .step(AppendAssistant(output_key="output"), id="append")
    )
    return flow, tools
