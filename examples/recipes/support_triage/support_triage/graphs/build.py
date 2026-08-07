"""Supervisor graph builder for support triage.

A ``Flow.supervisor`` router picks one of two auto-answer specialists
(billing / tech) or hands the ticket to a human.  Auto-answers terminate
the loop when their output slot is filled (``done_keys``).  The human
escalation is a sub-flow that **pauses on an ``Interrupt``** so an operator
can answer later and the run resumes.

Flow::

    supervisor ─ next_agent=billing ──► agent_step(billing) ──┐
       ▲                                                     │
       ├ next_agent=tech ──► agent_step(tech) ───────────────┤
       ├ next_agent=human ──► (draft human note → Interrupt) ─┘
       └──────────── supervisor ◀─────────────────────────────┘
       (done_keys filled → finish, bounded by supervisor_rounds)
"""

from __future__ import annotations

from teff.flow import Flow, agent_step
from teff.node import LLM, Interrupt
from teff.node.context import AppendAssistant
from teff.provider import ProviderRegistry

from .prompts import BILLING_PROMPT, HUMAN_PROMPT, SUPERVISOR_PROMPT, TECH_PROMPT

MODEL_DEFAULT = "llama3.1:8b"
PROVIDER_DEFAULT = "ollama"


def human_chain(*, model: str, provider: str) -> list:
    """Top-level chain for a human escalation: draft note → pause → append.

    Kept as flat nodes (not a ``SubFlow``) so the ``Interrupt`` lives in the
    outer graph: a mid-``SubFlow`` interrupt cannot resume cleanly because
    the sub-flow re-enters from its entry on resume.  At the top level the
    runtime resumes right after the ``escalate`` pause.
    """
    return [
        LLM(
            model=model,
            provider=provider,
            system=HUMAN_PROMPT,
            prompt="Составь заметку для оператора по тикету: {ticket}",
            output_key="human_note",
        ),
        Interrupt(key="escalate", prompt="Эскалация: {human_note}"),
        AppendAssistant(output_key="human_note", messages_key="messages"),
    ]


def build_flow(
    model: str = MODEL_DEFAULT,
    *,
    provider: str = PROVIDER_DEFAULT,
) -> Flow:
    """Assemble the support-triage supervisor and rerouting subflows."""
    flow = Flow("support_triage", providers=ProviderRegistry.from_presets(provider))
    flow.supervisor(
        system=SUPERVISOR_PROMPT,
        model=model,
        provider=provider,
        route_keys={"billing": "billing", "tech": "tech", "human": "human"},
        done_keys={"billing", "tech", "human_note"},
        done_mode="any",
        fallback_agent="billing",
    )
    flow.route(
        "next_agent",
        billing=agent_step(BILLING_PROMPT, "billing", model=model, provider=provider),
        tech=agent_step(TECH_PROMPT, "tech", model=model, provider=provider),
        human=human_chain(model=model, provider=provider),
    )
    return flow
