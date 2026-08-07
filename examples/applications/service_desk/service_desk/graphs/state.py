"""Typed graph state for the service-desk example.

``messages`` accumulates the whole conversation (append reducer).  Each
specialist writes into its own output slot (``billing`` / ``incident`` /
``deploy`` / ``fallback``); the supervisor's ``done_keys`` guard watches
those slots so a turn finishes deterministically once one expert answers.
``deploy_approved`` receives the operator's answer on the deploy gateway. The
per-turn scratch (``input`` / ``final`` / the specialist slots / the router
counter) is reset by the entry ``ContextBuilder``, so a follow-up message
routes to a fresh specialist without duplicating the previous answer.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from teff.state import reducers_from_typeddict


def add_messages(current: list | None, new: list | None) -> list:
    """Reducer: append new messages to the conversation."""
    return list(current or []) + list(new or [])


class ServiceDeskState(TypedDict):
    """State carried through the service-desk graph."""

    messages: Annotated[list, add_messages]

    next_agent: str
    supervisor_rounds: int

    billing: str
    incident: str
    deploy: str
    fallback: str
    deploy_approved: str

    input: str
    final: str


STATE_REDUCERS: dict[str, Any] = reducers_from_typeddict(ServiceDeskState)


def initial_state() -> ServiceDeskState:
    """Fresh state for a brand-new support session."""
    return {
        "messages": [],
        "next_agent": "",
        "supervisor_rounds": 0,
        "billing": "",
        "incident": "",
        "deploy": "",
        "fallback": "",
        "deploy_approved": "",
        "input": "",
        "final": "",
    }
