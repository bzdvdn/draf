"""Typed graph state with per-key reducers."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from teff.state import reducers_from_typeddict


def add_messages(current: list | None, new: list | None) -> list:
    """Reducer: append new items to a shared list."""
    return list(current or []) + list(new or [])


class TriageState(TypedDict):
    """State carried through the support-triage graph.

    ``billing`` / ``tech`` are the output slots of the two auto-answer
    specialists; ``human_note`` is what a human-escalation handler drafted
    before pausing on the ``escalate`` interrupt.  ``escalate`` holds the
    operator's answer after resume.
    """

    ticket: str
    messages: Annotated[list, add_messages]
    next_agent: str
    input: str
    billing: str
    tech: str
    human_note: str
    escalate: str
    supervisor_rounds: int


STATE_REDUCERS: dict[str, Any] = reducers_from_typeddict(TriageState)


def initial_state() -> TriageState:
    """Fresh state for a brand-new session."""
    return {
        "ticket": "",
        "messages": [],
        "next_agent": "",
        "input": "",
        "billing": "",
        "tech": "",
        "human_note": "",
        "escalate": "",
        "supervisor_rounds": 0,
    }
