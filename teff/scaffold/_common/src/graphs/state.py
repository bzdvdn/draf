"""Typed graph state with per-key reducers.

HOW TO EXTEND
    Add a key here for every value your agents share.  When two or more
    nodes write the same key, give it an ``Annotated[..., reducer]`` type
    so ``reducers_from_typeddict`` knows how to combine updates (the
    ``messages`` list uses ``add_messages``; plain ``str`` keys just
    overwrite).
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from teff.state import reducers_from_typeddict


def add_messages(current: list | None, new: list | None) -> list:
    """Reducer: append new messages to the conversation."""
    return list(current or []) + list(new or [])


class AppState(TypedDict):
    """State carried through the graph and persisted to a checkpointer.

    ``plan`` / ``draft`` / ``review`` are the output slots of the three
    routed agents; ``input`` is the plain-text context each agent receives.
    ``supervisor_rounds`` is the supervisor's loop budget (resets each turn).
    """

    messages: Annotated[list, add_messages]

    next_agent: str
    input: str
    plan: str
    draft: str
    review: str
    supervisor_rounds: int


STATE_REDUCERS: dict[str, Any] = reducers_from_typeddict(AppState)


def initial_state() -> AppState:
    """Fresh state for a brand-new session."""
    return {
        "messages": [],
        "next_agent": "",
        "input": "",
        "plan": "",
        "draft": "",
        "review": "",
        "supervisor_rounds": 0,
    }
