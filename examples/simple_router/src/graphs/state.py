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

from draf.state import reducers_from_typeddict


def add_messages(current: list | None, new: list | None) -> list:
    """Reducer: append new messages to the conversation."""
    return list(current or []) + list(new or [])


class RouterState(TypedDict):
    """State carried through the graph.

    ``code`` / ``talk`` are the output slots of the two routed agents;
    ``input`` is the plain-text context each agent receives.
    ``supervisor_rounds`` is the supervisor's loop budget (resets each turn).
    """

    messages: Annotated[list, add_messages]

    next_agent: str
    input: str
    code: str
    talk: str
    supervisor_rounds: int


STATE_REDUCERS: dict[str, Any] = reducers_from_typeddict(RouterState)


def initial_state() -> RouterState:
    """Fresh state for a brand-new session."""
    return {
        "messages": [],
        "next_agent": "",
        "input": "",
        "code": "",
        "talk": "",
        "supervisor_rounds": 0,
    }
