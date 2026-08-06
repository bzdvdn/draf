"""Graph state shared across the repair workflow.

``messages`` accumulates the whole conversation (append reducer) and is the
single source of conversation history for the chat API.  ``project_info``
holds the running extraction (merge reducer: ``None`` updates are skipped).

The coordinator's internal conversation lives in ``_coordinator_messages``
(a private, reducer-append slot recomputed each turn) so the tool-call
round-trips never pollute the user-visible ``messages``; only the final
answer reaches ``messages`` via ``AppendAssistant``.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from draf.state import reducers_from_typeddict
from src.domain.models import ProjectInfo, merge_project_info


def add_messages(current: list | None, new: list | None) -> list:
    """Reducer: append the new messages to the conversation."""
    return list(current or []) + list(new or [])


class RepairState(TypedDict):
    """Per-turn state carried through the graph and persisted to the checkpointer."""

    messages: Annotated[list, add_messages]
    project_info: Annotated[ProjectInfo, merge_project_info]

    plan: str
    estimate: str
    material_findings: str
    qa_feedback: str

    input: str
    output: str
    _coordinator_messages: Annotated[list, add_messages]


STATE_REDUCERS: dict[str, Any] = reducers_from_typeddict(RepairState)


def initial_state() -> RepairState:
    """Fresh state for a brand-new chat session."""
    return {
        "messages": [],
        "project_info": {},
        "plan": "",
        "estimate": "",
        "material_findings": "",
        "qa_feedback": "",
        "input": "",
        "output": "",
        "_coordinator_messages": [],
    }
