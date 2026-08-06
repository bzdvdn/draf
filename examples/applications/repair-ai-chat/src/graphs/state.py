"""Graph state shared across the repair workflow.

``messages`` accumulates the whole conversation (append reducer) and is the
single source of conversation history for the chat API.  ``project_info``
holds the running extraction (merge reducer: ``None`` updates are skipped).
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from draf.state import reducers_from_typeddict
from src.domain.models import merge_project_info


def add_messages(current: list | None, new: list | None) -> list:
    """Reducer: append the new messages to the conversation."""
    return list(current or []) + list(new or [])


class RepairState(TypedDict):
    """Per-turn state carried through the graph and persisted to the checkpointer."""

    messages: Annotated[list, add_messages]
    project_info: Annotated[dict, merge_project_info]

    next_agent: str
    plan_approved: str
    plan_ok: str
    plan_verdict: dict
    plan_rounds: int
    plan: str
    estimate: str
    material_findings: str
    qa_feedback: str
    qa_verdict: dict
    qa_ok: str
    qa_rounds: int
    estim_approved: str
    est_ok: str
    est_verdict: dict
    est_rounds: int
    final_answer: str
    input: str
    supervisor_rounds: int


STATE_REDUCERS: dict[str, Any] = reducers_from_typeddict(RepairState)


def initial_state() -> RepairState:
    """Fresh state for a brand-new chat session."""
    return {
        "messages": [],
        "project_info": {},
        "next_agent": "",
        "plan_approved": "",
        "plan_ok": "",
        "plan_verdict": {},
        "plan_rounds": 0,
        "plan": "",
        "estimate": "",
        "material_findings": "",
        "qa_feedback": "",
        "qa_verdict": {},
        "qa_ok": "",
        "qa_rounds": 0,
        "estim_approved": "",
        "est_ok": "",
        "est_verdict": {},
        "est_rounds": 0,
        "final_answer": "",
        "input": "",
        "supervisor_rounds": 0,
    }
