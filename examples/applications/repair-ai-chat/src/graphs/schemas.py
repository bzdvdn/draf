"""Structured LLM output schemas for the repair workflow."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ProjectInfo(BaseModel):
    """Project details the extractor may pull from the conversation.

    Every field is optional — the extractor prompt instructs the model to
    return ``null`` for anything it cannot find.  ``extra="forbid"`` rejects
    any invented extra key (local models love to add ``{"type": "text"}``
    noise), which Pydantic renders as ``additionalProperties: False``.
    """

    model_config = {"extra": "forbid"}

    room_type: Optional[str] = None
    area: Optional[float] = None
    ceiling_height: Optional[float] = None
    budget: Optional[float] = None
    style: Optional[str] = None
    walls_area: Optional[float] = None
    floor_area: Optional[float] = None
    ceiling_area: Optional[float] = None


#: JSON Schema handed to the framework validator / re-ask loop.  Pydantic
#: emits ``anyOf`` for ``Optional`` fields, which ``draf.schema.validate_json``
#: now treats as alternative branches (same as ``oneOf``).
PROJECT_INFO_SCHEMA: dict = ProjectInfo.model_json_schema()


class QaVerdict(BaseModel):
    """QA agent's verdict on the current plan / estimate.

    ``ok`` is ``True`` when everything is consistent; ``message`` carries
    the concrete remarks to feed back to the fix agents, and is empty when
    the verdict passes.  ``extra="forbid"`` rejects invented keys (same as
    :class:`ProjectInfo`).
    """

    model_config = {"extra": "forbid"}

    ok: bool
    message: str = ""


#: JSON Schema for the QA verdict, validated with re-asking on failure.
QA_VERDICT_SCHEMA: dict = QaVerdict.model_json_schema()


class PlanVerdict(BaseModel):
    """Classifier's reading of the human's answer to a plan-approval question.

    ``ok`` is ``True`` when the user approved (да, конечно, хорошо, ок …) and
    ``False`` when they did not (нет, переделай …); ``message`` carries an
    optional echo of the answer.  Same shape as :class:`QaVerdict` so a
    :class:`~draf.node.ask.Validate` can decode it into a ``flow.loop``
    decider.
    """

    model_config = {"extra": "forbid"}

    ok: bool
    message: str = ""


#: JSON Schema for the plan-approval classifier (see ``flow.interrupt_loop``).
PLAN_VERDICT_SCHEMA: dict = PlanVerdict.model_json_schema()

#: JSON Schema for the estimate-approval classifier — same verdict shape
#: (``ok`` / ``message``) as the plan classifier.
ESTIMATE_VERDICT_SCHEMA: dict = PLAN_VERDICT_SCHEMA
