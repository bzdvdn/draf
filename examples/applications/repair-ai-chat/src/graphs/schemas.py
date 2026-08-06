"""Structured LLM output schemas for the repair workflow."""

from __future__ import annotations

from dataclasses import dataclass

from draf.schema import json_schema_from_type


@dataclass
class QaVerdict:
    """QA agent's verdict on the current plan / estimate.

    ``ok`` is ``True`` when everything is consistent; ``message`` carries
    the concrete remarks to feed back to the fix agents, and is empty when
    the verdict passes.  Unknown keys are rejected (local models invent
    noise), mirroring ``extra="forbid"`` in the derived schema.
    """

    ok: bool
    message: str = ""


#: JSON Schema for the QA verdict, validated with re-asking on failure.
QA_VERDICT_SCHEMA: dict = dict(json_schema_from_type(QaVerdict))
QA_VERDICT_SCHEMA["required"] = ["ok"]
QA_VERDICT_SCHEMA["additionalProperties"] = False
