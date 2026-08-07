"""Domain models shared across the repair chat application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from teff.schema import json_schema_from_type

#: Human-readable labels for rendering :class:`ProjectInfo` to an LLM.
PROJECT_INFO_LABELS: dict[str, str] = {
    "room_type": "Тип помещения",
    "area": "Площадь",
    "ceiling_height": "Высота потолков",
    "budget": "Бюджет",
    "style": "Стиль",
    "walls_area": "Площадь стен",
    "floor_area": "Площадь пола",
    "ceiling_area": "Площадь потолка",
}


@dataclass
class ProjectInfo:
    """Information about a renovation project extracted from the chat.

    This dataclass is the single source of truth for the project model: the
    JSON Schema handed to the extractor is derived from it (see
    :data:`PROJECT_INFO_SCHEMA`).  The graph stores the same facts as a
    plain dict under ``project_info`` so it serializes cleanly to the
    checkpointer; :func:`project_info_from_dict` bridges dicts to this
    model.  Every field is optional — the extractor returns ``null``
    for anything it cannot find.
    """

    room_type: str | None = None
    area: float | None = None
    ceiling_height: float | None = None
    budget: float | None = None
    style: str | None = None
    walls_area: float | None = None
    floor_area: float | None = None
    ceiling_area: float | None = None


#: JSON Schema derived from :class:`ProjectInfo`.  All fields are optional
#: (the extractor returns ``null`` for unknown ones) and unknown keys are
#: rejected, mirroring the strict ``extra="forbid"`` contract.  Kept here so
#: the dataclass is the one model the state and the LLM both refer to.
PROJECT_INFO_SCHEMA: dict = dict(json_schema_from_type(ProjectInfo))
PROJECT_INFO_SCHEMA.pop("required", None)
PROJECT_INFO_SCHEMA["additionalProperties"] = False


@dataclass
class MaterialRequirement:
    name: str = ""
    quantity: float = 0.0
    unit: str = ""
    unit_price: float = 0.0
    total: float = 0.0


@dataclass
class Estimate:
    items: list[MaterialRequirement] = field(default_factory=list)
    labor_cost: float | None = None
    overhead_percent: float = 10.0
    overhead_amount: float = 0.0
    total: float = 0.0


def merge_project_info(current: dict | None, update: dict | None) -> dict:
    """State reducer: merge extracted fields into the running project info.

    ``None`` values in the update are skipped so an extraction that only
    returns new fields never wipes previously known ones.
    """
    merged = dict(current or {})
    if update:
        for key, value in update.items():
            if value is not None:
                merged[key] = value
    return merged


def project_info_text(state: dict) -> str:
    """Render the project info as LLM-readable ``label: value`` lines.

    Uses :data:`PROJECT_INFO_LABELS` so the sub-agents read the same,
    schema-backed project facts instead of raw JSON.
    """
    info = state.get("project_info") or {}
    if not info:
        return "(информация о проекте не извлечена)"
    lines = [
        f"{PROJECT_INFO_LABELS[field]}: {info[field]}"
        for field in PROJECT_INFO_LABELS
        if info.get(field) is not None
    ]
    return "; ".join(lines)


def project_info_from_dict(data: dict[str, Any]) -> ProjectInfo:
    """Build a ProjectInfo, ignoring unknown keys."""
    return ProjectInfo(
        **{k: v for k, v in data.items() if k in ProjectInfo.__dataclass_fields__}
    )
