"""Domain models shared across the repair chat application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProjectInfo:
    """Information about a renovation project extracted from the chat."""

    room_type: str | None = None
    area: float | None = None
    ceiling_height: float | None = None
    budget: int | None = None
    style: str | None = None
    walls_area: float | None = None
    floor_area: float | None = None
    ceiling_area: float | None = None


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


def project_info_from_dict(data: dict[str, Any]) -> ProjectInfo:
    """Build a ProjectInfo, ignoring unknown keys."""
    return ProjectInfo(
        **{k: v for k, v in data.items() if k in ProjectInfo.__dataclass_fields__}
    )


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
