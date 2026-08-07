"""Domain value objects for the fraud gate.

Pure Python models that carry the *business result* of a review across the
layer boundary (domain service -> HTTP schema).  They hold no FastAPI/pydantic
dependency, so the domain logic stays framework-agnostic; the API layer maps
them onto its pydantic response schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..graphs.state import FinalVerdict  # noqa: F401  (re-export for typing)


class Conflict(Exception):
    """The requested operation conflicts with the session's current state."""


@dataclass
class ReviewOutcome:
    """The business outcome of screening a payment.

    ``decision`` is ``approve`` | ``review`` | ``deny``.  When ``waiting`` is
    true the run paused for a human analyst (:class:`~teff.node.Interrupt`);
    ``prompt`` carries the question and the caller should resume via
    :meth:`ReviewService.decide` with the analyst's answer.
    """

    session_id: str
    decision: str | None
    waiting: bool
    risk: float | None = None
    prompt: str | None = None
    final: dict | None = None
    events: list = field(default_factory=list)
    run_id: str | None = None


@dataclass
class DecideOutcome:
    """The resulting verdict after a human resumes a paused review."""

    session_id: str
    decision: str | None
    waiting: bool
    final: dict | None = None
    run_id: str | None = None
