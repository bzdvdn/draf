"""Pydantic schemas for the review API (input validation only).

Pydantic is used only where FastAPI genuinely needs it: parsing **and
validating** request bodies.  Outgoing responses are the pure domain
dataclasses (:class:`~fraud_gate.domain.models.ReviewOutcome` /
:class:`~fraud_gate.domain.models.DecideOutcome`) returned directly from the
handlers — FastAPI derives the response schema and serialization from them
via the return annotation, so the core framework never depends on pydantic
and there is no mirroring response DTO.
"""

from __future__ import annotations

from pydantic import BaseModel


class ReviewRequest(BaseModel):
    """Body for ``POST /api/review`` — the payment to review."""

    session_id: str | None = None
    tx: dict


class DecideRequest(BaseModel):
    """Body for ``POST /api/review/{session_id}/decide`` — the analyst's answer."""

    answer: str
