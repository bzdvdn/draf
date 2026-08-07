"""Review endpoints for the fraud gate — thin HTTP layer.

Handlers only validate the request body (pydantic), delegate to the domain
:class:`~fraud_gate.domain.review_service.ReviewService` on ``app.state`` and
return its pure dataclass value objects unchanged — FastAPI builds the
response schema/serialization from the return annotations, so no response
DTO is needed.  All business logic (durable sessions, driving the graph,
normalising the structured verdict) lives in the domain layer.

Endpoints:
    POST   /api/review                         screen a payment (may pause)
    POST   /api/review/{session_id}/decide     resume a paused review
    GET    /api/review/{session_id}            inspect a review's state
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from fraud_gate.api.auth.router import require_api_key
from fraud_gate.api.review.schemas import DecideRequest, ReviewRequest
from fraud_gate.domain.models import Conflict, DecideOutcome, ReviewOutcome
from fraud_gate.domain.review_service import ReviewService
from teff.checkpoint import DEFAULT_OWNER
from teff.observability import GraphObserver

router = APIRouter(dependencies=[Depends(require_api_key)])


def _owner(x_user_id: str | None) -> str:
    return x_user_id or DEFAULT_OWNER


def _service(request: Request) -> ReviewService:
    return request.app.state.review_service


def _observer(request: Request, owner: str, session_id: str) -> GraphObserver | None:
    """A fresh GraphObserver for one review (None when tracing is disabled)."""
    exporter = request.app.state.traces_exporter
    if exporter is None:
        return None
    return GraphObserver(
        "fraud_gate",
        exporter=exporter,
        topology=request.app.state.trace_topology,
        owner=owner,
        checkpoint_id=session_id,
    )


def _trace_kwargs(observer: GraphObserver | None) -> dict:
    """The tracer/on_llm_payload kwargs for one review."""
    return {
        "tracer": observer.tracer if observer else None,
        "on_llm_payload": observer.on_llm_payload if observer else None,
    }


def _trace_finish(observer: GraphObserver | None):
    """Persist the captured run (returns its id) when tracing is active."""
    if observer is None:
        return None
    return observer.export


@router.post("")
async def screen(
    req: ReviewRequest,
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> ReviewOutcome:
    """Screen a payment; may pause for a human analyst (``waiting: true``)."""
    owner = _owner(x_user_id)
    session_id = req.session_id or f"review:{owner}:{uuid.uuid4().hex}"
    observer = _observer(request, owner, session_id)
    return await _service(request).screen(
        owner,
        dict(req.tx),
        session_id=session_id,
        **_trace_kwargs(observer),
        trace_finish=_trace_finish(observer),
    )


@router.post("/{session_id}/decide")
async def decide(
    session_id: str,
    req: DecideRequest,
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> DecideOutcome:
    """Resume a paused mid-risk review with the analyst's answer."""
    owner = _owner(x_user_id)
    observer = _observer(request, owner, session_id)
    try:
        return await _service(request).decide(
            owner,
            session_id,
            req.answer,
            **_trace_kwargs(observer),
            trace_finish=_trace_finish(observer),
        )
    except Conflict as exc:
        raise HTTPException(
            status_code=409, detail=f"review not awaiting a decision: {exc}"
        ) from exc


@router.get("/{session_id}")
async def get_review(
    session_id: str,
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> dict:
    state = await _service(request).get(_owner(x_user_id), session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="review not found")
    return {"session_id": session_id, "state": state}
