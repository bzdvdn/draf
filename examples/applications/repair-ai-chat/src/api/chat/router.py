"""Chat endpoints — single-shot reply and an SSE token stream.

Handlers are thin: they read the :class:`~src.service.assistant.Assistant`
off ``request.app.state`` and delegate one turn to it.  Sessions are scoped
to a user id (``X-User-Id`` header) and durable across requests and process
restarts.

Endpoints:
    POST   /api/chat         single-shot reply (runs the flow once)
    POST   /api/chat/stream  SSE event stream over ``graph.stream()``
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from draf.checkpoint import DEFAULT_OWNER
from draf.observability import GraphObserver
from src.api.auth.router import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


class ChatRequest(BaseModel):
    """Body for ``POST /api/chat`` and ``POST /api/chat/stream``."""

    message: str = "Помоги спланировать ремонт ванной комнаты, 5 м²."
    session_id: str | None = None
    max_iterations: int = 80


def _session(req: ChatRequest, request: Request, x_user_id: str | None):
    """The durable session assets plus the session id for one turn."""
    owner = x_user_id or DEFAULT_OWNER
    session_id = req.session_id or uuid.uuid4().hex
    return request.app.state.assistant, owner, session_id


def _observer(request: Request, owner: str, session_id: str) -> GraphObserver | None:
    """A fresh GraphObserver for one turn (None when tracing is disabled)."""
    exporter = request.app.state.traces_exporter
    if exporter is None:
        return None
    return GraphObserver(
        "chat",
        exporter=exporter,
        topology=request.app.state.trace_topology,
        owner=owner,
        checkpoint_id=session_id,
    )


def _finish(observer: GraphObserver | None) -> None:
    """Persist the captured run when an observer was active.

    The exporter itself is owned by the app (``app.state.traces_exporter``)
    and shared across requests, so only the run is written here — never
    closed.
    """
    if observer is not None:
        observer.export()


@router.post("")
async def chat(
    req: ChatRequest,
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> dict:
    assistant, owner, session_id = _session(req, request, x_user_id)
    observer = _observer(request, owner, session_id)
    try:
        result = await assistant.run_turn(
            session_id,
            req.message,
            owner=owner,
            max_iterations=req.max_iterations,
            tracer=observer.tracer if observer else None,
            on_llm_payload=observer.on_llm_payload if observer else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _finish(observer)
    return {"session_id": session_id, "result": result}


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> EventSourceResponse:
    assistant, owner, session_id = _session(req, request, x_user_id)

    async def events():
        yield {"event": "chat_id", "data": json.dumps({"session_id": session_id})}
        observer = _observer(request, owner, session_id)
        try:
            async for event in assistant.stream_turn(
                session_id,
                req.message,
                owner=owner,
                max_iterations=req.max_iterations,
                tracer=observer.tracer if observer else None,
                on_llm_payload=observer.on_llm_payload if observer else None,
            ):
                data = {"session_id": session_id}
                if event.node_id is not None:
                    data["node_id"] = event.node_id
                if event.node_type is not None:
                    data["node_type"] = event.node_type
                data.update(event.data)
                yield {"event": event.type, "data": json.dumps(data)}
        finally:
            _finish(observer)

    return EventSourceResponse(events())
