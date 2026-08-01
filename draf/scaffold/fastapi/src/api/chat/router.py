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

from src.api.auth.router import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


class ChatRequest(BaseModel):
    """Body for ``POST /api/chat`` and ``POST /api/chat/stream``."""

    message: str = "Hi! Where should we start?"
    session_id: str | None = None
    max_iterations: int = 80


def _session(req: ChatRequest, request: Request, x_user_id: str | None):
    """The durable session assets plus the session id for one turn."""
    owner = x_user_id or DEFAULT_OWNER
    session_id = req.session_id or uuid.uuid4().hex
    return request.app.state.assistant, owner, session_id


@router.post("")
async def chat(
    req: ChatRequest,
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> dict:
    assistant, owner, session_id = _session(req, request, x_user_id)
    try:
        result = await assistant.run_turn(
            session_id,
            req.message,
            owner=owner,
            max_iterations=req.max_iterations,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
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
        async for event in assistant.stream_turn(
            session_id,
            req.message,
            owner=owner,
            max_iterations=req.max_iterations,
        ):
            data = {"session_id": session_id}
            if event.node_id is not None:
                data["node_id"] = event.node_id
            if event.node_type is not None:
                data["node_type"] = event.node_type
            data.update(event.data)
            yield {"event": event.type, "data": json.dumps(data)}

    return EventSourceResponse(events())
