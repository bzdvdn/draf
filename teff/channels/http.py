"""HTTP/SSE channel: serve one durable ``Assistant`` over FastAPI.

Sessions are scoped to a caller id (``X-User-Id`` header by default) and
their checkpoints are durable across requests and restarts, exactly like
the code-first ``examples/fastapi_server`` — but here the graph comes from
a ``workflow.yaml`` and the endpoint surface is fixed, so any workflow is
reachable over HTTP with zero application code.

This module requires the ``teff[channels]`` extra (FastAPI, uvicorn,
sse-starlette).  It is deliberately **not** imported by
``teff.channels`` — importing that package stays dependency-free; the HTTP
adapter is reached through :func:`teff.channels.create_http_app` or a
direct ``from teff.channels.http import HTTPChannel``.

Endpoints:
    GET   /api/health              server status
    POST  /api/chat                one durable turn (``message`` in)
    POST  /api/chat/stream         the same turn as an SSE stream
    GET   /api/runs/{session_id}   durable state of a conversation
    DELETE /api/runs/{session_id}  delete a conversation

Body / headers::
    POST /api/chat  {"message": "..."}            X-User-Id: alice
    POST /api/chat  {"message": "..."}            (interrupt) ->
        {"session_id", "waiting": true, "message": "<prompt>", "key"}
    POST /api/chat  {"message": "<answer>"}       resume the pause

The interrupt flow is folded into :class:`TurnResult` by
:class:`~teff.assistant.Assistant`, so clients resume simply by sending the
operator's answer as the next ``message``.  A turn always answers with
``{"session_id", "waiting", "message"}`` — ``message`` is the reply when
``waiting`` is false and the interrupt prompt when true.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from teff.assistant import Assistant
from teff.channels.reply import reply_from_state, turn_response
from teff.checkpoint import DEFAULT_OWNER


class ChatRequest(BaseModel):
    """Body for ``POST /api/chat`` and ``POST /api/chat/stream``."""

    message: str = ""
    max_iterations: int = 80


def _session(owner: str) -> str:
    """A fresh, server-generated session id in the caller's namespace.

    The session id is **always** generated here — never taken from the
    client — so callers cannot address or hijack another user's session by
    supplying an id.
    """
    return uuid.uuid4().hex


def create_http_app(assistant: Assistant) -> FastAPI:
    """Build a FastAPI app bound to *assistant* (one durable service).

    Requires the ``teff[channels]`` extra.
    """
    app = FastAPI(title="teff channels (workflow.yaml over HTTP)", version="0.1")

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "workflow": "yaml"}

    @app.post("/api/chat")
    async def chat(
        req: ChatRequest,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> dict:
        owner = x_user_id or DEFAULT_OWNER
        session_id = _session(owner)
        result = await assistant.run(
            session_id,
            req.message,
            owner=owner,
            max_iterations=req.max_iterations,
        )
        return turn_response(result, session_id)

    @app.post("/api/chat/stream")
    async def chat_stream(
        req: ChatRequest,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ):
        owner = x_user_id or DEFAULT_OWNER
        session_id = _session(owner)
        from sse_starlette.sse import EventSourceResponse

        async def events() -> AsyncIterator[dict]:
            yield {
                "event": "chat_id",
                "data": json.dumps({"session_id": session_id}),
            }
            async for event in assistant.stream(
                session_id,
                req.message,
                owner=owner,
                max_iterations=req.max_iterations,
            ):
                data: dict[str, Any] = {"session_id": session_id}
                if event.node_id is not None:
                    data["node_id"] = event.node_id
                if event.node_type is not None:
                    data["node_type"] = event.node_type
                data.update(event.data)
                yield {"event": event.type, "data": json.dumps(data)}
            reply = await assistant.last_reply(session_id, owner=owner)
            if not reply:
                state = await assistant.get_state(session_id, owner=owner)
                reply = reply_from_state(state)
            yield {
                "event": "message",
                "data": json.dumps({"session_id": session_id, "message": reply}),
            }

        return EventSourceResponse(events())

    @app.get("/api/runs/{session_id}")
    async def get_run(
        session_id: str,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> dict:
        owner = x_user_id or DEFAULT_OWNER
        if assistant.checkpointer is None:
            raise HTTPException(status_code=404, detail="no checkpointer configured")
        saved = await assistant.checkpointer.load(session_id, owner=owner)
        if saved is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {"session_id": session_id, "state": saved.state}

    @app.delete("/api/runs/{session_id}")
    async def delete_run(
        session_id: str,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> dict:
        owner = x_user_id or DEFAULT_OWNER
        if assistant.checkpointer is None:
            raise HTTPException(status_code=404, detail="no checkpointer configured")
        await assistant.checkpointer.delete(session_id, owner=owner)
        return {"session_id": session_id, "status": "deleted"}

    return app


class HTTPChannel:
    """An HTTP/SSE adapter over a shared :class:`Assistant`.

    Requires the ``teff[channels]`` extra.  Usage::

        assistant = build_assistant("workflow.yaml")
        channel = HTTPChannel(assistant)
        uvicorn.run(channel.app, host=..., port=...)
    """

    def __init__(self, assistant: Assistant):
        self.assistant = assistant
        self.app = create_http_app(assistant)
