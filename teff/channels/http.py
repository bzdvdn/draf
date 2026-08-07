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

Hooks: ``create_http_app`` / :class:`HTTPChannel` accept FastAPI
*dependencies* (auth gates applied to every non-health endpoint) and a
``turn_kwargs`` factory ``(owner, session_id) -> kwargs`` whose result is
merged into every ``Assistant.run``/``Assistant.stream`` call — the hook
used to attach an observability ``tracer``/``on_llm_payload`` per turn.
The endpoints also live on an :class:`APIRouter` (``HTTPChannel.router``)
so the channel can be mounted into an existing app with
``app.include_router(channel.router)`` instead of being served standalone.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Callable, Sequence

from fastapi import APIRouter, FastAPI, Header, HTTPException
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


def create_http_router(
    assistant: Assistant,
    *,
    dependencies: Sequence[Any] | None = None,
    turn_kwargs: Callable[[str, str], dict] | None = None,
) -> APIRouter:
    """Build the HTTP/SSE routes bound to *assistant* as a mountable router.

    *dependencies* (FastAPI ``Depends`` objects) are attached to every
    endpoint except ``GET /api/health`` — e.g. ``[Depends(require_api_key)]``.
    *turn_kwargs* is called as ``turn_kwargs(owner, session_id)`` before each
    turn; its result is merged into the ``Assistant.run``/``Assistant.stream``
    call (and may override ``max_iterations``).
    """
    router = APIRouter()
    endpoint_deps = list(dependencies) if dependencies else None

    @router.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "workflow": "yaml"}

    @router.post("/api/chat", dependencies=endpoint_deps)
    async def chat(
        req: ChatRequest,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> dict:
        owner = x_user_id or DEFAULT_OWNER
        session_id = _session(owner)
        kwargs: dict[str, Any] = {"max_iterations": req.max_iterations}
        if turn_kwargs is not None:
            kwargs.update(turn_kwargs(owner, session_id) or {})
        result = await assistant.run(session_id, req.message, owner=owner, **kwargs)
        return turn_response(result, session_id)

    @router.post("/api/chat/stream", dependencies=endpoint_deps)
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
            kwargs: dict[str, Any] = {"max_iterations": req.max_iterations}
            if turn_kwargs is not None:
                kwargs.update(turn_kwargs(owner, session_id) or {})
            async for event in assistant.stream(
                session_id, req.message, owner=owner, **kwargs
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

    @router.get("/api/runs/{session_id}", dependencies=endpoint_deps)
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

    @router.delete("/api/runs/{session_id}", dependencies=endpoint_deps)
    async def delete_run(
        session_id: str,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> dict:
        owner = x_user_id or DEFAULT_OWNER
        if assistant.checkpointer is None:
            raise HTTPException(status_code=404, detail="no checkpointer configured")
        await assistant.checkpointer.delete(session_id, owner=owner)
        return {"session_id": session_id, "status": "deleted"}

    return router


def create_http_app(
    assistant: Assistant,
    *,
    dependencies: Sequence[Any] | None = None,
    turn_kwargs: Callable[[str, str], dict] | None = None,
) -> FastAPI:
    """Build a FastAPI app bound to *assistant* (one durable service).

    See :func:`create_http_router` for the ``dependencies`` / ``turn_kwargs``
    hooks.  Requires the ``teff[channels]`` extra.
    """
    app = FastAPI(title="teff channels (workflow.yaml over HTTP)", version="0.1")
    app.include_router(
        create_http_router(assistant, dependencies=dependencies, turn_kwargs=turn_kwargs)
    )
    return app


class HTTPChannel:
    """An HTTP/SSE adapter over a shared :class:`Assistant`.

    Requires the ``teff[channels]`` extra.  Usage::

        assistant = build_assistant("workflow.yaml")
        channel = HTTPChannel(assistant)
        uvicorn.run(channel.app, host=..., port=...)

    The endpoints also live on :attr:`router`, so the channel can be mounted
    into an existing app::

        app = FastAPI()
        app.include_router(HTTPChannel(assistant).router)
    """

    def __init__(
        self,
        assistant: Assistant,
        *,
        dependencies: Sequence[Any] | None = None,
        turn_kwargs: Callable[[str, str], dict] | None = None,
    ):
        self.assistant = assistant
        self.router = create_http_router(
            assistant, dependencies=dependencies, turn_kwargs=turn_kwargs
        )
        self.app = FastAPI(title="teff channels (workflow.yaml over HTTP)", version="0.1")
        self.app.include_router(self.router)
