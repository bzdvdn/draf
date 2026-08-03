"""FastAPI server running DRAFTFLOW graphs as durable HTTP endpoints.

Demonstrates the code-first pattern: graphs are defined in ``graphs.py``
(Flow builder, no workflow.yaml), registered by name, and executed over
HTTP with a shared checkpointer.  Every conversation is scoped to a user
id (``X-User-Id`` header) and its checkpoints are durable across requests
and process restarts.

Requires Ollama running locally with llama3.1:8b:

    ollama pull llama3.1:8b

Run (from the repo root):

    uv sync --extra fastapi
    uv run uvicorn --app-dir examples/fastapi_server app:app --port 8000

Endpoints:
    GET    /api/health          server status
    GET    /api/graphs          list registered code-first graphs
    POST   /api/chat            run a graph (single-shot or continuation)
    POST   /api/chat/stream     same, as an SSE event stream
    GET    /api/runs/{chat_id}  durable state of a conversation
    DELETE /api/runs/{chat_id}  delete a conversation
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from graphs import (
    DEFAULT_GRAPH,
    DEFAULT_MODEL,
    GRAPH_META,
    GRAPHS,
    TOOLS,
)
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from draf import set_defaults
from draf.checkpoint import DEFAULT_OWNER, Checkpoint, JSONFileCheckpointer
from draf.graph import Graph
from draf.node.interrupt import GraphInterrupt
from draf.tool import Tool

set_defaults(provider="ollama")

app = FastAPI(title="DRAFTFLOW FastAPI server", version="0.1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_checkpoint_dir = Path(
    os.environ.get(
        "DRAF_CHAT_CHECKPOINT_DIR", os.path.join(_HERE, "data", "checkpoints")
    )
)
checkpointer = JSONFileCheckpointer(str(_checkpoint_dir))


class ChatRequest(BaseModel):
    """Body for ``POST /api/chat`` and ``POST /api/chat/stream``."""

    graph: str = DEFAULT_GRAPH
    message: str = ""
    chat_id: str | None = None
    resume: dict[str, str] | None = None
    max_iterations: int = 20


def _get_graph(name: str) -> Graph:
    if name not in GRAPHS:
        raise HTTPException(status_code=404, detail=f"unknown graph: {name}")
    return GRAPHS[name]


def _tools_for(name: str) -> list[Tool]:
    return [TOOLS[t] for t in GRAPH_META[name].get("tools", []) if t in TOOLS]


async def _prepare_turn(
    graph_name: str, chat_id: str, owner: str, message: str, resume: dict | None
) -> tuple[dict, dict]:
    """Return ``(state, run_kwargs)`` for ``graph.run``/``graph.stream``.

    * Fresh chat          -> seed state with the user message.
    * Continuation        -> load the durable conversation, append the new
      user message, and re-enter the graph at its entry point so the
      accumulated history drives the reply (checkpoints are the
      conversation store).
    * Interrupt resume    -> pass the ``resume`` dict through untouched.
    """
    if resume:
        return {}, {"resume": resume}

    meta = GRAPH_META[graph_name]
    messages_key = meta.get("messages_key")
    graph = GRAPHS[graph_name]

    if messages_key is None:
        # Single-shot graph (e.g. summarize): overwrite the durable input.
        state = {"text": message}
        await checkpointer.save(
            chat_id,
            Checkpoint(state=state, next_node_id=graph.entry_point, iteration=0),
            owner=owner,
        )
        return {}, {}

    saved = await checkpointer.load(chat_id, owner=owner)
    if saved is None:
        return {"query": message, "messages": []}, {}

    state = dict(saved.state)
    messages = list(state.get(messages_key) or [])
    messages.append({"role": "user", "content": message})
    state[messages_key] = messages
    for stale in ("answer", "_tool_call_name", "_tool_calls"):
        state.pop(stale, None)
    await checkpointer.save(
        chat_id,
        Checkpoint(state=state, next_node_id=graph.entry_point, iteration=0),
        owner=owner,
    )
    return {}, {}


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "provider": "ollama",
        "model": DEFAULT_MODEL,
        "checkpoint_dir": str(_checkpoint_dir),
    }


@app.get("/api/graphs")
async def list_graphs() -> dict:
    return {"graphs": [{"name": name, **GRAPH_META[name]} for name in GRAPHS]}


@app.post("/api/chat")
async def chat(
    req: ChatRequest, x_user_id: str | None = Header(default=None, alias="X-User-Id")
) -> dict:
    owner = x_user_id or DEFAULT_OWNER
    graph_name = req.graph or DEFAULT_GRAPH
    graph = _get_graph(graph_name)
    chat_id = req.chat_id or uuid.uuid4().hex

    state, run_kwargs = await _prepare_turn(
        graph_name, chat_id, owner, req.message, req.resume
    )
    try:
        result = await graph.run(
            state=state,
            tools=_tools_for(graph_name),
            checkpointer=checkpointer,
            checkpoint_id=chat_id,
            owner=owner,
            max_iterations=req.max_iterations,
            **run_kwargs,
        )
    except GraphInterrupt as exc:
        return {
            "chat_id": chat_id,
            "status": "interrupt",
            "key": exc.key,
            "prompt": exc.prompt,
        }

    answer = result.get("answer") or result.get("summary") or ""
    return {"chat_id": chat_id, "status": "ok", "answer": answer}


@app.post("/api/chat/stream")
async def chat_stream(
    req: ChatRequest, x_user_id: str | None = Header(default=None, alias="X-User-Id")
) -> EventSourceResponse:
    owner = x_user_id or DEFAULT_OWNER
    graph_name = req.graph or DEFAULT_GRAPH
    graph = _get_graph(graph_name)
    chat_id = req.chat_id or uuid.uuid4().hex

    state, run_kwargs = await _prepare_turn(
        graph_name, chat_id, owner, req.message, req.resume
    )

    async def events():
        yield {
            "event": "chat_id",
            "data": json.dumps({"chat_id": chat_id}),
        }
        async for event in graph.stream(
            state=state,
            tools=_tools_for(graph_name),
            checkpointer=checkpointer,
            checkpoint_id=chat_id,
            owner=owner,
            max_iterations=req.max_iterations,
            **run_kwargs,
        ):
            data = {"chat_id": chat_id}
            if event.node_id is not None:
                data["node_id"] = event.node_id
            if event.node_type is not None:
                data["node_type"] = event.node_type
            data.update(event.data)
            yield {"event": event.type, "data": json.dumps(data)}

    return EventSourceResponse(events())


@app.get("/api/runs/{chat_id}")
async def get_run(
    chat_id: str, x_user_id: str | None = Header(default=None, alias="X-User-Id")
) -> dict:
    owner = x_user_id or DEFAULT_OWNER
    saved = await checkpointer.load(chat_id, owner=owner)
    if saved is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "chat_id": chat_id,
        "owner": owner,
        "next_node_id": saved.next_node_id,
        "iteration": saved.iteration,
        "state": saved.state,
    }


@app.delete("/api/runs/{chat_id}")
async def delete_run(
    chat_id: str, x_user_id: str | None = Header(default=None, alias="X-User-Id")
) -> dict:
    owner = x_user_id or DEFAULT_OWNER
    await checkpointer.delete(chat_id, owner=owner)
    return {"chat_id": chat_id, "status": "deleted"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000)
