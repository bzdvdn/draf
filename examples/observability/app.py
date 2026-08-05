"""Observability dashboard: run a graph, capture full traces, browse in a UI.

Starts a FastAPI app that serves two things on one process:

- ``POST /api/run`` — executes a small ``llm -> tools -> llm`` graph inside a
  :class:`~draf.observability.GraphObserver`: a ReAct agent (``assistant/agent``
  ``react_agent`` node ↔ ``assistant/tool`` ``tool_exec`` node) answers with
  the two tools ``current_time`` and ``uppercase``, then a final ``summarize``
  LLM node condenses the draft.  The observer records the graph topology,
  per-node spans, every LLM call with its full request (the messages) and
  response, and — newly — each tool call (name, args, result) as a
  first-class entry, all into a SQLite trace store.
- ``GET /obs/...`` — the read-only dashboard built by
  :func:`draf.observability.dashboard_router` over the same store.

Open ``http://localhost:8000/obs/ui`` after a few runs to inspect what was
sent to the model, per node, per run — including the tool-call chain.

Requires Ollama running locally (``ollama pull qwen2.5:7b``).

Usage:
    uv sync --extra observability
    uv run python examples/observability/app.py
    curl -X POST http://localhost:8000/api/run -H 'Content-Type: application/json' \
      -d '{"query": "Назови текущее время и переведи слово devops в верхний регистр"}'
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from draf.flow import Flow
from draf.observability import GraphObserver, SQLiteExporter, topology_from_graph
from draf.observability.api import dashboard_router
from draf.provider import ProviderRegistry
from draf.tool.tool import Tool

SYSTEM = "Ты краткий ассистент по DevOps."
TRACES_DB = "./traces.db"

_providers = ProviderRegistry.from_presets("ollama")


class CurrentTime(Tool):
    """Current UTC date and time."""

    name = "current_time"
    description = "Return the current UTC date and time."

    def run(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class Uppercase(Tool):
    """Convert a string to uppercase."""

    name = "uppercase"
    description = "Convert a string to uppercase letters."

    def run(self, text: str) -> str:
        return text.upper()


def build_flow() -> "Flow":
    """A three-leg chain: ReAct agent (llm <-> tools) then a summarising LLM."""
    flow = (
        Flow(
            "devops-assistant",
            providers=_providers,
            default_provider="ollama",
        )
        .react(
            id="assistant",
            model="qwen2.5:7b",
            system=SYSTEM,
            use_tools=["current_time", "uppercase"],
            input_key="input",
            output_key="draft",
            messages_key="messages",
        )
        .llm(
            id="summarize",
            model="qwen2.5:7b",
            system=SYSTEM,
            prompt="Сжато, 1-2 предложения:\n{draft}",
            output_key="answer",
        )
    )
    return flow


class RunRequest(BaseModel):
    query: str = "Назови текущее время и переведи слово devops в верхний регистр"


def create_app() -> FastAPI:
    flow = build_flow()
    graph = flow.compile()
    exporter = SQLiteExporter(TRACES_DB)
    tools = [CurrentTime(), Uppercase()]

    app = FastAPI(title="DRAFTFLOW observability", version="0.1")
    app.state.graph = graph
    app.state.exporter = exporter

    @app.post("/api/run")
    async def run_graph(req: RunRequest) -> JSONResponse:
        observer = GraphObserver(
            "devops-assistant",
            exporter=exporter,
            topology=topology_from_graph(graph),
            owner="console",
        )
        state = await graph.run(
            {"input": req.query},
            tools=tools,
            tracer=observer.tracer,
            on_llm_payload=observer.on_llm_payload,
            max_iterations=20,
        )
        observer.export()
        return JSONResponse({"answer": state.get("answer"), "stored": True})

    app.include_router(dashboard_router(exporter))
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
