"""Observability dashboard: run a graph, capture full traces, browse in a UI.

Starts a FastAPI app that serves two things on one process:

- ``POST /api/run`` — executes a small graph (a ``react`` agent with two
  LLM calls), wrapping it in a :class:`~draf.observability.GraphObserver`
  that records the graph topology, per-node spans, and every LLM call with
  its full request (the messages) and response into a SQLite trace store.
- ``GET /obs/...`` — the read-only dashboard built by
  :func:`draf.observability.dashboard_router` over the same store.

Open ``http://localhost:8000/obs/ui`` after a few runs to inspect what was
sent to the model, per node, per run.

Requires Ollama running locally (``ollama pull llama3.1:8b``).

Usage:
    uv sync --extra api
    uv run python examples/observability/app.py
    curl -X POST http://localhost:8000/api/run -H 'Content-Type: application/json' -d '{"query": "расскажи про DevOps"}'
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from draf.flow import Flow
from draf.observability import GraphObserver, SQLiteExporter, topology_from_graph
from draf.observability.api import dashboard_router
from draf.provider import ProviderRegistry

SYSTEM = "Ты краткий ассистент по DevOps."
TRACES_DB = "./traces.db"

_providers = ProviderRegistry.from_presets("ollama")


def build_flow() -> "Flow":
    """A two-node graph: route a query through a summarising LLM."""
    flow = Flow(
        "devops-assistant",
        providers=_providers,
        default_provider="ollama",
    )
    flow.llm(
        model="llama3.1:8b",
        system=SYSTEM,
        prompt="{query}",
        output_key="draft",
    ).llm(
        model="llama3.1:8b",
        system=SYSTEM,
        prompt="Сжато, 1-2 предложения:\n{draft}",
        output_key="answer",
    )
    return flow


class RunRequest(BaseModel):
    query: str = "объясни что такое CI/CD"


def create_app() -> FastAPI:
    flow = build_flow()
    graph = flow.compile()
    exporter = SQLiteExporter(TRACES_DB)

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
            {"query": req.query},
            tracer=observer.tracer,
            on_llm_payload=observer.on_llm_payload,
        )
        observer.export()
        return JSONResponse({"answer": state.get("answer"), "stored": True})

    app.include_router(dashboard_router(exporter))
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
