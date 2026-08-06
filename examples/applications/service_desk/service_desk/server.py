"""FastAPI application factory for the ``service_desk`` application.

Run (from the example root)::

    uv sync --extra fastapi
    uv run python main.py              # or: uv run uvicorn app:create_app

Endpoint groups live in :mod:`service_desk.api` and are assembled by
:mod:`service_desk.api.router`; this module only builds the app and wires the
durable assets (graph, tools, checkpointer) onto ``app.state``.  Nothing
runs at import time, and the LLM provider/model come from
:class:`service_desk.config.config.Settings` — no global defaults are mutated.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from draf import Assistant
from draf.observability import SQLiteExporter, topology_from_graph
from draf.observability.api import attach_dashboard
from service_desk.api.auth.router import require_api_key
from service_desk.api.router import api_router
from service_desk.config.config import Settings, get_settings
from service_desk.core.deps import build_deps
from service_desk.graphs.build import build_flow
from service_desk.graphs.state import STATE_REDUCERS, initial_state
from service_desk.storage import TRANSIENT_KEYS, build_checkpointer


def create_app(
    settings: Settings | None = None,
    *,
    checkpoint_dir: str | None = None,
    knowledge=None,
    traces_db: str | None = None,
) -> FastAPI:
    """Build the FastAPI app with its graph, tools and checkpointer.

    Assets are built once and carried on ``app.state``.  Pass a
    ``Settings`` to override environment defaults (tests do this);
    ``checkpoint_dir`` / ``knowledge`` / ``traces_db`` are convenience
    overrides for storage / RAG seeding / trace persistence (tests inject a
    stub knowledge base and an in-memory trace db without touching `.env``).
    """
    settings = settings or get_settings()
    if checkpoint_dir is not None:
        settings = settings.model_copy(update={"checkpoint_dir": checkpoint_dir})

    knowledge = (
        knowledge if knowledge is not None else build_deps(provider=settings.provider)
    )
    flow, tools = build_flow(
        model=settings.model,
        provider=settings.provider,
        knowledge=knowledge,
    )
    compiled = flow.compile()
    assistant = Assistant(
        compiled,
        tools,
        build_checkpointer(settings.checkpoint_dir),
        reducers=STATE_REDUCERS,
        initial_state=initial_state,
        transient_keys=TRANSIENT_KEYS,
    )

    app = FastAPI(
        title=settings.app_title,
        description=settings.app_description,
        version=settings.version,
    )
    app.state.assistant = assistant
    app.state.knowledge = knowledge
    app.state.model = settings.model
    app.state.settings = settings
    app.include_router(api_router)

    # Trace dashboard: every chat turn is captured by a GraphObserver into a
    # local SQLite store and browsable at /obs/ui (prefix from settings).
    # It shares the API-key gate — traces expose full prompts and responses.
    trace_path = traces_db or (
        settings.traces_db
        or str(Path(__file__).resolve().parent / "data" / "traces.db")
    )
    traces_exporter = SQLiteExporter(trace_path)
    app.state.traces_exporter = traces_exporter
    app.state.trace_topology = topology_from_graph(compiled)
    attach_dashboard(
        app, traces_exporter, prefix=settings.traces_prefix, auth=require_api_key
    )

    return app
