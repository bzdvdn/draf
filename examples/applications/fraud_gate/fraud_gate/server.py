"""FastAPI application factory for the ``fraud_gate`` application.

Run (from the example root)::

    uv sync --extra fastapi
    uv run python main.py              # or: uv run uvicorn app:create_app

Endpoint groups live in :mod:`fraud_gate.api` and are assembled by
:mod:`fraud_gate.api.router`; this module only builds the app and wires the
durable assets (graph, tools, checkpointer) onto ``app.state``.  The LLM
provider/model come from :class:`fraud_gate.config.config.Settings` — no
global defaults are mutated.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from draf import Assistant
from draf.observability import SQLiteExporter, topology_from_graph
from draf.observability.api import attach_dashboard
from fraud_gate.api.auth.router import require_api_key
from fraud_gate.api.router import api_router
from fraud_gate.config.config import Settings, get_settings
from fraud_gate.domain.review_service import ReviewService
from fraud_gate.graphs.build import build_flow
from fraud_gate.graphs.state import STATE_REDUCERS, initial_state
from fraud_gate.storage import TRANSIENT_KEYS, build_checkpointer


def create_app(
    settings: Settings | None = None,
    *,
    checkpoint_dir: str | None = None,
    traces_db: str | None = None,
) -> FastAPI:
    """Build the FastAPI app with its graph and checkpointer.

    Assets are built once and carried on ``app.state``.  Pass a ``Settings``
    to override environment defaults (tests do this); ``checkpooint_dir`` /
    ``traces_db`` are convenience overrides for storage / trace persistence
    without touching ``.env``.
    """
    settings = settings or get_settings()
    if checkpoint_dir is not None:
        settings = settings.model_copy(update={"checkpoint_dir": checkpoint_dir})

    graph = build_flow(
        model=settings.model,
        provider=settings.provider,
    ).compile()
    assistant = Assistant(
        graph,
        [],
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
    app.state.review_service = ReviewService(assistant)
    app.state.settings = settings
    app.include_router(api_router)

    # Trace dashboard: every review is captured by a GraphObserver into a
    # local SQLite store and browsable at ``<traces_prefix>/ui`` (settings).
    # It shares the API-key gate — traces expose full prompts and responses.
    trace_path = traces_db or (
        settings.traces_db
        or str(Path(__file__).resolve().parent / "data" / "traces.db")
    )
    traces_exporter = SQLiteExporter(trace_path)
    app.state.traces_exporter = traces_exporter
    app.state.trace_topology = topology_from_graph(graph)
    attach_dashboard(
        app, traces_exporter, prefix=settings.traces_prefix, auth=require_api_key
    )

    return app
