"""FastAPI application factory for {{PROJECT_NAME}}.

Run (from the project root)::

    uv sync --extra api
    uv run python main.py              # or: uv run uvicorn app:create_app

Endpoint groups live in :mod:`src.api` and are assembled by
:mod:`src.api.router`; this module only builds the app and wires the
durable assets (graph, tools, checkpointer) onto ``app.state``.  Nothing
runs at import time, and the LLM provider/model come from
:class:`src.config.config.Settings` — no global defaults are mutated.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from src.api.router import api_router
from src.config.config import Settings
from src.core import build_container

from draf.observability import SQLiteExporter, topology_from_graph
from draf.observability.api import attach_dashboard


def create_app(
    settings: Settings | None = None,
    *,
    checkpoint_dir: str | None = None,
) -> FastAPI:
    """Build the FastAPI app with its graph, tools and checkpointer.

    Assets are built once by :func:`src.core.build_container` and carried on
    ``app.state``.  Pass a ``Settings`` to override environment defaults
    (tests do this); ``checkpoint_dir`` is a convenience override for the
    storage location.
    """
    container = build_container(settings, checkpoint_dir=checkpoint_dir)

    app = FastAPI(
        title=container.settings.app_title,
        description=container.settings.app_description,
        version=container.settings.version,
    )
    app.state.container = container
    app.state.assistant = container.assistant
    app.state.catalog = container.catalog
    app.state.model = container.settings.model
    app.state.settings = container.settings
    app.include_router(api_router)

    # Trace dashboard: every chat turn is captured by a GraphObserver into a
    # local SQLite store and browsable at /obs/ui (prefix from settings).
    traces_path = container.settings.traces_db or (
        Path(__file__).resolve().parent / "data" / "traces.db"
    )
    traces_exporter = SQLiteExporter(str(traces_path))
    app.state.traces_exporter = traces_exporter
    app.state.trace_topology = topology_from_graph(container.assistant.graph)
    attach_dashboard(app, traces_exporter, prefix=container.settings.traces_prefix)

    return app
