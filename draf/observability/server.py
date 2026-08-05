"""Standalone trace server: ingest + dashboard (``draf obs-server``).

Workflows that have no API (declared purely as ``workflow.yaml``) push their
traces here over HTTP; the same process serves the dashboard UI::

    # side A: a machine running workflows
    draf run -f wf.yaml        # observability.export.webhook.url -> server

    # side B: the central collector
    draf obs-server --db traces.db --host 0.0.0.0 --port 8001
    # open http://localhost:8001/obs/ui

Requires ``draf[observability]`` (fastapi + uvicorn).
"""

from __future__ import annotations

from fastapi import FastAPI

from draf.observability.api import attach_dashboard, attach_ingest
from draf.observability.exporter import SQLiteExporter


def build_server(db: str = "traces.db", *, prefix: str = "/obs") -> FastAPI:
    """Assemble a FastAPI app serving ingest + dashboard over *exporter*."""
    exporter = SQLiteExporter(db)
    app = FastAPI(title="draf trace server")
    app.state.traces_exporter = exporter
    attach_ingest(app, exporter, prefix=prefix)
    attach_dashboard(app, exporter, prefix=prefix)
    return app


def serve(
    db: str = "traces.db",
    *,
    host: str = "127.0.0.1",
    port: int = 8001,
    prefix: str = "/obs",
) -> None:
    """Run the trace server with uvicorn (blocks)."""
    import uvicorn

    uvicorn.run(build_server(db, prefix=prefix), host=host, port=port)
