"""Standalone trace server: ingest + dashboard (``teff obs-server``).

Workflows that have no API (declared purely as ``workflow.yaml``) push their
traces here over HTTP; the same process serves the dashboard UI::

    # side A: a machine running workflows
    teff run -f wf.yaml        # observability.export.webhook.url -> server

    # side B: the central collector
    teff obs-server --db traces.db --host 0.0.0.0 --port 8001 --api-key <key>
    # open http://localhost:8001/obs/ui

Security: traces contain full prompts and responses.  When binding beyond
``127.0.0.1`` you **must** pass ``--api-key``; without one the server refuses
to start on a non-loopback host.  ``teff obs-server --help`` documents this.

Requires ``teff[observability]`` (fastapi + uvicorn).
"""

from __future__ import annotations

import hmac

from fastapi import FastAPI, Header, HTTPException

from teff.observability.api import attach_dashboard, attach_ingest
from teff.observability.exporter import SQLiteExporter

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _api_key_auth(expected: str):
    """A FastAPI dependency enforcing a single shared ``X-API-Key`` header."""

    async def _auth(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> str | None:
        if x_api_key is not None and hmac.compare_digest(x_api_key, expected):
            return x_api_key
        raise HTTPException(status_code=401, detail="invalid API key")

    return _auth


def build_server(
    db: str = "traces.db",
    *,
    prefix: str = "/obs",
    api_key: str | None = None,
) -> FastAPI:
    """Assemble a FastAPI app serving ingest + dashboard over *exporter*.

    When *api_key* is set it protects both the dashboard and the ingest
    endpoint with a shared ``X-API-Key`` header.
    """
    exporter = SQLiteExporter(db)
    app = FastAPI(title="teff trace server")
    app.state.traces_exporter = exporter
    auth = _api_key_auth(api_key) if api_key else None
    attach_ingest(app, exporter, prefix=prefix, auth=auth)
    attach_dashboard(app, exporter, prefix=prefix, auth=auth)
    return app


def serve(
    db: str = "traces.db",
    *,
    host: str = "127.0.0.1",
    port: int = 8001,
    prefix: str = "/obs",
    api_key: str | None = None,
) -> None:
    """Run the trace server with uvicorn (blocks)."""
    import uvicorn

    uvicorn.run(build_server(db, prefix=prefix, api_key=api_key), host=host, port=port)
