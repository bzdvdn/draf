"""FastAPI router exposing stored traces as a dashboard API.

Mount it against a :class:`~draf.observability.exporter.SQLiteExporter`
that the same process (or another one) writes into::

    from fastapi import FastAPI
    from draf.observability import SQLiteExporter, dashboard_router

    app = FastAPI()
    app.include_router(dashboard_router(SQLiteExporter("./traces.db")))

Endpoints:

- ``GET /obs/ui`` — the dashboard (``ui.html`` next to this module).
- ``GET /obs/ui/runs/{run_id}`` — the run-detail page for browsers (under
  the dashboard path).
- ``GET /obs/runs`` — recent runs (no payloads), with filters
  (``status``, ``name``, ``owner``, ``tag``) and pagination
  (``limit``/``offset``); returns ``{"items": [...], "total": n}``.
- ``GET /obs/runs/{run_id}`` — a dedicated HTML page for browsers
  (``ui_run.html``); returns the full run JSON for API clients.
- ``PATCH /obs/runs/{run_id}`` — update ``tags`` / ``notes`` on a run
  (body: ``{"tags": [...], "notes": "..."}``).

Run ids are the stable ``Run.run_id`` uuid4 values assigned when a run is
collected and shared across all exporters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from draf.observability.exporter import SQLiteExporter
from draf.observability.model import Run

_UI_PATH = Path(__file__).parent / "ui.html"
_UI_RUN_PATH = Path(__file__).parent / "ui_run.html"


def _ui_html(base: str) -> str:
    return _UI_PATH.read_text(encoding="utf-8").replace("__BASE_PATH__", base)


def _ui_run_html(run_id: str, base: str) -> str:
    return (
        _UI_RUN_PATH.read_text(encoding="utf-8")
        .replace("__RUN_ID__", json.dumps(run_id))
        .replace("__BASE_PATH__", base)
    )


class RunPatch(BaseModel):
    """Fields updatable on an existing run."""

    tags: list[str] | None = None
    notes: str | None = None


def dashboard_router(exporter: SQLiteExporter, *, prefix: str = "/obs") -> APIRouter:
    """Build the trace dashboard router over *exporter*.

    Mount it anywhere in your FastAPI app, under any prefix::

        app.include_router(dashboard_router(exporter))            # /obs/*
        app.include_router(dashboard_router(exporter, prefix="/dash"))  # /dash/*

    The HTML pages resolve their own links and fetches against *prefix*,
    so a custom prefix keeps the UI working.
    """

    router = APIRouter(prefix=prefix)

    @router.get("/ui")
    async def ui() -> Any:
        return HTMLResponse(_ui_html(prefix))

    @router.get("/runs")
    async def runs(
        limit: int = Query(20, ge=1, le=500),
        offset: int = Query(0, ge=0),
        status: str | None = Query(None),
        name: str | None = Query(None),
        owner: str | None = Query(None),
        tag: str | None = Query(None),
    ) -> dict[str, Any]:
        return exporter.list_runs(
            limit=limit,
            offset=offset,
            status=status,
            name=name,
            owner=owner,
            tag=tag,
        )

    @router.get("/ui/runs/{run_id}")
    async def ui_run_detail(run_id: str) -> Any:
        """The run-detail page under the dashboard path (``/obs/ui/runs/<id>``).

        The dashboard list links here for browsers; the pure JSON API stays at
        ``/obs/runs/<id>``.  The page resolves its own fetches against the
        dashboard *prefix*, so this URL is independent of the API path.
        """
        run = exporter.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return HTMLResponse(_ui_run_html(run_id, prefix))

    @router.get("/runs/{run_id}")
    async def run_detail(run_id: str, request: Request) -> Any:
        run = exporter.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        # Browsers get the dedicated page; API clients (fetch, curl) get JSON.
        if "text/html" in request.headers.get("accept", ""):
            return HTMLResponse(_ui_run_html(run_id, prefix))
        return run

    @router.patch("/runs/{run_id}")
    async def run_patch(run_id: str, patch: RunPatch) -> JSONResponse:
        ok = exporter.update_run(run_id, tags=patch.tags, notes=patch.notes)
        if not ok:
            raise HTTPException(status_code=404, detail="run not found")
        return JSONResponse({"run_id": run_id, "updated": True})

    return router


def attach_dashboard(
    app: FastAPI,
    exporter: SQLiteExporter,
    *,
    prefix: str = "/obs",
) -> None:
    """Mount the trace dashboard on an existing FastAPI *app*.

    Convenience wrapper around :func:`dashboard_router` for apps that
    assemble their endpoints elsewhere (e.g. ``app.include_router``):

        attach_dashboard(app, SQLiteExporter("./traces.db"))
    """
    app.include_router(dashboard_router(exporter, prefix=prefix))


def ingest_router(exporter: SQLiteExporter, *, prefix: str = "/obs") -> APIRouter:
    """Build the trace ingest router over *exporter*.

    ``POST {prefix}/ingest`` accepts a run in :meth:`Run.to_dict` shape
    (as produced by an :class:`~draf.observability.push.HttpExporter`) and
    persists it, so another machine — or a workflow with no API — can push
    traces into a shared dashboard::

        app.include_router(ingest_router(exporter))
    """
    router = APIRouter(prefix=prefix)

    @router.post("/ingest")
    async def ingest(payload: dict) -> JSONResponse:
        run = Run.from_dict(payload)
        exporter.export(run)
        return JSONResponse({"status": "ok"})

    return router


def attach_ingest(
    app: FastAPI,
    exporter: SQLiteExporter,
    *,
    prefix: str = "/obs",
) -> None:
    """Mount the trace ingest endpoint on an existing FastAPI *app*."""
    app.include_router(ingest_router(exporter, prefix=prefix))
