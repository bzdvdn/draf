"""Catalog ingestion endpoints — load CSVs and refresh the vector store.

In production the price/materials list is not embedded lazily on the first
chat message; the operator pre-fills (or refreshes) the store through these
handlers, driven by the CLI ``load`` command or the API.  Loading is batched
(multiples of ``batch_size`` documents per embedding call) so thousands of
rows stream in a few HTTP round-trips.

Endpoints:
    GET   /api/catalog                ingestion status (queued / stored)
    POST  /api/catalog/load           upload a CSV and ingest it
    POST  /api/catalog/update         rebuild the store from the whole catalog
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from src.api.auth.router import require_api_key
from src.core.deps import PRODUCT_FIELDMAP

router = APIRouter(dependencies=[Depends(require_api_key)])


def _fieldmap_for(head: str) -> dict[str, str] | None:
    """Auto-detect a Russian price-list layout from the header row."""
    if "Наименование" in head:
        return PRODUCT_FIELDMAP
    return None


@router.get("")
async def catalog_status(request: Request) -> dict:
    """Report how many documents are queued vs. embedded in the store."""
    catalog = request.app.state.catalog
    return {
        "queued": catalog.size,
        "stored": catalog.stored,
        "total": catalog.size,
        "top_k": catalog.top_k,
    }


@router.post("/load")
async def catalog_load(
    request: Request,
    file: UploadFile | None = File(default=None),
    batch_size: int = Form(default=250),
    path: str | None = Form(default=None),
) -> dict:
    """Ingest a CSV — either an uploaded file or a server-side *path*.

    The CSV layout is detected from its header (``Наименование`` => product
    price list; otherwise the plain ``description`` layout).  Documents are
    embedded in *batch_size* chunks and added to the store.
    """
    catalog = request.app.state.catalog
    if file is not None:
        raw = file.file.read()
        header = raw.split(b"\n", 1)[0].decode("utf-8", "replace")
        fieldmap = _fieldmap_for(header)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", encoding="utf-8", delete=True
        ) as tmp:
            tmp.write(raw.decode("utf-8", errors="replace"))
            tmp.flush()
            queued = catalog.add_csv(tmp.name, fieldmap=fieldmap)
    elif path:
        if not Path(path).exists():
            raise HTTPException(status_code=404, detail=f"file not found: {path}")
        head = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        fieldmap = _fieldmap_for(head[0]) if head else None
        queued = catalog.add_csv(str(path), fieldmap=fieldmap)
    else:
        raise HTTPException(status_code=400, detail="pass 'file' or 'path'")

    report = await catalog.ingest(batch_size=batch_size)
    return {
        "queued_this_file": queued,
        "batch_size": batch_size,
        "report": {
            "queued": report.queued,
            "added": report.added,
            "batches": report.batches,
            "stored": report.stored,
        },
    }


@router.post("/update")
async def catalog_update(request: Request, batch_size: int = Form(default=250)) -> dict:
    """Rebuild the vector store from the whole catalog (full refresh)."""
    report = await request.app.state.catalog.rebuild(batch_size=batch_size)
    return {
        "batch_size": batch_size,
        "report": {
            "queued": report.queued,
            "added": report.added,
            "batches": report.batches,
            "stored": report.stored,
        },
    }
