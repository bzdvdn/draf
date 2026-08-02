"""Top-level API router — aggregates every endpoint group.

``app.py`` includes only this router, so the whole surface stays visible in
one place.  Each feature owns a sub-package with its own ``router.py``:
:mod:`src.api.chat`, :mod:`src.api.run`, :mod:`src.api.auth`.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.auth.router import router as auth_router
from src.api.catalog.router import router as catalog_router
from src.api.chat.router import router as chat_router
from src.api.run.router import router as run_router

api_router = APIRouter()

api_router.include_router(chat_router, prefix="/api/chat", tags=["chat"])
api_router.include_router(run_router, prefix="/api/runs", tags=["runs"])
api_router.include_router(catalog_router, prefix="/api/catalog", tags=["catalog"])
api_router.include_router(auth_router, prefix="/api/auth", tags=["auth"])


@api_router.get("/api/health")
async def health(request: Request) -> dict:
    """Server + model status."""
    settings = request.app.state.settings
    return {"status": "ok", "provider": settings.provider, "model": settings.model}
