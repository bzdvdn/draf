"""Top-level API router — aggregates every endpoint group."""

from __future__ import annotations

from fastapi import APIRouter, Request

from fraud_gate.api.auth.router import router as auth_router
from fraud_gate.api.review.router import router as review_router

api_router = APIRouter()

api_router.include_router(review_router, prefix="/api/review", tags=["review"])
api_router.include_router(auth_router, prefix="/api/auth", tags=["auth"])


@api_router.get("/api/health")
async def health(request: Request) -> dict:
    """Server + model status."""
    settings = request.app.state.settings
    return {"status": "ok", "provider": settings.provider, "model": settings.model}
