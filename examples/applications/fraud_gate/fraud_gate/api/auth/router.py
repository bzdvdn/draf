"""API-key authentication for the fraud-gate server.

The example ships with a single shared key from ``settings.api_key``.  When
it is empty (the default) auth is disabled and every route is open; set
``DRAF_API_KEY`` (or ``api_key`` in ``.env``) to require the ``X-API-Key``
header on the review router.  Swap this dependency for a JWT/tenant flow
later without touching the route handlers.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request

router = APIRouter()


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str | None:
    """FastAPI dependency: enforce ``X-API-Key`` when one is configured."""
    api_key = request.app.state.settings.api_key
    if not api_key:
        return x_api_key
    if x_api_key != api_key:
        raise HTTPException(status_code=401, detail="invalid API key")
    return x_api_key


@router.get("/verify")
async def verify(_auth: str | None = Depends(require_api_key)) -> dict:
    """Probe endpoint — returns 200 when the caller is authorized."""
    return {"status": "ok"}
