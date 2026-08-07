"""API-key authentication for the service-desk server (fail-closed).

The example ships with a single shared key from ``settings.api_key``.
Auth is **fail-closed**: with no key configured every protected route
returns ``401`` with a hint to set ``TEFF_API_KEY`` (or ``api_key`` in
``.env``).  Set the key via ``TEFF_API_KEY``; the chat, run and
trace-dashboard routes then require the ``X-API-Key`` header.  Swap this
dependency for a JWT/cookie flow later without touching the route handlers.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, Request

router = APIRouter()


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str | None:
    """FastAPI dependency: enforce ``X-API-Key`` (fail-closed).

    Resolves the expected key from ``app.state.settings`` so tests can
    inject their own ``Settings`` without touching the environment.
    """
    api_key = request.app.state.settings.api_key
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key not configured; set TEFF_API_KEY (or api_key in .env)",
        )
    if x_api_key is None or not hmac.compare_digest(x_api_key, api_key):
        raise HTTPException(status_code=401, detail="invalid API key")
    return x_api_key


@router.get("/verify")
async def verify(_auth: str | None = Depends(require_api_key)) -> dict:
    """Probe endpoint — returns 200 when the caller is authorized."""
    return {"status": "ok"}
