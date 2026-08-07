"""API-key authentication for the {{PROJECT_NAME}} server.

The scaffold ships with a single shared key from ``settings.api_key``.
Authentication is **fail-closed**: when no key is configured every protected
route returns ``401`` with a hint to set ``TEFF_API_KEY``.  ``teff new``
generates a random key into the project's local ``.env`` (git-ignored), so a
fresh scaffold works out of the box while remaining secure by default.  Swap
this dependency for a JWT/cookie flow later without touching the route
handlers.

Tenant isolation: sessions are scoped to an explicit caller id.  Every route
requires the ``X-User-Id`` header — there is no fallback to a shared
``DEFAULT_OWNER`` bucket, so callers cannot silently collide in another
user's namespace.
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


def require_user_id(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> str:
    """FastAPI dependency: the caller's tenant id (``X-User-Id``).

    There is deliberately **no** fallback to a shared default owner: an
    anonymous caller would otherwise land in a common namespace and read or
    overwrite other users' sessions.  Wire this to your real authentication
    layer (JWT subject, session cookie, ...) later.
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-Id header is required")
    return x_user_id


@router.get("/verify")
async def verify(_auth: str | None = Depends(require_api_key)) -> dict:
    """Probe endpoint — returns 200 when the caller is authorized."""
    return {"status": "ok"}
