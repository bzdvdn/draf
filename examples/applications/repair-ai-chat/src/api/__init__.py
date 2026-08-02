"""HTTP/SSE server for the repair-supervisor app.

Run it with::

    uv sync --extra api
    uv run python main.py                 # or: uv run uvicorn app:create_app

The application factory lives in the project-root ``app.py``; endpoint
groups are split by feature in :mod:`src.api.chat`, :mod:`src.api.run`
and :mod:`src.api.auth`, aggregated by :mod:`src.api.router`.
"""

__all__ = ["api_router"]

from src.api.router import api_router
