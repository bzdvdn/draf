"""FastAPI application factory for {{PROJECT_NAME}}.

Run (from the project root)::

    uv sync --extra api
    uv run python main.py              # or: uv run uvicorn app:create_app

Endpoint groups live in :mod:`src.api` and are assembled by
:mod:`src.api.router`; this module only builds the app and wires the
durable assets (graph, tools, checkpointer) onto ``app.state``.  Nothing
runs at import time, and the LLM provider/model come from
:class:`src.config.config.Settings` — no global defaults are mutated.
"""

from __future__ import annotations

from fastapi import FastAPI

from src.config.config import Settings
from src.api.router import api_router
from src.core import build_container


def create_app(
    settings: Settings | None = None,
    *,
    checkpoint_dir: str | None = None,
) -> FastAPI:
    """Build the FastAPI app with its graph, tools and checkpointer.

    Assets are built once by :func:`src.core.build_container` and carried on
    ``app.state``.  Pass a ``Settings`` to override environment defaults
    (tests do this); ``checkpoint_dir`` is a convenience override for the
    storage location.
    """
    container = build_container(settings, checkpoint_dir=checkpoint_dir)

    app = FastAPI(
        title=container.settings.app_title,
        description=container.settings.app_description,
        version=container.settings.version,
    )
    app.state.container = container
    app.state.assistant = container.assistant
    app.state.catalog = container.catalog
    app.state.model = container.settings.model
    app.state.settings = container.settings
    app.include_router(api_router)
    return app
