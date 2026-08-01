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

from src.config.config import Settings, get_settings
from src.api.router import api_router
from src.graphs.build import build_flow
from src.service.assistant import Assistant
from src.storage import build_checkpointer


def create_app(
    settings: Settings | None = None,
    *,
    checkpoint_dir: str | None = None,
) -> FastAPI:
    """Build the FastAPI app with its graph, tools and checkpointer.

    Assets are built once and carried on ``app.state``.  Pass a
    ``Settings`` to override environment defaults (tests do this);
    ``checkpoint_dir`` is a convenience override for the storage location.
    """
    settings = settings or get_settings()
    if checkpoint_dir is not None:
        settings = settings.model_copy(update={"checkpoint_dir": checkpoint_dir})

    flow, tools = build_flow(model=settings.model, provider=settings.provider)
    assistant = Assistant(
        flow.compile(), tools, build_checkpointer(settings.checkpoint_dir)
    )

    app = FastAPI(
        title=settings.app_title,
        description=settings.app_description,
        version=settings.version,
    )
    app.state.assistant = assistant
    app.state.model = settings.model
    app.state.settings = settings
    app.include_router(api_router)
    return app
