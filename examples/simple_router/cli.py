"""CLI entry point for the simple_router example — run one turn.

Usage::

    uv run python cli.py run "write a python one-liner to list files"
    uv run python cli.py run --session abc123 "now make it a class"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.config import get_settings  # noqa: E402
from src.graphs.build import build_flow  # noqa: E402
from src.service.assistant import Assistant  # noqa: E402
from src.storage import build_checkpointer  # noqa: E402

app = typer.Typer(name="simple_router", help="Minimal draf route() example.")


def _build_assistant():
    settings = get_settings()
    flow = build_flow(model=settings.model, provider=settings.provider)
    return (
        Assistant(flow.compile(), build_checkpointer(settings.checkpoint_dir)),
        settings,
    )


def _render(event) -> None:
    if event.type == "run_start":
        print("\n-- run --")
    elif event.type == "node_start":
        print(f"\n-- {event.node_id} [{event.node_type}] --")
    elif event.type == "edge":
        print(f"  -> routed to {event.data['target_id']}")
    elif event.type == "token":
        print(event.data["token"], end="", flush=True)
    elif event.type == "run_end":
        print(f"\n== run_end: {event.data['status']} ==")
    elif event.type == "node_error":
        print(f"  !! {event.data['error']}")


@app.command()
def run(
    message: str = typer.Argument(..., help="The user message to route"),
    session: str = typer.Option("default", help="session id — reuse it to continue"),
    model: str | None = typer.Option(None, help="override the configured model"),
) -> None:
    """Run one conversation turn and stream the answer."""
    assistant, settings = _build_assistant()
    model = model or settings.model
    print(
        f"provider: {settings.provider}  model: {model}  "
        f"session: {session}  (requires a running Ollama)"
    )
    asyncio.run(_stream_turn(assistant, session, message))


async def _stream_turn(assistant: Assistant, session: str, message: str) -> None:
    async for event in assistant.stream_turn(session, message):
        _render(event)


if __name__ == "__main__":
    app()
