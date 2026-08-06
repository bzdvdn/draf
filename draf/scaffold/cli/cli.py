"""CLI app for {{PROJECT_NAME}} — run a turn or chat interactively.

The production interface here is the terminal (no server): the same
supervisor ``Flow`` from :mod:`src.graphs.build` runs through the same
:class:`~draf.assistant.Assistant`, so sessions are durable and
tokens stream live.

Usage::

    uv run python cli.py run "Help me draft a note"
    uv run python cli.py run --session abc123 "Now make it shorter"
    uv run python cli.py chat
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
from src.core import build_container  # noqa: E402

from draf import Assistant  # noqa: E402

app = typer.Typer(
    name="{{project_slug}}",
    help="Terminal-first draf app for {{PROJECT_NAME}}.",
    invoke_without_command=True,
)


@app.callback()
def _logging(
    log_level: str | None = typer.Option(
        None, "--log-level", help="draf log level (DEBUG/INFO/WARNING/ERROR)"
    ),
    log_format: str = typer.Option(
        "text", "--log-format", help="draf log format: text or json"
    ),
) -> None:
    """Configure draf's operational logging."""
    from draf import configure_logging

    configure_logging(log_level, format=log_format)


def _build_assistant():
    settings = get_settings()
    return build_container(settings), settings


def _render(event) -> None:
    if event.type == "run_start":
        print("\n-- run --")
    elif event.type == "node_start":
        print(f"\n-- {event.node_id} [{event.node_type}] --")
    elif event.type == "edge":
        print(f"  -> routed to {event.data['target_id']}")
    elif event.type == "token":
        print(event.data["token"], end="", flush=True)
    elif event.type == "checkpoint":
        print(f"  [checkpoint {event.data['action']}]")
    elif event.type == "run_end":
        print(f"\n== run_end: {event.data['status']} ==")
    elif event.type == "node_error":
        print(f"  !! {event.data['error']}")


async def _stream_turn(assistant: Assistant, session: str, message: str) -> None:
    async for event in assistant.stream_turn(session, message):
        _render(event)


@app.command()
def run(
    message: str = typer.Argument(..., help="The user message to send to the agents"),
    session: str = typer.Option("default", help="session id — reuse it to continue"),
    model: str | None = typer.Option(None, help="override the configured model"),
) -> None:
    """Run one conversation turn and stream the answer."""
    container, settings = _build_assistant()
    if model is not None and model != settings.model:
        settings = settings.model_copy(update={"model": model})
        container = build_container(settings)
    print(
        f"provider: {settings.provider}  model: {model or settings.model}  "
        f"session: {session}  (requires a running Ollama)"
    )
    asyncio.run(_stream_turn(container.assistant, session, message))
    print(f"\n== assistant ==\n{asyncio.run(container.assistant.last_reply(session))}")


@app.command()
def chat(
    session: str = typer.Option("default", help="session id — reuse it to continue"),
    model: str | None = typer.Option(None, help="override the configured model"),
) -> None:
    """Start an interactive chat loop (Ctrl-D to exit)."""
    container, settings = _build_assistant()
    if model is not None and model != settings.model:
        settings = settings.model_copy(update={"model": model})
        container = build_container(settings)
    print(
        f"provider: {settings.provider}  model: {model or settings.model}  "
        f"session: {session}  (requires a running Ollama)"
    )
    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        asyncio.run(_stream_turn(container.assistant, session, message))
        print(
            f"\n== assistant ==\n{asyncio.run(container.assistant.last_reply(session))}"
        )
        print()


if __name__ == "__main__":
    app()
