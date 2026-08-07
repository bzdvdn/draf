"""Debug CLI — run one conversation turn with live token streaming.

The server is the production interface; this script is a quick way to watch
a single turn against a local Ollama, streaming tokens as they arrive.
Provider/model/storage come from ``src.config.config`` (env / ``.env``).

Usage::

    uv run python cli.py "Help me draft a note"
    uv run python cli.py --session abc123 "Now make it shorter"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.config import get_settings  # noqa: E402
from src.core import build_container  # noqa: E402


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


async def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Run one conversation turn with token streaming."
    )
    parser.add_argument("message", nargs="?", default="Hi! Where should we start?")
    parser.add_argument("--model", default=settings.model, help="LLM model name")
    parser.add_argument(
        "--session", default="default", help="session id — reuse it to continue"
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="teff log level (DEBUG/INFO/WARNING/ERROR)",
    )
    parser.add_argument(
        "--log-format",
        default="text",
        choices=("text", "json"),
        help="teff log format",
    )
    args = parser.parse_args()

    from teff import configure_logging

    configure_logging(args.log_level, format=args.log_format)

    if args.model != settings.model:
        settings = settings.model_copy(update={"model": args.model})
    container = build_container(settings)

    print(
        f"provider: {settings.provider}  model: {args.model}  "
        f"session: {args.session}  (requires a running Ollama)"
    )
    async for event in container.assistant.stream(args.session, args.message):
        _render(event)
    reply = await container.assistant.last_reply(args.session)
    if reply:
        print(f"\n== assistant ==\n{reply}")


if __name__ == "__main__":
    asyncio.run(main())
