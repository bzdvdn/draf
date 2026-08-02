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
from src.graphs.build import build_flow  # noqa: E402
from src.service.assistant import Assistant  # noqa: E402
from src.storage import build_checkpointer  # noqa: E402


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
    args = parser.parse_args()

    flow, tools = build_flow(model=args.model, provider=settings.provider)
    assistant = Assistant(
        flow.compile(), tools, build_checkpointer(settings.checkpoint_dir)
    )

    print(
        f"provider: {settings.provider}  model: {args.model}  "
        f"session: {args.session}  (requires a running Ollama)"
    )
    async for event in assistant.stream_turn(args.session, args.message):
        _render(event)
    reply = await assistant.last_reply(args.session)
    if reply:
        print(f"\n== assistant ==\n{reply}")


if __name__ == "__main__":
    asyncio.run(main())
