"""Chat CLI — talk to the service-desk router interactively.

Runs the supervisor flow (billing / incident / deploy / fallback) as a chat:
type a request, the supervisor routes it to the right specialist (each backed
by its own knowledge-base RAG tool) and streams the answer back.  The deploy
specialist pauses for confirmation, which is answered in-line.  Sessions are
durable.

A separate ``load`` command pre-embeds / refreshes the knowledge base.

Usage::

    uv run python examples/applications/service_desk/cli.py                # chat
    uv run python examples/applications/service_desk/cli.py "message"      # one turn
    uv run python examples/applications/service_desk/cli.py load           # embed known facts
    uv run python examples/applications/service_desk/cli.py load --rebuild # clear + re-embed

Ctrl-D or Ctrl-C exits the chat loop.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service_desk.config.config import get_settings  # noqa: E402
from service_desk.core.deps import build_deps  # noqa: E402
from service_desk.graphs.build import build_flow  # noqa: E402
from service_desk.graphs.state import STATE_REDUCERS, initial_state  # noqa: E402
from service_desk.storage import TRANSIENT_KEYS, build_checkpointer  # noqa: E402

from draf import Assistant  # noqa: E402


async def _stream(assistant: Assistant, session: str, message: str) -> None:
    """Run one turn, streaming the supervisor routing and the specialist's answer.

    Pause handling is delegated to :meth:`Assistant.stream`: on the deploy
    gateway it emits an ``interrupt`` event and ends; the operator's answer is
    fed back as the next message so the flow continues in the same session.
    """
    streamed = False
    while True:
        async for event in assistant.stream(session, message):
            etype = event.type
            if etype == "node_start":
                print(f"\n—— {event.node_id} [{event.node_type}]")
            elif etype == "edge":
                print(f"    → {event.data.get('target_id')}")
            elif etype == "token":
                streamed = True
                print(event.data["token"], end="", flush=True)
            elif etype == "node_error":
                print(f"\n    !! {event.data['error']}")
            elif etype == "interrupt":
                print()
                prompt = event.data.get("prompt", "")
                if prompt:
                    print(f"\n{prompt}")
                message = input("\n> ").strip()
                break
        else:
            break
    print()
    if not streamed:
        reply = await assistant.last_reply(session)
        if reply:
            print(reply)
        else:
            print("(ответ не сформирован — проверьте модель/маршрутизацию)")


async def chat(assistant: Assistant, session: str) -> None:
    """Interactive loop: prompt -> answer -> repeat until Ctrl-D/Ctrl-C."""
    print("Просто введите запрос; Ctrl-D/Ctrl-C — выйти\n")
    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        await _stream(assistant, session, message)


async def _load(rebuild: bool, provider: str) -> None:
    """Embed (or re-embed) the knowledge-base CSVs into the vector store."""
    knowledge = build_deps(provider=provider)
    report = await (knowledge.rebuild if rebuild else knowledge.ingest)(batch_size=250)
    print(
        f"knowledge base: {report.queued} queued, {report.added} added, "
        f"{report.batches} batch(es), {report.stored} stored"
        + (" (rebuilt)" if rebuild else "")
    )


async def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="cli", description="Service-desk router chat (supervisor example)."
    )
    sub = parser.add_subparsers(dest="command")

    chat_p = sub.add_parser(
        "chat", help="interactive chat (no message) or a one-shot turn (with message)"
    )
    chat_p.add_argument("message", nargs="?", help="one-shot turn (omit to chat)")
    chat_p.add_argument("--model", default=settings.model, help="LLM model name")
    chat_p.add_argument(
        "--session", default="default", help="session id — reuse it to continue"
    )

    load_p = sub.add_parser("load", help="index the knowledge CSVs into the store")
    load_p.add_argument(
        "--rebuild", action="store_true", help="clear + re-embed the whole base"
    )
    load_p.add_argument(
        "--provider", default=settings.provider, help="embedding provider"
    )

    argv = sys.argv[1:]
    if not argv or argv[0] not in ("chat", "load"):
        argv = ["chat", *argv]
    args = parser.parse_args(argv)

    if args.command == "load":
        await _load(args.rebuild, args.provider)
        return

    flow, tools = build_flow(model=args.model, provider=settings.provider)
    assistant = Assistant(
        flow.compile(),
        tools,
        build_checkpointer(settings.checkpoint_dir),
        reducers=STATE_REDUCERS,
        initial_state=initial_state,
        transient_keys=TRANSIENT_KEYS,
    )

    print(
        f"provider: {settings.provider}  model: {args.model}  "
        f"session: {args.session}  (requires a running Ollama)"
    )
    if args.message:
        await _stream(assistant, args.session, args.message)
    else:
        await chat(assistant, args.session)


if __name__ == "__main__":
    asyncio.run(main())
