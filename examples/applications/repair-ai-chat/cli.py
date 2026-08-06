"""Chat CLI — talk to the repair assistant interactively.

Runs the whole supervisor graph (planner → estimator → materials → QA) as a
simple chat: type a request, the assistant plans/calculates and streams the
answer back.  Sessions are durable, so history is kept across prompts and
the conversation really feels like talking to a person.

Usage::

    uv run python examples/applications/repair-ai-chat/cli.py            # chat
    uv run python examples/applications/repair-ai-chat/cli.py "message"  # one turn

Ctrl-D or Ctrl-C exits the chat loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.config import get_settings  # noqa: E402
from src.graphs.build import build_flow  # noqa: E402
from src.graphs.state import STATE_REDUCERS, initial_state  # noqa: E402
from src.storage import TRANSIENT_KEYS, build_checkpointer  # noqa: E402

from draf import Assistant  # noqa: E402
from draf.checkpoint import DEFAULT_OWNER  # noqa: E402

#: State keys that add noise to the debug ledger — always hidden.
_HIDDEN_KEYS = {"next_agent", "input", "supervisor_rounds", "tool_approval"}


def _short(value, limit: int = 200) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _state_delta(prev: dict, cur: dict) -> list[str]:
    """Render keys that changed since the last snapshot (debug view)."""
    lines: list[str] = []
    for key, value in cur.items():
        if key.startswith("_") or key in _HIDDEN_KEYS:
            continue
        if value != prev.get(key):
            if value:
                lines.append(f"{key}={_short(value)}")
            else:
                lines.append(f"{key}=")  # cleared
    return lines


async def _stream(assistant: Assistant, session: str, message: str) -> None:
    """Run one turn, streaming the assistant's tokens plus debug ledger.

    Pause handling is delegated to :meth:`Assistant.stream`: on the plan- or
    estimate-approval interrupt it emits an ``interrupt`` event and ends;
    the operator's answer is fed back as the next message so the pipeline
    continues in the same session — no manual resume plumbing here.
    """
    prev: dict = {}
    streamed = False
    while True:
        async for event in assistant.stream(session, message):
            etype = event.type
            if etype == "node_start":
                print(f"\n—— {event.node_id} [{event.node_type}]")
            elif etype == "edge":
                print(f"    → {event.data.get('target_id')}")
            elif etype == "tool_call":
                args = event.data.get("args", "{}")
                try:
                    args = json.loads(args) if args else {}
                except (json.JSONDecodeError, TypeError):
                    pass
                print(f"    [tool] {event.data.get('name')}({_short(args, 160)})")
            elif etype == "token":
                streamed = True
                print(event.data["token"], end="", flush=True)
            elif etype == "node_error":
                print(f"\n    !! {event.data['error']}")
            elif etype == "node_end":
                saved = await assistant.checkpointer.load(session, owner=DEFAULT_OWNER)
                cur = saved.state if saved is not None else {}
                delta = _state_delta(prev, cur)
                if delta:
                    print("    · state: " + "; ".join(delta))
                prev = cur
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
        # Nothing streamed (e.g. the turn ended on extract-only) — show the
        # durable final reply so the user isn't left staring at silence.
        reply = await assistant.last_reply(session)
        if reply:
            print(reply)
        else:
            print("(ответ не сформирован — проверьте модель/маршрутизацию)")


async def chat(assistant: Assistant, session: str) -> None:
    """Interactive loop: prompt -> answer -> repeat until Ctrl-D/Ctrl-C."""
    print("просто вводите запросы, Ctrl-D/Ctrl-C — выйти\n")
    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        await _stream(assistant, session, message)


async def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="cli",
        description="Repair assistant chat + catalog loader for the "
        "repair-ai-chat application.",
    )
    sub = parser.add_subparsers(dest="command")

    chat_p = sub.add_parser(
        "chat", help="interactive chat (no message) or a one-shot turn (with message)"
    )
    chat_p.add_argument(
        "message",
        nargs="?",
        help="one-shot turn to run (omit to start interactive chat)",
    )
    chat_p.add_argument("--model", default=settings.model, help="LLM model name")
    chat_p.add_argument(
        "--session", default="default", help="session id — reuse it to continue"
    )

    load_p = sub.add_parser("load", help="ingest CSVs into the vector store in batches")
    load_p.add_argument("files", nargs="+", type=Path, help="CSV file(s) to load")
    load_p.add_argument("--batch-size", type=int, default=250)
    load_p.add_argument(
        "--rebuild", action="store_true", help="clear + re-embed the whole catalog"
    )
    load_p.add_argument(
        "--provider", default=settings.rag_embedder or settings.provider
    )  #: Durable vector-store file to load into (defaults to the shared catalog db).
    load_p.add_argument(
        "--db",
        dest="catalog_db",
        default=settings.database_url or settings.catalog_db,
    )

    # Backwards compatibility: `cli.py "msg"` and bare `cli.py` mean "chat".
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("chat", "load"):
        argv = ["chat", *argv]
    args = parser.parse_args(argv)

    if args.command == "load":
        await _load(
            args.files,
            args.batch_size,
            args.rebuild,
            args.provider,
            catalog_db=args.catalog_db,
        )
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


async def _load(
    files, batch_size: int, rebuild: bool, provider: str, catalog_db=None
) -> None:
    """Ingest CSV(s) into the durable catalog store in *batch_size* chunks."""
    from src.core.deps import DEFAULT_CATALOG_DB, PRODUCT_FIELDMAP
    from src.rag.catalog import MaterialCatalog

    from draf.rag.embedder import Embedder
    from draf.rag.stores import SQLiteVectorStore

    store = SQLiteVectorStore(path=str(catalog_db or DEFAULT_CATALOG_DB), dim=768)
    catalog = MaterialCatalog(embedder=Embedder(provider=provider), store=store)
    for f in files:
        if not f.exists():
            print(f"  skip: {f} not found")
            continue
        head = f.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
        fieldmap = PRODUCT_FIELDMAP if "Наименование" in head else None
        added = catalog.add_csv(str(f), fieldmap=fieldmap)
        print(f"  queued {added:>5} rows from {f.name}")
    if catalog.size == 0:
        print("nothing to load")
        return
    report = await (catalog.rebuild if rebuild else catalog.ingest)(
        batch_size=batch_size
    )
    print(
        f"  stored {report.stored} (of {report.queued} queued) "
        f"via {report.batches} batch(es) of <= {batch_size}"
    )


if __name__ == "__main__":
    asyncio.run(main())
