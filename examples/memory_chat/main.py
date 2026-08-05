"""Multi-user streaming chat with per-owner long-term memory.

A single graph serves every tenant: the operator types an owner id at the
console, and memory is scoped to ``("users", <owner>)`` automatically.
Each turn streams assistant tokens to the terminal, and afterwards a
:class:`~draf.memory.MemoryExtractor` turns the exchange into durable facts
written back under the same owner namespace.  On the next turn those facts
are injected into the agent's system prompt (``${owner}`` in the
``MemoryConfig`` namespace is resolved from the run context), so a second
session under the same owner starts with everything already known.

Requires Ollama running locally.  Two models are used: *llama3.1:8b* for
chat and extraction, and *nomic-embed-text* for embeddings.  Set the
``EXTRACTOR_PROMPT`` module constant below to scope what gets remembered.

Usage:
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
    python examples/memory_chat/main.py            # owner asked on stdin
    python examples/memory_chat/main.py --owner ana   # or pass it directly
"""

from __future__ import annotations

import argparse
import asyncio

from draf.flow import Flow
from draf.memory import MemoryConfig, MemoryExtractor
from draf.memory.tool import memory_from_config
from draf.provider import ProviderRegistry

CHAT_SYSTEM = (
    "You are a helpful personal assistant. Use the relevant memories "
    "provided in your system prompt to personalise your replies."
)

MODEL = "llama3.1:8b"
DEFAULT_DB = "./memories.db"

# Override the fact-extraction prompt via ``MemoryExtractor(system_prompt=...)``.
# ``None`` would keep the built-in few-shot default; setting it here makes the
# example self-contained.  Scope what gets remembered: only preferences, only
# budget, one language, ... Adjust freely — this is a working RU/EN version.
EXTRACTOR_PROMPT = """\
You extract durable, long-term facts about the user from a conversation.
A durable fact stays true beyond this session: identity, profession,
preferences, relationships, habits, recurring needs, explicit decisions.

Only extract facts about the user — ignore what the assistant says. Do not
extract greetings, thanks, timestamps, or one-off requests.

Examples:
User: "Привет, я DevOps инженер и люблю кофе"
-> [{"text": "The user is a DevOps engineer"}, {"text": "The user likes coffee"}]

User: "thanks, bye"
-> []

Return ONLY a JSON array of objects with a "text" field, one per fact each a
self-contained sentence. Respond in the language of the conversation. If
there are no durable facts, return an empty array: []"""


async def emit_tokens(event) -> None:
    """Print ``token`` events as they arrive; terminate the line on run end."""
    if event.type == "token":
        print(event.data["token"], end="", flush=True)
    elif event.type == "run_end":
        print()


async def recall_count(memory, owner: str) -> int:
    return len(await memory.list(("users", owner), limit=1000))


async def run(owner: str, *, extract: bool, db: str) -> None:
    providers = ProviderRegistry.from_presets("ollama")

    # SQLite store: fact survive process restarts, so a later session under
    # the same owner picks up where the last one left off.
    memory = memory_from_config(
        {
            "store": {"type": "sqlite", "path": db, "dim": 768},
            "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
        },
        providers=providers,
    )
    extractor = MemoryExtractor(
        model=MODEL, provider="ollama", system_prompt=EXTRACTOR_PROMPT
    )

    flow = Flow(
        "memory_chat",
        providers=providers,
        default_provider="ollama",
    )
    flow.react(
        model=MODEL,
        system=CHAT_SYSTEM,
        messages_key="messages",
        stream=True,
        memory=MemoryConfig(
            store=memory,
            namespace=["users", "${owner}"],  # resolved from the run owner
            k=5,
        ),
    )
    graph = flow.compile()

    known = await recall_count(memory, owner)
    print(f"\n[owner] {owner!r} — {known} memory item(s) on file")
    print("(type '/quit' to leave)\n")

    state: dict = {"messages": []}
    while True:
        user_text = input("you> ").strip()
        if user_text in ("/quit", "/exit", "q"):
            break
        if not user_text:
            continue

        state["messages"].append({"role": "user", "content": user_text})
        print("assistant> ", end="", flush=True)
        final = await graph.run(state, owner=owner, emit=emit_tokens)

        # Keep only the conversation; reset the react loop's transient keys.
        state = {"messages": list(final["messages"])}

        if extract and len(state["messages"]) >= 2:
            turn = state["messages"][-2:]
            written = await extractor.save(memory, turn, ("users", owner))
            if written:
                print(f"[memory] +{len(written)} fact(s) stored")
    print("\nbye")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-user streaming memory chat")
    parser.add_argument("--owner", default="", help="owner id (asked on stdin if empty)")
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="skip fact extraction after each turn",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help="SQLite database file for long-term memory",
    )
    args = parser.parse_args()

    owner = args.owner.strip() or input("owner> ").strip() or "guest"
    asyncio.run(run(owner, extract=not args.no_extract, db=args.db))


if __name__ == "__main__":
    main()
