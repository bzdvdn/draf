"""Long-term memory: extract facts, store them, and inject them.

Demonstrates the three memory pieces on top of a vector store:

1. ``MemoryExtractor`` — turn a conversation into durable facts via an LLM.
2. ``MemoryStore`` — persist facts under a per-user namespace, built with
   the same config dict the ``memory`` tool uses (so a provider registered
   in the :class:`~draf.provider.ProviderRegistry` feeds its ``base_url``
   into the embedder unless the config overrides it).
3. ``memory_context`` — recall relevant facts and format them so they can
   be injected into an agent's system prompt (this is what a
   ``react_agent`` with a ``memory`` config does on every turn).

Requires Ollama running locally. Two models are used: *llama3.1:8b* for
the extractor and *nomic-embed-text* for embeddings.

Usage:
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
    python examples/memory_assistant/main.py
"""

import asyncio

from draf.flow import Flow
from draf.memory import MemoryConfig, MemoryExtractor, memory_context
from draf.memory.tool import memory_from_config
from draf.provider import ProviderRegistry


async def main():
    # Declare the provider once; the embedder inherits its base_url.
    providers = ProviderRegistry.from_presets("ollama")

    # 1. The MemoryStore: same config dict the `memory` tool accepts.
    memory = memory_from_config(
        {
            "store": {"type": "in_memory", "dim": 768},
            "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
        },
        providers=providers,
    )

    # 2. Extract durable facts from a conversation.
    extractor = MemoryExtractor(model="llama3.1:8b", provider="ollama")
    conversation = [
        {
            "role": "user",
            "content": "Hi, I'm Ana. I mostly work with "
            "TypeScript and I'm based in Lisbon.",
        },
        {"role": "assistant", "content": "Nice to meet you, Ana!"},
        {
            "role": "user",
            "content": "Also, I strongly prefer video calls over written docs.",
        },
    ]
    written = await extractor.save(memory, conversation, ("users", "ana"))
    print(f"Extracted {len(written)} facts.")

    # 3. Recall them on a later turn, formatted for a system prompt.
    block = await memory_context(
        memory,
        "how should we meet?",
        namespace=("users", "ana"),
        k=5,
    )
    print("\n=== injected context ===\n")
    print(block or "(nothing recalled)")

    # 4. A react_agent can do the same injection on every turn with a
    #    `memory` config; the store and provider come from the flow:
    flow = Flow(
        "memory_assistant",
        providers=providers,
        default_provider="ollama",
    )
    flow.react(
        model="llama3.1:8b",
        system="You are a helpful personal assistant.",
        input_key="query",
        output_key="answer",
        memory=MemoryConfig(store=memory, namespace=("users", "ana"), k=5),
    )
    print("\n=== react_agent with memory injection is ready ===")
    print(flow.compile().nodes.keys())


if __name__ == "__main__":
    asyncio.run(main())
