"""RAG agent on any store, built with the Python Flow API.

Mirrors the YAML workflows next to it: same agent, same documents, but
assembled from code instead of loaded from workflow.yaml.

Usage:
    uv run python examples/rag_stores/flow.py [store]

Supported stores: in_memory, sqlite, chroma, qdrant, pgvector
(default: in_memory). See each store's README for install steps.
"""

import asyncio
import os
import sys

from draf import set_defaults
from draf.flow import Flow
from draf.rag import RAGTool

set_defaults(provider="ollama")

_HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_CSV = os.path.join(_HERE, "docs.csv")

STORE_CONFIGS = {
    "in_memory": {"type": "in_memory", "dim": 768},
    "sqlite": {
        "type": "sqlite",
        "path": os.path.join(_HERE, "sqlite", "vectors.db"),
        "dim": 768,
    },
    "chroma": {
        "type": "chroma",
        "path": os.path.join(_HERE, "chroma", "chroma_db"),
        "collection": "draf",
    },
    "qdrant": {
        "type": "qdrant",
        "host": "localhost",
        "port": 6333,
        "collection": "draf",
    },
    "pgvector": {
        "type": "pgvector",
        "dsn": "postgresql://postgres:postgres@localhost:5433/postgres",
        "table": "draf_vectors",
    },
}


async def main(store: str):
    if store not in STORE_CONFIGS:
        print(f"unknown store: {store}")
        print("choose one of:", ", ".join(STORE_CONFIGS))
        raise SystemExit(1)

    rag = RAGTool(
        config={
            "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
            "store": STORE_CONFIGS[store],
            "documents": [{"type": "csv", "path": DOCS_CSV}],
        }
    )

    flow = Flow(f"rag_{store}")
    flow.react(
        model="llama3.1:8b",
        system=(
            "You are a RAG assistant over the Draf knowledge base. Always call "
            "the 'rag' tool to search before answering, and answer strictly "
            "from the retrieved snippets."
        ),
        input_key="query",
        output_key="answer",
    )

    graph = flow.compile()
    result = await graph.run(
        state={"query": "What is the mascot of Draf, and what does it ride?"},
        tools=[rag],
        max_iterations=10,
    )
    tool_calls = sum(1 for m in result.get("messages", []) if m.get("role") == "tool")
    print("Store:", store)
    print("Query:", result.get("query"))
    print("Answer:", result.get("answer"))
    print("Tool calls:", tool_calls)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "in_memory"))
