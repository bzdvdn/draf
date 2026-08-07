"""Real RAG: local Ollama embeddings + vector search + LLM answer.

A small knowledge base about "Teff" is embedded with ``nomic-embed-text``
into an in-memory vector store. The chat model has never seen this
content, so it MUST retrieve the relevant snippet via the ``rag`` tool
before it can answer.

Requires Ollama running locally with:
    llama3.1:8b          (chat model)
    nomic-embed-text     (embedding model)

Usage:
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
    python examples/rag_search/run.py
"""

import asyncio

from teff.flow import Flow
from teff.provider import ProviderRegistry
from teff.rag import Embedder, RAGTool
from teff.rag.stores import InMemoryVectorStore

# Fictional knowledge base — the model cannot know any of this.
DOCUMENTS = [
    (
        "Teff is a Python framework for building AI agent workflows as data. "
        "Workflows are graphs: nodes transform state, edges carry conditions, "
        "and the graph owns all behaviour.",
        {"id": "doc_1", "topic": "intro"},
    ),
    (
        "The Teff constitution: the graph owns behaviour, nodes transform "
        "state, conditions live on edges, and a workflow is YAML data that "
        "compiles into a graph.",
        {"id": "doc_2", "topic": "constitution"},
    ),
    (
        "Teff ships built-in nodes: Transform, LLM, ReActAgent, ToolExec, "
        "and Retry. Tools and RAG primitives are first-class citizens.",
        {"id": "doc_3", "topic": "nodes"},
    ),
    (
        "The official mascot of Teff is a river otter named Flux. "
        "Flux rides the graph edges and never gets lost in a cycle.",
        {"id": "doc_4", "topic": "mascot"},
    ),
]


async def main():
    store = InMemoryVectorStore(dim=768)
    embedder = Embedder(provider="ollama", model="nomic-embed-text")
    rag = RAGTool(store=store, embedder=embedder)
    await rag.add_documents(DOCUMENTS)

    flow = Flow(
        "rag_agent",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
    )
    flow.react(
        model="llama3.1:8b",
        system=(
            "You are a RAG assistant over the Teff knowledge base. "
            "Always call the 'rag' tool to search before answering, and "
            "answer strictly from the retrieved snippets."
        ),
        input_key="query",
        output_key="answer",
    )

    graph = flow.compile()
    query = "What is the mascot of Teff, and what does it ride?"
    result = await graph.run(
        state={"query": query},
        tools=[rag],
        max_iterations=10,
    )
    print("Query:", result.get("query"))
    print("Answer:", result.get("answer"))
    print(
        "Tool calls:",
        sum(1 for m in result.get("messages", []) if m.get("role") == "tool"),
    )


if __name__ == "__main__":
    asyncio.run(main())
