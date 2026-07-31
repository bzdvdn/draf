"""Parallel RAG from two sources, report file, and LLM review.

Demonstrates the full pipeline the user described:

    1. Two RAG searches run in PARALLEL branches — each queries its own
       knowledge base and an LLM drafts a per-source summary.
    2. ``converge`` rejoins the branches; a third LLM writes a combined
       report from both summaries.
    3. The report is saved to a file.
    4. A final LLM REVIEWS the report and returns a verdict.

Requires Ollama running locally with:
    llama3.1:8b          (chat model)
    nomic-embed-text     (embedding model)

Usage:
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
    python examples/parallel/rag_report.py
"""

import asyncio
import os

from draf import set_defaults
from draf.flow import Flow
from draf.node import LLM, Node
from draf.rag import Embedder, RAGTool
from draf.rag.stores import InMemoryVectorStore
from draf.trace import RunTracer

set_defaults(provider="ollama")

REPORT_PATH = os.path.join(os.path.dirname(__file__), "rag_report.md")

# Two fictional knowledge bases about "Draf".
KB_CONSTITUTION = [
    (
        "The Draf constitution: the graph owns behaviour, nodes transform "
        "state, conditions live on edges, and a workflow is YAML data that "
        "compiles into a graph.",
        {"id": "const_1", "topic": "constitution"},
    ),
    (
        "Draf state is a flat dict. Nodes receive state and return state. "
        "Branching decisions read from state keys, never from hidden logic.",
        {"id": "const_2", "topic": "state"},
    ),
]

KB_RAG = [
    (
        "Draf RAG combines a vector store, an embedder, and a chunker into "
        "a single 'rag' tool. Stores: in-memory, sqlite, chroma, qdrant, "
        "pgvector.",
        {"id": "rag_1", "topic": "rag"},
    ),
    (
        "Draf runs parallel branches concurrently with asyncio.gather. "
        "Each branch reads an isolated copy of state; per-key reducers "
        "merge the results back.",
        {"id": "rag_2", "topic": "parallel"},
    ),
]


class RagSearch(Node):
    """Search a configured RAG tool and store the formatted results."""

    type = "rag_search"

    def __init__(self, tool_name: str, query: str, output_key: str, **kwargs):
        super().__init__(
            tool_name=tool_name, query=query, output_key=output_key, **kwargs
        )

    async def execute(self, ctx, state):
        tool = ctx.tool(self.config["tool_name"])
        results = await tool.arun(query=self.config["query"])
        return {self.config["output_key"]: results}


class Combine(Node):
    """Concatenate the two per-source summaries into one prompt block."""

    type = "combine"

    async def execute(self, ctx, state):
        return {
            "combined": (
                "Source A (constitution):\n"
                f"{state.get('summary_a', '')}\n\n"
                "Source B (rag):\n"
                f"{state.get('summary_b', '')}"
            )
        }


class WriteReport(Node):
    """Write the report text to a file."""

    type = "write_report"

    def __init__(self, path: str, **kwargs):
        super().__init__(path=path, **kwargs)

    async def execute(self, ctx, state):
        content = str(state.get("report", ""))
        with open(self.config["path"], "w") as f:
            f.write(content)
        return {"file_path": self.config["path"], "bytes": len(content)}


SYNTH_SYSTEM = (
    "You are a research analyst. Summarize the retrieved context below in "
    "2-3 sentences, keeping all concrete facts. Output only the summary."
)

REPORT_SYSTEM = (
    "You are a report writer. Write a concise markdown report titled "
    "'Draf at a glance' that merges BOTH sections below into one coherent "
    "overview. The report MUST explicitly mention these three topics by "
    "name: (1) the Draf constitution, (2) parallel branches, (3) the RAG "
    "tool. Preserve concrete facts. Use 3-5 bullet points."
)

REVIEW_SYSTEM = (
    "You are a strict reviewer. Read the report below. Reply on the first "
    "line with 'VERDICT: pass' if the report explicitly mentions all three "
    "of these topics: (1) the Draf constitution, (2) parallel branches, "
    "(3) the RAG tool. Otherwise reply 'VERDICT: fail'. Then add one short "
    "sentence of feedback."
)


async def main():
    # Two independent RAG tools, each with its own vector store.
    # ``graph.run`` keys tools by name, so give them distinct names.
    embedder = Embedder(provider="ollama", model="nomic-embed-text")

    rag_a = RAGTool(
        store=InMemoryVectorStore(dim=768),
        embedder=embedder,
        name="rag_a",
    )
    await rag_a.add_documents(KB_CONSTITUTION)

    rag_b = RAGTool(
        store=InMemoryVectorStore(dim=768),
        embedder=embedder,
        name="rag_b",
    )
    await rag_b.add_documents(KB_RAG)

    flow = Flow("parallel-rag-report")
    flow.parallel(
        [
            RagSearch(
                "rag_a", "What does the constitution say about state?", "context_a"
            ),
            LLM(
                model="llama3.1:8b",
                system=SYNTH_SYSTEM,
                input_key="context_a",
                output_key="summary_a",
            ),
        ],
        [
            RagSearch(
                "rag_b", "How does parallel execution work in Draf?", "context_b"
            ),
            LLM(
                model="llama3.1:8b",
                system=SYNTH_SYSTEM,
                input_key="context_b",
                output_key="summary_b",
            ),
        ],
    ).converge(Combine()).step(
        LLM(
            model="llama3.1:8b",
            system=REPORT_SYSTEM,
            input_key="combined",
            output_key="report",
        )
    )
    flow.step(WriteReport(path=REPORT_PATH)).step(
        LLM(
            model="llama3.1:8b",
            system=REVIEW_SYSTEM,
            input_key="report",
            output_key="review",
        )
    )

    graph = flow.compile()
    tracer = RunTracer()
    result = await graph.run(
        state={},
        tools=[rag_a, rag_b],
        max_iterations=50,
        tracer=tracer,
    )

    print("=== Review ===")
    print(result.get("review"))
    print()
    print("=== Report saved ===")
    print(f"{result.get('file_path')} ({result.get('bytes')} bytes)")
    print()
    print("=== Trace summary ===")
    summary = tracer.summary()
    print(
        f"status: {summary.status}  total: {summary.total_ms:.1f}ms  "
        f"llm_calls: {summary.llm_calls}  tokens: {summary.tokens.total}"
    )
    for node_id in sorted(summary.nodes):
        st = summary.nodes[node_id]
        print(f"  {node_id:20s} runs={st.runs} errors={st.errors}")


if __name__ == "__main__":
    asyncio.run(main())
