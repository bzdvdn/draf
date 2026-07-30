"""RAG tool — retrieve context from a vector store for LLM use."""

from draf.tool.tool import Tool
from draf.rag.base import VectorStore
from draf.rag.embedder import Embedder
from draf.rag.chunker import Chunker


class RAGTool(Tool):
    """Tool that searches a vector store and returns ranked results.

    Usage::

        store = InMemoryVectorStore(dim=768)
        embedder = Embedder(provider="openai")
        tool = RAGTool(store, embedder)
        await tool.add_document("some long text")
        result = await tool.arun(query="find this")
    """

    name = "rag"
    description = "Search documents using RAG"

    def __init__(self, store: VectorStore, embedder: Embedder, chunker: Chunker | None = None):
        self.store = store
        self.embedder = embedder
        self.chunker = chunker or Chunker()

    async def arun(self, query: str = "", k: int = 5) -> str:  # type: ignore[override]
        """Search documents and return formatted results."""
        query_vec = await self.embedder.embed(query)
        results = await self.store.search(query_vec, k=k)
        if not results:
            return ""
        context_parts = []
        for i, (doc_id, score, meta) in enumerate(results):
            text = meta.get("text", doc_id)
            context_parts.append(f"[{i + 1}] (score: {score:.3f}) {text}")
        return "\n\n".join(context_parts)

    async def add_document(self, text: str, metadata: dict | None = None) -> None:
        """Chunk, embed, and store a document."""
        metadata = metadata or {}
        chunks = self.chunker.chunk(text)
        vectors = []
        embeddings = await self.embedder.embed_many(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            doc_id = f"{metadata.get('id', 'doc')}_{i}" if metadata.get("id") else f"chunk_{i}"
            meta = {**metadata, "text": chunk, "chunk_index": i}
            vectors.append((doc_id, vec, meta))
        await self.store.add(vectors)

    async def add_documents(self, docs: list[tuple[str, dict]]) -> None:
        """Add multiple documents at once."""
        for text, meta in docs:
            await self.add_document(text, meta)
