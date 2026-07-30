"""Vector store implementations."""

from draf.rag.stores.memory import InMemoryVectorStore

try:
    from draf.rag.stores.qdrant import QdrantVectorStore
except ImportError:
    class QdrantVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install qdrant-client for QdrantVectorStore")

try:
    from draf.rag.stores.chroma import ChromaVectorStore
except ImportError:
    class ChromaVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install chromadb for ChromaVectorStore")

try:
    from draf.rag.stores.pgvector import PGVectorStore
except ImportError:
    class PGVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install asyncpg + sqlalchemy + pgvector for PGVectorStore")

__all__ = ["InMemoryVectorStore", "QdrantVectorStore", "ChromaVectorStore", "PGVectorStore"]
