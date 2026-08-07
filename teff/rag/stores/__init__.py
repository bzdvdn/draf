"""Vector store implementations."""

from teff.rag.stores.memory import InMemoryVectorStore
from teff.rag.stores.sqlite import SQLiteVectorStore

try:
    from teff.rag.stores.qdrant import QdrantVectorStore
except ImportError:

    class QdrantVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install qdrant-client for QdrantVectorStore")


try:
    from teff.rag.stores.chroma import ChromaVectorStore
except ImportError:

    class ChromaVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install chromadb for ChromaVectorStore")


try:
    from teff.rag.stores.pgvector import PGVectorStore
except ImportError:

    class PGVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError(
                "install asyncpg + sqlalchemy + pgvector for PGVectorStore"
            )


try:
    from teff.rag.stores.faiss import FAISSVectorStore
except ImportError:

    class FAISSVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install faiss-cpu for FAISSVectorStore")


try:
    from teff.rag.stores.lance import LanceVectorStore
except ImportError:

    class LanceVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install lancedb for LanceVectorStore")


try:
    from teff.rag.stores.milvus import MilvusVectorStore
except ImportError:

    class MilvusVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install pymilvus for MilvusVectorStore")


try:
    from teff.rag.stores.weaviate import WeaviateVectorStore
except ImportError:

    class WeaviateVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install weaviate-client for WeaviateVectorStore")


try:
    from teff.rag.stores.pinecone import PineconeVectorStore
except ImportError:

    class PineconeVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install pinecone for PineconeVectorStore")


__all__ = [
    "InMemoryVectorStore",
    "SQLiteVectorStore",
    "QdrantVectorStore",
    "ChromaVectorStore",
    "PGVectorStore",
    "FAISSVectorStore",
    "LanceVectorStore",
    "MilvusVectorStore",
    "WeaviateVectorStore",
    "PineconeVectorStore",
]
