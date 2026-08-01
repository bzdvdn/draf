"""Vector store implementations."""

from draf.rag.stores.memory import InMemoryVectorStore
from draf.rag.stores.sqlite import SQLiteVectorStore

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
            raise ImportError(
                "install asyncpg + sqlalchemy + pgvector for PGVectorStore"
            )


try:
    from draf.rag.stores.faiss import FAISSVectorStore
except ImportError:

    class FAISSVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install faiss-cpu for FAISSVectorStore")


try:
    from draf.rag.stores.lance import LanceVectorStore
except ImportError:

    class LanceVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install lancedb for LanceVectorStore")


try:
    from draf.rag.stores.milvus import MilvusVectorStore
except ImportError:

    class MilvusVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install pymilvus for MilvusVectorStore")


try:
    from draf.rag.stores.weaviate import WeaviateVectorStore
except ImportError:

    class WeaviateVectorStore:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("install weaviate-client for WeaviateVectorStore")


try:
    from draf.rag.stores.pinecone import PineconeVectorStore
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
