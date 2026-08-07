"""Build a :class:`VectorStore` from a declarative ``store:`` config dict.

Shared by the ``rag`` (search) and ``rag_ingest`` (write) tools so both
read the same ``{type, ...}`` block in a workflow YAML.
"""

from __future__ import annotations

from typing import Any

from teff.rag.base import VectorStore


def store_from_config(config: dict) -> VectorStore:
    """Instantiate the store named by ``config["type"]``.

    Supported types: ``in_memory`` (default), ``sqlite``, ``chroma``,
    ``qdrant``, ``pgvector``, ``faiss``, ``lance``/``lancedb``, ``milvus``,
    ``weaviate``, ``pinecone``.  Raises ``ValueError`` for unknown types.
    """
    store_type = config.get("type", "in_memory")
    if store_type == "in_memory":
        from teff.rag.stores import InMemoryVectorStore

        return InMemoryVectorStore(dim=config.get("dim", 768))
    if store_type == "sqlite":
        from teff.rag.stores import SQLiteVectorStore

        return SQLiteVectorStore(
            path=config.get("path", "./vectors.db"),
            dim=config.get("dim"),
        )
    if store_type == "chroma":
        from teff.rag.stores import ChromaVectorStore

        return ChromaVectorStore(
            path=config.get("path", "./chroma"),
            collection=config.get("collection", "teff"),
        )
    if store_type == "qdrant":
        from teff.rag.stores import QdrantVectorStore

        return QdrantVectorStore(
            host=config.get("host", "localhost"),
            port=config.get("port", 6333),
            collection=config.get("collection", "teff"),
        )
    if store_type == "pgvector":
        from teff.rag.stores import PGVectorStore

        return PGVectorStore(
            dsn=config.get("dsn", ""),
            table=config.get("table", "teff_vectors"),
        )
    if store_type == "faiss":
        from teff.rag.stores import FAISSVectorStore

        return FAISSVectorStore(
            dim=config.get("dim", 1536),
            path=config.get("path"),
        )
    if store_type in ("lance", "lancedb"):
        from teff.rag.stores import LanceVectorStore

        return LanceVectorStore(
            path=config.get("path", "./lance"),
            table=config.get("table", "vectors"),
            dim=config.get("dim"),
        )
    if store_type == "milvus":
        from teff.rag.stores import MilvusVectorStore

        return MilvusVectorStore(
            uri=config.get("uri", "./milvus.db"),
            token=config.get("token", ""),
            collection=config.get("collection", "teff"),
            dim=config.get("dim"),
        )
    if store_type == "weaviate":
        from teff.rag.stores import WeaviateVectorStore

        return WeaviateVectorStore(
            collection=config.get("collection", "teff"),
            embedded=bool(config.get("embedded", False)),
            host=config.get("host", "localhost"),
            http_port=config.get("http_port", 8080),
            http_secure=bool(config.get("http_secure", False)),
            grpc_port=config.get("grpc_port", 50051),
            grpc_secure=bool(config.get("grpc_secure", False)),
            api_key=config.get("api_key", ""),
            headers=config.get("headers"),
            dim=config.get("dim"),
        )
    if store_type == "pinecone":
        from teff.rag.stores import PineconeVectorStore

        return PineconeVectorStore(
            index_name=config.get("index_name", "teff"),
            api_key=config.get("api_key", ""),
            host=config.get("host", ""),
            namespace=config.get("namespace", ""),
            dim=config.get("dim"),
        )
    msg = f"unsupported store type: {store_type}"
    raise ValueError(msg)


def _abs_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Return *config* with ``store.path``/``file``/``path`` made absolute."""
    import os

    result = dict(config)

    def _abs(item: dict) -> dict:
        item = dict(item)
        for key in ("file", "path"):
            if key in item and not os.path.isabs(item[key]):
                item[key] = os.path.abspath(item[key])
        return item

    store = config.get("store")
    if isinstance(store, dict):
        result["store"] = _abs(store)
    return result


__all__ = ["store_from_config", "_abs_paths"]
