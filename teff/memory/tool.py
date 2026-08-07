"""Agent-facing tool for long-term memory (remember / recall / forget)."""

from __future__ import annotations

import uuid

from teff.memory.base import MemoryStore
from teff.rag.base import VectorStore
from teff.rag.embedder import Embedder, embedder_from_config
from teff.tool.tool import Tool


class MemoryTool(Tool):
    """Tool that lets an agent read and write long-term memory.

    Usage::

        memory = MemoryTool(
            store=SQLiteVectorStore(path="./memory.db", dim=768),
            embedder=Embedder(provider="ollama", model="nomic-embed-text"),
            namespace=("users", "u1"),
        )
        await memory.arun(action="remember", text="prefers email over Slack")
        result = await memory.arun(action="recall", query="how to reach them?")

    Actions (passed as ``action``):

    - ``remember`` — upsert a fact (``text`` plus optional ``metadata``).
      When ``similarity_threshold`` is set and a semantically close item
      already exists in the namespace, the new text overwrites that item
      instead of creating a duplicate.
    - ``recall`` — return top-*k* memories for a ``query`` (or the most
      recent if no query is given), formatted for a prompt.
    - ``forget`` — delete the memory at ``key``.
    - ``list`` — enumerate stored keys.

    Can be built from a config dict (e.g. a ``tools:`` entry in a
    workflow YAML)::

        {
          "name": "memory",
          "store": {"type": "sqlite", "path": "./memory.db", "dim": 768},
          "embedder": {"provider": "ollama", "model": "nomic-embed-text"},
          "namespace": ["users", "${USER_ID}"],
          "default_k": 5,
          "similarity_threshold": 0.6,
        }

    Supported store types match ``RAGTool``: ``in_memory`` (default),
    ``sqlite``, ``chroma``, ``qdrant``, ``pgvector``, ``faiss``, ``lance``,
    ``milvus``, ``weaviate``, ``pinecone``.
    """

    name = "memory"
    description = (
        "Long-term memory: remember facts, recall relevant memories, "
        "forget, and list what is stored."
    )

    def __init__(
        self,
        config: dict | None = None,
        *,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        namespace: tuple[str, ...] | list[str] = (),
        default_k: int = 5,
        similarity_threshold: float | None = None,
        ttl: float | None = None,
    ):
        self._memory: MemoryStore | None = None
        self._namespace = tuple(namespace)
        self._default_k = default_k
        self._threshold = similarity_threshold
        self._ttl = ttl
        if isinstance(config, dict):
            self._apply_config(config)
        elif store is not None and embedder is not None:
            self.memory = MemoryStore(store=store, embedder=embedder, ttl=ttl)

    @property
    def memory(self) -> MemoryStore:
        if self._memory is None:
            raise RuntimeError("memory store not initialised")
        return self._memory

    @memory.setter
    def memory(self, value: MemoryStore) -> None:
        self._memory = value

    def _apply_config(self, config: dict) -> None:
        self.memory = memory_from_config(config, default_ttl=self._ttl)
        ns = config.get("namespace")
        if ns:
            self._namespace = tuple(str(part) for part in ns)
        if config.get("default_k") is not None:
            self._default_k = int(config["default_k"])
        if config.get("similarity_threshold") is not None:
            self._threshold = float(config["similarity_threshold"])

    async def arun(  # type: ignore[override]
        self,
        action: str = "recall",
        key: str = "",
        text: str = "",
        value: dict | None = None,
        query: str = "",
        metadata: dict | None = None,
        k: int | None = None,
    ) -> str:
        """Run a memory operation and return a human-readable result.

        The namespace is fixed at construction time and can never be
        overridden by the caller — an agent cannot address another owner's
        memories by passing a namespace.  Per-owner isolation is achieved by
        building one tool per owner (``namespace=("users", owner)``).
        """
        ns = self._namespace
        eff_k = int(k) if k is not None else self._default_k
        mem = self.memory

        if action == "remember":
            return await self._remember(ns, key, text, value, metadata)
        if action == "recall":
            items = await mem.search(ns, query=query or None, k=eff_k)
            return _format_recall(items)
        if action == "forget":
            if not key:
                return "forget requires a `key`"
            await mem.delete(ns, key)
            return f"forgotten {key!r}"
        if action == "list":
            keys = await mem.list(ns, limit=1000)
            return "\n".join(keys) if keys else "(no memories)"
        raise ValueError(f"unknown memory action: {action!r}")

    async def _remember(
        self,
        ns: tuple[str, ...],
        key: str,
        text: str,
        value: dict | None,
        metadata: dict | None,
    ) -> str:
        if value is None:
            if not text:
                return "remember requires `text`"
            value = {"text": text, **(metadata or {})}
        elif "text" not in value:
            return "remember `value` requires a 'text' field"

        if self._threshold is not None:
            similar = await self.memory.search(ns, query=value["text"], k=1)
            if (
                similar
                and similar[0].score is not None
                and similar[0].score >= self._threshold
            ):
                key = similar[0].key

        final_key = key or uuid.uuid4().hex[:12]
        await self.memory.put(ns, final_key, value)
        return f"remembered {final_key!r}"


def memory_from_config(
    config: dict,
    *,
    default_ttl: float | None = None,
    providers=None,
    default_provider: str | None = None,
) -> MemoryStore:
    """Build a :class:`MemoryStore` from a config dict.

    Mirrors ``RAGTool`` / ``MemoryTool`` config: ``{"store": {...},
    "embedder": {...}, "ttl": ...}``.  Used by workflow YAML loading and
    by :class:`~teff.node.agent.ReActAgent` context injection.

    *providers* (a registry) lets the embedder inherit a provider's
    ``base_url`` / ``api_key_env`` when the config does not set them
    explicitly.
    """
    embedder = embedder_from_config(
        config,
        providers=providers,
        default_provider=default_provider,
    )
    store = _build_store(config.get("store") or {})
    return MemoryStore(
        store=store,
        embedder=embedder,
        ttl=config.get("ttl", default_ttl),
    )


def _format_recall(items) -> str:
    if not items:
        return "(no memories)"
    lines = []
    for i, item in enumerate(items, start=1):
        score = f" (score: {item.score:.3f})" if item.score is not None else ""
        lines.append(f"[{i}]{score} {item.value.get('text', '')}")
    return "\n".join(lines)


def _build_store(cfg: dict) -> VectorStore:
    """Build a :class:`VectorStore` from a config dict (mirrors RAGTool)."""
    store_type = cfg.get("type", "in_memory")
    if store_type == "in_memory":
        from teff.rag.stores import InMemoryVectorStore

        return InMemoryVectorStore(dim=cfg.get("dim", 768))
    if store_type == "sqlite":
        from teff.rag.stores import SQLiteVectorStore

        return SQLiteVectorStore(
            path=cfg.get("path", "./memory.db"),
            dim=cfg.get("dim"),
        )
    if store_type == "chroma":
        from teff.rag.stores import ChromaVectorStore

        return ChromaVectorStore(
            path=cfg.get("path", "./chroma"),
            collection=cfg.get("collection", "teff"),
        )
    if store_type == "qdrant":
        from teff.rag.stores import QdrantVectorStore

        return QdrantVectorStore(
            host=cfg.get("host", "localhost"),
            port=cfg.get("port", 6333),
            collection=cfg.get("collection", "teff"),
        )
    if store_type == "pgvector":
        from teff.rag.stores import PGVectorStore

        return PGVectorStore(
            dsn=cfg.get("dsn", ""),
            table=cfg.get("table", "teff_vectors"),
            dim=cfg.get("dim", 768),
        )
    if store_type == "faiss":
        from teff.rag.stores import FAISSVectorStore

        return FAISSVectorStore(
            dim=cfg.get("dim", 1536),
            path=cfg.get("path"),
        )
    if store_type in ("lance", "lancedb"):
        from teff.rag.stores import LanceVectorStore

        return LanceVectorStore(
            path=cfg.get("path", "./lance"),
            table=cfg.get("table", "vectors"),
            dim=cfg.get("dim"),
        )
    if store_type == "milvus":
        from teff.rag.stores import MilvusVectorStore

        return MilvusVectorStore(
            uri=cfg.get("uri", "./milvus.db"),
            token=cfg.get("token", ""),
            collection=cfg.get("collection", "teff"),
            dim=cfg.get("dim"),
        )
    if store_type == "weaviate":
        from teff.rag.stores import WeaviateVectorStore

        return WeaviateVectorStore(
            collection=cfg.get("collection", "teff"),
            embedded=bool(cfg.get("embedded", False)),
            host=cfg.get("host", "localhost"),
            http_port=cfg.get("http_port", 8080),
            grpc_port=cfg.get("grpc_port", 50051),
            api_key=cfg.get("api_key", ""),
            dim=cfg.get("dim"),
        )
    if store_type == "pinecone":
        from teff.rag.stores import PineconeVectorStore

        return PineconeVectorStore(
            index_name=cfg.get("index_name", "teff"),
            api_key=cfg.get("api_key", ""),
            host=cfg.get("host", ""),
            namespace=cfg.get("namespace", ""),
            dim=cfg.get("dim"),
        )
    raise ValueError(f"unsupported store type: {store_type}")
