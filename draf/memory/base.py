"""Long-term memory: a namespace store over a vector store.

Memory is *knowledge that outlives a single run* — facts, preferences,
and history kept per user/agent and looked up semantically.  Checkpoints
snapshot one run; this module stores small cross-session facts.

The :class:`MemoryStore` is a thin namespace layer over a
:class:`~draf.rag.base.VectorStore` plus an
:class:`~draf.rag.embedder.Embedder`:

- A memory item is a ``{text, ...}`` dict; ``text`` is what gets embedded.
- Items live under hierarchical *namespaces* (tuples of strings, rooted
  on the user/session id — never the checkpoint id).  A search under a
  namespace also matches items stored in deeper sub-namespaces.
- ``put`` is an upsert keyed by ``namespace::key``; expired items (TTL)
  are skipped on read.

The same store/embedder pair powers :class:`MemoryTool` for agents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from draf.rag.base import VectorStore, match_filter
from draf.rag.embedder import Embedder

#: Metadata keys reserved by the memory layer (never leaked to the value).
_META_KEYS = ("updated_at", "expires_at")

#: Sentinel meaning "use the store-level TTL" (distinct from ``None``,
#: which means "never expires").
_UNSET = object()


def _ns_filter(namespace: tuple[str, ...]) -> dict:
    """Build a metadata filter that matches *namespace* and its sub-namespaces."""
    return {f"_ns_{i}": part for i, part in enumerate(namespace)}


def _item_id(namespace: tuple[str, ...], key: str) -> str:
    return "::".join((*namespace, key))


@dataclass
class MemoryItem:
    """A single stored memory.

    Attributes:
        key: Item key within its namespace.
        value: The stored ``{text, ...}`` dict (namespace metadata stripped).
        namespace: The namespace the item lives under.
        updated_at: Unix timestamp of the last write.
        score: Similarity score from a semantic search, or ``None`` for a
            recency-only lookup.
    """

    key: str
    value: dict
    namespace: tuple[str, ...]
    updated_at: float
    score: float | None = None


class MemoryStore:
    """Namespace-scoped semantic memory over a :class:`VectorStore`.

    Args:
        store: Backing vector store (in-memory, sqlite, qdrant, ...).
        embedder: Embedding service used for ``put`` and ``search``.
        ttl: Default seconds an item lives unless overridden at ``put``
            time.  ``None`` (default) means items never expire.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        ttl: float | None = None,
    ):
        self._store = store
        self._embedder = embedder
        self._ttl = ttl

    @property
    def store(self) -> VectorStore:
        """The backing vector store (exposed for lifecycle tools)."""
        return self._store

    async def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict,
        *,
        ttl: float | None | object = _UNSET,
    ) -> None:
        """Upsert *value* under ``namespace::key``.

        Args:
            namespace: Hierarchical path (``("users", "u1")``).
            key: Unique key within the namespace; writing to an existing
                key overwrites it.
            value: The memory dict; must contain a non-empty ``text``
                field (the part that is embedded).
            ttl: ``None`` disables expiry for this item; a number overrides
                the store-level TTL; omitted uses the store default.
        """
        text = value.get("text")
        if not text:
            raise ValueError("memory value requires a non-empty 'text' field")
        vec = await self._embedder.embed(text)
        now = time.time()
        meta: dict[str, Any] = {
            **value,
            **_ns_filter(namespace),
            "updated_at": now,
        }
        eff_ttl: float | None
        if ttl is _UNSET:
            eff_ttl = self._ttl
        elif isinstance(ttl, (int, float)):
            eff_ttl = ttl
        else:
            eff_ttl = None
        if eff_ttl is not None:
            meta["expires_at"] = now + eff_ttl
        await self._store.add([(_item_id(namespace, key), vec, meta)])

    async def get(self, namespace: tuple[str, ...], key: str) -> MemoryItem | None:
        """Return the item under ``namespace::key``, or ``None``.

        Expired items are reported as missing.
        """
        rows = await self._store.get([_item_id(namespace, key)])
        if not rows:
            return None
        _id, meta = rows[0]
        if self._is_expired(meta):
            return None
        return self._to_item(namespace, key, meta)

    async def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Remove the item under ``namespace::key`` (no-op if absent)."""
        await self._store.delete([_item_id(namespace, key)])

    async def search(
        self,
        namespace: tuple[str, ...],
        *,
        query: str | None = None,
        k: int = 10,
        filter: dict | None = None,
    ) -> list[MemoryItem]:
        """Return the *k* most relevant items under *namespace*.

        With *query*, items are ranked by semantic similarity to it; a
        ``namespace`` match also covers deeper sub-namespaces (prefix
        semantics).  Without *query*, the most recently written items are
        returned instead.

        Args:
            namespace: Namespace subtree to search.
            query: Natural-language query; when ``None`` the search falls
                back to recency order.
            k: Maximum number of results.
            filter: Extra metadata filter DSL (see
                :func:`~draf.rag.base.match_filter`).
        """
        eff_filter = {**_ns_filter(namespace), **(filter or {})}
        if query:
            qvec = await self._embedder.embed(query)
            results = await self._store.search(
                qvec,
                k=max(k, k * 2),
                filter=eff_filter,
                query_text=query,
            )
        else:
            rows = await self._store.entries(limit=100_000)
            results = [
                (_id, 0.0, meta) for _id, meta in rows if match_filter(meta, eff_filter)
            ]
            results.sort(key=lambda r: r[2].get("updated_at", 0.0), reverse=True)

        items: list[MemoryItem] = []
        for _id, score, meta in results:
            if self._is_expired(meta):
                continue
            ns, key = self._split_id(_id)
            item = self._to_item(ns, key, meta)
            item.score = score if query else None
            items.append(item)
            if len(items) >= k:
                break
        return items

    async def list(
        self, namespace: tuple[str, ...], limit: int = 100, offset: int = 0
    ) -> list[str]:
        """Return the keys stored under *namespace* (recency order)."""
        items = await self.search(namespace, k=limit + offset)
        keys = [i.key for i in items if i.key]
        return keys[offset : offset + limit]

    async def cleanup(self, *, max_age: float | None = None) -> int:
        """Delete expired items; return how many were removed.

        ``max_age`` additionally removes items whose ``updated_at`` is
        older than that many seconds.  Expired items are removed from the
        backing store (not merely hidden).
        """
        rows = await self._store.entries(limit=100_000)
        now = time.time()
        to_delete: list[str] = []
        for _id, meta in rows:
            if self._is_expired(meta):
                to_delete.append(_id)
                continue
            if max_age is not None:
                updated = meta.get("updated_at", 0.0)
                if updated and now - updated > max_age:
                    to_delete.append(_id)
        if to_delete:
            await self._store.delete(to_delete)
        return len(to_delete)

    def _is_expired(self, meta: dict) -> bool:
        expires = meta.get("expires_at")
        return isinstance(expires, (int, float)) and expires <= time.time()

    def _to_item(self, namespace: tuple[str, ...], key: str, meta: dict) -> MemoryItem:
        value = {k: v for k, v in meta.items() if not _is_meta_key(k)}
        return MemoryItem(
            key=key,
            value=value,
            namespace=namespace,
            updated_at=float(meta.get("updated_at", 0.0)),
        )

    def _split_id(self, item_id: str) -> tuple[tuple[str, ...], str]:
        parts = item_id.split("::")
        if len(parts) < 2:
            return (), parts[0]
        return tuple(parts[:-1]), parts[-1]


def _is_meta_key(key: str) -> bool:
    return key in _META_KEYS or key.startswith("_ns_")
