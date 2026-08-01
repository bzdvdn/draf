"""Vector store abstract base and similarity utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math


def match_filter(metadata: dict, filter: dict | None) -> bool:
    """Return ``True`` if *metadata* satisfies the filter DSL.

    A filter is a dict of field -> condition pairs:

    - scalar value — equality: ``{"category": "news"}``
    - list value — membership: ``{"category": ["news", "tech"]}``; when the
      stored field value is itself a list, any shared element matches
    - ``"$and"`` / ``"$or"`` keys combine sub-filters (lists of filters).

    A missing field never matches a scalar or list condition.
    """
    if not filter:
        return True
    for key, cond in filter.items():
        if key == "$and":
            if not all(match_filter(metadata, sub) for sub in cond):
                return False
        elif key == "$or":
            if not any(match_filter(metadata, sub) for sub in cond):
                return False
        else:
            value = metadata.get(key)
            if isinstance(cond, list):
                if isinstance(value, list):
                    if not set(value).intersection(cond):
                        return False
                elif value not in cond:
                    return False
            elif value != cond:
                return False
    return True


def blend_scores(cosine: float, text: str, query: str, alpha: float = 0.4) -> float:
    """Blend a cosine score with a lexical overlap score.

    Used for hybrid search: ``alpha`` weights the lexical (keyword) share,
    ``1 - alpha`` the semantic (cosine) share.  When the query has no
    alphabetic tokens the cosine score is returned unchanged.
    """
    tokens = {t for t in query.lower().split() if t}
    if not tokens:
        return cosine
    ltext = text.lower()
    hits = sum(1 for t in tokens if t in ltext)
    lexical = hits / len(tokens)
    return (1 - alpha) * cosine + alpha * lexical


def finalize_results(
    candidates: list[tuple[str, float, dict]],
    k: int,
    filter: dict | None = None,
    hybrid: bool = False,
    query_text: str | None = None,
) -> list[tuple[str, float, dict]]:
    """Apply the filter, optional hybrid blending, rank, and cap at *k*.

    Stores that retrieve candidates (e.g. brute-force scans) use this to
    post-process results consistently: drop non-matching metadata, blend
    a lexical score for ``hybrid`` search, sort descending, and trim.
    """
    out: list[tuple[str, float, dict]] = []
    for vid, score, meta in candidates:
        if not match_filter(meta, filter):
            continue
        if hybrid and query_text:
            score = blend_scores(score, meta.get("text", ""), query_text)
        out.append((vid, score, meta))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:k]


class VectorStore(ABC):
    """Abstract interface for vector storage and similarity search.

    Implementations must provide *add*, *search*, and *delete*.  The
    extended operations (*count*, *list*, *get*, *update_metadata*,
    *clear*) default to :class:`NotImplementedError` and are implemented
    by the built-in stores.
    """

    @abstractmethod
    async def add(self, vectors: list[tuple[str, list[float], dict]]) -> None:
        """Store vectors with IDs and metadata.

        Args:
            vectors: List of ``(id, embedding, metadata)`` tuples.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: list[float],
        k: int = 10,
        filter: dict | None = None,
        hybrid: bool = False,
        query_text: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Search for the *k* nearest neighbours.

        Args:
            query: Query embedding.
            k: Maximum number of results.
            filter: Optional metadata filter DSL (see :func:`match_filter`).
            hybrid: When ``True``, blend a lexical keyword score with the
                cosine score (stores that support it; others ignore it).
            query_text: Original query text, required for ``hybrid``.

        Returns:
            List of ``(id, score, metadata)`` tuples sorted by score
            descending.  Scores are similarity-like (higher = more similar).
        """
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Remove vectors by ID."""
        ...

    async def count(self) -> int:
        """Return the number of stored vectors."""
        raise NotImplementedError(f"{type(self).__name__} does not implement count()")

    async def entries(
        self, limit: int = 100, offset: int = 0
    ) -> list[tuple[str, dict]]:
        """Return ``(id, metadata)`` pairs with pagination."""
        raise NotImplementedError(f"{type(self).__name__} does not implement list()")

    async def get(self, ids: list[str]) -> list[tuple[str, dict]]:
        """Return ``(id, metadata)`` pairs for existing IDs."""
        raise NotImplementedError(f"{type(self).__name__} does not implement get()")

    async def update_metadata(self, id: str, metadata: dict) -> None:
        """Merge *metadata* into the metadata of an existing ID."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement update_metadata()"
        )

    async def clear(self) -> None:
        """Remove all stored vectors."""
        raise NotImplementedError(f"{type(self).__name__} does not implement clear()")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
