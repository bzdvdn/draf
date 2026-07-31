"""Vector store abstract base and similarity utilities."""

from abc import ABC, abstractmethod
import math


class VectorStore(ABC):
    """Abstract interface for vector storage and similarity search.

    Implementations must provide *add*, *search*, and *delete*.
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
        self, query: list[float], k: int = 10
    ) -> list[tuple[str, float, dict]]:
        """Search for the *k* nearest neighbours.

        Returns:
            List of ``(id, score, metadata)`` tuples sorted by score descending.
        """
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Remove vectors by ID."""
        ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
