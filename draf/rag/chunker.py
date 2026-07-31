"""Text chunking strategies for RAG document splitting."""

from dataclasses import dataclass


@dataclass
class Chunker:
    """Split text into chunks for embedding and retrieval.

    Supports three strategies:

    - ``token`` — Split on whitespace into token windows.
    - ``sentence`` — Split on sentence boundaries.
    - ``fixed`` — Split by fixed character count.

    Attributes:
        strategy: Chunking strategy name.
        chunk_size: Target chunk size (tokens, sentences, or chars).
        overlap: Overlap between consecutive chunks.
    """

    strategy: str = "token"
    chunk_size: int = 500
    overlap: int = 50

    def chunk(self, text: str) -> list[str]:
        """Split *text* into chunks using the configured strategy."""
        if self.strategy == "token":
            return self._chunk_token(text)
        if self.strategy == "sentence":
            return self._chunk_sentence(text)
        if self.strategy == "fixed":
            return self._chunk_fixed(text)
        raise ValueError(f"unknown chunk strategy: {self.strategy}")

    def _chunk_token(self, text: str) -> list[str]:
        tokens = text.split()
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk = " ".join(tokens[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
            if self.chunk_size - self.overlap <= 0:
                break
        return chunks

    def _chunk_sentence(self, text: str) -> list[str]:
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = []
        for s in sentences:
            current.append(s)
            if len(current) >= self.chunk_size:
                chunks.append(" ".join(current))
                overlap_start = max(0, len(current) - self.overlap)
                current = current[overlap_start:]
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _chunk_fixed(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
            if self.chunk_size - self.overlap <= 0:
                break
        return chunks
