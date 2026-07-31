"""Retrieval-augmented generation primitives."""

from draf.rag.base import VectorStore
from draf.rag.embedder import Embedder
from draf.rag.chunker import Chunker
from draf.rag.tool import RAGTool
from draf.tool.registry import default_tool_registry

default_tool_registry.register(RAGTool)

__all__ = ["VectorStore", "Embedder", "Chunker", "RAGTool"]
