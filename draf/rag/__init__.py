"""Retrieval-augmented generation primitives."""

from draf.rag.base import VectorStore
from draf.rag.chunker import Chunker
from draf.rag.embedder import Embedder
from draf.rag.image_tool import ImageTool
from draf.rag.pdf_tool import PDFTool
from draf.rag.tool import RAGTool
from draf.tool.registry import default_tool_registry

default_tool_registry.register(RAGTool)
default_tool_registry.register(PDFTool)
default_tool_registry.register(ImageTool)

__all__ = ["VectorStore", "Embedder", "Chunker", "RAGTool", "PDFTool", "ImageTool"]
