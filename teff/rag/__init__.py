"""Retrieval-augmented generation primitives."""

from teff.rag.base import VectorStore
from teff.rag.chunker import Chunker
from teff.rag.embedder import Embedder
from teff.rag.image_tool import ImageTool
from teff.rag.pdf_tool import PDFTool
from teff.rag.tool import RAGTool
from teff.tool.registry import default_tool_registry

default_tool_registry.register(RAGTool)
default_tool_registry.register(PDFTool)
default_tool_registry.register(ImageTool)

__all__ = ["VectorStore", "Embedder", "Chunker", "RAGTool", "PDFTool", "ImageTool"]
