"""PDF extraction tool — turn a PDF into per-page text for RAG."""

from teff.rag.tool import load_documents_pdf
from teff.tool.tool import Tool


class PDFTool(Tool):
    """Extract text from a PDF file, one section per page.

    Text-based PDFs are read with ``pypdf`` (extra ``teff[rag-pdf]``).
    Scanned / image-only pages yield no text — feed those pages to
    :class:`~teff.rag.image_tool.ImageTool` instead.

    Args:
        config: Optional dict.  ``max_chars`` sets the default output
            limit (default 50000).  Kept for config parity with other
            tools in a workflow ``tools:`` block.
    """

    name = "pdf"
    description = "Extract text from a PDF file, one section per page"

    def __init__(self, config: dict | None = None):
        self.max_chars: int = 50000
        if isinstance(config, dict):
            self.max_chars = int(config.get("max_chars", 50000))

    def run(self, path: str, max_chars: int | None = None) -> str:  # type: ignore[override]
        """Return the PDF text as ``--- page N ---`` sections."""
        if not path:
            raise ValueError("path is required")
        docs = load_documents_pdf(path)
        if not docs:
            return "no text found in pdf"
        parts = [f"--- page {meta['page']} ---\n{text}" for text, meta in docs]
        result = "\n".join(parts)
        limit = max_chars if max_chars is not None else self.max_chars
        if limit and limit > 0:
            return result[:limit]
        return result
