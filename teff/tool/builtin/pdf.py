"""PDF read tool — extract text from a PDF file."""

from teff.tool.tool import Tool


class PDFReadTool(Tool):
    """Extract text from a PDF file.

    Requires ``pypdf`` (from ``teff[tools]``). Returns the text of each
    page, optionally limited to *max_chars* characters.

    Args:
        config: Optional dict. Currently unused, kept for config parity.
    """

    name = "read_pdf"
    description = "Extract text from a PDF file"

    def __init__(self, config: dict | None = None):
        pass

    def run(self, path: str = "", max_chars: int = 50000) -> str:  # type: ignore[override]
        if not path:
            raise ValueError("path is required")
        try:
            from pypdf import PdfReader
        except ImportError as e:
            msg = "read_pdf requires 'pypdf' (pip install teff[tools])"
            raise ImportError(msg) from e

        reader = PdfReader(path)
        parts: list[str] = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            parts.append(f"--- page {i} ---\n{text}")
        result = "\n".join(parts).strip()
        if not result:
            return "no text found in pdf"
        return result[:max_chars]
