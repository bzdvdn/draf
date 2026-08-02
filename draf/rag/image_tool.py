"""Image extraction tool — OCR via an OpenAI-compatible vision model."""

import base64
import os

import httpx

from draf.tool.tool import Tool

_VISION_DEFAULTS = {
    "ollama": ("http://localhost:11434/v1", "", "llava"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
}


def _guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "application/octet-stream")


class ImageTool(Tool):
    """Extract text from an image with an OpenAI-compatible vision model.

    The image is base64-encoded and sent to a chat-completions vision
    endpoint (default ``ollama``/``llava``; ``openai``/``gpt-4o-mini`` is
    the API alternative).  Use it for OCR on screenshots, scans, charts
    and photos.

    Args:
        config: Optional dict with ``provider``, ``model``, ``base_url``,
            ``api_key`` (falls back to the ``<PROVIDER>_API_KEY`` env var)
            and a default ``prompt``.
    """

    name = "image"
    description = "Extract text from an image using a vision model (OCR)"

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        provider = cfg.get("provider", "ollama")
        default_url, default_env, default_model = _VISION_DEFAULTS.get(
            provider, _VISION_DEFAULTS["ollama"]
        )
        self.provider: str = provider
        self.model: str = cfg.get("model") or default_model
        self.base_url: str = (
            cfg.get("base_url")
            or os.environ.get(f"{provider.upper()}_BASE_URL", default_url)
        )
        self.api_key: str = (
            cfg.get("api_key")
            or os.environ.get(f"{provider.upper()}_API_KEY", "")
            or os.environ.get(default_env, "")
        )
        self.prompt: str = cfg.get(
            "prompt", "Extract all text visible in this image. Return only the text."
        )

    async def arun(  # type: ignore[override]
        self,
        path: str,
        prompt: str | None = None,
        max_chars: int = 50000,
    ) -> str:
        """OCR the image at *path* and return the transcribed text."""
        if not path:
            raise ValueError("path is required")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            data_url = (
                f"data:{_guess_mime(path)};base64,"
                f"{base64.b64encode(f.read()).decode('ascii')}"
            )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
        }

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        text = data["choices"][0]["message"].get("content") or ""
        if max_chars and max_chars > 0:
            return text[:max_chars]
        return text
