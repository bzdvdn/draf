"""Server entry point — runs the ``repair-ai-chat`` FastAPI app.

Usage::

    uv run python examples/applications/repair-ai-chat/main.py
    uv run python examples/applications/repair-ai-chat/main.py --port 8001
"""

from __future__ import annotations

import argparse

from src.config.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Run the repair-ai-chat API server."
    )
    parser.add_argument("--host", default=settings.host, help="bind host")
    parser.add_argument("--port", type=int, default=settings.port, help="bind port")
    args = parser.parse_args()

    import uvicorn

    from app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
