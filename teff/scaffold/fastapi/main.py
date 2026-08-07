"""Server entry point — runs the {{PROJECT_NAME}} FastAPI app.

Usage::

    uv run python main.py
    uv run python main.py --port 9000
"""

from __future__ import annotations

import argparse

from src.config.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the {{PROJECT_NAME}} API server.")
    parser.add_argument("--host", default=settings.host, help="bind host")
    parser.add_argument("--port", type=int, default=settings.port, help="bind port")
    parser.add_argument(
        "--log-level",
        default=None,
        help="teff log level (DEBUG/INFO/WARNING/ERROR)",
    )
    parser.add_argument(
        "--log-format",
        default="text",
        choices=("text", "json"),
        help="teff log format",
    )
    args = parser.parse_args()

    from teff import configure_logging

    configure_logging(args.log_level, format=args.log_format)

    import uvicorn
    from app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
