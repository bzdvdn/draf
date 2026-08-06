"""Server entry point — runs the ``service_desk`` FastAPI app.

Usage::

    uv run python examples/applications/service_desk/main.py
    uv run python examples/applications/service_desk/main.py --port 8001
"""

from __future__ import annotations

import argparse

from service_desk.config.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the service-desk API server.")
    parser.add_argument("--host", default=settings.host, help="bind host")
    parser.add_argument("--port", type=int, default=settings.port, help="bind port")
    args = parser.parse_args()

    import uvicorn
    from service_desk.server import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
