"""Channels variant entry point — reach the code-first app over Telegram.

Runs the same compiled :class:`~teff.assistant.Assistant` the other entry
points use (``main.py`` / ``cli.py`` / ``daemon.py``) through the
``teff.channels`` adapters, so the graph, tools, checkpoints and interrupt
handling are identical on every surface.  No rewriting needed: the
code-first graph stays as-is, and the channels layer just binds transports.

Usage::

    TELEGRAM_BOT_TOKEN=... uv run python bot.py            # long-polling
    TELEGRAM_BOT_TOKEN=... uv run python bot.py --webhook https://bot.example.com/api/telegram
    uv run python bot.py --once                            # drain pending updates

HOW TO EXTEND
    Mount more adapters here (e.g. a generic ``WebhookChannel``) — they all
    take the same ``container.assistant``.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from src.config.config import get_settings
from src.core import build_container

from teff import configure_logging


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the {{PROJECT_NAME}} channels.")
    parser.add_argument(
        "--webhook",
        default=None,
        help="Register this public URL as the Telegram webhook and serve it",
    )
    parser.add_argument("--once", action="store_true", help="Drain updates and exit")
    parser.add_argument("--log-level", default=None, help="teff log level")
    parser.add_argument(
        "--log-format", default="text", choices=("text", "json"), help="log format"
    )
    args = parser.parse_args()

    configure_logging(args.log_level, format=args.log_format)

    from teff.channels import TelegramChannel

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")

    container, _ = build_container()
    bot = TelegramChannel(container.assistant, token, owner=settings.app_title)

    if args.webhook:
        import uvicorn
        from fastapi import FastAPI, Request

        app = FastAPI(title=f"{settings.app_title} telegram webhook")

        @app.post("/api/telegram")
        async def _telegram(request: Request) -> dict:
            update = await request.json()
            await bot.handle_update(update)
            return {"ok": True}

        async def _serve() -> None:
            await bot.set_webhook(args.webhook)
            await uvicorn.Server(
                uvicorn.Config(app, host="127.0.0.1", port=settings.port)
            ).serve()

        asyncio.run(_serve())
        return

    print(f"telegram bot polling (model {settings.model})")
    asyncio.run(bot.run(once=args.once))


if __name__ == "__main__":
    main()
