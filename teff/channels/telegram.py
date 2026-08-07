"""Telegram channel: run a workflow from Telegram messages.

A thin Bot API adapter over the shared :class:`~teff.assistant.Assistant`,
with two transport modes:

* ``polling`` — the bot loops over ``getUpdates`` (default, no public
  server needed).
* ``webhook`` — the bot registers a ``setWebhook`` URL and FastAPI receives
  Telegram's POSTs; ideal behind TLS (``mode: webhook`` + ``url``).

Each Telegram chat is a durable session: ``session_id = chat_id``, so a
multi-turn workflow pauses on an interrupt, asks the operator a question
in-chat, and resumes when they answer.  The checkpoint owner is the
sender's Telegram user id, so each user's conversations are isolated.  The
transport is plain ``httpx`` (already a core dependency), so no extra
package is required.

``channels:`` block::

    channels:
      telegram:
        token_env: TELEGRAM_BOT_TOKEN
        mode: polling            # polling | webhook
        url: https://bot.example.com/api/telegram   # for webhook mode
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from teff.assistant import Assistant

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org/bot"


class TelegramChannel:
    """A Telegram Bot API adapter over a shared ``Assistant``."""

    def __init__(
        self,
        assistant: Assistant,
        token: str,
        *,
        owner: str = "telegram",
        poll_timeout: int = 30,
    ):
        self.assistant = assistant
        self.token = token
        self.owner = owner
        self.poll_timeout = poll_timeout
        self._base = API_BASE + token
        self._offset: int | None = None

    def session_id_for(self, chat_id: int | str) -> str:
        """Telegram chats map one-to-one to durable sessions."""
        return f"tg-{chat_id}"

    async def _api(self, method: str, **params: Any) -> dict:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.post(f"{self._base}/{method}", json=params)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                raise RuntimeError(f"telegram {method} failed: {data}")
            return data["result"]

    async def send_message(self, chat_id: int, text: str) -> None:
        """Send a plain text reply (interrupt prompts included)."""
        await self._api("sendMessage", chat_id=chat_id, text=text)

    async def handle_update(self, update: dict[str, Any]) -> None:
        """Process one Telegram update: run a turn and reply in-chat.

        The checkpoint owner is the sender's Telegram user id
        (``message.from.id``), so every user's sessions are isolated.
        """
        message = update.get("message") or update.get("edited_message")
        if not message:
            return
        chat_id = message["chat"]["id"]
        text = message.get("text")
        if not text:
            return
        owner = str(message.get("from", {}).get("id") or chat_id)
        session_id = self.session_id_for(chat_id)
        result = await self.assistant.run(session_id, text, owner=owner)
        if result.waiting:
            prompt = result.prompt or "?"
            await self.send_message(chat_id, f"⏳ {prompt}")
        else:
            from teff.channels.reply import reply_text

            await self.send_message(chat_id, reply_text(result))

    async def run(self, *, once: bool = False) -> None:
        """Long-poll for updates forever (or a single pass with ``once``)."""
        logger.info("telegram channel polling for updates")
        while True:
            params: dict[str, Any] = {"timeout": self.poll_timeout}
            if self._offset is not None:
                params["offset"] = self._offset
            updates = await self._api("getUpdates", **params)
            for update in updates:
                self._offset = int(update["update_id"]) + 1
                try:
                    await self.handle_update(update)
                except Exception:
                    logger.exception("error handling telegram update")
            if once:
                return
            await asyncio.sleep(0.1)

    async def set_webhook(self, url: str) -> None:
        """Point Telegram at ``url`` (call once, then serve the POSTs)."""
        await self._api("setWebhook", url=url)

    async def delete_webhook(self) -> None:
        await self._api("deleteWebhook")
