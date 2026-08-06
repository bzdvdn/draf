"""Extractor node — structured pull of project info from the conversation."""

from __future__ import annotations

from draf.node.llm import LLM
from src.graphs.prompts import EXTRACTOR_SYSTEM_PROMPT
from src.graphs.schemas import PROJECT_INFO_SCHEMA

#: Russian room keywords → canonical ``room_type`` value.  Local models are
#: unreliable at this task, so when the LLM leaves ``room_type`` empty the
#: node falls back to scanning the first user message for these.
ROOM_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ванн", "санузел", "с/у", "сануз"), "bathroom"),
    (("кухн",), "kitchen"),
    (("спальн",), "bedroom"),
    (("гостин", "зал", "living"), "living_room"),
    (("детск",), "kids_room"),
    (("коридор", "прихож"), "hallway"),
)


def detect_room_type(first_user_message: str) -> str | None:
    """Map the first user message to a ``room_type`` via keywords."""
    text = first_user_message.lower()
    for keywords, room in ROOM_KEYWORDS:
        if any(k in text for k in keywords):
            return room
    return None


class Extractor(LLM):
    """Extract project details from the whole conversation as JSON.

    Subclasses :class:`~draf.node.llm.LLM` so it inherits the structured
    output loop (schema validation with re-asking) and streaming support.
    The full message history is fed to the model together with the
    extractor system prompt; the parsed object lands on ``project_info``.

    If the model returns an empty ``room_type`` the node falls back to the
    deterministic keyword map (:func:`detect_room_type`) over the first
    user message — a small model frequently drops the room even when the
    user named it explicitly.
    """

    type = "extractor"

    def __init__(
        self,
        config: dict | None = None,
        *,
        system: str = EXTRACTOR_SYSTEM_PROMPT,
        messages_key: str = "messages",
        output_key: str = "project_info",
        json_schema: dict | None = PROJECT_INFO_SCHEMA,
        **kwargs,
    ):
        merged = {
            "system": system,
            "messages_key": messages_key,
            "output_key": output_key,
            "json_schema": json_schema,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        messages_key = cfg.get("messages_key", "messages")
        system = cfg.get("system", "")

        history = list(state.get(messages_key, []))
        work = dict(state)
        work[messages_key] = [{"role": "system", "content": system}, *history]
        result = await super().execute(ctx, work)

        info = result.get(cfg.get("output_key", "project_info"))
        if isinstance(info, dict) and not info.get("room_type"):
            for message in history:
                if message.get("role") == "user" and message.get("content"):
                    room = detect_room_type(str(message["content"]))
                    if room:
                        result[cfg.get("output_key", "project_info")] = {
                            **info,
                            "room_type": room,
                        }
                    break
        return result
