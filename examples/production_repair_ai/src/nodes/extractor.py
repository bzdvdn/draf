"""Extractor node — structured pull of project info from the conversation."""

from __future__ import annotations

from draf.node.llm import LLM

from src.graphs.prompts import EXTRACTOR_SYSTEM_PROMPT
from src.graphs.schemas import PROJECT_INFO_SCHEMA


class Extractor(LLM):
    """Extract project details from the whole conversation as JSON.

    Subclasses :class:`~draf.node.llm.LLM` so it inherits the structured
    output loop (schema validation with re-asking) and streaming support.
    The full message history is fed to the model together with the
    extractor system prompt; the parsed object lands on ``project_info``.
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
        output_key = cfg.get("output_key", "project_info")
        system = cfg.get("system", "")

        history = list(state.get(messages_key, []))
        work = dict(state)
        work[messages_key] = [{"role": "system", "content": system}, *history]
        return await super().execute(ctx, work)
