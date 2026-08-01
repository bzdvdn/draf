"""Context builders and conversation helpers."""

from __future__ import annotations

from draf.node.node import Node


class ContextBuilder(Node):
    """Compose a plain-text ``input`` from shared state for an agent.

    Args:
        sections: State key → section label mapping rendered as
            ``<label>:\\n<value>`` blocks.
        messages_key: State key holding the conversation.
        output_key: State key receiving the composed text.
        reset_keys: Scratch state keys to clear before the agent runs.
    """

    type = "context_builder"

    def __init__(
        self,
        config: dict | None = None,
        *,
        sections: dict[str, str] | None = None,
        messages_key: str = "messages",
        output_key: str = "input",
        reset_keys: tuple[str, ...] = (),
        **kwargs,
    ):
        merged = {
            "sections": sections or {},
            "messages_key": messages_key,
            "output_key": output_key,
            "reset_keys": list(reset_keys),
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        parts: list[str] = []
        for key, label in self.config["sections"].items():
            value = state.get(key)
            if value:
                parts.append(f"{label}:\n{value}")
        last_user = _last_user_message(state.get(self.config["messages_key"], []))
        if last_user:
            parts.append(f"User: {last_user}")
        output_key = self.config["output_key"]
        out: dict = {output_key: "\n\n".join(parts)}
        for key in self.config["reset_keys"]:
            if key != output_key:
                out[key] = []
        return out


class AppendAssistant(Node):
    """Append an agent's response to the shared conversation as assistant."""

    type = "append_assistant"

    def __init__(
        self,
        config: dict | None = None,
        *,
        output_key: str = "code",
        messages_key: str = "messages",
        **kwargs,
    ):
        merged = {
            "output_key": output_key,
            "messages_key": messages_key,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        content = state.get(self.config["output_key"], "")
        if not content:
            return {}
        return {self.config["messages_key"]: [{"role": "assistant", "content": content}]}


def _last_user_message(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""
