"""Transform node — simple string transformations."""

from draf.node.node import Node


class Transform(Node):
    """Apply a string transform to state values.

    Supported actions: ``uppercase``, ``lowercase``, ``trim``,
    ``count_lines``, ``value``.
    """

    type = "transform"

    async def execute(self, ctx, state: dict) -> dict:
        action = self.config.get("action", "")
        input_key = self.config.get("input_key", "")
        output_key = self.config.get("output_key", "")
        value = self.config.get("value")

        source = value if value is not None else state.get(input_key, "")
        result = self._apply(action, source)
        state[output_key] = result
        return state

    def _apply(self, action: str, text: str) -> str:
        if action == "uppercase":
            return text.upper()
        if action == "lowercase":
            return text.lower()
        if action == "trim":
            return text.strip() if isinstance(text, str) else str(text).strip()
        if action == "count_lines":
            return str(len(text.splitlines()))
        if action == "value":
            return text if isinstance(text, str) else str(text)
        msg = f"unknown transform action: {action}"
        raise ValueError(msg)
