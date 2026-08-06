"""Transform node — simple string transformations."""

from typing import Any

from draf.node.node import Node
from draf.prompt import render_template


class Transform(Node):
    """Apply a string transform to state values.

    Supported actions: ``uppercase``, ``lowercase``, ``trim``,
    ``count_lines``, ``value``, ``render``, ``json_get``, ``append``.

    ``render`` formats a ``template`` (``{key}`` placeholders rendered from
    state) and stores the resulting string under *output_key* — the scalar
    counterpart of ``append`` (which accumulates into a list).

    ``json_get`` extracts ``field`` from a dict in *input_key*.  Non-string
    values are stringified by default; pass ``raw=True`` to keep the value
    as-is (e.g. to hand a parsed list to a ``Map``).

    ``append`` formats a ``template`` (``{key}`` placeholders rendered from
    state) and appends the result to the list in *output_key*.  When no
    template is given, *input_key*/*value* supplies the item instead.  The
    list is created if absent — the common "accumulate formatted results"
    pattern (report sections, chapter text, step logs).

    Parameters:
        action: Transform action name.
        input_key: State key to read from.
        output_key: State key to write to.
        value: Literal value (used with ``action="value"``).
        template: Template string for ``action="append"``.
        raw: Return ``json_get`` values without stringifying.
    """

    type = "transform"

    def __init__(
        self,
        config: dict | None = None,
        *,
        action: str = "",
        input_key: str = "",
        output_key: str = "",
        value: str | None = None,
        template: str | None = None,
        raw: bool = False,
        **kwargs,
    ):
        merged = {
            "action": action,
            "input_key": input_key,
            "output_key": output_key,
            "value": value,
            "template": template,
            "raw": raw,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        action = self.config.get("action", "")
        input_key = self.config.get("input_key", "")
        output_key = self.config.get("output_key", "")
        value = self.config.get("value")
        field = self.config.get("field")

        if action == "json_get":
            data = state.get(input_key) if input_key else value
            result = self._json_get(
                data, field, raw=bool(self.config.get("raw", False))
            )
            state[output_key] = result
            return {output_key: result}

        if action == "append":
            item = self._render_item(state, input_key, value)
            items = list(state.get(output_key, []))
            items.append(item)
            state[output_key] = items
            return {output_key: items}

        if action == "render":
            result = render_template(self.config.get("template", ""), state)
            state[output_key] = result
            return {output_key: result}

        source = value if value is not None else state.get(input_key, "")
        result = self._apply(action, source)
        state[output_key] = result
        return {output_key: result}

    def _render_item(self, state: dict, input_key: str, value: str | None) -> str:
        template = self.config.get("template")
        if template:
            return render_template(template, state)
        if value is not None:
            return value
        return str(state.get(input_key, ""))

    def _json_get(self, data, field: str | None, raw: bool = False) -> Any:
        if not isinstance(data, dict):
            raise ValueError(
                f"json_get requires a dict in state key, got {type(data).__name__}"
            )
        if not field:
            raise ValueError("json_get requires 'field'")
        if field not in data:
            raise KeyError(f"json_get: no field {field!r} in object")
        value = data[field]
        if raw:
            return value
        return value if isinstance(value, str) else str(value)

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
