"""Transform node — simple string/data transformations."""

from __future__ import annotations

import datetime as _dt
from typing import Any

from teff.node.node import Node
from teff.prompt import render_template


class Transform(Node):
    """Apply a transform to state values.

    Supported actions: ``uppercase``, ``lowercase``, ``trim``,
    ``count_lines``, ``value``, ``render``, ``json_get``, ``append``,
    plus the pipeline-building actions ``contains``, ``compare``, ``split``,
    ``join``, ``replace``, ``coalesce``, ``pick``, ``to_int``, ``to_float``,
    ``now``.

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

    ``contains`` outputs ``"true"``/``"false"`` when *input_key* contains
    ``value``; ``compare`` does the same for ``input_key`` against ``value``
    with ``op`` in ``eq/ne/gt/ge/lt/le`` (numeric when both sides parse as
    numbers).  ``split``/``join`` convert between strings and lists with
    ``sep`` (default ``,``).  ``replace`` swaps ``old``→``new`` in
    *input_key*.  ``coalesce`` returns *input_key* unless it is empty, then
    ``value``.  ``pick`` reads ``field`` out of a dict (like ``json_get``).
    ``to_int``/``to_float`` coerce *input_key* to a number (as a string).
    ``now`` writes the current UTC ISO timestamp.  Every action writes to
    *output_key*; boolean-like actions emit ``"true"``/``"false"`` so they
    can drive ``edges:`` conditions like ``has_refund=true``.

    Parameters:
        action: Transform action name.
        input_key: State key to read from.
        output_key: State key to write to.
        value: Literal value (used with ``action="value"``, the needle for
            ``contains``, the right-hand side of ``compare``, the fallback
            for ``coalesce``).
        template: Template string for ``action="append"``.
        raw: Return ``json_get``/``pick`` values without stringifying.
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

        if action == "pick":
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

        if action == "contains":
            needle = value if value is not None else self.config.get("needle", "")
            result = "true" if str(needle) in str(state.get(input_key, "")) else "false"
            state[output_key] = result
            return {output_key: result}

        if action == "compare":
            op = self.config.get("op", "eq")
            result = (
                "true" if self._compare(state.get(input_key), value, op) else "false"
            )
            state[output_key] = result
            return {output_key: result}

        if action == "split":
            sep = self.config.get("sep", ",")
            result = str(state.get(input_key, "")).split(sep)
            state[output_key] = result
            return {output_key: result}

        if action == "join":
            sep = self.config.get("sep", ",")
            items = state.get(input_key) or []
            result = sep.join(str(i) for i in items)
            state[output_key] = result
            return {output_key: result}

        if action == "replace":
            old = self.config.get("old", "")
            new = self.config.get("new", "")
            result = str(state.get(input_key, "")).replace(old, new)
            state[output_key] = result
            return {output_key: result}

        if action == "coalesce":
            source = state.get(input_key)
            if source in (None, ""):
                source = value if value is not None else ""
            result = source if isinstance(source, str) else str(source)
            state[output_key] = result
            return {output_key: result}

        if action in ("to_int", "to_float"):
            source = str(state.get(input_key, "")).strip()
            result = (
                str(int(float(source))) if action == "to_int" else str(float(source))
            )
            state[output_key] = result
            return {output_key: result}

        if action == "now":
            result = _dt.datetime.now(_dt.timezone.utc).isoformat()
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

    def _compare(self, lhs, rhs: Any, op: str) -> bool:
        a, b = lhs, rhs
        try:
            a, b = float(str(a).strip()), float(str(b).strip())
        except (ValueError, TypeError):
            a, b = str(a), str(b)
        if op == "eq":
            return a == b
        if op == "ne":
            return a != b
        if op == "gt":
            return a > b
        if op == "ge":
            return a >= b
        if op == "lt":
            return a < b
        if op == "le":
            return a <= b
        raise ValueError(f"unknown compare op: {op}")

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
