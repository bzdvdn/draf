"""Typed state with per-key reducers for graph workflows."""

import typing
from typing import Any, Callable

Reducer = Callable[[Any, Any], Any] | str
"""Merge strategy: ``"override"``, ``"append"``, ``"keep"``, or a callable ``(old, new) -> value``."""


def reducer_appends(reducer: Reducer | None) -> bool:
    """True when *reducer* accumulates list contributions (append semantics).

    ``"append"`` and any callable treat a node's returned value as new items
    to merge into the existing value. ``None`` / ``"override"`` / ``"keep"``
    treat it as a full replacement (or a keep-if-absent), so a node that
    writes a whole value back must return that whole value under those
    strategies and only its delta under append semantics.
    """
    if reducer is None or reducer == "override" or reducer == "keep":
        return False
    return True


def reducers_from_yaml_schema(schema: dict) -> dict[str, Reducer]:
    """Convert a YAML state schema dict into a reducer map.

    YAML format::

        state:
          schema:
            messages:
              reducer: append
              type: list
            status:
              reducer: keep

    Returns a dict like ``{"messages": "append", "status": "keep"}``.
    Keys without a ``reducer`` field default to ``"override"``.
    """
    reducers: dict[str, Reducer] = {}
    for key, spec in schema.items():
        if isinstance(spec, dict):
            reducer = spec.get("reducer", "override")
        elif isinstance(spec, str):
            reducer = spec
        else:
            continue
        if reducer in ("override", "append", "keep"):
            reducers[key] = reducer
    return reducers


def reducers_to_yaml_schema(reducers: dict[str, Reducer]) -> dict:
    """Convert a reducer map back into a YAML state schema dict.

    Only string reducers (``"override"``/``"append"``/``"keep"``) can be
    serialised; callable reducers are skipped.  Field types are unknown
    after the reverse mapping, so entries carry just the ``reducer`` key::

        reducers_to_yaml_schema({"messages": "append"})
        # -> {"messages": {"reducer": "append"}}
    """
    schema: dict = {}
    for key, reducer in reducers.items():
        if isinstance(reducer, str):
            schema[key] = {"reducer": reducer}
    return schema


def reducers_from_typeddict(cls: type) -> dict[str, Reducer]:
    """Extract per-key reducers from a TypedDict's ``Annotated`` metadata.

    Usage::

        def add_messages(old: list, new: list) -> list:
            return old + new

        class ChatState(TypedDict):
            messages: Annotated[list[str], add_messages]
            status: str

        reducers = reducers_from_typeddict(ChatState)  # {"messages": add_messages}
    """
    reducers: dict[str, Reducer] = {}
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception:
        return reducers
    for key, annotation in hints.items():
        origin = typing.get_origin(annotation)
        if origin is typing.Annotated:
            args = typing.get_args(annotation)
            if len(args) >= 2:
                reducer = args[1]
                if callable(reducer) or isinstance(reducer, str):
                    reducers[key] = reducer
    return reducers


def apply_reducers(state: dict, new_values: dict, reducers: dict[str, Reducer]) -> None:
    """Merge *new_values* into *state* using the provided per-key *reducers*.

    Keys without a reducer are overridden (backward-compatible default).
    A callable reducer for a key that is missing from *state* receives no
    ``old`` value — the new value is stored as-is instead of calling it
    with ``None`` (so ``add_messages(old, new) = old + new`` works from a
    fresh state).
    """
    for key, new_val in new_values.items():
        reducer = reducers.get(key)
        if reducer is None or reducer == "override":
            state[key] = new_val
        elif reducer == "append":
            old = state.get(key)
            if isinstance(old, list):
                old.extend(new_val if isinstance(new_val, list) else [new_val])
            else:
                state[key] = new_val
        elif reducer == "keep":
            if key not in state:
                state[key] = new_val
        elif callable(reducer):
            if key in state:
                state[key] = reducer(state[key], new_val)
            else:
                state[key] = new_val


_TYPE_MAP = {
    "str": "string",
    "string": "string",
    "text": "string",
    "int": "integer",
    "integer": "integer",
    "number": "number",
    "float": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
    "map": "object",
}


def state_schema_to_jsonschema(schema: dict) -> dict:
    """Convert a YAML ``state.schema`` block into a JSON Schema dict.

    The YAML format associates a type with each state key::

        state:
          schema:
            status: string
            count: {type: integer, minimum: 0}
            tags: {type: list}

    Each entry may be a plain type name (``string``, ``integer``,
    ``number``, ``boolean``, ``list``, ``object``, ``null``/``any``) or a
    dict whose ``type`` key holds the type plus any JSON Schema keywords
    (``minimum``, ``items``, ``enum``, ...).  Keys listed as ``required: true``
    are marked required.  Unknown types are left unconstrained.

    Returns:
        A JSON Schema with ``type: object`` and per-key ``properties``.
    """
    properties: dict = {}
    required: list[str] = []
    for key, spec in schema.items():
        if isinstance(spec, dict):
            prop = _yaml_type_to_schema(spec)
            if spec.get("required") is True:
                required.append(key)
        else:
            prop = _yaml_type_to_schema(spec)
        properties[key] = prop or {}
    jsonschema: dict = {"type": "object", "properties": properties}
    if required:
        jsonschema["required"] = required
    return jsonschema


def _yaml_type_to_schema(spec) -> dict:
    """Turn a YAML state-schema entry into a JSON Schema fragment."""
    if isinstance(spec, dict):
        spec = dict(spec)
        type_name = spec.pop("type", None)
        spec.pop("reducer", None)
        spec.pop("required", None)
        schema = _yaml_type_to_schema(type_name)
        schema.update(spec)
        if schema.get("type") == "array" and "items" in spec:
            schema["items"] = _yaml_type_to_schema(spec["items"])
        return schema
    if isinstance(spec, str):
        mapped = _TYPE_MAP.get(spec.strip().lower())
        if mapped:
            return {"type": mapped}
        if spec.strip().lower() in ("null", "any", "none"):
            return {}
        return {}
    return {}


def validate_state(state: dict, schema: dict) -> list[str]:
    """Validate *state* against a YAML ``state.schema`` dict.

    Returns a list of human-readable errors (empty when *state* conforms).
    *schema* is converted with :func:`state_schema_to_jsonschema` and
    validated with :func:`draf.schema.validate_json`.
    """
    from draf.schema import validate_json

    return validate_json(state, state_schema_to_jsonschema(schema))


class State(dict):
    """Typed workflow state that applies per-key reducers on merge.

    Wraps a ``dict`` with reducers extracted from a TypedDict schema::

        class MyState(TypedDict):
            messages: Annotated[list, "append"]
            status: str

        state = State(MyState, {"status": "ok"})
        state.merge({"messages": ["hello"]})
        state.merge({"messages": ["world"]})
        assert state["messages"] == ["hello", "world"]
    """

    def __init__(self, schema: type, data: dict | None = None):
        super().__init__(data or {})
        self._reducers = reducers_from_typeddict(schema)

    def merge(self, new_values: dict) -> None:
        """Merge *new_values* using per-key reducers."""
        apply_reducers(self, new_values, self._reducers)

    @property
    def reducers(self) -> dict[str, Reducer]:
        """Return this state's per-key reducers.

        Exposed so nested components (e.g. parallel branches) apply the
        same merge strategies as the top-level ``graph.run()`` merge.
        """
        return self._reducers
