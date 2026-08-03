"""Typed state with per-key reducers for graph workflows."""

import typing
from typing import Any, Callable

Reducer = Callable[[Any, Any], Any] | str
"""Merge strategy: ``"override"``, ``"append"``, ``"keep"``, or a callable ``(old, new) -> value``."""


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
    """
    for key, new_val in new_values.items():
        reducer = reducers.get(key)
        if reducer is None or reducer == "override":
            state[key] = new_val
        elif reducer == "append":
            if key in state:
                state[key].extend(new_val)
            else:
                state[key] = new_val
        elif reducer == "keep":
            if key not in state:
                state[key] = new_val
        elif callable(reducer):
            state[key] = reducer(state.get(key), new_val)


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
