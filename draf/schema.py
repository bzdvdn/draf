"""Lightweight JSON Schema validation for structured LLM output.

Constitution Principle VI: minimal dependencies.  This module provides a
compact JSON Schema subset — objects, arrays, primitives, ``enum``,
``oneOf``, and string/number limits — implemented with stdlib only, so
no ``jsonschema`` or Pydantic dependency is required.

The subset covers the practical shapes an LLM is asked to produce:

- ``{"type": "object", "properties": {...}, "required": [...]}``
- ``{"type": "array", "items": {...}, "minItems"/"maxItems"}``
- primitives with ``enum``, ``minimum``/``maximum``,
  ``minLength``/``maxLength``/``pattern``
- ``oneOf`` for unions / nullable fields
"""

from __future__ import annotations

import json
import re
import types
import typing
from typing import Any, get_args, get_origin, get_type_hints


def json_schema_from_type(spec: Any) -> dict:
    """Build a JSON Schema from a Python type spec.

    Accepts:

    - a raw JSON Schema dict (returned unchanged);
    - a ``dict[str, type]`` field map, e.g.
      ``{"name": str, "age": int}``;
    - a ``TypedDict`` (or dataclass) class whose fields become
      ``properties``.

    Raises:
        TypeError: If *spec* is none of the supported forms.
    """
    if isinstance(spec, dict):
        if any(key in spec for key in ("type", "properties", "$schema")):
            return spec
        return {
            "type": "object",
            "properties": {k: _py_to_schema(v) for k, v in spec.items()},
            "required": list(spec.keys()),
        }
    if isinstance(spec, type):
        return _py_to_schema(spec)
    raise TypeError(f"cannot derive JSON Schema from {spec!r}")


def validate_json(value: Any, schema: dict) -> list[str]:
    """Validate *value* against *schema*.

    Returns:
        A list of human-readable error strings.  An empty list means the
        value conforms to the schema.
    """
    errors: list[str] = []
    _validate("$", value, schema, errors)
    return errors


def parse_json_object(content: str) -> Any:
    """Extract and parse a JSON value from LLM output.

    Tries ``json.loads`` on the whole text first, then falls back to the
    first balanced ``{...}`` object embedded in surrounding prose (a
    common failure mode for local models).

    Returns:
        The parsed value.

    Raises:
        ValueError: If no valid JSON can be found in *content*.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    brace = content.find("{")
    while brace != -1:
        obj = extract_json_object(content, brace)
        if obj is not None:
            try:
                return json.loads(obj)
            except json.JSONDecodeError:
                brace = content.find("{", brace + 1)
                continue
        brace = content.find("{", brace + 1)
    raise ValueError("no valid JSON object found in LLM output")


def extract_json_object(text: str, start: int) -> str | None:
    """Return the balanced JSON object starting at ``text[start] == '{'``."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Python type -> JSON Schema
# ---------------------------------------------------------------------------


def _py_to_schema(tp: Any) -> dict:
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is typing.Annotated:
        return _py_to_schema(args[0])
    if origin is typing.Literal:
        return {"enum": list(args)}
    if origin is typing.Union or origin is types.UnionType:
        return _union_to_schema(args)
    if origin is list:
        items = _py_to_schema(args[0]) if args else {}
        return {"type": "array", "items": items}
    if origin is dict:
        value = _py_to_schema(args[1]) if len(args) > 1 else {}
        return {"type": "object", "additionalProperties": value}
    if isinstance(tp, type):
        if tp is str:
            return {"type": "string"}
        if tp is int:
            return {"type": "integer"}
        if tp is float:
            return {"type": "number"}
        if tp is bool:
            return {"type": "boolean"}
        if tp is type(None):
            return {"type": "null"}
        if hasattr(tp, "__annotations__"):
            hints = get_type_hints(tp, include_extras=True)
            return {
                "type": "object",
                "properties": {k: _py_to_schema(v) for k, v in hints.items()},
                "required": list(hints.keys()),
            }
    return {}


def _union_to_schema(args: tuple[Any, ...]) -> dict:
    """Build a JSON Schema for a union of types (``typing.Union`` /
    ``types.UnionType``), special-casing optional single types."""
    non_null = [a for a in args if a is not type(None)]
    if len(non_null) == 1:
        schema = _py_to_schema(non_null[0])
        if len(non_null) != len(args):
            schema = {"oneOf": [{"type": "null"}, schema]}
        return schema
    choices = [_py_to_schema(a) for a in non_null]
    if len(non_null) != len(args):
        choices.append({"type": "null"})
    return {"oneOf": choices}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _validate(path: str, value: Any, schema: dict, errors: list[str]) -> None:
    if not isinstance(schema, dict) or not schema:
        return
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        for sub in one_of:
            sub_errors: list[str] = []
            _validate(path, value, sub if isinstance(sub, dict) else {}, sub_errors)
            if not sub_errors:
                return
        errors.append(f"{path}: does not match any oneOf branch")
        return
    enum = schema.get("enum")
    if isinstance(enum, list):
        if value not in enum:
            errors.append(f"{path}: {value!r} not in enum {enum!r}")
        return
    typ = schema.get("type")
    if typ == "object":
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object, got {_type_name(value)}")
            return
        _validate_object(path, value, schema, errors)
    elif typ == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: expected array, got {_type_name(value)}")
            return
        _validate_array(path, value, schema, errors)
    elif typ == "null":
        if value is not None:
            errors.append(f"{path}: expected null, got {_type_name(value)}")
    elif typ == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean, got {_type_name(value)}")
    elif typ == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{path}: expected integer, got {_type_name(value)}")
        else:
            _validate_number(path, value, schema, errors)
    elif typ == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{path}: expected number, got {_type_name(value)}")
        else:
            _validate_number(path, value, schema, errors)
    elif typ == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string, got {_type_name(value)}")
        else:
            _validate_string(path, value, schema, errors)


def _validate_object(path: str, value: dict, schema: dict, errors: list[str]) -> None:
    props = schema.get("properties")
    props = props if isinstance(props, dict) else {}
    for key in schema.get("required", []):
        if key not in value:
            errors.append(f"{path}.{key}: missing required property")
    additional = schema.get("additionalProperties", True)
    for key, item in value.items():
        sub = props.get(key)
        if sub is not None:
            _validate(f"{path}.{key}", item, sub, errors)
        elif additional is False:
            errors.append(f"{path}: unexpected property {key!r}")
        elif isinstance(additional, dict):
            _validate(f"{path}.{key}", item, additional, errors)


def _validate_array(path: str, value: list, schema: dict, errors: list[str]) -> None:
    if schema.get("minItems") is not None and len(value) < schema["minItems"]:
        errors.append(f"{path}: fewer than {schema['minItems']} items")
    if schema.get("maxItems") is not None and len(value) > schema["maxItems"]:
        errors.append(f"{path}: more than {schema['maxItems']} items")
    items = schema.get("items")
    if isinstance(items, dict):
        for i, item in enumerate(value):
            _validate(f"{path}[{i}]", item, items, errors)


def _validate_number(
    path: str, value: int | float, schema: dict, errors: list[str]
) -> None:
    if schema.get("minimum") is not None and value < schema["minimum"]:
        errors.append(f"{path}: below minimum {schema['minimum']}")
    if schema.get("maximum") is not None and value > schema["maximum"]:
        errors.append(f"{path}: above maximum {schema['maximum']}")


def _validate_string(path: str, value: str, schema: dict, errors: list[str]) -> None:
    if schema.get("minLength") is not None and len(value) < schema["minLength"]:
        errors.append(f"{path}: shorter than minLength {schema['minLength']}")
    if schema.get("maxLength") is not None and len(value) > schema["maxLength"]:
        errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        try:
            if not re.search(pattern, value):
                errors.append(f"{path}: does not match pattern {pattern!r}")
        except re.error:
            pass
