"""Tool schema conversion — Python signatures to OpenAI-style function schemas."""

from __future__ import annotations

import inspect
import typing

from draf.provider.providers import _JSON_TYPE_MAP
from draf.schema import _py_to_schema
from draf.tool.tool import Tool


def _py_type_to_json(tp: type) -> str:
    return _JSON_TYPE_MAP.get(tp, "string")


def _py_type_to_schema(tp: typing.Any) -> dict:
    """Convert a Python type hint to a (possibly nested) JSON Schema.

    Falls back to a flat ``{"type": ...}`` mapping for types the
    schema builder does not understand.
    """
    try:
        schema = _py_to_schema(tp)
    except Exception:
        schema = {}
    if isinstance(schema, dict) and schema:
        return schema
    return {"type": _py_type_to_json(tp)}


def tool_to_schema(tool: Tool) -> dict:
    """Convert a :class:`~draf.tool.Tool` to an OpenAI-style function schema.

    Uses the tool's ``schema`` attribute when set (e.g. by MCP tools);
    otherwise infers ``parameters`` from the ``run``/``arun`` signature.
    Nested type hints (``list[dict]``, ``dict[str, str]``, dataclasses,
    ``TypedDict``) expand to nested JSON Schemas.
    """
    provider_schema = tool.schema
    if isinstance(provider_schema, dict):
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": provider_schema,
            },
        }
    run_method = tool.run
    if type(tool).run is Tool.run and type(tool).arun is not Tool.arun:
        run_method = tool.arun
    sig = inspect.signature(run_method)
    try:
        hints = typing.get_type_hints(run_method)
    except Exception:
        hints = {}

    properties: dict = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "kwargs", "args"):
            continue
        prop: dict = _py_type_to_schema(hints.get(pname, str))
        if param.default is not inspect.Parameter.empty:
            if param.default is not None:
                prop["default"] = param.default
        else:
            required.append(pname)
        properties[pname] = prop

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
