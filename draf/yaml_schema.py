"""Validation of workflow YAML documents against a JSON Schema.

A workflow file is validated *before* execution so that typos and
structural mistakes are reported with a clear path and message instead
of a stack trace mid-run.

Use :func:`validate_workflow_file` / :func:`validate_workflow` directly,
or rely on :func:`draf.yaml.load_workflow`, which validates by default
and raises :class:`~draf.errors.ConfigError` with all findings.
"""

from __future__ import annotations

import os
from typing import Any

import jsonschema
import yaml

from draf.errors import ConfigError

WORKFLOW_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Draf workflow",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "config": {"type": "object"},
                },
                "additionalProperties": True,
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from", "to"],
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "condition": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {"type": "string"},
                    "config": {"type": "object"},
                },
                "additionalProperties": True,
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "schema": {"type": "object"},
                "initial": {},
            },
            "additionalProperties": True,
        },
        "plugins": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ]
        },
        "plugins_folder": {"type": "string"},
    },
    "additionalProperties": True,
}

_VALIDATOR = jsonschema.Draft202012Validator(WORKFLOW_JSON_SCHEMA)


def _node_types() -> list[str]:
    from draf.node.registry import default_registry

    return default_registry.list()


def _tool_types() -> list[str]:
    import draf.tool.builtin  # noqa: F401 — registers built-in tools
    import draf.rag  # noqa: F401 — registers the "rag" tool
    from draf.tool.registry import default_tool_registry

    return default_tool_registry.list()


def validate_workflow(
    data: dict,
    *,
    node_types: list[str] | None = None,
    tool_types: list[str] | None = None,
) -> list[dict]:
    """Validate a parsed workflow dict.

    Checks the structural JSON Schema plus node/tool type membership and
    edge references.

    Args:
        data: The parsed workflow document.
        node_types: Allowed node type names (defaults to the registry).
        tool_types: Allowed tool type names (defaults to the registry).

    Returns:
        A list of ``{"path", "message"}`` errors (empty when valid).
    """
    errors: list[dict] = []
    for err in _VALIDATOR.iter_errors(data):
        path = _err_path(err.absolute_path)
        errors.append({"path": path, "message": err.message})

    node_types = node_types or _node_types()
    tool_types = tool_types or _tool_types()

    steps = data.get("steps") or []
    step_ids: set[str] = set()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        sid = step.get("id")
        if isinstance(sid, str):
            step_ids.add(sid)
        stype = step.get("type")
        if isinstance(stype, str) and stype not in node_types:
            errors.append(
                {
                    "path": f"steps[{i}].type",
                    "message": (
                        f"unknown node type {stype!r} (registered: "
                        f"{', '.join(sorted(node_types))})"
                    ),
                }
            )

    for i, tool in enumerate(data.get("tools") or []):
        if not isinstance(tool, dict):
            continue
        ttype = tool.get("type")
        if isinstance(ttype, str) and ttype not in tool_types:
            errors.append(
                {
                    "path": f"tools[{i}].type",
                    "message": f"unknown tool type {ttype!r}",
                }
            )

    for i, edge in enumerate(data.get("edges") or []):
        if not isinstance(edge, dict):
            continue
        for key in ("from", "to"):
            target = edge.get(key)
            if isinstance(target, str) and step_ids and target not in step_ids:
                errors.append(
                    {
                        "path": f"edges[{i}].{key}",
                        "message": f"edge references unknown step {target!r}",
                    }
                )

    return errors


def _err_path(parts: Any) -> str:
    out = ""
    for part in parts:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f"{'.' if out else ''}{part}"
    return out or "$"


def format_errors(errors: list[dict], *, source: str = "workflow") -> str:
    """Render validation *errors* as human-readable lines."""
    lines = []
    for err in errors:
        lines.append(f"{source}: {err['path']}: {err['message']}")
    return "\n".join(lines)


def validate_workflow_file(path: str) -> list[dict]:
    """Validate a workflow YAML file on disk.

    Loads any plugins referenced by the ``plugins`` key (or the default
    ``plugins/`` folder) so custom node/tool types are registered before
    validation.

    Returns a list of ``{"path", "message"}`` errors (empty when valid).
    A missing or unparseable file raises :class:`ConfigError`.
    """
    if not os.path.exists(path):
        raise ConfigError(f"workflow file not found: {path}")
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return []
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path}: workflow must be a mapping, got {type(data).__name__}"
        )
    from draf.plugins import load_plugins_from_document

    load_plugins_from_document(data, os.path.dirname(os.path.abspath(path)))
    return validate_workflow(data)


def raise_for_validation(errors: list[dict], *, source: str = "workflow") -> None:
    """Raise :class:`ConfigError` listing *errors* if any exist."""
    if errors:
        raise ConfigError(format_errors(errors, source=source))
