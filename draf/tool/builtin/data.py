"""Data tools — JSON/YAML parsing, a persistent key-value store, and safe eval."""

import ast
import json
import operator
import os
from typing import Any, Callable

from draf.tool.tool import Tool

_ALLOWED_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
}

_MATH_CONSTANTS = {"pi": "pi", "e": "e", "tau": "tau", "inf": "inf", "nan": "nan"}


class JsonParseTool(Tool):
    """Parse a JSON string and pretty-print it (or validate it)."""

    name = "json_parse"
    description = "Parse and pretty-print a JSON string"

    def run(self, text: str = "", indent: int = 2) -> str:  # type: ignore[override]
        if not text:
            raise ValueError("text is required")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}") from e
        return json.dumps(data, ensure_ascii=False, indent=indent)


class YamlParseTool(Tool):
    """Parse a YAML string and dump it as pretty JSON."""

    name = "yaml_parse"
    description = "Parse a YAML string and dump it as JSON"

    def run(self, text: str = "", indent: int = 2) -> str:  # type: ignore[override]
        if not text:
            raise ValueError("text is required")
        try:
            import yaml
        except ImportError as e:
            msg = "yaml_parse requires 'pyyaml' (a core dependency)"
            raise ImportError(msg) from e
        data = yaml.safe_load(text)
        return json.dumps(data, ensure_ascii=False, indent=indent)


class KVStoreTool(Tool):
    """A persistent JSON-backed key-value store.

    Data lives in a single JSON file (config key ``path``). Operations
    are selected with ``action``: ``get``, ``set``, ``delete``, ``list``.

    Args:
        config: Optional dict with ``path`` (default ``./kv_store.json``).
    """

    name = "kv_store"
    description = "Read/write a persistent key-value store (get, set, delete, list)"

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.path = cfg.get("path", "./kv_store.json")
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def run(  # type: ignore[override]
        self, action: str = "get", key: str = "", value: str = ""
    ) -> str:
        if action == "get":
            if key not in self._data:
                return "not found"
            return json.dumps(self._data[key], ensure_ascii=False)
        if action == "set":
            if not key:
                raise ValueError("key is required")
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
            self._data[key] = parsed
            self._save()
            return f"set {key}"
        if action == "delete":
            if key in self._data:
                del self._data[key]
                self._save()
                return f"deleted {key}"
            return "not found"
        if action == "list":
            return "\n".join(sorted(self._data.keys())) if self._data else "empty"
        raise ValueError(f"unknown action: {action}")


class PythonEvalTool(Tool):
    """Safely evaluate a Python expression using an AST whitelist.

    Supports numbers, arithmetic, ``math`` constants/functions, strings,
    lists, tuples, dicts, and comparisons. Imports, attributes beyond
    ``math.``, and calls outside the whitelist are rejected.
    """

    name = "python_eval"
    description = "Safely evaluate a Python expression"

    _BUILTIN_FUNCS: dict[str, Callable[..., Any]] = {
        "abs": abs,
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "range": range,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "isinstance": isinstance,
    }

    def run(self, expression: str = "") -> str:  # type: ignore[override]
        if not expression:
            raise ValueError("expression is required")
        tree = ast.parse(expression, mode="eval")
        return str(self._eval(tree.body))

    def _eval(self, node) -> object:
        import math

        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            binop = _ALLOWED_BINOPS.get(type(node.op))
            if binop is None:
                raise ValueError(f"operator not allowed: {type(node.op).__name__}")
            return binop(self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp):
            unop = _ALLOWED_UNARYOPS.get(type(node.op))
            if unop is None:
                raise ValueError(f"operator not allowed: {type(node.op).__name__}")
            return unop(self._eval(node.operand))
        if isinstance(node, ast.Name):
            if node.id in _MATH_CONSTANTS:
                return getattr(math, _MATH_CONSTANTS[node.id])
            if node.id in ("True", "False", "None"):
                return {"True": True, "False": False, "None": None}[node.id]
            raise ValueError(f"name not allowed: {node.id}")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "math":
                return getattr(math, node.attr)
            raise ValueError(f"attribute not allowed: {node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fn = self._BUILTIN_FUNCS.get(node.func.id)
                if fn is None:
                    raise ValueError(f"function not allowed: {node.func.id}")
                return fn(*(self._eval(a) for a in node.args))
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                if node.func.value.id == "math":
                    fn = getattr(math, node.func.attr, None)
                    if fn is None:
                        raise ValueError(f"function not allowed: {node.func.attr}")
                    return fn(*(self._eval(a) for a in node.args))
            raise ValueError("call not allowed")
        if isinstance(node, ast.List):
            return [self._eval(e) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(e) for e in node.elts)
        if isinstance(node, ast.Dict):
            return {
                self._eval(k): self._eval(v)
                for k, v in zip(node.keys, node.values)
                if k is not None
            }
        if isinstance(node, ast.Compare):
            left: Any = self._eval(node.left)
            for cmp, comparator in zip(node.ops, node.comparators):
                right: Any = self._eval(comparator)
                if isinstance(cmp, ast.Eq):
                    ok = left == right
                elif isinstance(cmp, ast.NotEq):
                    ok = left != right
                elif isinstance(cmp, ast.Lt):
                    ok = left < right
                elif isinstance(cmp, ast.LtE):
                    ok = left <= right
                elif isinstance(cmp, ast.Gt):
                    ok = left > right
                elif isinstance(cmp, ast.GtE):
                    ok = left >= right
                else:
                    raise ValueError(f"operator not allowed: {type(cmp).__name__}")
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = True
                for v in node.values:
                    result = result and bool(self._eval(v))
            elif isinstance(node.op, ast.Or):
                result = False
                for v in node.values:
                    result = result or bool(self._eval(v))
            else:
                raise ValueError("boolean operator not allowed")
            return result
        raise ValueError(f"expression not allowed: {ast.dump(node)}")
