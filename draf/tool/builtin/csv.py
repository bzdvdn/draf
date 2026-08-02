"""CSV tools — read, filter and aggregate tabular data from CSV files.

A single tool with an ``action`` selector (read, columns, filter,
aggregate) for quickly exploring and summarizing CSV files — exports,
report dumps, lookup tables — without an agent having to shell out to
Python or load a spreadsheet.
"""

import csv

from draf.tool.tool import Tool


class CsvQueryTool(Tool):
    """Read, filter and aggregate a CSV file.

    Args:
        action: ``read`` | ``columns`` | ``filter`` | ``aggregate``.
        path: CSV file path (falls back to config ``path``).
        column: Column name for ``filter``/``aggregate``.
        value: Exact value to match for ``filter``.
        op: ``count`` | ``sum`` | ``avg`` | ``min`` | ``max``.
        group_by: Optional column to group ``aggregate`` by.
        limit: Max rows to return (default 100).

    Args (config):
        path: Default CSV file path.
    """

    name = "csv_query"
    description = (
        "Read, filter and aggregate a CSV file (read, columns, filter, "
        "aggregate)"
    )

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.path = cfg.get("path", "")

    def _rows(self, path: str):
        path = path or self.path
        if not path:
            raise ValueError("path is required")
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"no header row in {path}")
                return reader.fieldnames, [dict(r) for r in reader]
        except FileNotFoundError as e:
            raise ValueError(f"file not found: {path}") from e

    @staticmethod
    def _fmt(v) -> str:
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return f"{v:.4f}".rstrip("0").rstrip(".")

    def _render(self, fields, rows, limit: int) -> str:
        out = ["\t".join(fields)]
        for r in rows[:limit]:
            out.append("\t".join(str(r.get(c, "")) for c in fields))
        return "\n".join(out)

    def run(  # type: ignore[override]
        self,
        action: str,
        path: str = "",
        column: str = "",
        value: str = "",
        op: str = "count",
        group_by: str = "",
        limit: int = 100,
    ) -> str:
        if not action:
            raise ValueError("action is required (read, columns, filter, aggregate)")
        fields, rows = self._rows(path)
        a = action.lower()
        if a == "columns":
            return "\n".join(fields)
        if a == "read":
            out = self._render(fields, rows, int(limit))
            return out if len(out) > 1 else f"no rows (columns: {', '.join(fields)})"
        if a == "filter":
            if not column:
                raise ValueError("column is required")
            needle = str(value)
            matches = [r for r in rows if str(r.get(column, "")) == needle]
            if not matches:
                return f"no rows match {column}={needle}"
            return self._render(fields, matches, int(limit))
        if a == "aggregate":
            if not column:
                raise ValueError("column is required")
            groups: dict[str, list] = {}
            for r in rows:
                gkey = str(r.get(group_by, "")) if group_by else ""
                groups.setdefault(gkey, []).append(r)
            lines: list[str] = []
            for gkey in sorted(groups):
                vals: list[float] = []
                for r in groups[gkey]:
                    raw = r.get(column)
                    if raw in (None, ""):
                        continue
                    try:
                        vals.append(float(raw))
                    except (TypeError, ValueError):
                        continue
                label = f"{gkey}: {column}" if group_by else column
                if not vals:
                    lines.append(f"{label} (no numeric values)")
                    continue
                if op == "count":
                    res: float | int = len(groups[gkey])
                elif op == "sum":
                    res = sum(vals)
                elif op == "avg":
                    res = sum(vals) / len(vals)
                elif op == "min":
                    res = min(vals)
                elif op == "max":
                    res = max(vals)
                else:
                    raise ValueError(f"unknown op: {op}")
                lines.append(f"{label} {op}={self._fmt(res)}")
            return "\n".join(lines) if lines else "no rows"
        raise ValueError(f"unknown action: {a}")


__all__ = ["CsvQueryTool"]
