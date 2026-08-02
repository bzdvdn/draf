"""Offline tests for the csv_query tool (real CSV files, no network)."""

import pytest


class TestCsvQueryTool:
    def _tool(self, **cfg):
        from draf.tool.builtin import CsvQueryTool

        return CsvQueryTool(cfg or {})

    def _write(self, tmp_path, rows):
        import csv

        p = tmp_path / "data.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        return str(p)

    def test_read(self, tmp_path):
        p = self._write(tmp_path, [["name", "score"], ["a", "1"], ["b", "2"]])
        out = self._tool().run(action="read", path=p)
        assert "name\tscore" in out
        assert "a\t1" in out
        assert "b\t2" in out

    def test_read_limit(self, tmp_path):
        p = self._write(
            tmp_path, [["name", "score"], ["a", "1"], ["b", "2"], ["c", "3"]]
        )
        out = self._tool().run(action="read", path=p, limit=2)
        assert "a\t1" in out and "b\t2" in out
        assert "c\t3" not in out

    def test_columns(self, tmp_path):
        p = self._write(tmp_path, [["name", "score"], ["a", "1"]])
        assert self._tool().run(action="columns", path=p) == "name\nscore"

    def test_filter(self, tmp_path):
        p = self._write(
            tmp_path, [["name", "team"], ["a", "x"], ["b", "y"], ["c", "x"]]
        )
        out = self._tool().run(action="filter", path=p, column="team", value="x")
        assert "a\tx" in out and "c\tx" in out
        assert "b\ty" not in out

    def test_filter_no_match(self, tmp_path):
        p = self._write(tmp_path, [["name", "team"], ["a", "x"]])
        assert "no rows match team=z" in self._tool().run(
            action="filter", path=p, column="team", value="z"
        )

    def test_aggregate_sum(self, tmp_path):
        p = self._write(tmp_path, [["name", "score"], ["a", "1"], ["b", "2"]])
        assert "score sum=3" in self._tool().run(
            action="aggregate", path=p, column="score", op="sum"
        )

    def test_aggregate_avg(self, tmp_path):
        p = self._write(tmp_path, [["name", "score"], ["a", "2"], ["b", "4"]])
        assert "score avg=3" in self._tool().run(
            action="aggregate", path=p, column="score", op="avg"
        )

    def test_aggregate_count_min_max(self, tmp_path):
        p = self._write(
            tmp_path, [["name", "score"], ["a", "1"], ["b", "5"], ["c", "3"]]
        )
        assert "score min=1" in self._tool().run(
            action="aggregate", path=p, column="score", op="min"
        )
        assert "score max=5" in self._tool().run(
            action="aggregate", path=p, column="score", op="max"
        )
        assert "score count=3" in self._tool().run(
            action="aggregate", path=p, column="score", op="count"
        )

    def test_aggregate_group_by(self, tmp_path):
        p = self._write(
            tmp_path, [["team", "score"], ["x", "1"], ["x", "2"], ["y", "5"]]
        )
        out = self._tool().run(
            action="aggregate", path=p, column="score", op="sum", group_by="team"
        )
        assert "x: score sum=3" in out
        assert "y: score sum=5" in out

    def test_aggregate_no_numeric(self, tmp_path):
        p = self._write(tmp_path, [["name", "score"], ["a", "n/a"], ["b", "?"]])
        assert "no numeric values" in self._tool().run(
            action="aggregate", path=p, column="score", op="sum"
        )

    def test_default_path_from_config(self, tmp_path):
        p = self._write(tmp_path, [["name", "score"], ["a", "1"]])
        out = self._tool(path=p).run(action="read")
        assert "a\t1" in out

    def test_missing_file(self):
        with pytest.raises(ValueError, match="file not found"):
            self._tool().run(action="read", path="/nonexistent/x.csv")

    def test_missing_path(self):
        with pytest.raises(ValueError, match="path is required"):
            self._tool().run(action="read")

    def test_filter_missing_column(self, tmp_path):
        p = self._write(tmp_path, [["name", "score"], ["a", "1"]])
        with pytest.raises(ValueError, match="column is required"):
            self._tool().run(action="filter", path=p, value="x")

    def test_unknown_action(self, tmp_path):
        p = self._write(tmp_path, [["name", "score"], ["a", "1"]])
        with pytest.raises(ValueError, match="unknown action: join"):
            self._tool().run(action="join", path=p)

    def test_action_required(self, tmp_path):
        p = self._write(tmp_path, [["name", "score"], ["a", "1"]])
        with pytest.raises(ValueError, match="action is required"):
            self._tool().run(action="", path=p)

    def test_schema_action_required(self):
        from draf.harness import tool_to_schema
        from draf.tool.builtin import CsvQueryTool

        schema = tool_to_schema(CsvQueryTool({}))["function"]["parameters"]
        assert "action" in schema["required"]

    def test_registered(self):
        from draf.tool.registry import default_tool_registry

        assert "csv_query" in default_tool_registry.list()
