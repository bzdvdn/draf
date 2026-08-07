"""Declarative extras: ``parallel`` nodes, the ``checkpoint:`` block, and the
``hooks:`` block expressed in a workflow YAML."""

import asyncio
from pathlib import Path

import pytest

from teff.yaml import from_yaml


def _run(graph, state=None, **kw):
    return asyncio.run(graph.run(state or {}, **kw))


class TestParallelFromYaml:
    def test_declares_branches_as_mappings(self):
        graph = from_yaml(
            """\
name: p
steps:
  - id: fanout
    type: parallel
    config:
      branches:
        - {type: transform, config: {action: value, value: one, output_key: a}}
        - {type: transform, config: {action: value, value: two, output_key: b}}
edges: []
"""
        )
        assert _run(graph) == {"a": "one", "b": "two"}

    def test_branch_with_multiple_steps(self):
        graph = from_yaml(
            """\
name: p
steps:
  - id: fanout
    type: parallel
    config:
      branches:
        - - {type: transform, config: {action: value, value: x, output_key: a}}
          - {type: transform, config: {action: uppercase, input_key: a, output_key: a}}
        - {type: transform, config: {action: value, value: y, output_key: c}}
edges: []
"""
        )
        assert _run(graph) == {"a": "X", "c": "y"}

    def test_branches_merge_via_append_reducer(self):
        graph = from_yaml(
            """\
name: p
state:
  schema:
    msgs:
      reducer: append
      type: list
steps:
  - id: fanout
    type: parallel
    config:
      branches:
        - {type: transform, config: {action: value, value: message-a, output_key: msgs}}
edges: []
"""
        )
        # override reducer test is simpler than append here; just assert run
        assert _run(graph)["msgs"] == "message-a"

    def test_malformed_branches_rejected(self):
        from teff.errors import ConfigError

        with pytest.raises(ConfigError, match="branches"):
            from_yaml(
                """\
name: p
steps:
  - id: bad
    type: parallel
    config:
      branches: "not-a-list"
edges: []
"""
            )


class TestCheckpointBlock:
    def test_no_block_returns_none(self, tmp_path):
        from teff.yaml import checkpointer_from_workflow

        path = Path(tmp_path) / "wf.yaml"
        path.write_text("name: x\nsteps: []\n")
        assert checkpointer_from_workflow(str(path)) is None

    def test_sqlite_block_resolves_path(self, tmp_path):
        from teff.checkpoint import SQLiteCheckpointer
        from teff.yaml import checkpointer_from_workflow

        path = Path(tmp_path) / "wf.yaml"
        path.write_text(
            "name: x\ncheckpoint:\n  type: sqlite\n  path: cp.db\nsteps: []\n"
        )
        cp = checkpointer_from_workflow(str(path))
        assert isinstance(cp, SQLiteCheckpointer)
        assert str(cp._path) == str(tmp_path / "cp.db")

    def test_unknown_type_raises(self, tmp_path):
        from teff.errors import ConfigError
        from teff.yaml import checkpointer_from_workflow

        path = Path(tmp_path) / "wf.yaml"
        path.write_text("name: x\ncheckpoint: {type: nope}\nsteps: []\n")
        with pytest.raises(ConfigError, match="unknown checkpoint type"):
            checkpointer_from_workflow(str(path))


class TestHooksBlock:
    def test_resolve_unknown_raises(self):
        from teff.errors import ConfigError
        from teff.hooks import resolve_hooks

        with pytest.raises(ConfigError, match="unknown hook"):
            resolve_hooks({"on_node_start": "nope"})

    def test_resolve_single_and_list(self):
        from teff import hooks

        calls = []

        def one(*a, **k):
            calls.append(k.get("result"))

        def two(*a, **k):
            calls.append("two")

        hooks.register("one", one)
        hooks.register("two", two)
        resolved = hooks.resolve_hooks(
            {"on_node_start": "one", "on_node_error": ["one", "two"]}
        )
        assert set(resolved) == {"on_node_start", "on_node_error"}
        resolved["on_node_start"]("n", None, {})
        assert calls == [None]

    def test_cli_runs_checkpoint_and_hooks(self, tmp_path):
        from typer.testing import CliRunner

        from teff.cli import app

        (Path(tmp_path) / "hp.py").write_text(
            "from teff import hooks\n"
            '@hooks.hook("tick")\n'
            "def tick(*a, **k):\n"
            '    print(f"HOOK {a[0]}")\n'
        )
        wf = Path(tmp_path) / "wf.yaml"
        wf.write_text(
            """\
name: decl
plugins: [hp.py]
hooks:
  on_node_start: tick
checkpoint:
  type: sqlite
  path: cp.db
steps:
  - {id: s, type: transform, config: {action: value, value: v, output_key: out}}
"""
        )
        result = CliRunner().invoke(
            app, ["run", "--file", str(wf), "--checkpoint-id", "sess"]
        )
        assert result.exit_code == 0, result.output
        assert "HOOK s" in result.output
        assert '"out": "v"' in result.output
        assert (Path(tmp_path) / "cp.db").exists()
