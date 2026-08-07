"""Tests for ``include:`` workflow composition in YAML."""

import pytest

from teff.yaml import from_yaml, load_workflow


class TestIncludeMerge:
    def _shared(self, tmp_path, name="shared.yaml") -> str:
        path = tmp_path / name
        path.write_text(
            """\
name: shared
steps:
  - id: shared_step
    type: transform
    config: {action: uppercase, input_key: text, output_key: upper}
"""
        )
        return str(path)

    def test_merges_steps_and_edges(self, tmp_path):
        shared = self._shared(tmp_path)
        graph = from_yaml(
            f"""\
name: composed
include: {shared}
steps:
  - id: final
    type: transform
    config: {{action: value, value: done, output_key: final}}
edges:
  - from: shared_step
    to: final
"""
        )
        assert set(graph.nodes) == {"shared_step", "final"}
        r = __import__("asyncio").run(graph.run({"text": "hi"}, max_iterations=10))
        assert r["upper"] == "HI"
        assert r["final"] == "done"

    def test_prefix_avoids_id_collision(self, tmp_path):
        self._shared(tmp_path)
        shared = self._shared(tmp_path, name="other.yaml")
        graph = from_yaml(
            f"""\
name: composed
include:
  - path: {shared}
    prefix: a_
steps:
  - id: shared_step
    type: transform
    config: {{action: value, value: mine, output_key: upper}}
edges:
  - from: a_shared_step
    to: shared_step
"""
        )
        assert set(graph.nodes) == {"a_shared_step", "shared_step"}
        r = __import__("asyncio").run(graph.run({"text": "hi"}, max_iterations=10))
        assert r["upper"] == "mine"

    def test_tools_are_merged(self, tmp_path):
        path = tmp_path / "wf.yaml"
        path.write_text(
            """\
name: composed
include:
  - path: ./tool.yaml
steps:
  - id: s
    type: transform
    config: {action: value, value: x, output_key: out}
"""
        )
        (tmp_path / "tool.yaml").write_text(
            """\
name: toolfile
tools:
  - type: calculator
"""
        )
        graph, tools, _, _ = load_workflow(str(path))
        assert [t.name for t in tools] == ["calculator"]

    def test_recursive_include(self, tmp_path):
        (tmp_path / "leaf.yaml").write_text(
            """\
name: leaf
steps:
  - id: leaf_step
    type: transform
    config: {action: value, value: leaf, output_key: leaf}
"""
        )
        (tmp_path / "mid.yaml").write_text(
            """\
name: mid
include:
  - path: ./leaf.yaml
    prefix: l_
steps:
  - id: mid_step
    type: transform
    config: {action: value, value: mid, output_key: mid}
"""
        )
        graph = from_yaml(
            f"""\
name: top
include:
  - path: {tmp_path}/mid.yaml
    prefix: m_
steps:
  - id: top_step
    type: transform
    config: {{action: value, value: top, output_key: top}}
"""
        )
        assert set(graph.nodes) == {"m_l_leaf_step", "m_mid_step", "top_step"}

    def test_missing_include_raises(self, tmp_path):
        from teff.errors import ConfigError

        with pytest.raises(ConfigError):
            from_yaml(
                f"""\
name: composed
include: {tmp_path}/nope.yaml
steps:
  - id: s
    type: transform
    config: {{action: value, value: x, output_key: out}}
"""
            )

    def test_include_entry_requires_path(self, tmp_path):
        from teff.errors import ConfigError

        with pytest.raises(ConfigError, match="path"):
            from_yaml(
                """\
name: composed
include:
  - prefix: x
steps:
  - id: s
    type: transform
    config: {action: value, value: x, output_key: out}
"""
            )


class TestIncludeCommandPrefix:
    def test_command_goto_is_prefixed(self, tmp_path):
        (tmp_path / "sub.yaml").write_text(
            """\
name: sub
steps:
  - id: route
    type: command
    config: {goto: STOP}
  - id: target
    type: transform
    config: {action: value, value: hit, output_key: hit}
"""
        )
        graph = from_yaml(
            f"""\
name: top
include:
  - path: {tmp_path}/sub.yaml
    prefix: sub_
steps:
  - id: s
    type: transform
    config: {{action: value, value: x, output_key: out}}
"""
        )
        assert set(graph.nodes) == {"sub_route", "sub_target", "s"}
