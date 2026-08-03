"""Offline tests for folder-based plugin discovery of custom nodes/tools."""

import pytest


CUSTOM_NODE = """\
from draf.node.registry import node
from draf.tool.registry import tool


@node("greet_node")
async def greet_node(ctx, state):
    return {"greeting": f"hi {state.get('name', '')}".strip()}


@tool("greet_tool", "Say hello")
def greet_tool(name: str = "") -> str:
    return f"hello {name}"
"""


def _write_workflow(dirpath, body, name="wf.yaml"):
    p = dirpath / name
    p.write_text(body, encoding="utf-8")
    return str(p)


class TestPlugins:
    def test_plugins_key_loads_custom_node_and_tool(self, tmp_path):
        from draf.plugins import reset_plugins
        from draf.yaml import load_workflow
        from draf.yaml_schema import validate_workflow_file

        reset_plugins()
        (tmp_path / "greet.py").write_text(CUSTOM_NODE, encoding="utf-8")
        wf = _write_workflow(
            tmp_path,
            """\
plugins:
  - greet.py
steps:
  - id: s
    type: greet_node
tools:
  - type: greet_tool
""",
        )
        assert validate_workflow_file(wf) == []
        graph, tools, _, _ = load_workflow(wf)
        assert graph.entry_point == "s"
        assert type(graph.nodes["s"]).type == "greet_node"
        assert [t.name for t in tools] == ["greet_tool"]

    def test_default_plugins_dir_auto_loaded(self, tmp_path):
        from draf.plugins import reset_plugins
        from draf.yaml_schema import validate_workflow_file

        reset_plugins()
        (tmp_path / "plugins").mkdir()
        (tmp_path / "plugins" / "extra.py").write_text(
            "from draf.node.registry import node\n\n"
            '@node("extra_node")\n'
            "async def extra_node(ctx, state):\n"
            '    return {"ok": True}\n',
            encoding="utf-8",
        )
        wf = _write_workflow(
            tmp_path,
            """\
steps:
  - id: s
    type: extra_node
""",
        )
        assert validate_workflow_file(wf) == []

    def test_missing_plugin_raises_config_error(self, tmp_path):
        from draf.errors import ConfigError
        from draf.yaml_schema import validate_workflow_file

        wf = _write_workflow(
            tmp_path,
            """\
plugins:
  - nope.py
steps:
  - id: s
    type: transform
""",
        )
        with pytest.raises(ConfigError, match="nope.py"):
            validate_workflow_file(wf)

    def test_unknown_type_without_plugin_still_fails(self, tmp_path):
        from draf.yaml_schema import validate_workflow_file

        wf = _write_workflow(
            tmp_path,
            """\
steps:
  - id: s
    type: not_a_real_node_type_xyz
""",
        )
        errors = validate_workflow_file(wf)
        assert any("unknown node type" in e["message"] for e in errors)

    def test_load_plugin_idempotent(self, tmp_path):
        from draf.plugins import load_plugin_dir, load_plugin_file, reset_plugins

        reset_plugins()
        p = tmp_path / "once.py"
        p.write_text(CUSTOM_NODE, encoding="utf-8")
        load_plugin_file(str(p))
        load_plugin_file(str(p))  # no-op, must not raise
        load_plugin_dir(str(tmp_path))
        assert p.name.endswith(".py")

    def test_plugins_folder_custom_location(self, tmp_path):
        from draf.plugins import reset_plugins
        from draf.yaml_schema import validate_workflow_file

        reset_plugins()
        (tmp_path / "vendor").mkdir()
        (tmp_path / "vendor" / "ext.py").write_text(
            "from draf.node.registry import node\n\n"
            '@node("vendor_node")\n'
            "async def vendor_node(ctx, state):\n"
            '    return {"ok": True}\n',
            encoding="utf-8",
        )
        wf = _write_workflow(
            tmp_path,
            """\
plugins_folder: vendor
steps:
  - id: s
    type: vendor_node
""",
        )
        assert validate_workflow_file(wf) == []

    def test_plugins_folder_defaults_to_plugins(self, tmp_path):
        """Without plugins_folder the default plugins/ dir is still used."""
        from draf.plugins import reset_plugins
        from draf.yaml_schema import validate_workflow_file

        reset_plugins()
        (tmp_path / "plugins").mkdir()
        (tmp_path / "plugins" / "def.py").write_text(
            "from draf.node.registry import node\n\n"
            '@node("default_node")\n'
            "async def default_node(ctx, state):\n"
            '    return {"ok": True}\n',
            encoding="utf-8",
        )
        wf = _write_workflow(
            tmp_path,
            """\
steps:
  - id: s
    type: default_node
""",
        )
        assert validate_workflow_file(wf) == []
