import json

import pytest
from typer.testing import CliRunner

from draf.cli import app

runner = CliRunner()

SIMPLE_YAML = """\
name: test-workflow
steps:
  - id: step1
    type: transform
    config: {action: uppercase, input_key: text, output_key: out}
"""

TOOL_YAML = """\
name: tool-workflow
tools:
  - type: calculator
steps:
  - id: step1
    type: transform
    config: {action: uppercase, input_key: text, output_key: out}
"""


class TestLoadWorkflow:
    def test_basic_workflow(self, tmp_path):
        from draf.yaml import load_workflow
        path = tmp_path / "wf.yaml"
        path.write_text(SIMPLE_YAML)
        graph, tools, initial_state, reducers = load_workflow(str(path))
        assert graph.entry_point == "step1"
        assert tools == []
        assert initial_state == {}
        assert reducers == {}

    def test_with_tools(self, tmp_path):
        from draf.yaml import load_workflow
        path = tmp_path / "wf.yaml"
        path.write_text(TOOL_YAML)
        graph, tools, initial_state, reducers = load_workflow(str(path))
        assert len(tools) == 1
        assert tools[0].name == "calculator"

    def test_runs_to_completion(self, tmp_path):
        from draf.yaml import load_workflow
        import asyncio
        path = tmp_path / "wf.yaml"
        path.write_text(SIMPLE_YAML)
        graph, tools, initial_state, reducers = load_workflow(str(path))
        result = asyncio.run(graph.run({"text": "hello"}, tools=tools))
        assert result["out"] == "HELLO"


STATE_YAML = """\
name: test-workflow
state:
  initial:
    text: hello
steps:
  - id: step1
    type: transform
    config: {action: uppercase, input_key: text, output_key: out}
"""

STATE_HI_YAML = """\
name: test-workflow
state:
  initial:
    text: hi
steps:
  - id: step1
    type: transform
    config: {action: uppercase, input_key: text, output_key: out}
"""


class TestCLI:
    def test_run_basic(self, tmp_path):
        path = tmp_path / "wf.yaml"
        path.write_text(STATE_YAML)
        result = runner.invoke(app, ["--file", str(path)])
        assert result.exit_code == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["out"] == "HELLO"

    def test_run_pretty(self, tmp_path):
        path = tmp_path / "wf.yaml"
        path.write_text(STATE_HI_YAML)
        result = runner.invoke(app, ["--file", str(path), "--pretty"])
        assert result.exit_code == 0, result.stderr
        assert '"out"' in result.stdout

    def test_run_output_file(self, tmp_path):
        path = tmp_path / "wf.yaml"
        path.write_text(STATE_HI_YAML)
        out_path = tmp_path / "result.json"
        result = runner.invoke(
            app,
            ["--file", str(path), "--output", str(out_path)],
        )
        assert result.exit_code == 0, result.stderr
        data = json.loads(out_path.read_text())
        assert data["out"] == "HI"

    def test_run_missing_file_errors(self, tmp_path):
        result = runner.invoke(app, ["--file", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0
        assert "error" in result.stderr

    def test_no_command_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code != 0
        assert "Usage" in result.stderr
