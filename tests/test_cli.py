import json

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

    def test_run_trace_prints_json_report(self, tmp_path):
        path = tmp_path / "wf.yaml"
        path.write_text(STATE_YAML)
        result = runner.invoke(app, ["--file", str(path), "--trace"])
        assert result.exit_code == 0, result.stderr
        report = json.loads(result.stderr)
        assert report["summary"]["status"] == "ok"
        assert report["summary"]["node_count"] == 1
        assert report["events"][0]["kind"] == "run_start"

    def test_run_missing_file_errors(self, tmp_path):
        result = runner.invoke(app, ["--file", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0
        assert "error" in result.stderr

    def test_no_command_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code != 0
        assert "Usage" in result.stderr


class TestNew:
    def test_new_scaffolds_project(self, tmp_path):
        dest = tmp_path / "my_app"
        result = runner.invoke(app, ["new", "My App", "--dest", str(dest)])
        assert result.exit_code == 0, result.stderr
        assert (dest / "src" / "graphs" / "build.py").is_file()
        assert (dest / "app.py").is_file()
        assert (dest / "src" / "config" / "config.py").is_file()
        assert (dest / "src" / "api" / "router.py").is_file()
        assert (dest / "src" / "api" / "chat" / "router.py").is_file()
        assert (dest / "main.py").is_file()
        readme = (dest / "README.md").read_text()
        assert "My App" in readme
        assert "my_app" in readme

    def test_new_cli_template(self, tmp_path):
        dest = tmp_path / "cli_app"
        result = runner.invoke(
            app, ["new", "Cli App", "--dest", str(dest), "--template", "cli"]
        )
        assert result.exit_code == 0, result.stderr
        assert (dest / "cli.py").is_file()
        assert not (dest / "app.py").exists()
        assert not (dest / "src" / "api").exists()
        assert (dest / "src" / "graphs" / "build.py").is_file()
        assert (dest / "tests" / "test_cli.py").is_file()

    def test_new_daemon_template(self, tmp_path):
        dest = tmp_path / "daemon_app"
        result = runner.invoke(
            app, ["new", "Daemon App", "--dest", str(dest), "--template", "daemon"]
        )
        assert result.exit_code == 0, result.stderr
        assert (dest / "daemon.py").is_file()
        assert (dest / "src" / "queue" / "__init__.py").is_file()
        assert not (dest / "app.py").exists()
        assert (dest / "src" / "graphs" / "build.py").is_file()

    def test_new_unknown_template_rejects(self, tmp_path):
        dest = tmp_path / "nope"
        result = runner.invoke(
            app, ["new", "X", "--dest", str(dest), "--template", "bogus"]
        )
        assert result.exit_code != 0
        assert "unknown template" in result.stderr

    def test_new_refuses_existing_nonempty(self, tmp_path):
        dest = tmp_path / "taken"
        dest.mkdir()
        (dest / "x.txt").write_text("x")
        result = runner.invoke(app, ["new", "X", "--dest", str(dest)])
        assert result.exit_code != 0
        assert "already exists" in result.stderr
