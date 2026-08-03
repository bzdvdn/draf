import json
import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from draf.cli import app

ROOT = Path(__file__).resolve().parents[1]

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
        import asyncio

        from draf.yaml import load_workflow

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

    def test_new_rag_variant_copies_rag_files(self, tmp_path):
        dest = tmp_path / "rag_app"
        result = runner.invoke(
            app, ["new", "Rag App", "--dest", str(dest), "--with", "rag"]
        )
        assert result.exit_code == 0, result.stderr
        assert (dest / "src" / "rag" / "catalog.py").is_file()
        assert (dest / "src" / "rag" / "wiring.py").is_file()
        assert (dest / "src" / "tools" / "rag.py").is_file()
        assert (dest / "data" / "documents" / "sample.md").is_file()
        assert "variants: rag" in result.output

    def test_new_celery_variant_copies_queue_files(self, tmp_path):
        dest = tmp_path / "celery_app"
        result = runner.invoke(
            app, ["new", "Celery App", "--dest", str(dest), "--with", "celery"]
        )
        assert result.exit_code == 0, result.stderr
        assert (dest / "src" / "queue" / "celery_app.py").is_file()
        assert (dest / "src" / "queue" / "ingest.py").is_file()

    def test_new_postgres_variant_copies_deploy_files(self, tmp_path):
        dest = tmp_path / "pg_app"
        result = runner.invoke(
            app, ["new", "Pg App", "--dest", str(dest), "--with", "postgres"]
        )
        assert result.exit_code == 0, result.stderr
        assert (dest / "deploy" / "compose.yaml").is_file()
        assert (dest / ".env.example").is_file()
        compose = (dest / "deploy" / "compose.yaml").read_text()
        assert "pgvector/pgvector:pg16" in compose
        assert "pg_app" in compose  # placeholders rendered

    def test_new_unknown_variant_rejects(self, tmp_path):
        dest = tmp_path / "nope"
        result = runner.invoke(
            app, ["new", "X", "--dest", str(dest), "--with", "bogus"]
        )
        assert result.exit_code != 0
        assert "unknown variant" in result.stderr

    def test_new_does_not_copy_template_manifest(self, tmp_path):
        dest = tmp_path / "clean"
        runner.invoke(app, ["new", "Clean App", "--dest", str(dest)])
        assert not (dest / "template.toml").exists()

    def test_templates_registry(self):
        from draf.scaffold import TEMPLATES, VARIANTS

        assert set(TEMPLATES) == {"fastapi", "cli", "daemon"}
        assert TEMPLATES["fastapi"].entry == "python main.py"
        assert TEMPLATES["cli"].entry == "python cli.py run"
        assert TEMPLATES["daemon"].entry == "python daemon.py --once"
        for manifest in TEMPLATES.values():
            assert {"postgres", "rag", "celery"} <= set(manifest.variants)
        assert {"postgres", "rag", "celery"} <= set(VARIANTS)

    def test_new_rag_project_builds_container(self, tmp_path):
        """The generated rag project builds a container offline (no LLM)."""
        dest = tmp_path / "rag_app"
        result = runner.invoke(
            app, ["new", "Rag App", "--dest", str(dest), "--with", "rag"]
        )
        assert result.exit_code == 0, result.stderr
        env = {**os.environ, "DRAF_CHECKPOINT_DIR": str(tmp_path / "cp")}
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(dest)!r})\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from src.core import build_container\n"
            "c = build_container()\n"
            "assert c.catalog is not None, 'catalog missing'\n"
            "names = {t.name for t in c.tools}\n"
            "assert 'search_catalog' in names and 'find_similar' in names, names\n"
            "print('container-ok')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(dest),
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "container-ok" in proc.stdout


class TestPruneCommand:
    def test_prune_keep_last(self, tmp_path):
        import asyncio

        from draf.checkpoint import Checkpoint, SQLiteCheckpointer

        db = str(tmp_path / "cp.db")
        ck = SQLiteCheckpointer(db)
        try:
            for i in range(3):
                asyncio.run(
                    ck.save(
                        f"run-{i}",
                        Checkpoint(state={}, next_node_id=None, iteration=i),
                    )
                )
        finally:
            ck.close()

        cfg = json.dumps({"type": "sqlite", "path": db})
        result = runner.invoke(
            app,
            ["prune", "--checkpoint", cfg, "--keep-last", "1"],
        )
        assert result.exit_code == 0, result.stderr
        assert "removed 2 checkpoint(s)" in result.stdout

        ck = SQLiteCheckpointer(db)
        try:
            assert asyncio.run(ck.list()) == ["run-2"]
        finally:
            ck.close()

    def test_prune_unknown_type_errors(self):
        cfg = json.dumps({"type": "nope"})
        result = runner.invoke(app, ["prune", "--checkpoint", cfg])
        assert result.exit_code == 1
        assert "unknown checkpoint type" in result.stderr

    def test_prune_noop_without_args(self, tmp_path):
        db = str(tmp_path / "cp.db")
        cfg = json.dumps({"type": "sqlite", "path": db})
        result = runner.invoke(app, ["prune", "--checkpoint", cfg])
        assert result.exit_code == 0, result.stderr
        assert "removed 0 checkpoint(s)" in result.stdout
