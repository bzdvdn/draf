"""Tests for the evaluation harness (draf eval)."""

import asyncio
import json

import httpx
import pytest


def _mock_response(data: dict):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return data

    return MockResponse()


class TestDatasetLoading:
    def test_jsonl(self, tmp_path):
        from draf.eval import load_dataset

        path = tmp_path / "d.jsonl"
        path.write_text(
            json.dumps({"id": "a", "x": 1}) + "\n" + json.dumps({"id": "b"}) + "\n"
        )
        items = load_dataset(str(path))
        assert [i["id"] for i in items] == ["a", "b"]

    def test_json_list(self, tmp_path):
        from draf.eval import load_dataset

        path = tmp_path / "d.json"
        path.write_text(json.dumps([{"id": "a"}, {"id": "b"}]))
        assert len(load_dataset(str(path))) == 2

    def test_csv(self, tmp_path):
        from draf.eval import load_dataset

        path = tmp_path / "d.csv"
        path.write_text("id,input,expected\na,hello,HELLO\nb,hi,HI\n")
        items = load_dataset(str(path))
        assert items[0]["expected"] == "HELLO"

    def test_missing_file(self, tmp_path):
        from draf.eval import load_dataset

        with pytest.raises(FileNotFoundError):
            load_dataset(str(tmp_path / "nope.jsonl"))

    def test_item_state_excludes_metadata(self):
        from draf.eval import item_state

        assert item_state({"id": "a", "query": "q", "expected": "e"}) == {"query": "q"}


class _MockTransport:
    """Drop-in for httpx.AsyncClient.post returning a canned LLM reply."""

    def __init__(self, text: str):
        self._text = text

    async def __call__(self, *args, **kwargs):
        return _mock_response(
            {
                "choices": [{"message": {"role": "assistant", "content": self._text}}],
                "usage": {},
            }
        )


def _simple_graph():
    from draf.graph import Graph
    from draf.node import Transform

    return Graph(
        nodes={
            "n1": Transform(
                {"action": "uppercase", "input_key": "text", "output_key": "answer"}
            )
        },
        edges=[],
        entry_point="n1",
    )


class TestRunEval:
    @pytest.mark.asyncio
    async def test_exact_match(self):
        from draf.eval import run_eval

        graph = _simple_graph()
        report = await run_eval(
            (graph, [], {}, {}),
            [
                {"id": "a", "text": "hello", "expected": "HELLO"},
                {"id": "b", "text": "bye", "expected": "WRONG"},
            ],
            exact=True,
        )
        assert report["total"] == 2
        assert report["passed"] == 1
        assert report["failed"] == 1
        assert report["cases"][0]["verdict"] == "PASS"

    @pytest.mark.asyncio
    async def test_llm_judge(self, monkeypatch):
        from draf.eval import run_eval

        monkeypatch.setattr(
            httpx.AsyncClient, "post", _MockTransport("PASS close enough")
        )
        graph = _simple_graph()
        report = await run_eval(
            (graph, [], {}, {}),
            [{"id": "a", "text": "hello", "expected": "hello world"}],
            judge_model="gpt-4o",
            judge_provider="openai",
        )
        assert report["cases"][0]["verdict"] == "PASS"
        assert "close enough" in report["cases"][0]["reason"]

    @pytest.mark.asyncio
    async def test_llm_judge_fail(self, monkeypatch):
        from draf.eval import run_eval

        monkeypatch.setattr(httpx.AsyncClient, "post", _MockTransport("FAIL wrong"))
        graph = _simple_graph()
        report = await run_eval(
            (graph, [], {}, {}),
            [{"id": "a", "text": "hello", "expected": "totally different"}],
            judge_model="gpt-4o",
            judge_provider="openai",
        )
        assert report["cases"][0]["verdict"] == "FAIL"

    @pytest.mark.asyncio
    async def test_output_key_override(self):
        from draf.eval import run_eval

        graph = _simple_graph()
        report = await run_eval(
            (graph, [], {}, {}),
            [{"id": "a", "text": "hello", "expected": "HELLO"}],
            exact=True,
            output_key="answer",
        )
        assert report["cases"][0]["verdict"] == "PASS"

    @pytest.mark.asyncio
    async def test_max_examples(self):
        from draf.eval import run_eval

        graph = _simple_graph()
        report = await run_eval(
            (graph, [], {}, {}),
            [{"id": "a", "text": "hello"}, {"id": "b", "text": "hi"}],
            exact=True,
            max_examples=1,
        )
        assert report["total"] == 1

    @pytest.mark.asyncio
    async def test_unscored_without_expected(self):
        from draf.eval import run_eval

        graph = _simple_graph()
        report = await run_eval((graph, [], {}, {}), [{"id": "a", "text": "hello"}])
        assert report["unscored"] == 1
        assert report["cases"][0]["verdict"] is None

    @pytest.mark.asyncio
    async def test_case_error_is_collected(self):
        from draf.eval import run_eval
        from draf.graph import Graph
        from draf.node.node import Node

        class Boom(Node):
            type = "boom"

            async def execute(self, ctx, state: dict) -> dict:  # type: ignore[override]
                raise RuntimeError("kaboom")

        graph = Graph(nodes={"n1": Boom()}, edges=[], entry_point="n1")
        report = await run_eval(
            (graph, [], {}, {}), [{"id": "a", "text": "x", "expected": "y"}]
        )
        assert report["errors"] == 1
        assert report["cases"][0]["status"] == "error"

    def test_format_report(self):
        from draf.eval import format_report

        report = {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "unscored": 0,
            "errors": 0,
            "cases": [
                {"id": "a", "verdict": "PASS", "duration_ms": 1.5, "status": "ok"}
            ],
        }
        text = format_report(report)
        assert "passed=1" in text
        assert "a: PASS" in text


class TestCLIEval:
    def test_eval_command(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from draf.cli import app

        runner = CliRunner()
        wf = tmp_path / "wf.yaml"
        wf.write_text(
            "name: t\nsteps:\n  - id: s1\n    type: transform\n    config: "
            "{action: uppercase, input_key: text, output_key: answer}\n"
        )
        ds = tmp_path / "d.jsonl"
        ds.write_text(json.dumps({"id": "a", "text": "hi", "expected": "HI"}) + "\n")

        result = runner.invoke(
            app,
            ["eval", str(wf), "--data", str(ds), "--exact", "--output-key", "answer"],
        )
        assert result.exit_code == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["passed"] == 1

    def test_validate_command_ok(self, tmp_path):
        from typer.testing import CliRunner

        from draf.cli import app

        runner = CliRunner()
        wf = tmp_path / "wf.yaml"
        wf.write_text(
            "name: t\nsteps:\n  - id: s1\n    type: transform\n    config: {action: uppercase}\n"
        )
        result = runner.invoke(app, ["validate", str(wf)])
        assert result.exit_code == 0
        assert "valid workflow" in result.stdout

    def test_validate_command_fails(self, tmp_path):
        from typer.testing import CliRunner

        from draf.cli import app

        runner = CliRunner()
        wf = tmp_path / "wf.yaml"
        wf.write_text("name: t\nsteps:\n  - id: s1\n    type: react_agnt\n")
        result = runner.invoke(app, ["validate", str(wf)])
        assert result.exit_code == 1
        assert "react_agnt" in result.stderr

    def test_version_command(self):
        from typer.testing import CliRunner

        from draf.cli import app

        result = CliRunner().invoke(app, ["version"])
        assert result.exit_code == 0
        assert "draf" in result.stdout

    def test_inspect_command(self, tmp_path):

        from typer.testing import CliRunner

        from draf.checkpoint import Checkpoint, JSONFileCheckpointer
        from draf.cli import app

        cp_dir = tmp_path / "cps"
        cp = JSONFileCheckpointer(str(cp_dir))
        asyncio.run(
            cp.save("r1", Checkpoint(state={"x": 1}, next_node_id=None, iteration=2))
        )

        result = CliRunner().invoke(
            app,
            [
                "inspect",
                "--checkpoint",
                json.dumps({"type": "file", "path": str(cp_dir)}),
                "--checkpoint-id",
                "r1",
            ],
        )
        assert result.exit_code == 0
        assert '"x": 1' in result.stdout
