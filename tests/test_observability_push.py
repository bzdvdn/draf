"""Tests for remote trace exporters, the YAML observability builder, and
the standalone obs-server ingest path."""

import base64
import json
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from draf.errors import ConfigError
from draf.observability.exporter import CompositeExporter, SQLiteExporter, TraceExporter
from draf.observability.model import LLMCall, NodeSpan, Run
from draf.observability.push import (
    HttpExporter,
    LangfuseExporter,
    LangsmithExporter,
)

pytest.importorskip("fastapi")


class _FakeExporter(TraceExporter):
    def __init__(self):
        self.runs = []
        self.closed = False

    def export(self, run: Run) -> None:
        self.runs.append(run)

    def close(self) -> None:
        self.closed = True


class _Handler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).requests.append(
            {"path": self.path, "headers": dict(self.headers), "body": body}
        )
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


@pytest.fixture
def http_server():
    _Handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _sample_run() -> Run:
    node = NodeSpan(node_id="n1", node_type="agent", start_ms=0.0, end_ms=100.0)
    node.llm_calls.append(
        LLMCall(
            node_id="n1",
            provider="ollama",
            model="llama3",
            messages=[{"role": "user", "content": "hi"}],
            response="hello",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=50.0,
            cached=False,
        )
    )
    return Run(
        name="test-flow",
        status="ok",
        total_ms=100.0,
        owner="alice",
        checkpoint_id="c1",
        created_at=1700000000.0,
        nodes=[node],
    )


def _base(http_server) -> str:
    return f"http://127.0.0.1:{http_server.server_address[1]}"


# ---------------------------------------------------------------------------
# CompositeExporter
# ---------------------------------------------------------------------------


def test_composite_exporter_fans_out():
    a, b = _FakeExporter(), _FakeExporter()
    exp = CompositeExporter([a, b])
    run = _sample_run()
    exp.export(run)
    assert a.runs == [run]
    assert b.runs == [run]
    exp.close()
    assert a.closed and b.closed


def test_composite_exporter_isolates_failures():
    class _Boom(_FakeExporter):
        def export(self, run: Run) -> None:
            raise RuntimeError("boom")

    a, b = _Boom(), _FakeExporter()
    exp = CompositeExporter([a, b])
    exp.export(_sample_run())  # must not raise
    assert len(b.runs) == 1


# ---------------------------------------------------------------------------
# HttpExporter
# ---------------------------------------------------------------------------


def test_http_exporter_posts_run(http_server):
    exp = HttpExporter(_base(http_server) + "/ingest", retries=0)
    exp.export(_sample_run())
    exp.close()
    assert len(_Handler.requests) == 1
    req = _Handler.requests[0]
    assert req["path"] == "/ingest"
    assert req["headers"]["Content-Type"] == "application/json"
    body = req["body"]
    assert body["name"] == "test-flow"
    assert body["owner"] == "alice"
    assert body["created_at"] == 1700000000.0
    assert body["nodes"][0]["node_id"] == "n1"
    assert body["nodes"][0]["llm_calls"][0]["response"] == "hello"


def test_http_exporter_swallows_connection_errors():
    exp = HttpExporter("http://127.0.0.1:1/nope", retries=0)
    exp.export(_sample_run())
    exp.close()  # must not raise


# ---------------------------------------------------------------------------
# LangfuseExporter
# ---------------------------------------------------------------------------


def test_langfuse_exporter_mapping(http_server):
    exp = LangfuseExporter(_base(http_server), "pk", "sk", retries=0)
    exp.export(_sample_run())
    exp.close()
    req = _Handler.requests[0]
    assert req["path"] == "/api/public/traces"
    expected_auth = "Basic " + base64.b64encode(b"pk:sk").decode()
    assert req["headers"]["Authorization"] == expected_auth

    body = req["body"]
    assert body["name"] == "test-flow"
    assert body["userId"] == "alice"
    assert body["sessionId"] == "c1"
    spans = [o for o in body["observations"] if o["type"] == "span"]
    gens = [o for o in body["observations"] if o["type"] == "generation"]
    assert len(spans) == 1
    assert spans[0]["id"] == "n1"
    assert len(gens) == 1
    assert gens[0]["parentObservationId"] == "n1"
    assert gens[0]["model"] == "llama3"
    assert gens[0]["input"][0]["content"] == "hi"
    assert gens[0]["usage"] == {"input": 10, "output": 5}


def _lower_headers(req: dict) -> dict:
    return {k.lower(): v for k, v in req["headers"].items()}


# ---------------------------------------------------------------------------
# LangsmithExporter
# ---------------------------------------------------------------------------


def test_langsmith_exporter_mapping(http_server):
    exp = LangsmithExporter(_base(http_server), "secret", project="p", retries=0)
    exp.export(_sample_run())
    exp.close()
    req = _Handler.requests[0]
    assert req["path"] == "/runs/batch"
    headers = _lower_headers(req)
    assert headers["x-api-key"] == "secret"
    assert headers["x-langchain-project"] == "p"

    runs = req["body"]
    assert runs[0]["run_type"] == "chain"
    assert runs[1]["run_type"] == "chain"
    assert runs[1]["parent_run_id"] == runs[0]["id"]
    assert runs[2]["run_type"] == "llm"
    assert runs[2]["parent_run_id"] == runs[1]["id"]
    assert runs[2]["inputs"]["messages"][0]["content"] == "hi"


# ---------------------------------------------------------------------------
# Builder from the workflow observability: block
# ---------------------------------------------------------------------------


def test_build_observability_persists_to_db(tmp_path):
    from draf.observability import build_observability

    db = tmp_path / "data" / "traces.db"
    observer = build_observability({"db": str(db)}, name="wf")
    assert observer is not None
    observer.export()

    exp = SQLiteExporter(str(db))
    try:
        assert exp.list_runs()["total"] == 1
        assert exp.list_runs()["items"][0]["name"] == "wf"
    finally:
        exp.close()


def test_build_observability_fans_out_to_db_and_webhook(tmp_path, http_server):
    from draf.observability import build_observability

    db = tmp_path / "t.db"
    url = _base(http_server) + "/ingest"
    observer = build_observability(
        {"db": str(db), "export": [{"type": "webhook", "url": url}]},
        name="wf",
    )
    observer.export()

    exp = SQLiteExporter(str(db))
    try:
        assert exp.list_runs()["total"] == 1
    finally:
        exp.close()
    assert len(_Handler.requests) == 1
    assert _Handler.requests[0]["path"] == "/ingest"
    assert _Handler.requests[0]["body"]["name"] == "wf"


def test_build_observability_none_without_config():
    from draf.observability import build_observability

    assert build_observability(None) is None
    assert build_observability({}) is None


def test_build_remote_exporter_unknown_type():
    from draf.observability import build_remote_exporter

    with pytest.raises(ConfigError):
        build_remote_exporter({"type": "bogus"})


def test_build_remote_exporter_missing_secret():
    from draf.observability import build_remote_exporter

    with pytest.raises(ConfigError):
        build_remote_exporter(
            {"type": "langfuse", "host": "http://x", "public_key_env": "NO_SUCH_1"}
        )


def test_build_observer_factory_shares_exporter(tmp_path):
    from draf.observability import build_observer_factory

    factory = build_observer_factory({"db": str(tmp_path / "t.db")}, name="wf")
    assert factory is not None
    first, second = factory(), factory()
    assert first.exporter is second.exporter
    first.export()
    second.export()

    exp = SQLiteExporter(str(tmp_path / "t.db"))
    try:
        assert exp.list_runs()["total"] == 2
    finally:
        exp.close()


def test_workflow_yaml_accepts_observability_block():
    from draf.yaml_schema import validate_workflow

    errors = validate_workflow(
        {
            "name": "wf",
            "steps": [{"id": "a", "type": "transform", "config": {"action": "trim"}}],
            "observability": {
                "db": "./data/traces.db",
                "export": [{"type": "webhook", "url": "http://x/ingest"}],
            },
        }
    )
    assert errors == []


# ---------------------------------------------------------------------------
# Ingest + standalone obs server
# ---------------------------------------------------------------------------


def test_ingest_router_persists_run(tmp_path):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from draf.observability.api import attach_ingest

    app = FastAPI()
    exp = SQLiteExporter(str(tmp_path / "t.db"))
    attach_ingest(app, exp)
    client = TestClient(app)

    resp = client.post("/obs/ingest", json=_sample_run().to_dict())
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert exp.list_runs()["total"] == 1
    assert exp.get_run(1)["name"] == "test-flow"


def test_obs_server_app(tmp_path):
    from starlette.testclient import TestClient

    from draf.observability.server import build_server

    client = TestClient(build_server(str(tmp_path / "t.db")))
    assert client.get("/obs/ui").status_code == 200

    resp = client.post("/obs/ingest", json=_sample_run().to_dict())
    assert resp.status_code == 200
    page = client.get("/obs/runs").json()
    assert page["total"] == 1
    assert page["items"][0]["owner"] == "alice"


# ---------------------------------------------------------------------------
# CLI: draf run traces a YAML workflow automatically
# ---------------------------------------------------------------------------


def test_cli_run_persists_trace(tmp_path):
    from typer.testing import CliRunner

    from draf.cli import app

    wf = tmp_path / "wf.yaml"
    wf.write_text(
        textwrap.dedent(
            """
            name: traced-workflow
            observability:
              db: ./data/traces.db
            state:
              initial:
                title: "  hi  "
            steps:
              - id: trim
                type: transform
                config:
                  action: trim
                  input_key: title
                  output_key: out
            """
        )
    )
    result = CliRunner().invoke(app, ["run", "-f", str(wf)])
    assert result.exit_code == 0, result.stderr

    exp = SQLiteExporter(str(tmp_path / "data" / "traces.db"))
    try:
        assert exp.list_runs()["total"] == 1
        run = exp.get_run(1)
        assert run["name"] == "traced-workflow"
        assert run["nodes"][0]["node_id"] == "trim"
    finally:
        exp.close()


def test_cli_run_without_observability_is_untouched(tmp_path):
    from typer.testing import CliRunner

    from draf.cli import app

    wf = tmp_path / "wf.yaml"
    wf.write_text(
        textwrap.dedent(
            """
            name: plain
            steps:
              - id: trim
                type: transform
                config: {action: trim, input_key: title, output_key: out}
            state:
              initial: {title: "  x  "}
            """
        )
    )
    result = CliRunner().invoke(app, ["run", "-f", str(wf)])
    assert result.exit_code == 0, result.stderr
    assert not (tmp_path / "traces.db").exists()
