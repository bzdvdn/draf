"""Tests for the observability layer: collector wiring and exporters."""

import json

import pytest

from draf.harness.loop import ModelReply


class _FakeHarness:
    @classmethod
    def from_config(cls, cfg, **kwargs):
        return _FakeHarness()

    async def call(self, messages, tools=None, stream=False, **kwargs):
        if getattr(self, "on_llm_payload", None) is not None:
            await self.on_llm_payload(
                "ollama",
                "m",
                messages,
                "hello",
                {"prompt": 5, "completion": 2},
                1.5,
                False,
            )
        return ModelReply(
            data={},
            message={"role": "assistant", "content": "hello"},
            content="hello",
            usage={"prompt": 5, "completion": 2},
            latency_ms=1.5,
        )


class _CtxHarness(_FakeHarness):
    """Harness that surfaces its on_llm_payload hook after construction."""

    captured = None

    def __init__(self):
        self.on_llm_payload = None


@pytest.fixture
def fake_agent_harness(monkeypatch):
    from draf.node import agent as agent_mod

    monkeypatch.setattr(agent_mod, "Harness", _CtxHarness)
    yield _CtxHarness


async def _run_single(graph, owner="ana"):
    state = {"messages": [{"role": "user", "content": "hi"}]}
    return await graph.run(state, owner=owner)


def test_run_model_to_dict():
    from draf.observability import Run

    run = Run(name="r", status="ok", total_ms=1.0, owner="ana")
    data = run.to_dict()
    assert data["owner"] == "ana"
    assert data["llm_calls"] == []
    assert data["prompt_tokens"] == 0


def test_topology_from_graph():
    from draf.flow import Flow
    from draf.observability import topology_from_graph

    flow = Flow("t")
    flow.llm(model="m", system="s", output_key="o")
    graph = flow.compile()
    topo = topology_from_graph(graph)
    assert topo.to_dict()["nodes"] == [{"id": "llm_chat_1", "type": "llm_chat"}]
    assert topo.to_dict()["edges"] == []


def test_observer_without_exporter(fake_agent_harness):
    import asyncio

    from draf.flow import Flow
    from draf.observability import GraphObserver

    flow = Flow("t")
    flow.react(model="m", system="s", messages_key="messages")
    graph = flow.compile()

    observer = GraphObserver("flow")
    asyncio.run(_run_single_with(observer, graph))

    run = observer.build()
    assert run.name == "flow"
    assert any(node.node_type == "react_agent" for node in run.nodes)
    calls = run.llm_calls
    assert calls, "expected an LLM call captured"
    assert calls[0].messages[0]["role"] == "user"
    assert calls[0].response == "hello"


async def _run_single_with(observer, graph):
    state = {"messages": [{"role": "user", "content": "hi"}]}
    await graph.run(
        state,
        owner="ana",
        tracer=observer.tracer,
        on_llm_payload=observer.on_llm_payload,
    )


def test_sqlite_exporter_roundtrip(tmp_path, fake_agent_harness):
    import asyncio

    from draf.flow import Flow
    from draf.observability import GraphObserver, SQLiteExporter, topology_from_graph

    flow = Flow("roundtrip")
    flow.react(model="m", system="s", messages_key="messages")
    graph = flow.compile()

    db = tmp_path / "traces.db"
    observer = GraphObserver(
        "roundtrip",
        exporter=SQLiteExporter(str(db)),
        topology=topology_from_graph(graph),
        owner="ana",
        checkpoint_id="c1",
    )
    asyncio.run(_run_single_with(observer, graph))
    observer.export()
    observer.close()

    exp = SQLiteExporter(str(db))
    try:
        page = exp.list_runs()
        rows = page["items"]
        assert len(rows) == 1
        assert page["total"] == 1
        assert rows[0]["name"] == "roundtrip"
        assert rows[0]["owner"] == "ana"
        assert rows[0]["checkpoint_id"] == "c1"
        assert rows[0]["tags"] == []
        assert rows[0]["notes"] == ""

        run = exp.get_run(rows[0]["run_id"])
        assert run["topology"]["nodes"] == [
            {"id": "react_agent_1", "type": "react_agent"},
            {"id": "tool_exec_2", "type": "tool_exec"},
        ]
        assert run["nodes"], "expected node spans"
        llm_calls = run["llm_calls"]
        assert llm_calls and llm_calls[0]["messages"][0]["role"] == "user"
        assert llm_calls[0]["response"] == "hello"
        assert run["prompt_tokens"] >= 5
        assert run["tags"] == []
        assert run["notes"] == ""
    finally:
        exp.close()


def test_sqlite_filters_and_pagination(tmp_path):
    from draf.observability import Run, SQLiteExporter

    db = tmp_path / "traces.db"
    exp = SQLiteExporter(str(db))
    try:
        runs = [
            Run(name="alpha", status="ok", total_ms=1.0, owner="ana", tags=["perf"]),
            Run(
                name="alpha",
                status="err",
                total_ms=2.0,
                owner="bob",
                tags=["perf", "debug"],
                notes="flaky",
            ),
            Run(name="beta", status="ok", total_ms=3.0, owner="ana", tags=["prod"]),
            Run(name="gamma", status="ok", total_ms=4.0, owner="carol", tags=[]),
        ]
        for run in runs:
            exp.export(run)

        page = exp.list_runs()
        assert page["total"] == 4
        assert [r["run_id"] for r in page["items"]] == [4, 3, 2, 1]

        page = exp.list_runs(status="ok")
        assert page["total"] == 3

        page = exp.list_runs(name="alp")
        assert page["total"] == 2

        page = exp.list_runs(owner="BOB")  # case-insensitive substring
        assert page["total"] == 1
        assert page["items"][0]["owner"] == "bob"

        page = exp.list_runs(tag="perf")
        assert page["total"] == 2

        page = exp.list_runs(limit=2, offset=0)
        assert len(page["items"]) == 2
        assert page["total"] == 4
        first = page["items"][0]["run_id"]
        page = exp.list_runs(limit=2, offset=2)
        assert len(page["items"]) == 2
        assert page["items"][0]["run_id"] != first
    finally:
        exp.close()


def test_sqlite_update_run(tmp_path):
    from draf.observability import Run, SQLiteExporter

    db = tmp_path / "traces.db"
    exp = SQLiteExporter(str(db))
    try:
        exp.export(Run(name="r", status="ok", total_ms=1.0))
        run_id = exp.list_runs()["items"][0]["run_id"]

        assert exp.update_run(run_id, tags=["a", "b"]) is True
        assert exp.update_run(run_id, notes="hello world") is True
        run = exp.get_run(run_id)
        assert run["tags"] == ["a", "b"]
        assert run["notes"] == "hello world"

        # partial update keeps the other field
        assert exp.update_run(run_id, tags=["only"]) is True
        run = exp.get_run(run_id)
        assert run["tags"] == ["only"]
        assert run["notes"] == "hello world"

        # unknown run / no fields to update
        assert exp.update_run(999, tags=["x"]) is False
        assert exp.update_run(run_id) is False
    finally:
        exp.close()


def test_sqlite_migrates_missing_columns(tmp_path):
    import sqlite3

    from draf.observability import SQLiteExporter

    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
        "status TEXT NOT NULL, total_ms REAL NOT NULL, owner TEXT, checkpoint_id TEXT, "
        "prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL "
        "DEFAULT 0, topology TEXT NOT NULL, created_at REAL NOT NULL)"
    )
    conn.commit()
    conn.close()

    exp = SQLiteExporter(str(db))
    try:
        page = exp.list_runs()
        assert page == {"items": [], "total": 0}
    finally:
        exp.close()


def test_jsonl_exporter(tmp_path, fake_agent_harness):
    import asyncio

    from draf.flow import Flow
    from draf.observability import GraphObserver, JsonlExporter

    flow = Flow("j")
    flow.react(model="m", system="s", messages_key="messages")
    graph = flow.compile()

    path = tmp_path / "traces.jsonl"
    observer = GraphObserver("j", exporter=JsonlExporter(str(path)))
    asyncio.run(_run_single_with(observer, graph))
    observer.export()
    observer.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["name"] == "j"
    assert data["llm_calls"][0]["response"] == "hello"
