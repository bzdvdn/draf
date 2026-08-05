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


class _ToolCallHarness(_CtxHarness):
    """Emulates an agent loop: tool call on the first call, answer after."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    @classmethod
    def from_config(cls, cfg, **kwargs):
        return cls()

    async def call(self, messages, tools=None, stream=False, **kwargs):
        if getattr(self, "on_llm_payload", None) is not None:
            await self.on_llm_payload(
                "ollama",
                "m",
                messages,
                "ok",
                {"prompt": 5, "completion": 2},
                1.5,
                False,
            )
        # Once a tool result is in the history, answer for real.
        if any(m.get("role") == "tool" for m in messages):
            return ModelReply(
                data={},
                message={"role": "assistant", "content": "HI"},
                content="HI",
                usage={"prompt": 5, "completion": 2},
                latency_ms=1.5,
            )
        return ModelReply(
            data={},
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "uppercase",
                            "arguments": '{"text": "hi"}',
                        },
                    }
                ],
            },
            content="",
            usage={"prompt": 5, "completion": 2},
            latency_ms=1.5,
        )


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


def test_tool_call_model_roundtrip():
    from draf.observability import NodeSpan, SpanEvent, ToolCall

    span = NodeSpan(
        node_id="agent",
        node_type="react_agent",
        start_ms=0.0,
        tool_calls=[
            ToolCall(name="current_time", args='{"x": 1}', result="12:00", ok=True),
            ToolCall(name="failing", args="{}", result="Error: boom", ok=False),
        ],
        events=[SpanEvent(kind="llm", index=0), SpanEvent(kind="tool", index=0)],
    )
    data = span.to_dict()
    assert data["tool_calls"] == [
        {"name": "current_time", "args": '{"x": 1}', "result": "12:00", "ok": True},
        {"name": "failing", "args": "{}", "result": "Error: boom", "ok": False},
    ]
    assert data["events"] == [{"kind": "llm", "index": 0}, {"kind": "tool", "index": 0}]
    restored = NodeSpan.from_dict(data)
    assert restored.tool_calls[0].name == "current_time"
    assert restored.tool_calls[1].ok is False
    assert [(e.kind, e.index) for e in restored.events] == [("llm", 0), ("tool", 0)]


async def _emit_payload(observer, messages, response="done"):
    await observer.on_llm_payload(
        "ollama", "m", messages, response, {"prompt": 5, "completion": 2}, 1.5, False
    )


def test_collector_captures_tool_calls():
    import asyncio

    from draf.observability import GraphObserver

    observer = GraphObserver("t")
    observer.tracer.node_start("agent", "react_agent")

    asyncio.run(
        _emit_payload(
            observer,
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "current_time", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "12:00 UTC"},
            ],
        )
    )

    span = observer.build().nodes[0]
    assert len(span.tool_calls) == 1
    call = span.tool_calls[0]
    assert call.name == "current_time"
    assert call.args == "{}"
    assert call.result == "12:00 UTC"
    assert call.ok is True
    # The tool ran before this call (it is part of the request history).
    assert [(e.kind, e.index) for e in span.events] == [("tool", 0), ("llm", 0)]


def test_collector_backfills_tool_result_without_duplication():
    import asyncio

    from draf.observability import GraphObserver

    observer = GraphObserver("t")
    observer.tracer.node_start("agent", "react_agent")

    # Round 1: the model asks for a tool; its result is not here yet.
    asyncio.run(
        _emit_payload(
            observer,
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "x"}'},
                        }
                    ],
                },
            ],
        )
    )
    # Round 2: full history — the previous tool_call block repeats, now
    # followed by its result and a fresh tool call.
    asyncio.run(
        _emit_payload(
            observer,
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "x"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "three results"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "y"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_2", "content": "Error: timeout"},
            ],
        )
    )

    span = observer.build().nodes[0]
    assert [c.name for c in span.tool_calls] == ["search", "search"]
    assert span.tool_calls[0].result == "three results"  # backfilled
    assert span.tool_calls[0].ok is True
    assert span.tool_calls[1].result == "Error: timeout"
    assert span.tool_calls[1].ok is False
    # Chronology: llm round 1 (requested tool 1) -> tool 1 -> llm round 2
    # (re-saw tool 1, requested tool 2) -> tool 2.  No duplicates.
    assert [(e.kind, e.index) for e in span.events] == [
        ("tool", 0),
        ("llm", 0),
        ("tool", 1),
        ("llm", 1),
    ]


def test_collector_accepts_graph_signal_tool_shape():
    import asyncio

    from draf.observability import GraphObserver

    observer = GraphObserver("t")
    observer.tracer.node_start("agent", "react_agent")

    asyncio.run(
        _emit_payload(
            observer,
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_9", "name": "read_file", "args": {"p": "a.txt"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_9", "content": "file body"},
            ],
        )
    )

    call = observer.build().nodes[0].tool_calls[0]
    assert call.name == "read_file"
    assert call.args == '{"p": "a.txt"}'  # dict args are serialised
    assert call.result == "file body"


def test_full_loop_captures_tool_call(monkeypatch):
    import asyncio

    from draf.flow import Flow
    from draf.node import agent as agent_mod
    from draf.observability import GraphObserver
    from draf.tool.tool import Tool

    class Uppercase(Tool):
        name = "uppercase"
        description = "Convert a string to uppercase."

        def run(self, **kwargs) -> str:
            return kwargs["text"].upper()

    monkeypatch.setattr(agent_mod, "Harness", _ToolCallHarness)

    flow = Flow("t")
    flow.react(
        model="m",
        system="s",
        messages_key="messages",
        use_tools=["uppercase"],
        input_key="input",
        output_key="output",
    )
    graph = flow.compile()

    observer = GraphObserver("flow")

    async def run():
        return await graph.run(
            {"input": "hello", "messages": []},
            tools=[Uppercase()],
            tracer=observer.tracer,
            on_llm_payload=observer.on_llm_payload,
            max_iterations=10,
        )

    state = asyncio.run(run())
    assert state.get("output") == "HI"

    agent_span = next(n for n in observer.build().nodes if n.node_type == "react_agent")
    assert len(agent_span.tool_calls) == 1
    call = agent_span.tool_calls[0]
    assert call.name == "uppercase"
    assert call.args == '{"text": "hi"}'
    assert call.result == "HI"
    assert call.ok is True
    # Sequence across the agent's two LLM rounds: llm -> tool -> llm.
    assert [(e.kind, e.index) for e in agent_span.events] == [
        ("llm", 0),
        ("tool", 0),
        ("llm", 1),
    ]


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


def test_sqlite_exporter_roundtrip(tmp_path, monkeypatch):
    import asyncio

    from draf.flow import Flow
    from draf.node import agent as agent_mod
    from draf.observability import GraphObserver, SQLiteExporter, topology_from_graph
    from draf.tool.tool import Tool

    class Uppercase(Tool):
        name = "uppercase"
        description = "Convert a string to uppercase."

        def run(self, **kwargs) -> str:
            return kwargs["text"].upper()

    monkeypatch.setattr(agent_mod, "Harness", _ToolCallHarness)

    flow = Flow("roundtrip")
    flow.react(
        model="m",
        system="s",
        messages_key="messages",
        use_tools=["uppercase"],
        input_key="input",
        output_key="output",
    )
    graph = flow.compile()

    db = tmp_path / "traces.db"
    observer = GraphObserver(
        "roundtrip",
        exporter=SQLiteExporter(str(db)),
        topology=topology_from_graph(graph),
        owner="ana",
        checkpoint_id="c1",
    )
    asyncio.run(
        graph.run(
            {"input": "hello", "messages": []},
            tools=[Uppercase()],
            tracer=observer.tracer,
            on_llm_payload=observer.on_llm_payload,
            max_iterations=10,
        )
    )
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
        assert llm_calls
        assert any(m["role"] == "user" for m in llm_calls[0]["messages"])
        assert run["prompt_tokens"] >= 5
        assert run["tags"] == []
        assert run["notes"] == ""
        # Tool calls + the llm/tool sequence survive the SQLite round-trip.
        agent = next(n for n in run["nodes"] if n["node_type"] == "react_agent")
        assert [t["name"] for t in agent["tool_calls"]] == ["uppercase"]
        assert agent["tool_calls"][0]["result"] == "HI"
        assert agent["events"] == [
            {"kind": "llm", "index": 0},
            {"kind": "tool", "index": 0},
            {"kind": "llm", "index": 1},
        ]
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
