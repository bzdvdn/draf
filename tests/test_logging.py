import asyncio
import json
import logging

import pytest


def _make_node(state_update=None):
    from draf.node import Node

    class Simple(Node):
        type = "simple"

        async def execute(self, ctx, state):
            if isinstance(state_update, dict):
                state.update(state_update)
            return state_update or {}

    return Simple({})


class TestLoggerNaming:
    def test_get_logger_prefixes_draf(self):
        from draf.logging import get_logger

        assert get_logger("my_app").name == "draf.my_app"
        assert get_logger("draf.graph.execution").name == "draf.graph.execution"
        assert get_logger().name == "draf"
        assert get_logger("").name == "draf"

    def test_get_logger_does_not_add_handlers(self):
        from draf.logging import get_logger

        logger = get_logger("app")
        assert logger.handlers == []


class TestRunContext:
    def test_run_id_ctx_sets_and_restores(self):
        from draf.logging import new_run_id, run_id, run_id_ctx

        assert run_id() == ""
        rid = new_run_id()
        assert len(rid) == 12
        with run_id_ctx(run_id=rid, session_id="sess"):
            assert run_id() == rid
        assert run_id() == ""

    def test_context_propagates_to_child_tasks(self):
        from draf.logging import run_id_ctx

        outer = []

        async def inner():
            from draf.logging import run_id

            outer.append(run_id())

        async def main():
            from draf.logging import run_id

            with run_id_ctx(run_id="task-a"):
                await asyncio.create_task(inner())
                assert run_id() == "task-a"

        asyncio.run(main())
        assert outer == ["task-a"]


class TestContextFilter:
    def test_attaches_ids_to_record(self):
        from draf.logging import ContextFilter, node_id_ctx, run_id_ctx

        record = logging.LogRecord("draf.x", logging.INFO, "", 0, "msg", (), None)
        f = ContextFilter()
        with (
            run_id_ctx(run_id="r1", session_id="s1"),
            node_id_ctx(node_id="n1", node_type="t1"),
        ):
            assert f.filter(record)
        assert record.run_id == "r1"
        assert record.session_id == "s1"
        assert record.node_id == "n1"
        assert record.node_type == "t1"


class TestFormatters:
    def test_text_formatter_renders_context(self):
        from draf.logging import ContextFilter, TextFormatter

        fmt = TextFormatter()
        record = logging.LogRecord(
            "draf.x", logging.INFO, "", 0, "node_start", (), None
        )
        ContextFilter().filter(record)
        record.run_id = "r1"
        record.session_id = "sess"
        record.node_id = "n1"
        record.node_type = "simple"
        out = fmt.format(record)
        assert "[run=r1 session=sess node=n1 type=simple]" in out
        assert out.endswith("node_start")

    def test_text_formatter_omits_empty_context(self):
        from draf.logging import TextFormatter

        fmt = TextFormatter()
        record = logging.LogRecord("draf.x", logging.INFO, "", 0, "run_start", (), None)
        record.run_id = ""
        record.session_id = ""
        record.node_id = ""
        record.node_type = ""
        out = fmt.format(record)
        assert "[run=" not in out

    def test_json_formatter_round_trips(self):
        from draf.logging import ContextFilter, JsonFormatter

        fmt = JsonFormatter()
        record = logging.LogRecord(
            "draf.x", logging.INFO, "", 0, "node_start", (), None
        )
        ContextFilter().filter(record)
        record.run_id = "r1"
        payload = json.loads(fmt.format(record))
        assert payload["event"] == "node_start"
        assert payload["run_id"] == "r1"
        assert payload["level"] == "INFO"
        assert "logger" in payload


class TestConfigureLogging:
    def test_idempotent(self):
        from draf import configure_logging

        configure_logging("INFO")
        configure_logging("DEBUG", format="json")
        root = logging.getLogger()
        draf_handlers = [h for h in root.handlers if getattr(h, "_draf_handler", False)]
        assert len(draf_handlers) == 1
        assert draf_handlers[0].level == logging.DEBUG

    def test_default_level_from_env(self, monkeypatch):
        from draf import configure_logging

        monkeypatch.setenv("DRAF_LOG_LEVEL", "WARNING")
        configure_logging()
        assert logging.getLogger().level == logging.WARNING

    def test_filters_on_handler(self):
        from draf import configure_logging

        configure_logging("INFO")
        root = logging.getLogger()
        handler = next(h for h in root.handlers if getattr(h, "_draf_handler", False))
        filters = [type(f).__name__ for f in handler.filters]
        assert "ContextFilter" in filters
        assert "_DrafOnlyFilter" in filters


class TestGraphLogging:
    async def _run(self, graph, **kwargs):
        return await graph.run(state={}, **kwargs)

    @pytest.mark.asyncio
    async def test_info_skeleton(self, caplog):
        from draf import configure_logging
        from draf.graph import Graph

        configure_logging("INFO")
        caplog.set_level(logging.INFO)
        g = Graph(nodes={"a": _make_node({"x": 1})}, edges=[], entry_point="a")
        await g.run(state={}, checkpoint_id="thread-42")

        messages = [r.message for r in caplog.records]
        assert any(m.startswith("run_start") for m in messages)
        assert "node_start" in messages
        assert any(m.startswith("node_end") for m in messages)
        assert any(m.startswith("run_end status=ok") for m in messages)
        run_start = next(r for r in caplog.records if r.message.startswith("run_start"))
        assert run_start.name.startswith("draf")

    @pytest.mark.asyncio
    async def test_edge_logged(self, caplog):
        from draf import configure_logging
        from draf.graph import Edge, Graph

        configure_logging("INFO")
        caplog.set_level(logging.INFO)
        g = Graph(
            nodes={"a": _make_node({"go": True}), "b": _make_node({})},
            edges=[Edge("a", "b", "go=true")],
            entry_point="a",
        )
        await g.run(state={})

        edges = [r for r in caplog.records if r.message.startswith("edge")]
        assert edges
        assert "a -> b" in edges[0].message

    @pytest.mark.asyncio
    async def test_node_error_logged(self, caplog):
        from draf import configure_logging
        from draf.graph import Graph
        from draf.node import Node

        class Boom(Node):
            type = "boom"

            async def execute(self, ctx, state):
                raise RuntimeError("kaput")

        configure_logging("INFO")
        caplog.set_level(logging.INFO)
        g = Graph(nodes={"a": Boom({})}, edges=[], entry_point="a")
        with pytest.raises(RuntimeError):
            await g.run(state={})

        errors = [r for r in caplog.records if r.message.startswith("node_error")]
        assert errors
        assert errors[0].levelno == logging.ERROR
        assert "kaput" in errors[0].message
        assert any(r.message.startswith("run_end status=error") for r in caplog.records)

    @pytest.mark.asyncio
    async def test_checkpoint_is_debug_only(self, caplog, tmp_path):
        from draf import configure_logging
        from draf.checkpoint import JSONFileCheckpointer
        from draf.graph import Graph

        configure_logging("INFO")
        caplog.set_level(logging.DEBUG)
        cp = JSONFileCheckpointer(str(tmp_path))
        g = Graph(nodes={"a": _make_node({})}, edges=[], entry_point="a")
        await g.run(state={}, checkpointer=cp, checkpoint_id="ck")

        checkpoints = [r for r in caplog.records if r.message.startswith("checkpoint")]
        assert checkpoints

        # INFO must not surface checkpoint detail
        configure_logging("INFO")
        caplog.clear()
        caplog.set_level(logging.INFO)
        await g.run(state={}, checkpointer=cp, checkpoint_id="ck")
        assert not [r for r in caplog.records if r.message.startswith("checkpoint")]

    @pytest.mark.asyncio
    async def test_run_id_context_visible_in_node(self, caplog):
        from draf import configure_logging
        from draf.graph import Graph
        from draf.logging import run_id
        from draf.node import Node

        class Probe(Node):
            type = "probe"
            captured = []

            async def execute(self, ctx, state):
                Probe.captured.append(run_id())
                return {}

        configure_logging("INFO")
        g = Graph(nodes={"a": Probe({})}, edges=[], entry_point="a")
        await g.run(state={}, checkpoint_id="sess-1")
        assert Probe.captured and Probe.captured[0]

    @pytest.mark.asyncio
    async def test_json_format_logs_correlation(self, capsys):
        from draf import configure_logging
        from draf.graph import Graph

        configure_logging("INFO", format="json")
        g = Graph(nodes={"a": _make_node({})}, edges=[], entry_point="a")
        await g.run(state={}, checkpoint_id="thread-42")

        payloads = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("{")
        ]
        start = next(p for p in payloads if p["event"].startswith("run_start"))
        assert start["session_id"] == "thread-42"
        assert start["run_id"]
        node = next(p for p in payloads if p["event"] == "node_start")
        assert node["node_id"] == "a"
        assert node["node_type"] == "simple"


class TestHarnessLogging:
    @pytest.mark.asyncio
    async def test_llm_call_info_and_debug_content(self, mock_llm, caplog):
        from draf import configure_logging
        from draf.flow import Flow
        from draf.node import LLM

        mock_llm.content = "reply with secret=sk-abc123 " + "word " * 400
        flow = Flow("qa")
        flow.step(LLM(model="gpt-4o-mini", prompt="Say {topic}", output_key="a"))
        graph = flow.compile()

        configure_logging("DEBUG")
        caplog.set_level(logging.DEBUG)
        await graph.run(state={"topic": "hi"})

        info = [r for r in caplog.records if r.message.startswith("llm_call")]
        assert info
        assert info[0].levelno == logging.INFO
        assert "model=gpt-4o-mini" in info[0].message
        assert "prompt_tokens=10" in info[0].message

        reqs = [r for r in caplog.records if r.message.startswith("llm_request")]
        assert reqs
        assert "hi" in reqs[0].message

        resps = [r for r in caplog.records if r.message.startswith("llm_response")]
        assert resps
        assert "(truncated" in resps[0].message
        assert "sk-abc123" not in resps[0].message
        assert "word word" in resps[0].message

    @pytest.mark.asyncio
    async def test_llm_content_hidden_at_info(self, mock_llm, caplog):
        from draf import configure_logging
        from draf.flow import Flow
        from draf.node import LLM

        mock_llm.content = "super secret reply"
        flow = Flow("qa")
        flow.step(LLM(model="gpt-4o-mini", prompt="hi", output_key="a"))
        graph = flow.compile()

        configure_logging("INFO")
        caplog.set_level(logging.INFO)
        await graph.run(state={})

        debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert not debug
        assert "super secret reply" not in [r.getMessage() for r in caplog.records]
