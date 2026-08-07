import asyncio

import pytest

from teff.provider import ProviderRegistry


def _make_node(state_update):
    from teff.node import Node

    class Simple(Node):
        type = "simple"

        async def execute(self, ctx, state):
            state.update(state_update)
            return state_update

    return Simple({})


class TestRunTracerBasics:
    @pytest.mark.asyncio
    async def test_records_run_and_node_events(self):
        from teff.graph import Graph
        from teff.trace import RunTracer

        g = Graph(nodes={"a": _make_node({"x": 1})}, edges=[], entry_point="a")
        tracer = RunTracer()
        await g.run(state={}, tracer=tracer)

        kinds = [ev.kind for ev in tracer.events]
        assert kinds[0] == "run_start"
        assert kinds[-1] == "run_end"
        assert "node_start" in kinds
        assert "node_end" in kinds

        start = next(ev for ev in tracer.events if ev.kind == "node_start")
        assert start.node_id == "a"
        assert start.node_type == "simple"

    @pytest.mark.asyncio
    async def test_node_timing_measured(self):
        from teff.graph import Graph
        from teff.node import Node
        from teff.trace import RunTracer

        class Slow(Node):
            type = "slow"

            async def execute(self, ctx, state):
                await asyncio.sleep(0.005)
                return {}

        g = Graph(nodes={"a": Slow({})}, edges=[], entry_point="a")
        tracer = RunTracer()
        await g.run(state={}, tracer=tracer)

        end = next(ev for ev in tracer.events if ev.kind == "node_end")
        assert end.duration_ms is not None
        assert end.duration_ms >= 4.0

    @pytest.mark.asyncio
    async def test_summary_aggregates_nodes(self):
        from teff.graph import Graph
        from teff.node import Node
        from teff.trace import RunTracer

        class One(Node):
            type = "one"

            async def execute(self, ctx, state):
                return {"s": "1"}

        g = Graph(
            nodes={"a": One({}), "b": One({})},
            edges=[from_to("a", "b")],
            entry_point="a",
        )
        tracer = RunTracer()
        await g.run(state={}, tracer=tracer)

        s = tracer.summary()
        assert s.status == "ok"
        assert s.node_count == 2
        assert set(s.nodes) == {"a", "b"}
        assert s.nodes["a"].runs == 1
        assert s.total_ms >= 0


def from_to(src, dst):
    from teff.graph import Edge

    return Edge(src, dst)


class TestRunTracerEdges:
    @pytest.mark.asyncio
    async def test_conditional_edge_recorded(self):
        from teff.graph import Graph
        from teff.trace import RunTracer

        g = Graph(
            nodes={
                "s": _make_node({"lang": "go"}),
                "t": _make_node({"ok": True}),
            },
            edges=[from_to_cond("s", "t", "lang=go")],
            entry_point="s",
        )
        tracer = RunTracer()
        await g.run(state={}, tracer=tracer)

        edge = next(ev for ev in tracer.events if ev.kind == "edge")
        assert edge.node_id == "s"
        assert edge.data["target_id"] == "t"
        assert edge.data["condition"] == "lang=go"


def from_to_cond(src, dst, cond):
    from teff.graph import Edge

    return Edge(src, dst, cond)


class TestRunTracerErrors:
    @pytest.mark.asyncio
    async def test_error_edge_records_node_error_and_fallback(self):
        from teff.graph import Edge, Graph
        from teff.node import Node
        from teff.trace import RunTracer

        class Crash(Node):
            type = "crash"

            async def execute(self, ctx, state):
                raise ValueError("boom")

        class Fallback(Node):
            type = "fb"

            async def execute(self, ctx, state):
                state["handled"] = True
                return state

        g = Graph(
            nodes={"a": Crash({}), "b": Fallback({})},
            edges=[Edge("a", "b", "__error__")],
            entry_point="a",
        )
        tracer = RunTracer()
        await g.run(state={}, tracer=tracer)

        err = next(ev for ev in tracer.events if ev.kind == "node_error")
        assert err.node_id == "a"
        assert err.data["error"] == "boom"

        # fallback still ran
        starts = [ev.node_id for ev in tracer.events if ev.kind == "node_start"]
        assert starts == ["a", "b"]
        assert tracer.summary().status == "ok"
        assert tracer.summary().nodes["a"].errors == 1

    @pytest.mark.asyncio
    async def test_unhandled_error_marks_run_as_error(self):
        from teff.graph import Graph
        from teff.node import Node
        from teff.trace import RunTracer

        class Crash(Node):
            type = "crash"

            async def execute(self, ctx, state):
                raise RuntimeError("fatal")

        g = Graph(nodes={"a": Crash({})}, edges=[], entry_point="a")
        tracer = RunTracer()
        with pytest.raises(RuntimeError, match="fatal"):
            await g.run(state={}, tracer=tracer)

        end = next(ev for ev in tracer.events if ev.kind == "run_end")
        assert end.data["status"] == "error"
        assert end.data["error"] == "fatal"
        assert tracer.summary().status == "error"


class TestRunTracerCheckpoints:
    @pytest.mark.asyncio
    async def test_checkpoint_events_recorded(self, tmp_path):
        from teff.checkpoint import JSONFileCheckpointer
        from teff.graph import Graph
        from teff.trace import RunTracer

        g = Graph(nodes={"a": _make_node({"x": 1})}, edges=[], entry_point="a")
        cp = JSONFileCheckpointer(str(tmp_path / "ckpt"))
        tracer = RunTracer()
        await g.run(state={}, checkpointer=cp, checkpoint_id="run-1", tracer=tracer)

        ckpts = [ev for ev in tracer.events if ev.kind == "checkpoint"]
        assert ckpts[0].data["action"] == "load"
        assert ckpts[-1].data["action"] == "save"
        assert ckpts[-1].data["next_node_id"] is None

    @pytest.mark.asyncio
    async def test_resume_records_load(self, tmp_path):
        from teff.checkpoint import JSONFileCheckpointer
        from teff.graph import Graph
        from teff.trace import RunTracer

        g = Graph(nodes={"a": _make_node({"x": 1})}, edges=[], entry_point="a")
        cp = JSONFileCheckpointer(str(tmp_path / "ckpt"))
        await g.run(state={}, checkpointer=cp, checkpoint_id="run-1")

        tracer = RunTracer()
        await g.run(state={}, checkpointer=cp, checkpoint_id="run-1", tracer=tracer)
        loads = [
            ev
            for ev in tracer.events
            if ev.kind == "checkpoint" and ev.data["action"] == "load"
        ]
        assert len(loads) == 1


class TestRunTracerRetry:
    @pytest.mark.asyncio
    async def test_retry_events_recorded(self):
        from teff.graph import Graph
        from teff.node import Node
        from teff.node.retry import Retry
        from teff.trace import RunTracer

        attempts = {"n": 0}

        class Flaky(Node):
            type = "flaky"

            async def execute(self, ctx, state):
                attempts["n"] += 1
                if attempts["n"] < 3:
                    raise ValueError("flaky")
                return {"ok": True}

        g = Graph(
            nodes={"a": Retry(Flaky({}), max_retries=3)},
            edges=[],
            entry_point="a",
        )
        tracer = RunTracer()
        await g.run(state={}, tracer=tracer)

        retries = [ev for ev in tracer.events if ev.kind == "retry"]
        assert len(retries) == 2
        assert retries[0].data["attempt"] == 1
        assert tracer.summary().nodes["a"].runs == 1


class TestRunTracerLLM:
    @pytest.mark.asyncio
    async def test_llm_usage_recorded(self, monkeypatch):
        from teff.graph import Graph
        from teff.node import LLM
        from teff.trace import RunTracer

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "choices": [{"message": {"content": "hi"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    }

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        g = Graph(
            nodes={
                "a": LLM({"model": "gpt-4", "output_key": "out", "provider": "openai"})
            },
            edges=[],
            entry_point="a",
            providers=ProviderRegistry.from_presets("openai"),
        )
        tracer = RunTracer()
        await g.run(state={}, tracer=tracer)

        llms = [ev for ev in tracer.events if ev.kind == "llm"]
        assert len(llms) == 1
        assert llms[0].data["prompt_tokens"] == 10
        assert llms[0].data["completion_tokens"] == 5
        assert llms[0].data["model"] == "gpt-4"

        s = tracer.summary()
        assert s.llm_calls == 1
        assert s.tokens.prompt_tokens == 10
        assert s.tokens.completion_tokens == 5
        assert s.tokens.total == 15

    @pytest.mark.asyncio
    async def test_ollama_style_usage_extracted(self, monkeypatch):
        from teff.node.llm import _extract_usage

        data = {"prompt_eval_count": 7, "eval_count": 3, "message": {"content": "x"}}
        assert _extract_usage(data) == (7, 3)

        data = {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}
        assert _extract_usage(data) == (1, 2)

        assert _extract_usage({}) == (0, 0)

    @pytest.mark.asyncio
    async def test_llm_without_tracer_is_noop(self, monkeypatch):
        from teff.node import LLM, ExecContext

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        async def mock_post(*a, **kw):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "hi"}}]}

            return MockResponse()

        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        node = LLM({"model": "gpt-4", "provider": "openai"})
        ctx = ExecContext(state={}, tools={})
        result = await node.execute(ctx, {})
        assert result["output"] == "hi"


class TestRunTracerOutput:
    @pytest.mark.asyncio
    async def test_timeline_is_json_serializable(self):
        import json

        from teff.graph import Graph
        from teff.trace import RunTracer

        g = Graph(nodes={"a": _make_node({"x": 1})}, edges=[], entry_point="a")
        tracer = RunTracer()
        await g.run(state={}, tracer=tracer)

        timeline = tracer.timeline()
        json.dumps(timeline)  # must not raise
        report = json.loads(tracer.to_json())
        assert report["summary"]["status"] == "ok"
        assert report["events"][0]["kind"] == "run_start"

    @pytest.mark.asyncio
    async def test_tracer_reusable_across_runs(self):
        from teff.graph import Graph
        from teff.trace import RunTracer

        g = Graph(nodes={"a": _make_node({"x": 1})}, edges=[], entry_point="a")
        tracer = RunTracer()
        await g.run(state={}, tracer=tracer)
        await g.run(state={}, tracer=tracer)

        assert len([ev for ev in tracer.events if ev.kind == "run_start"]) == 2
