"""Tests for human-in-the-loop interrupts."""

import asyncio

import pytest

from draf.flow import Flow
from draf.graph import Edge
from draf.node import Interrupt, Transform
from draf.node.interrupt import GraphInterrupt


def _build_flow():
    flow = Flow("approval")
    flow.step(
        Transform({"action": "uppercase", "input_key": "text", "output_key": "draft"})
    )
    flow.interrupt("approved", prompt="Approve the draft?")
    flow.step(
        Transform({"action": "uppercase", "input_key": "draft", "output_key": "final"})
    )
    return flow.compile()


class TestInterruptNode:
    def test_raises_with_key_and_prompt(self):
        g = _build_flow()
        with pytest.raises(GraphInterrupt) as excinfo:
            asyncio.run(g.run(state={"text": "hello"}))
        exc = excinfo.value
        assert exc.key == "approved"
        assert exc.prompt == "Approve the draft?"
        assert exc.checkpoint_id is None

    def test_registerable_and_type(self):
        node = Interrupt(key="k", prompt="p")
        assert node.type == "interrupt"
        assert node.config["key"] == "k"
        assert node.config["prompt"] == "p"


class TestInterruptResume:
    def test_resume_after_checkpoint(self, tmp_path):
        from draf.checkpoint import JSONFileCheckpointer

        g = _build_flow()
        cp = JSONFileCheckpointer(str(tmp_path))

        with pytest.raises(GraphInterrupt) as excinfo:
            asyncio.run(
                g.run(
                    state={"text": "hello"},
                    checkpointer=cp,
                    checkpoint_id="run-1",
                )
            )
        exc = excinfo.value
        assert exc.key == "approved"
        assert exc.node_id == "interrupt_2"

        result = asyncio.run(
            g.run(
                state={"text": "ignored"},
                checkpointer=cp,
                checkpoint_id="run-1",
                resume={"approved": "yes"},
            )
        )
        assert result["draft"] == "HELLO"
        assert result["approved"] == "yes"
        assert result["final"] == "HELLO"
        assert "__interrupt__" not in result

    def test_resume_without_answer_reraises(self, tmp_path):
        from draf.checkpoint import JSONFileCheckpointer

        g = _build_flow()
        cp = JSONFileCheckpointer(str(tmp_path))

        with pytest.raises(GraphInterrupt):
            asyncio.run(
                g.run(state={"text": "hi"}, checkpointer=cp, checkpoint_id="r1")
            )
        with pytest.raises(GraphInterrupt) as excinfo:
            asyncio.run(g.run(state={}, checkpointer=cp, checkpoint_id="r1"))
        assert excinfo.value.key == "approved"

    def test_flow_interrupt_method_returns_self(self):
        flow = Flow("a")
        ret = flow.interrupt("k", "p")
        assert ret is flow

    def test_resume_reroutes_by_answer(self, tmp_path):
        from draf import Edge, Graph
        from draf.checkpoint import JSONFileCheckpointer

        g = Graph(
            nodes={
                "ask": Interrupt(key="approved", prompt="ok?"),
                "yes": Transform(
                    {"action": "value", "value": "APPROVED", "output_key": "final"}
                ),
                "no": Transform(
                    {"action": "value", "value": "REWORK", "output_key": "final"}
                ),
            },
            edges=[
                Edge("ask", "yes", "approved=да"),
                Edge("ask", "no", "approved!=да"),
            ],
            entry_point="ask",
        )
        cp = JSONFileCheckpointer(str(tmp_path))

        with pytest.raises(GraphInterrupt):
            asyncio.run(g.run(state={}, checkpointer=cp, checkpoint_id="r1"))
        result = asyncio.run(
            g.run(
                state={},
                checkpointer=cp,
                checkpoint_id="r1",
                resume={"approved": "да"},
            )
        )
        assert result["final"] == "APPROVED"

        with pytest.raises(GraphInterrupt):
            asyncio.run(g.run(state={}, checkpointer=cp, checkpoint_id="r2"))
        result = asyncio.run(
            g.run(
                state={},
                checkpointer=cp,
                checkpoint_id="r2",
                resume={"approved": "нет"},
            )
        )
        assert result["final"] == "REWORK"


class TestInterruptYAML:
    def test_interrupt_from_yaml(self, tmp_path):
        import yaml

        from draf.yaml import from_yaml

        data = {
            "name": "y-interrupt",
            "steps": [
                {
                    "id": "s1",
                    "type": "transform",
                    "config": {
                        "action": "uppercase",
                        "input_key": "text",
                        "output_key": "draft",
                    },
                },
                {
                    "id": "s2",
                    "type": "interrupt",
                    "config": {"key": "approved", "prompt": "ok?"},
                },
                {
                    "id": "s3",
                    "type": "transform",
                    "config": {
                        "action": "value",
                        "value": "done",
                        "output_key": "final",
                    },
                },
            ],
            "edges": [
                {"from": "s1", "to": "s2"},
                {"from": "s2", "to": "s3"},
            ],
        }
        g = from_yaml(yaml.safe_dump(data))

        with pytest.raises(GraphInterrupt) as excinfo:
            asyncio.run(g.run(state={"text": "hi"}))
        assert excinfo.value.key == "approved"

    def test_interrupt_to_yaml_roundtrip(self):
        g = _build_flow()
        yaml_text = g.to_yaml()
        assert "interrupt_2" in yaml_text
        assert "approved" in yaml_text

    def test_yaml_loop_with_resume(self, tmp_path):
        import yaml

        from draf.yaml import from_yaml

        data = {
            "name": "y-loop",
            "steps": [
                {
                    "id": "ask",
                    "type": "interrupt",
                    "config": {"key": "approved", "prompt": "ok?"},
                },
                {
                    "id": "final",
                    "type": "transform",
                    "config": {
                        "action": "value",
                        "value": "APPROVED",
                        "output_key": "final",
                    },
                },
                {
                    "id": "edit",
                    "type": "transform",
                    "config": {
                        "action": "value",
                        "value": "REWORK",
                        "output_key": "final",
                    },
                },
            ],
            "edges": [
                {"from": "ask", "to": "final", "condition": "approved=да"},
                {"from": "ask", "to": "edit", "condition": "approved!=да"},
                {"from": "edit", "to": "ask"},
            ],
        }
        from draf.checkpoint import JSONFileCheckpointer

        g = from_yaml(yaml.safe_dump(data))
        cp = JSONFileCheckpointer(str(tmp_path))

        with pytest.raises(GraphInterrupt):
            asyncio.run(g.run(state={}, checkpointer=cp, checkpoint_id="y1"))
        with pytest.raises(GraphInterrupt):
            asyncio.run(
                g.run(
                    state={},
                    checkpointer=cp,
                    checkpoint_id="y1",
                    resume={"approved": "нет"},
                )
            )
        result = asyncio.run(
            g.run(
                state={},
                checkpointer=cp,
                checkpoint_id="y1",
                resume={"approved": "да"},
            )
        )
        assert result["final"] == "APPROVED"


class TestFlowLoop:
    def _build_loop_flow(self):
        flow = Flow("loop")
        flow.interrupt("approved", prompt="ok?")
        flow.loop(
            key="approved",
            until="да",
            done=Transform(
                {"action": "value", "value": "APPROVED", "output_key": "final"}
            ),
            body=Transform(
                {"action": "value", "value": "REWORK", "output_key": "final"}
            ),
        )
        return flow.compile()

    def test_loop_wires_cycle(self):
        g = self._build_loop_flow()
        ask_id = [nid for nid, n in g.nodes.items() if n.type == "interrupt"][0]
        done_id = [
            nid
            for nid, n in g.nodes.items()
            if n.type == "transform" and n.config.get("value") == "APPROVED"
        ][0]
        body_id = [
            nid
            for nid, n in g.nodes.items()
            if n.type == "transform" and n.config.get("value") == "REWORK"
        ][0]
        assert Edge(ask_id, done_id, "approved=да") in g.edges
        assert Edge(ask_id, body_id, "approved!=да") in g.edges
        assert Edge(body_id, ask_id) in g.edges

    def test_loop_with_interrupt_resumes(self, tmp_path):
        from draf.checkpoint import JSONFileCheckpointer

        g = self._build_loop_flow()
        cp = JSONFileCheckpointer(str(tmp_path))

        with pytest.raises(GraphInterrupt):
            asyncio.run(g.run(state={}, checkpointer=cp, checkpoint_id="r1"))

        # первый ответ «нет» → body, потом снова interrupt
        with pytest.raises(GraphInterrupt):
            asyncio.run(
                g.run(
                    state={},
                    checkpointer=cp,
                    checkpoint_id="r1",
                    resume={"approved": "нет"},
                )
            )

        result = asyncio.run(
            g.run(
                state={},
                checkpointer=cp,
                checkpoint_id="r1",
                resume={"approved": "да"},
            )
        )
        assert result["final"] == "APPROVED"

    def test_loop_requires_decider(self):
        flow = Flow("empty")
        with pytest.raises(ValueError, match="preceding node"):
            flow.loop(
                "k",
                "да",
                done=Transform({"action": "value", "value": "x", "output_key": "o"}),
                body=Transform({"action": "value", "value": "y", "output_key": "o"}),
            )

    def test_loop_without_interrupt_self_check(self, tmp_path):
        async def checker(ctx, state):
            if state.get("score", 0) >= 2:
                return {"final": "ok"}
            return {"score": state.get("score", 0) + 1}

        from draf.node import node

        node("score")(checker)
        node("bump")(checker)
        from draf.node.registry import default_registry

        flow = Flow("self-check")
        flow.step(default_registry.create("score"))
        flow.loop(
            key="score",
            until="2",
            done=Transform({"action": "value", "value": "DONE", "output_key": "final"}),
            body=default_registry.create("bump"),
        )
        result = asyncio.run(flow.compile().run(state={}))
        assert result["final"] == "DONE"


class TestInterruptTrace:
    def test_tracer_records_interrupt(self, tmp_path):
        from draf.checkpoint import JSONFileCheckpointer
        from draf import RunTracer

        g = _build_flow()
        cp = JSONFileCheckpointer(str(tmp_path))
        tracer = RunTracer()

        with pytest.raises(GraphInterrupt):
            asyncio.run(
                g.run(
                    state={"text": "hi"},
                    checkpointer=cp,
                    checkpoint_id="t1",
                    tracer=tracer,
                )
            )

        kinds = [ev.kind for ev in tracer.events]
        assert "interrupt" in kinds

        tracer2 = RunTracer()
        asyncio.run(
            g.run(
                state={},
                checkpointer=cp,
                checkpoint_id="t1",
                resume={"approved": "yes"},
                tracer=tracer2,
            )
        )
        kinds2 = [ev.kind for ev in tracer2.events]
        assert "interrupt_resume" in kinds2
