import asyncio

import pytest


def _node(state_update):
    from draf.node import Node

    class Simple(Node):
        type = "simple"

        async def execute(self, ctx, state):
            state.update(state_update)
            return dict(state_update)

    return Simple({})


class TestParallelNode:
    @pytest.mark.asyncio
    async def test_branches_merge_results(self):
        from draf.graph import Graph
        from draf.node import Parallel

        node = Parallel([[_node({"a": 1})], [_node({"b": 2})]])
        g = Graph(nodes={"p": node}, edges=[], entry_point="p")
        r = await g.run(state={})
        assert r["a"] == 1
        assert r["b"] == 2

    @pytest.mark.asyncio
    async def test_branches_run_concurrently(self):
        from draf.graph import Graph
        from draf.node import Node
        from draf.node import Parallel

        elapsed = {"t": 0.0}

        class Slow(Node):
            type = "slow"

            async def execute(self, ctx, state):
                await asyncio.sleep(0.05)
                elapsed["t"] += 0.05
                return {"slow": True}

        node = Parallel([[Slow({})], [Slow({})]])
        g = Graph(nodes={"p": node}, edges=[], entry_point="p")

        start = asyncio.get_event_loop().time()
        await g.run(state={})
        total = asyncio.get_event_loop().time() - start

        # two 50ms branches in parallel should take ~50ms, not ~100ms
        assert total < 0.09

    @pytest.mark.asyncio
    async def test_sequential_nodes_within_branch(self):
        from draf.graph import Graph
        from draf.node import Node
        from draf.node import Parallel

        class Count(Node):
            type = "count"

            async def execute(self, ctx, state):
                state["n"] = state.get("n", 0) + 1
                return {"n": state["n"]}

        node = Parallel([[Count({}), Count({})], [_node({"b": 1})]])
        g = Graph(nodes={"p": node}, edges=[], entry_point="p")
        r = await g.run(state={})
        assert r["n"] == 2
        assert r["b"] == 1

    @pytest.mark.asyncio
    async def test_append_reducer_accumulates_across_branches(self):
        from draf.graph import Graph
        from draf.node import Node
        from draf.node import Parallel
        from draf.state import reducers_from_yaml_schema

        class Msg(Node):
            type = "msg"

            async def execute(self, ctx, state):
                return {"messages": [self.config["m"]]}

        node = Parallel([[Msg({"m": "a"})], [Msg({"m": "b"})]])
        g = Graph(nodes={"p": node}, edges=[], entry_point="p")
        reducers = reducers_from_yaml_schema(
            {"messages": {"reducer": "append", "type": "list"}}
        )
        r = await g.run(state={}, reducers=reducers)
        assert sorted(r["messages"]) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_typed_state_reducers_apply_inside_branches(self):
        from typing import Annotated, TypedDict

        from draf.graph import Graph
        from draf.node import Node
        from draf.node import Parallel
        from draf.state import State

        class S(TypedDict):
            messages: Annotated[list, "append"]

        class Msg(Node):
            type = "msg"

            async def execute(self, ctx, state):
                return {"messages": [self.config["m"]]}

        node = Parallel([[Msg({"m": "a"})], [Msg({"m": "b"})]])
        g = Graph(nodes={"p": node}, edges=[], entry_point="p")
        r = await g.run(state=State(S, {}))
        assert sorted(r["messages"]) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_branch_exception_propagates(self):
        from draf.graph import Graph
        from draf.node import Node
        from draf.node import Parallel

        class Boom(Node):
            type = "boom"

            async def execute(self, ctx, state):
                raise ValueError("branch failure")

        node = Parallel([[_node({"a": 1})], [Boom({})]])
        g = Graph(nodes={"p": node}, edges=[], entry_point="p")
        with pytest.raises(ValueError, match="branch failure"):
            await g.run(state={})

    @pytest.mark.asyncio
    async def test_branch_state_isolation(self):
        from draf.graph import Graph
        from draf.node import Node
        from draf.node import Parallel

        class AddOne(Node):
            type = "ao"

            async def execute(self, ctx, state):
                return {"v": state.get("v", 0) + 1}

        # both branches read v=1 and write 2; without isolation the
        # second branch would see the first branch's write
        node = Parallel([[AddOne({})], [AddOne({})]])
        g = Graph(nodes={"p": node}, edges=[], entry_point="p")
        r = await g.run(state={"v": 1})
        assert r["v"] == 2


class TestFlowParallel:
    @pytest.mark.asyncio
    async def test_parallel_in_flow_with_converge(self):
        from draf.flow import Flow
        from draf.node import Transform

        flow = (
            Flow("p")
            .step(Transform(action="uppercase", input_key="text", output_key="text"))
            .parallel(
                [Transform(action="value", value="(a)", output_key="a")],
                [Transform(action="value", value="(b)", output_key="b")],
            )
            .converge(Transform(action="uppercase", input_key="a", output_key="merged"))
        )
        g = flow.compile()
        r = await g.run(state={"text": "hi"})
        assert r["a"] == "(a)"
        assert r["b"] == "(b)"
        assert r["merged"] == "(A)"

    @pytest.mark.asyncio
    async def test_parallel_with_embedded_flow(self):
        from draf.flow import Flow
        from draf.node import Transform

        sub = Flow("sub").step(
            Transform(action="uppercase", input_key="text", output_key="out")
        )

        flow = Flow("p").parallel(
            sub, [Transform(action="value", value="x", output_key="x")]
        )
        g = flow.compile()
        r = await g.run(state={"text": "hi"})
        assert r["out"] == "HI"
        assert r["x"] == "x"

    @pytest.mark.asyncio
    async def test_parallel_single_node_branch(self):
        from draf.flow import Flow
        from draf.node import Transform

        flow = (
            Flow("p")
            .step(Transform(action="value", value="start", output_key="s"))
            .parallel(
                Transform(action="value", value="one", output_key="one"),
                Transform(action="value", value="two", output_key="two"),
            )
        )
        g = flow.compile()
        r = await g.run(state={})
        assert r["s"] == "start"
        assert r["one"] == "one"
        assert r["two"] == "two"


class TestParallelTelemetry:
    @pytest.mark.asyncio
    async def test_branch_nodes_traced_with_nested_ids(self):
        from draf.graph import Graph
        from draf.node import Node
        from draf.node import Parallel
        from draf.trace import RunTracer

        class Tag(Node):
            type = "tag"

            async def execute(self, ctx, state):
                state["last_id"] = ctx.node_id
                return {"last_id": ctx.node_id}

        node = Parallel([[Tag({})], [Tag({})]])
        g = Graph(nodes={"p": node}, edges=[], entry_point="p")
        tracer = RunTracer()
        await g.run(state={}, tracer=tracer)

        starts = [ev for ev in tracer.events if ev.kind == "node_start"]
        ids = {ev.node_id for ev in starts}
        assert ids == {"p", "p.b0.0", "p.b1.0"}
        assert any(ev.node_type == "tag" for ev in starts)
