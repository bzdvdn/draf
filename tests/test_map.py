import asyncio

import pytest


class TestMapNode:
    @pytest.mark.asyncio
    async def test_fans_out_over_list(self):
        from teff.graph import Graph
        from teff.node import Map, Transform

        node = Map(
            processor=Transform(
                action="uppercase", input_key="chunks", output_key="summary"
            ),
            input_keys=["chunks"],
            output_key="summaries",
        )
        g = Graph(nodes={"m": node}, edges=[], entry_point="m")
        r = await g.run(state={"chunks": ["a", "bb", "ccc"]})
        assert r["summaries"] == ["A", "BB", "CCC"]

    @pytest.mark.asyncio
    async def test_zipped_input_keys(self):
        from teff.graph import Graph
        from teff.node import Map, Transform

        node = Map(
            processor=Transform(action="value", input_key="summ", output_key="s"),
            input_keys=["type", "summ"],
            output_key="summaries",
        )
        g = Graph(nodes={"m": node}, edges=[], entry_point="m")
        r = await g.run(state={"type": ["a", "b"], "summ": [1, 2]})
        assert r["summaries"] == ["1", "2"]

    @pytest.mark.asyncio
    async def test_empty_list(self):
        from teff.graph import Graph
        from teff.node import Map, Transform

        node = Map(
            processor=Transform(action="uppercase", input_key="chunks", output_key="s"),
            input_keys=["chunks"],
            output_key="summaries",
        )
        g = Graph(nodes={"m": node}, edges=[], entry_point="m")
        r = await g.run(state={"chunks": []})
        assert r["summaries"] == []

    @pytest.mark.asyncio
    async def test_chunk_size_batches(self):
        from teff.graph import Graph
        from teff.node import Map, Node

        class Join(Node):
            type = "join"

            async def execute(self, ctx, state):
                return {"summary": "|".join(state["chunks"])}

        node = Map(
            processor=Join({}),
            input_keys=["chunks"],
            output_key="summaries",
            result_key="summary",
            chunk_size=2,
        )
        g = Graph(nodes={"m": node}, edges=[], entry_point="m")
        r = await g.run(state={"chunks": ["a", "b", "c"]})
        assert r["summaries"] == ["a|b", "c"]

    @pytest.mark.asyncio
    async def test_result_key_override(self):
        from teff.graph import Graph
        from teff.node import Map, Node

        class Produce(Node):
            type = "produce"

            async def execute(self, ctx, state):
                return {"item": state["chunks"].upper()}

        node = Map(
            processor=Produce({}),
            input_keys=["chunks"],
            output_key="out",
            result_key="item",
        )
        g = Graph(nodes={"m": node}, edges=[], entry_point="m")
        r = await g.run(state={"chunks": ["x"]})
        assert r["out"] == ["X"]

    @pytest.mark.asyncio
    async def test_max_concurrency_limits_branches(self):
        from teff.graph import Graph
        from teff.node import Map, Node

        active = {"now": 0, "peak": 0}

        class Slow(Node):
            type = "slow"

            async def execute(self, ctx, state):
                active["now"] += 1
                active["peak"] = max(active["peak"], active["now"])
                await asyncio.sleep(0.02)
                active["now"] -= 1
                return {"summary": "ok"}

        node = Map(
            processor=Slow({}),
            input_keys=["chunks"],
            output_key="summaries",
            result_key="summary",
            max_concurrency=2,
        )
        g = Graph(nodes={"m": node}, edges=[], entry_point="m")
        r = await g.run(state={"chunks": ["a", "b", "c", "d"]})
        assert r["summaries"] == ["ok", "ok", "ok", "ok"]
        assert active["peak"] <= 2

    @pytest.mark.asyncio
    async def test_exception_propagates(self):
        from teff.graph import Graph
        from teff.node import Map, Node

        class Boom(Node):
            type = "boom"

            async def execute(self, ctx, state):
                raise ValueError("map failure")

        node = Map(processor=Boom({}), input_keys=["chunks"], output_key="s")
        g = Graph(nodes={"m": node}, edges=[], entry_point="m")
        with pytest.raises(ValueError, match="map failure"):
            await g.run(state={"chunks": ["a", "b"]})

    @pytest.mark.asyncio
    async def test_state_isolation_between_branches(self):
        from teff.graph import Graph
        from teff.node import Map, Node

        class Inc(Node):
            type = "inc"

            async def execute(self, ctx, state):
                return {"summary": state["v"] + 1}

        node = Map(
            processor=Inc({}),
            input_keys=["chunks"],
            output_key="summaries",
            result_key="summary",
        )
        g = Graph(nodes={"m": node}, edges=[], entry_point="m")
        r = await g.run(state={"chunks": ["a", "b"], "v": 1})
        # both branches read v=1 (isolated); without isolation second sees 2
        assert r["summaries"] == [2, 2]

    @pytest.mark.asyncio
    async def test_traced_with_nested_ids(self):
        from teff.graph import Graph
        from teff.node import Map, Transform
        from teff.trace import RunTracer

        node = Map(
            processor=Transform(
                action="uppercase", input_key="chunks", output_key="summary"
            ),
            input_keys=["chunks"],
            output_key="summaries",
        )
        g = Graph(nodes={"m": node}, edges=[], entry_point="m")
        tracer = RunTracer()
        await g.run(state={"chunks": ["a", "b"]}, tracer=tracer)

        starts = [ev for ev in tracer.events if ev.kind == "node_start"]
        ids = {ev.node_id for ev in starts}
        assert ids == {"m", "m.m0.0", "m.m1.0"}


class TestFlowMap:
    @pytest.mark.asyncio
    async def test_map_in_flow(self):
        from teff.flow import Flow
        from teff.node import Transform

        flow = (
            Flow("m")
            .step(Transform(action="value", value="START", output_key="s"))
            .map(
                Transform(action="uppercase", input_key="chunks", output_key="summary"),
                input_keys=["chunks"],
                output_key="summaries",
            )
        )
        g = flow.compile()
        r = await g.run(state={"chunks": ["a", "b"]})
        assert r["s"] == "START"
        assert r["summaries"] == ["A", "B"]

    @pytest.mark.asyncio
    async def test_map_with_templated_prompt_processor(self):
        from teff.flow import Flow
        from teff.node import LLM

        seen = []

        class FakeLLM(LLM):
            async def execute(self, ctx, state):
                from teff.prompt import render_template

                seen.append(render_template(self.config["prompt"], state))
                return {self.config["output_key"]: "ok"}

        flow = Flow("m").map(
            FakeLLM(prompt="план для {type} на {summ}", output_key="summary"),
            input_keys=["type", "summ"],
            output_key="summaries",
        )
        g = flow.compile()
        r = await g.run(state={"type": ["кухни", "ванной"], "summ": [1, 2]})
        assert r["summaries"] == ["ok", "ok"]
        assert seen == ["план для кухни на 1", "план для ванной на 2"]
