import pytest


class TestFlow:
    def test_compile_linear(self):
        from draf.flow import Flow
        from draf.node import Transform
        import asyncio

        flow = Flow("test")
        flow.step(
            Transform({"action": "uppercase", "input_key": "text", "output_key": "out"})
        )
        g = flow.compile()
        assert g.entry_point == "transform_1"
        r = asyncio.run(g.run(state={"text": "hello"}))
        assert r["out"] == "HELLO"

    def test_empty_flow_raises(self):
        from draf.flow import Flow

        with pytest.raises(ValueError, match="no nodes"):
            Flow("x").compile()

    def test_branch_routing(self):
        from draf.flow import Flow, Case
        from draf.node import Node
        import asyncio

        class CN(Node):
            type = "cn"

            async def execute(self, ctx, state):
                state["mode"] = "a"
                return state

        class AN(Node):
            type = "an"

            async def execute(self, ctx, state):
                state["result"] = "A"
                return state

        flow = Flow("t").step(CN({})).branch("mode", Case("a").add(AN({})))
        g = flow.compile()
        r = asyncio.run(g.run(state={}))
        assert r["result"] == "A"

    def test_default_fallback(self):
        from draf.flow import Flow, Case
        from draf.node import Node
        import asyncio

        class CN(Node):
            type = "cn"

            async def execute(self, ctx, state):
                state["mode"] = "unknown"
                return state

        class AN(Node):
            type = "an"

            async def execute(self, ctx, state):
                state["result"] = "A"
                return state

        class FN(Node):
            type = "fn"

            async def execute(self, ctx, state):
                state["result"] = "default"
                return state

        flow = (
            Flow("t").step(CN({})).branch("mode", Case("a").add(AN({}))).default(FN({}))
        )
        g = flow.compile()
        r = asyncio.run(g.run(state={}))
        assert r["result"] == "default"


class TestStep:
    def test_step_accepts_node_instance(self):
        from draf.node import Node
        from draf.flow import Flow
        import asyncio

        class MyNode(Node):
            type = "my"

            async def execute(self, ctx, state):
                state["x"] = 42
                return state

        flow = Flow("test").step(MyNode())
        g = flow.compile()
        r = asyncio.run(g.run(state={}))
        assert r["x"] == 42

    def test_step_rejects_string(self):
        from draf.flow import Flow

        flow = Flow("test")
        with pytest.raises(TypeError, match="Node instance"):
            flow.step("transform")  # type: ignore[arg-type]

    def test_step_rejects_non_node(self):
        from draf.flow import Flow

        flow = Flow("test")
        with pytest.raises(TypeError, match="Node instance"):
            flow.step({"action": "uppercase"})  # type: ignore[arg-type]

    def test_step_with_transform_node(self):
        from draf.flow import Flow
        from draf.node import Transform
        import asyncio

        flow = Flow("default").step(
            Transform(action="uppercase", input_key="text", output_key="out")
        )
        g = flow.compile()
        r = asyncio.run(g.run(state={"text": "hi"}))
        assert r["out"] == "HI"

    def test_step_chaining(self):
        from draf.flow import Flow
        from draf.node import Transform
        import asyncio

        flow = (
            Flow("chain")
            .step(Transform(action="trim", input_key="text", output_key="t"))
            .step(Transform(action="uppercase", input_key="t", output_key="out"))
        )
        g = flow.compile()
        r = asyncio.run(g.run(state={"text": "  hi  "}))
        assert r["out"] == "HI"


class TestSubFlow:
    def test_subflow_basic(self):
        from draf.flow import Flow
        from draf.node import Node
        import asyncio

        class AddOne(Node):
            type = "ao"

            async def execute(self, ctx, state):
                state["val"] = state.get("val", 0) + 1
                return state

        sub = Flow("counter").step(AddOne({}))

        parent = Flow("parent").step(AddOne({}))
        parent.add_flow(sub, max_iterations=5)
        g = parent.compile()

        r = asyncio.run(g.run({"val": 0}))
        assert r["val"] == 2

    def test_subflow_with_maps(self):
        from draf.flow import Flow
        from draf.node import Transform
        import asyncio

        sub = Flow("inner")
        sub.step(
            Transform({"action": "uppercase", "input_key": "x", "output_key": "y"})
        )

        parent = Flow("outer")
        parent.add_flow(sub, input_map={"text": "x"}, output_map={"y": "result"})
        g = parent.compile()

        r = asyncio.run(g.run({"text": "hello"}))
        assert r["result"] == "HELLO"

    def test_subflow_state_isolation(self):
        from draf.flow import Flow
        from draf.node import Node
        import asyncio

        class SetFoo(Node):
            type = "sf"

            async def execute(self, ctx, state):
                state["foo"] = "bar"
                return state

        sub = Flow("inner").step(SetFoo({}))
        parent = Flow("outer").step(SetFoo({}))
        parent.add_flow(sub)
        g = parent.compile()

        r = asyncio.run(g.run({}))
        assert r["foo"] == "bar"


class TestCyclicGraph:
    @pytest.mark.asyncio
    async def test_simple_cycle_terminates_by_condition(self):
        from draf.graph import Graph, Edge
        from draf.node import Node

        class Counter(Node):
            type = "ct"

            async def execute(self, ctx, state):
                state["n"] = state.get("n", 0) + 1
                return state

        class Done(Node):
            type = "dn"

            async def execute(self, ctx, state):
                state["done"] = True
                return state

        # counter -> counter (loop, condition: n<3) -> done (unconditional)
        g = Graph(
            nodes={"ct": Counter({}), "dn": Done({})},
            edges=[
                Edge("ct", "ct", "n=1,2"),
                Edge("ct", "dn"),
            ],
            entry_point="ct",
        )
        r = await g.run(state={}, max_iterations=10)
        assert r["n"] == 3
        assert r["done"] is True

    @pytest.mark.asyncio
    async def test_max_iterations_raises(self):
        from draf.graph import Graph, Edge
        from draf.node import Node

        class InfLoop(Node):
            type = "il"

            async def execute(self, ctx, state):
                state["n"] = state.get("n", 0) + 1
                return state

        # self-loop with unconditional edge -> infinite
        g = Graph(
            nodes={"il": InfLoop({})},
            edges=[Edge("il", "il")],
            entry_point="il",
        )
        with pytest.raises(RuntimeError, match="max_iterations"):
            await g.run(state={}, max_iterations=5)

    @pytest.mark.asyncio
    async def test_max_iterations_linear_completes(self):
        from draf.graph import Graph, Edge
        from draf.node import Node

        class AddOne(Node):
            type = "ao"

            async def execute(self, ctx, state):
                state["n"] = state.get("n", 0) + 1
                return state

        g = Graph(
            nodes={"a": AddOne({}), "b": AddOne({})},
            edges=[Edge("a", "b")],
            entry_point="a",
        )
        r = await g.run(state={}, max_iterations=10)
        assert r["n"] == 2
