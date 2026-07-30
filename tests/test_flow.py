import pytest


class TestFlow:
    def test_compile_linear(self):
        from draf.flow import Flow
        from draf.builtin import Transform
        import asyncio

        flow = Flow("test")
        flow.step(Transform({"action": "uppercase", "input_key": "text", "output_key": "out"}))
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

        flow = Flow("t").step(CN({})).branch("mode", Case("a").add(AN({}))).default(FN({}))
        g = flow.compile()
        r = asyncio.run(g.run(state={}))
        assert r["result"] == "default"


class TestRegistryIsolation:
    def test_custom_registry_works(self):
        from draf.node import NodeRegistry, Node
        from draf.flow import Flow
        import asyncio

        reg = NodeRegistry()

        class MyNode(Node):
            type = "my"
            async def execute(self, ctx, state):
                state["x"] = 42
                return state

        reg.register("my", lambda cfg: MyNode(cfg))
        flow = Flow("test", registry=reg)
        flow.step("my")
        g = flow.compile()
        r = asyncio.run(g.run(state={}))
        assert r["x"] == 42

    def test_default_registry_still_works(self):
        from draf.flow import Flow
        import asyncio

        flow = Flow("default")
        flow.step("transform", action="uppercase", input_key="text", output_key="out")
        g = flow.compile()
        r = asyncio.run(g.run(state={"text": "hi"}))
        assert r["out"] == "HI"

    def test_empty_registry_raises_on_unknown_type(self):
        from draf.node import NodeRegistry
        from draf.flow import Flow

        reg = NodeRegistry()
        flow = Flow("empty", registry=reg)
        with pytest.raises(KeyError):
            flow.step("transform", action="uppercase")

    def test_registry_copy_isolation(self):
        from draf.node import default_registry
        from draf.flow import Flow
        import asyncio

        reg = default_registry.copy()
        flow = Flow("copy", registry=reg)
        flow.step("transform", action="uppercase", input_key="text", output_key="out")
        g = flow.compile()
        r = asyncio.run(g.run(state={"text": "iso"}))
        assert r["out"] == "ISO"
        # default_registry is unmodified
        assert "transform" in default_registry.list()

    def test_two_isolated_registries_dont_interfere(self):
        from draf.node import NodeRegistry, Node
        from draf.flow import Flow
        import asyncio

        reg_a = NodeRegistry()
        reg_b = NodeRegistry()

        class ANode(Node):
            type = "a"
            async def execute(self, ctx, state):
                state["src"] = "A"
                return state

        class BNode(Node):
            type = "b"
            async def execute(self, ctx, state):
                state["src"] = "B"
                return state

        reg_a.register("x", lambda cfg: ANode(cfg))
        reg_b.register("x", lambda cfg: BNode(cfg))

        flow_a = Flow("a", registry=reg_a).step("x")
        flow_b = Flow("b", registry=reg_b).step("x")

        g_a = flow_a.compile()
        g_b = flow_b.compile()

        r_a = asyncio.run(g_a.run(state={}))
        r_b = asyncio.run(g_b.run(state={}))
        assert r_a["src"] == "A"
        assert r_b["src"] == "B"


class TestSubFlow:
    def test_subflow_basic(self):
        from draf.flow import Flow, Case
        from draf.node import Node
        import asyncio

        class AddOne(Node):
            type = "ao"
            async def execute(self, ctx, state):
                state["val"] = state.get("val", 0) + 1
                return state

        sub = Flow("counter").step(AddOne({}))
        sub_g = sub.compile()

        parent = Flow("parent").step(AddOne({}))
        parent.add_flow(sub, max_iterations=5)
        g = parent.compile()

        r = asyncio.run(g.run({"val": 0}))
        assert r["val"] == 2

    def test_subflow_with_maps(self):
        from draf.flow import Flow
        from draf.builtin import Transform
        import asyncio

        sub = Flow("inner")
        sub.step(Transform({"action": "uppercase", "input_key": "x", "output_key": "y"}))
        sub_g = sub.compile()

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
