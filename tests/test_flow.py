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
