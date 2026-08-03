import asyncio

import pytest


class TestEdgeConditions:
    @pytest.mark.asyncio
    async def test_equals(self):
        from draf.graph import Edge, Graph
        from draf.node import Node

        class PN(Node):
            type = "pn"

            async def execute(self, ctx, state):
                return {}

        class TN(Node):
            type = "tn"

            async def execute(self, ctx, state):
                state["ok"] = True
                return state

        g = Graph(
            nodes={"s": PN({}), "t": TN({})},
            edges=[Edge("s", "t", "x=1")],
            entry_point="s",
        )
        r = await g.run(state={"x": "1"})
        assert r.get("ok") is True

    @pytest.mark.asyncio
    async def test_not_equals(self):
        from draf.graph import Edge, Graph
        from draf.node import Node

        class PN(Node):
            type = "pn"

            async def execute(self, ctx, state):
                return {}

        class TN(Node):
            type = "tn"

            async def execute(self, ctx, state):
                state["ok"] = True
                return state

        g = Graph(
            nodes={"s": PN({}), "t": TN({})},
            edges=[Edge("s", "t", "x!=0")],
            entry_point="s",
        )
        r = await g.run(state={"x": "1"})
        assert r.get("ok") is True
        r2 = await g.run(state={"x": "0"})
        assert r2.get("ok") is not True

    @pytest.mark.asyncio
    async def test_comma_disjunction(self):
        from draf.graph import Edge, Graph
        from draf.node import Node

        class PN(Node):
            type = "pn"

            async def execute(self, ctx, state):
                return {}

        class TN(Node):
            type = "tn"

            async def execute(self, ctx, state):
                state["ok"] = True
                return state

        g = Graph(
            nodes={"s": PN({}), "t": TN({})},
            edges=[Edge("s", "t", "lang=py,go,rs")],
            entry_point="s",
        )
        r = await g.run(state={"lang": "go"})
        assert r.get("ok") is True
        r2 = await g.run(state={"lang": "java"})
        assert r2.get("ok") is not True


class TestErrorEdges:
    @pytest.mark.asyncio
    async def test_error_edge_catches_exception(self):
        from draf.graph import Edge, Graph
        from draf.node import Node

        class Crash(Node):
            type = "cr"

            async def execute(self, ctx, state):
                raise ValueError("crash")

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
        result = await g.run(state={})
        assert result["handled"] is True

    @pytest.mark.asyncio
    async def test_no_error_edge_re_raises(self):
        from draf.graph import Edge, Graph
        from draf.node import Node

        class Crash(Node):
            type = "cr"

            async def execute(self, ctx, state):
                raise ValueError("crash")

        class Other(Node):
            type = "ot"

            async def execute(self, ctx, state):
                return {"ok": True}

        g = Graph(
            nodes={"a": Crash({}), "b": Other({})},
            edges=[Edge("a", "b", "x=1")],
            entry_point="a",
        )
        with pytest.raises(ValueError, match="crash"):
            await g.run(state={})

    @pytest.mark.asyncio
    async def test_error_edge_does_not_interfere_with_normal_routing(self):
        from draf.graph import Edge, Graph
        from draf.node import Node

        class Normal(Node):
            type = "n"

            async def execute(self, ctx, state):
                state["x"] = "ok"
                return state

        g = Graph(
            nodes={"a": Normal({})},
            edges=[Edge("a", "nonexistent", "__error__")],
            entry_point="a",
        )
        result = await g.run(state={})
        assert result["x"] == "ok"


class TestHooks:
    @pytest.mark.asyncio
    async def test_on_node_start_called(self):
        from draf.graph import Graph
        from draf.node import Node

        class Simple(Node):
            type = "s"

            async def execute(self, ctx, state):
                return {"done": True}

        g = Graph(nodes={"a": Simple({})}, edges=[], entry_point="a")
        calls = []
        result = await g.run(
            state={}, hooks={"on_node_start": lambda nid, n, s: calls.append(nid)}
        )
        assert calls == ["a"]
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_on_node_end_called(self):
        from draf.graph import Graph
        from draf.node import Node

        class Simple(Node):
            type = "s"

            async def execute(self, ctx, state):
                return {"done": True}

        g = Graph(nodes={"a": Simple({})}, edges=[], entry_point="a")
        calls = []
        result = await g.run(
            state={},
            hooks={"on_node_end": lambda nid, n, s, r: calls.append((nid, r))},
        )
        assert calls == [("a", {"done": True})]
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_on_node_error_called_before_fallback(self):
        from draf.graph import Edge, Graph
        from draf.node import Node

        class Crash(Node):
            type = "cr"

            async def execute(self, ctx, state):
                raise RuntimeError("boom")

        class Fallback(Node):
            type = "fb"

            async def execute(self, ctx, state):
                state["ok"] = True
                return state

        g = Graph(
            nodes={"a": Crash({}), "b": Fallback({})},
            edges=[Edge("a", "b", "__error__")],
            entry_point="a",
        )
        errors = []
        result = await g.run(
            state={},
            hooks={"on_node_error": lambda nid, n, s, e: errors.append((nid, str(e)))},
        )
        assert errors == [("a", "boom")]
        assert result["ok"] is True


class TestNodeTimeout:
    @pytest.mark.asyncio
    async def test_timeout_triggers_error_edge(self):
        from draf.graph import Edge, Graph
        from draf.node import Node

        class Slow(Node):
            type = "slow"

            async def execute(self, ctx, state):
                await asyncio.sleep(10)
                return {}

        class Fallback(Node):
            type = "fb"

            async def execute(self, ctx, state):
                state["fallback"] = True
                return state

        g = Graph(
            nodes={"a": Slow({}), "b": Fallback({})},
            edges=[Edge("a", "b", "__error__")],
            entry_point="a",
        )
        result = await g.run(state={}, node_timeout=0.01)
        assert result["fallback"] is True

    @pytest.mark.asyncio
    async def test_timeout_raises_without_error_edge(self):
        from draf.graph import Graph
        from draf.node import Node

        class Slow(Node):
            type = "slow"

            async def execute(self, ctx, state):
                await asyncio.sleep(10)
                return {}

        g = Graph(
            nodes={"a": Slow({})},
            edges=[],
            entry_point="a",
        )
        with pytest.raises(asyncio.TimeoutError):
            await g.run(state={}, node_timeout=0.01)

    @pytest.mark.asyncio
    async def test_no_timeout_completes_normally(self):
        from draf.graph import Graph
        from draf.node import Node

        class Fast(Node):
            type = "fast"

            async def execute(self, ctx, state):
                state["done"] = True
                return state

        g = Graph(
            nodes={"a": Fast({})},
            edges=[],
            entry_point="a",
        )
        result = await g.run(state={}, node_timeout=10)
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_timeout_hook_called(self):
        from draf.graph import Graph
        from draf.node import Node

        class Slow(Node):
            type = "slow"

            async def execute(self, ctx, state):
                await asyncio.sleep(10)
                return {}

        g = Graph(
            nodes={"a": Slow({})},
            edges=[],
            entry_point="a",
        )
        errors = []
        with pytest.raises(asyncio.TimeoutError):
            await g.run(
                state={},
                node_timeout=0.01,
                hooks={
                    "on_node_error": lambda nid, n, s, e: errors.append(
                        type(e).__name__
                    )
                },
            )
        assert errors == ["TimeoutError"]


class TestYAML:
    def test_round_trip(self):
        from draf.graph import Graph
        from draf.node import Node

        class TN(Node):
            type = "tn"

            async def execute(self, ctx, state):
                state["x"] = "y"
                return state

        g = Graph(nodes={"a": TN({"k": "v"})}, edges=[], entry_point="a")
        yaml_str = g.to_yaml()
        assert "tn" in yaml_str
        assert "k" in yaml_str or "v" in yaml_str
