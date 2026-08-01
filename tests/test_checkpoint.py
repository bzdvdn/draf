from typing import TypedDict

import pytest


def _build_linear_graph():
    from draf.flow import Flow
    from draf.node import Transform

    flow = Flow("ckpt")
    flow.step(
        Transform({"action": "uppercase", "input_key": "text", "output_key": "a"})
    )
    flow.step(Transform({"action": "lowercase", "input_key": "a", "output_key": "b"}))
    return flow.compile()


class TestCheckpointBase:
    def test_roundtrip_dict(self):
        from draf.checkpoint import Checkpoint, checkpoint_from_dict, checkpoint_to_dict

        cp = Checkpoint(state={"a": 1}, next_node_id="node_2", iteration=3)
        assert checkpoint_from_dict(checkpoint_to_dict(cp)) == cp

    def test_checkpoint_id_required(self):
        from draf.checkpoint import SQLiteCheckpointer

        import asyncio

        g = _build_linear_graph()
        with pytest.raises(ValueError, match="checkpoint_id"):
            asyncio.run(
                g.run(
                    state={"text": "hi"}, checkpointer=SQLiteCheckpointer("/tmp/x.db")
                )
            )

    def test_missing_dep_raises(self):
        import importlib.util

        if importlib.util.find_spec("asyncpg") is not None:
            pytest.skip("asyncpg is installed")
        from draf.checkpoint.pg import PGCheckpointer

        with pytest.raises(ImportError):
            PGCheckpointer("postgresql://localhost/x")


class TestJSONFileCheckpointer:
    def test_save_load_delete(self, tmp_path):
        from draf.checkpoint import Checkpoint, JSONFileCheckpointer

        import asyncio

        ck = JSONFileCheckpointer(str(tmp_path))
        asyncio.run(
            ck.save("t1", Checkpoint(state={"a": 1}, next_node_id="n", iteration=2))
        )
        cp = asyncio.run(ck.load("t1"))
        assert cp is not None
        assert cp.state == {"a": 1}
        assert cp.next_node_id == "n"
        assert cp.iteration == 2

        asyncio.run(ck.delete("t1"))
        assert asyncio.run(ck.load("t1")) is None

    def test_load_missing(self, tmp_path):
        from draf.checkpoint import JSONFileCheckpointer

        import asyncio

        ck = JSONFileCheckpointer(str(tmp_path))
        assert asyncio.run(ck.load("missing")) is None

    def test_sanitizes_path(self, tmp_path):
        from draf.checkpoint import Checkpoint, JSONFileCheckpointer

        import asyncio

        ck = JSONFileCheckpointer(str(tmp_path))
        asyncio.run(
            ck.save("a/b", Checkpoint(state={}, next_node_id=None, iteration=0))
        )
        assert asyncio.run(ck.load("a/b")) is not None


class TestSQLiteCheckpointer:
    def test_save_load_overwrite(self, tmp_path):
        from draf.checkpoint import Checkpoint, SQLiteCheckpointer

        import asyncio

        ck = SQLiteCheckpointer(str(tmp_path / "ck.db"))
        try:
            asyncio.run(
                ck.save(
                    "t1", Checkpoint(state={"n": 1}, next_node_id="n1", iteration=1)
                )
            )
            asyncio.run(
                ck.save(
                    "t1", Checkpoint(state={"n": 2}, next_node_id="n2", iteration=2)
                )
            )
            cp = asyncio.run(ck.load("t1"))
            assert cp is not None
            assert cp.state == {"n": 2}
            assert cp.next_node_id == "n2"
            assert cp.iteration == 2

            asyncio.run(ck.delete("t1"))
            assert asyncio.run(ck.load("t1")) is None
        finally:
            ck.close()

    def test_independent_ids(self, tmp_path):
        from draf.checkpoint import Checkpoint, SQLiteCheckpointer

        import asyncio

        ck = SQLiteCheckpointer(str(tmp_path / "ck.db"))
        try:
            asyncio.run(
                ck.save("a", Checkpoint(state={"x": 1}, next_node_id=None, iteration=0))
            )
            asyncio.run(
                ck.save("b", Checkpoint(state={"y": 2}, next_node_id=None, iteration=0))
            )
            assert asyncio.run(ck.load("a")).state == {"x": 1}
            assert asyncio.run(ck.load("b")).state == {"y": 2}
        finally:
            ck.close()


def _cp(state: dict):
    from draf.checkpoint import Checkpoint

    return Checkpoint(state=state, next_node_id=None, iteration=0)


@pytest.mark.parametrize("kind", ["file", "sqlite"])
class TestOwnerIsolation:
    @pytest.fixture(autouse=True)
    def _make(self, kind, tmp_path):
        import asyncio

        if kind == "file":
            from draf.checkpoint import JSONFileCheckpointer

            self.ck = JSONFileCheckpointer(str(tmp_path))
        else:
            from draf.checkpoint import SQLiteCheckpointer

            self.ck = SQLiteCheckpointer(str(tmp_path / "ck.db"))
        self.run = lambda coro: asyncio.run(coro)
        yield
        close = getattr(self.ck, "close", None)
        if close is not None:
            close()

    async def test_same_id_different_owners_isolated(self):
        await self.ck.save("chat-1", _cp({"user": "alice"}), owner="alice")
        await self.ck.save("chat-1", _cp({"user": "bob"}), owner="bob")

        alice = await self.ck.load("chat-1", owner="alice")
        bob = await self.ck.load("chat-1", owner="bob")
        assert alice.state == {"user": "alice"}
        assert bob.state == {"user": "bob"}

        # default owner sees nothing
        assert await self.ck.load("chat-1") is None

    async def test_list_returns_owner_checkpoints(self):
        await self.ck.save("a", _cp({}), owner="alice")
        await self.ck.save("b", _cp({}), owner="alice")
        await self.ck.save("c", _cp({}), owner="bob")

        assert await self.ck.list("alice") == ["a", "b"]
        assert await self.ck.list("bob") == ["c"]
        assert await self.ck.list() == []

    async def test_delete_scoped_to_owner(self):
        await self.ck.save("x", _cp({}), owner="alice")
        await self.ck.save("x", _cp({}), owner="bob")

        await self.ck.delete("x", owner="alice")
        assert await self.ck.load("x", owner="alice") is None
        assert await self.ck.load("x", owner="bob") is not None

    async def test_overwrite_scoped_to_owner(self):
        await self.ck.save("x", _cp({"v": 1}), owner="alice")
        await self.ck.save("x", _cp({"v": 2}), owner="alice")
        assert (await self.ck.load("x", owner="alice")).state == {"v": 2}


class TestSQLiteOwnerMigration:
    def test_migrates_legacy_single_owner_table(self, tmp_path):
        import asyncio
        import sqlite3

        path = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                next_node_id TEXT,
                iteration INTEGER NOT NULL
            )
            """
        )
        conn.execute("INSERT INTO checkpoints VALUES ('old-1', '{\"x\": 1}', NULL, 2)")
        conn.commit()
        conn.close()

        from draf.checkpoint import SQLiteCheckpointer

        ck = SQLiteCheckpointer(path)
        try:
            assert asyncio.run(ck.load("old-1")).state == {"x": 1}
            assert asyncio.run(ck.list()) == ["old-1"]
            # save into the migrated owner-scoped schema works
            asyncio.run(ck.save("new", _cp({"y": 2}), owner="alice"))
            assert asyncio.run(ck.load("new", owner="alice")).state == {"y": 2}
        finally:
            ck.close()


class TestGraphOwnerPassThrough:
    @pytest.mark.parametrize("kind", ["file", "sqlite"])
    @pytest.mark.asyncio
    async def test_run_scopes_checkpoints(self, kind, tmp_path):
        if kind == "file":
            from draf.checkpoint import JSONFileCheckpointer

            ck = JSONFileCheckpointer(str(tmp_path))
        else:
            from draf.checkpoint import SQLiteCheckpointer

            ck = SQLiteCheckpointer(str(tmp_path / "ck.db"))
        try:
            g = _build_linear_graph()
            state = {"text": "hi"}
            await g.run(
                state=state,
                checkpointer=ck,
                checkpoint_id="run-1",
                owner="alice",
            )
            await g.run(
                state=state,
                checkpointer=ck,
                checkpoint_id="run-1",
                owner="bob",
            )

            assert await ck.list("alice") == ["run-1"]
            assert await ck.list("bob") == ["run-1"]

            # same id but different owners do not share state
            await ck.save("chat-9", _cp({"text": "alice-secret"}), owner="alice")
            await ck.save("chat-9", _cp({"text": "bob-secret"}), owner="bob")
            alice = await g.run(
                state={"text": "ignored"},
                checkpointer=ck,
                checkpoint_id="chat-9",
                owner="alice",
            )
            # resume used alice's saved state, not bob's
            assert alice["text"] == "alice-secret"
        finally:
            close = getattr(ck, "close", None)
            if close is not None:
                close()


@pytest.fixture
def checkpointer(request, tmp_path):
    kind = request.param
    if kind == "file":
        from draf.checkpoint import JSONFileCheckpointer

        return JSONFileCheckpointer(str(tmp_path))
    if kind == "sqlite":
        from draf.checkpoint import SQLiteCheckpointer

        return SQLiteCheckpointer(str(tmp_path / "ck.db"))


@pytest.mark.parametrize("checkpointer", ["file", "sqlite"], indirect=True)
class TestCheckpointResume:
    async def test_resume_from_checkpoint(self, checkpointer):
        """Fresh run writes checkpoints; re-running with same id resumes."""
        g = _build_linear_graph()

        state = {"text": "Hello World"}
        result = await g.run(
            state=state,
            checkpointer=checkpointer,
            checkpoint_id="run-1",
        )
        assert result["a"] == "HELLO WORLD"
        assert result["b"] == "hello world"

        # Terminal checkpoint: next_node_id is None -> resume returns state
        cp = await checkpointer.load("run-1")
        assert cp is not None
        assert cp.next_node_id is None

        again = await g.run(
            state={"text": "IGNORED"},
            checkpointer=checkpointer,
            checkpoint_id="run-1",
        )
        assert again["a"] == "HELLO WORLD"
        assert again["b"] == "hello world"

    async def test_untouched_new_id_ignores_state(self, checkpointer):
        g = _build_linear_graph()
        result = await g.run(
            state={"text": "abc"},
            checkpointer=checkpointer,
            checkpoint_id="fresh",
        )
        assert result["a"] == "ABC"

    async def test_crash_between_nodes_resumes_next(self, checkpointer):
        """Simulate a crash: save a checkpoint manually and resume."""
        from draf.checkpoint import Checkpoint

        g = _build_linear_graph()
        await checkpointer.save(
            "run-2",
            Checkpoint(
                state={"text": "xx", "a": "XX"}, next_node_id=None, iteration=99
            ),
        )
        # next_node_id None -> completed, returns saved state
        result = await g.run(
            state={"text": "ignored"},
            checkpointer=checkpointer,
            checkpoint_id="run-2",
        )
        assert result["a"] == "XX"

    async def test_resume_mid_graph(self, checkpointer):
        """A checkpoint pointing at node 2 skips node 1 on resume."""
        from draf.checkpoint import Checkpoint

        g = _build_linear_graph()
        node_ids = list(g.nodes)
        await checkpointer.save(
            "run-3",
            Checkpoint(
                state={"text": "Hi", "a": "HI"},
                next_node_id=node_ids[1],
                iteration=1,
            ),
        )
        result = await g.run(
            state={"text": "ignored"},
            checkpointer=checkpointer,
            checkpoint_id="run-3",
        )
        assert result["a"] == "HI"
        assert result["b"] == "hi"

    async def test_state_instance_keeps_schema(self, checkpointer):
        from draf.state import State

        class S(TypedDict):
            text: str
            a: str

        g = _build_linear_graph()
        st = State(S, {"text": "Hello"})
        result = await g.run(
            state=st,
            checkpointer=checkpointer,
            checkpoint_id="state-run",
        )
        assert isinstance(result, State)
        assert result["a"] == "HELLO"

    async def test_error_edge_checkpoint_points_to_fallback(self, checkpointer):
        """After a node fails and routes via __error__, resume goes to fallback."""
        from draf.graph import Graph, Edge
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
        result = await g.run(
            state={},
            checkpointer=checkpointer,
            checkpoint_id="err-run",
        )
        assert result["handled"] is True

        # checkpoint points at the fallback node (completed run -> None)
        cp = await checkpointer.load("err-run")
        assert cp is not None
        assert cp.next_node_id is None

    async def test_no_error_edge_keeps_checkpoint_at_failed_node(self, checkpointer):
        from draf.graph import Graph
        from draf.node import Node

        class Crash(Node):
            type = "cr"

            async def execute(self, ctx, state):
                raise ValueError("crash")

        g = Graph(
            nodes={"a": Crash({})},
            edges=[],
            entry_point="a",
        )
        with pytest.raises(ValueError, match="crash"):
            await g.run(
                state={"kept": True},
                checkpointer=checkpointer,
                checkpoint_id="fail-run",
            )

        # checkpoint still points at the failed node, so a resume retries it
        cp = await checkpointer.load("fail-run")
        assert cp is not None
        assert cp.next_node_id == "a"
        assert cp.state == {"kept": True}
