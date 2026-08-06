"""Tests for ``Graph.get_state`` / ``Graph.update_state`` (HITL state edits)."""

import asyncio

import pytest

from draf.checkpoint import JSONFileCheckpointer
from draf.flow import Flow
from draf.node import Transform
from draf.node.interrupt import GraphInterrupt


def _build_flow():
    flow = Flow("state")
    flow.step(
        Transform({"action": "uppercase", "input_key": "text", "output_key": "draft"})
    )
    flow.interrupt("approve", prompt="Approve?")
    flow.step(
        Transform({"action": "uppercase", "input_key": "fix", "output_key": "final"})
    )
    return flow.compile()


class TestGetState:
    def test_returns_none_without_checkpoint(self, tmp_path):
        g = _build_flow()
        cp = JSONFileCheckpointer(str(tmp_path))
        assert asyncio.run(g.get_state("nope", checkpointer=cp)) is None

    def test_reads_state_after_pause(self, tmp_path):
        g = _build_flow()
        cp = JSONFileCheckpointer(str(tmp_path))
        with pytest.raises(GraphInterrupt):
            asyncio.run(
                g.run(state={"text": "hello"}, checkpointer=cp, checkpoint_id="run-1")
            )
        state = asyncio.run(g.get_state("run-1", checkpointer=cp))
        assert state["draft"] == "HELLO"
        assert "__interrupt__" not in state

    def test_reads_state_after_completion(self, tmp_path):
        g = _build_flow()
        cp = JSONFileCheckpointer(str(tmp_path))
        with pytest.raises(GraphInterrupt):
            asyncio.run(
                g.run(state={"text": "hi"}, checkpointer=cp, checkpoint_id="run-2")
            )
        asyncio.run(
            g.run(
                state={},
                checkpointer=cp,
                checkpoint_id="run-2",
                resume={"approve": "yes", "fix": "done"},
            )
        )
        state = asyncio.run(g.get_state("run-2", checkpointer=cp))
        assert state["final"] == "DONE"
        assert "__interrupt__" not in state


class TestUpdateState:
    def test_keyerror_without_checkpoint(self, tmp_path):
        g = _build_flow()
        cp = JSONFileCheckpointer(str(tmp_path))
        with pytest.raises(KeyError, match="no checkpoint"):
            asyncio.run(g.update_state("ghost", {"draft": "X"}, checkpointer=cp))

    def test_overrides_keys_and_persists(self, tmp_path):
        g = _build_flow()
        cp = JSONFileCheckpointer(str(tmp_path))
        with pytest.raises(GraphInterrupt):
            asyncio.run(
                g.run(state={"text": "hello"}, checkpointer=cp, checkpoint_id="run-1")
            )
        edited = asyncio.run(
            g.update_state("run-1", {"draft": "EDITED"}, checkpointer=cp)
        )
        assert edited["draft"] == "EDITED"
        state = asyncio.run(g.get_state("run-1", checkpointer=cp))
        assert state["draft"] == "EDITED"
        assert state["text"] == "hello"

    def test_resume_uses_edited_state(self, tmp_path):
        g = _build_flow()
        cp = JSONFileCheckpointer(str(tmp_path))
        with pytest.raises(GraphInterrupt):
            asyncio.run(
                g.run(state={"text": "hello"}, checkpointer=cp, checkpoint_id="run-1")
            )
        asyncio.run(g.update_state("run-1", {"fix": "good"}, checkpointer=cp))
        final = asyncio.run(
            g.run(
                state={},
                checkpointer=cp,
                checkpoint_id="run-1",
                resume={"approve": "yes"},
            )
        )
        assert final["final"] == "GOOD"

    def test_keeps_paused_on_edit(self, tmp_path):
        g = _build_flow()
        cp = JSONFileCheckpointer(str(tmp_path))
        with pytest.raises(GraphInterrupt):
            asyncio.run(
                g.run(state={"text": "hello"}, checkpointer=cp, checkpoint_id="run-1")
            )
        asyncio.run(g.update_state("run-1", {"draft": "EDITED"}, checkpointer=cp))
        assert asyncio.run(g.pending("run-1", checkpointer=cp)) is not None


class TestAssistantWrappers:
    @pytest.mark.asyncio
    async def test_assistant_get_update_roundtrip(self, tmp_path):
        from draf.assistant import Assistant

        g = _build_flow()
        cp = JSONFileCheckpointer(str(tmp_path))
        assistant = Assistant(g, tools=[], checkpointer=cp)
        with pytest.raises(GraphInterrupt):
            await g.run(
                state={"text": "hello"}, checkpointer=cp, checkpoint_id="sess-1"
            )
        assert (await assistant.get_state("sess-1"))["draft"] == "HELLO"
        await assistant.update_state("sess-1", {"fix": "ready"})
        assert (await assistant.get_state("sess-1"))["fix"] == "ready"
        final = await g.run(
            state={},
            checkpointer=cp,
            checkpoint_id="sess-1",
            resume={"approve": "yes"},
        )
        assert final["final"] == "READY"
