"""Tests for the ``Extract`` structured-extraction recipe and ``Fallback`` node.

Covers the declarative recipe (LLM pass + deterministic fallbacks), the
``Fallback`` node's no-op semantics, and the end-to-end wiring of an
``Extract`` chain inside an ``interrupt_loop`` ``done`` (the repair-ai-chat
extractor pattern, in miniature).
"""

import pytest

from draf.flow import Flow
from draf.node import LLM, Ask, Extract, Fallback
from draf.node.interrupt import GraphInterrupt


async def _run(node, state: dict) -> dict:
    from draf.node import ExecContext

    ctx = ExecContext(state=state, tools={})
    return await node.execute(ctx, state)


def _room_from_first_user(state):
    for message in state.get("messages", []):
        if message.get("role") == "user" and message.get("content"):
            text = str(message["content"]).lower()
            return (
                "bathroom"
                if "ванн" in text
                else ("kitchen" if "кухн" in text else None)
            )
    return None


class TestExtract:
    def test_model_builds_llm_and_fallback_chain(self):
        extractor = Extract.model(
            system="Ты извлекаешь данные проекта.",
            schema={"type": "object", "properties": {"room_type": {"type": "string"}}},
            model="llama3.1:8b",
            provider="ollama",
            messages_key="messages",
            output_key="project_info",
            fallbacks=[Extract.fallback("room_type", _room_from_first_user)],
        )
        nodes = extractor.nodes()
        assert isinstance(nodes[0], LLM)
        assert isinstance(nodes[1], Fallback)

        llm_cfg = nodes[0].config
        assert llm_cfg["system"] == "Ты извлекаешь данные проекта."
        assert llm_cfg["messages_key"] == "messages"
        assert llm_cfg["output_key"] == "project_info"
        assert llm_cfg["json_schema"] is not None

        fb_cfg = nodes[1].config
        assert fb_cfg["input_key"] == "project_info"
        assert fb_cfg["field"] == "room_type"
        assert fb_cfg["fn"] is _room_from_first_user

    def test_nodes_are_empty_without_fallbacks(self):
        extractor = Extract.model(
            system="s", schema={}, model="m", provider="p", output_key="info"
        )
        assert len(extractor.nodes()) == 1

    def test_llm_threads_kwargs(self):
        extractor = Extract(
            system="s",
            schema={},
            model="m",
            provider="p",
            max_retries=5,
            temperature=0.2,
        )
        llm_cfg = extractor.llm().config
        assert llm_cfg["max_retries"] == 5
        assert llm_cfg["temperature"] == 0.2


class TestFallback:
    @pytest.mark.asyncio
    async def test_fills_empty_field(self):
        node = Fallback(input_key="info", field="room", fn=_room_from_first_user)
        out = await _run(
            node,
            {
                "messages": [{"role": "user", "content": "Помоги с ремонтом ванной."}],
                "info": {"area": 5.0},
            },
        )
        assert out["info"] == {"area": 5.0, "room": "bathroom"}

    @pytest.mark.asyncio
    async def test_preserves_model_value(self):
        node = Fallback(input_key="info", field="room", fn=_room_from_first_user)
        out = await _run(
            node,
            {
                "messages": [{"role": "user", "content": "Помоги с ремонтом ванной."}],
                "info": {"room": "kitchen"},
            },
        )
        assert out == {}

    @pytest.mark.asyncio
    async def test_skips_when_fn_returns_none(self):
        node = Fallback(input_key="info", field="room", fn=_room_from_first_user)
        out = await _run(
            node,
            {
                "messages": [{"role": "user", "content": "какой стиль выбрать?"}],
                "info": {},
            },
        )
        assert out == {}

    @pytest.mark.asyncio
    async def test_noop_without_callable(self):
        node = Fallback(input_key="info", field="room")
        out = await _run(node, {"messages": [], "info": {}})
        assert out == {}


class TestExtractInFlow:
    @pytest.mark.asyncio
    async def test_extraction_chain_runs_in_loop_done(self, monkeypatch, tmp_path):
        """A model that drops the room is patched by the Extract fallback."""
        from draf.checkpoint import JSONFileCheckpointer
        from draf.provider import ProviderRegistry

        called = {}

        async def fake_execute(self, ctx, state):
            if self.config.get("messages_key"):
                called["system"] = self.config.get("system", "")
                return {"project_info": {"area": 5.0}}  # model drops room_type
            return {"verdict": {"ok": True}}

        monkeypatch.setattr(LLM, "execute", fake_execute)

        flow = Flow(
            "extract", providers=ProviderRegistry.from_presets("ollama")
        ).interrupt_loop(
            key="approved",
            prompt="Одобрить план?",
            accept=Ask.model(
                system="s",
                user="Ответ: {approved}",
                schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
                model="m",
                provider="ollama",
                verdict_key="verdict",
                decision_key="approved_ok",
            ),
            body=[],
            done=Extract.model(
                system="Ты извлекаешь данные проекта.",
                schema={
                    "type": "object",
                    "properties": {"room_type": {"type": "string"}},
                },
                model="m",
                provider="ollama",
                messages_key="messages",
                output_key="project_info",
                fallbacks=[Extract.fallback("room_type", _room_from_first_user)],
            ).nodes(),
            id="approval",
        )
        graph = flow.compile()

        async def _resume(answers):
            cp = JSONFileCheckpointer(str(tmp_path))
            state = {
                "messages": [
                    {"role": "user", "content": "Помоги спланировать ремонт ванной."}
                ]
            }
            try:
                await graph.run(state=state, checkpointer=cp, checkpoint_id="x")
            except GraphInterrupt:
                pass
            last = {}
            for answer in answers:
                try:
                    last = await graph.run(
                        state=state, checkpointer=cp, checkpoint_id="x", resume=answer
                    )
                except GraphInterrupt:
                    continue
                break
            return last

        last = await _resume([{"approved": "да"}])
        assert last["project_info"] == {"area": 5.0, "room_type": "bathroom"}
        assert called["system"] == "Ты извлекаешь данные проекта."

    def test_flow_wires_extractor_nodes(self):
        extractor = Extract.model(
            system="s",
            schema={},
            model="m",
            provider="p",
            output_key="info",
            fallbacks=[Extract.fallback("room", _room_from_first_user)],
        )
        flow = Flow("wire").interrupt_loop(
            key="approved",
            prompt="Ок?",
            accept=Ask.equals("да", decision_key="ok"),
            body=[],
            done=extractor.nodes(),
            id="approval",
        )
        graph = flow.compile()
        node_types = {n.type for nid, n in graph.nodes.items()}
        assert {"interrupt", "validate", "llm_chat", "fallback"} <= node_types
