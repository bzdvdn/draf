"""Tests for the ``Ask`` validation strategy behind ``interrupt_loop``.

Covers the strategies (equals / any_of / regex / check / model), the
arbitrary-value capture (``value_key``) that plain yes/no gates can't do,
and the wiring ``interrupt -> [classifier] -> validate -> loop`` produced
by ``Flow.interrupt_loop``.
"""

import asyncio

import pytest

from draf.flow import Flow
from draf.node import Ask, Transform
from draf.node.ask import Validate
from draf.node.interrupt import GraphInterrupt
from draf.node.node import Node


async def _run(node, state: dict) -> dict:
    from draf.node import ExecContext

    ctx = ExecContext(state=state, tools={})
    return await node.execute(ctx, state)


async def _resume(graph, cp, answers: list[dict]) -> dict:
    """Trigger the opening interrupt, then feed answers until the graph ends."""
    try:
        await graph.run(state={}, checkpointer=cp, checkpoint_id="ask-run")
    except GraphInterrupt:
        pass
    last: dict = {}
    for answer in answers:
        try:
            last = await graph.run(
                state={}, checkpointer=cp, checkpoint_id="ask-run", resume=answer
            )
        except GraphInterrupt:
            continue
        break
    return last


class TestValidateStrategies:
    @pytest.mark.asyncio
    async def test_equals_is_normalized(self):
        v = Validate(input_key="answer", strategy="equals", equals="да")
        out = await _run(v, {"answer": "  Да "})
        assert out["decision"] == "да"

    @pytest.mark.asyncio
    async def test_any_of_accepts_variants(self):
        v = Validate(input_key="answer", strategy="any_of", any_of=["да", "ок", "конечно"])
        assert (await _run(v, {"answer": "ОК"}))["decision"] == "да"
        assert (await _run(v, {"answer": "нет"}))["decision"] == "нет"

    @pytest.mark.asyncio
    async def test_regex_captures_value_into_value_key(self):
        v = Validate(
            input_key="answer",
            strategy="regex",
            regex=r"^([A-Z]{2}-[0-9]{4})$",
            value_key="discount_code",
        )
        out = await _run(v, {"answer": "AB-1234"})
        assert out["decision"] == "да"
        assert out["discount_code"] == "AB-1234"
        out = await _run(v, {"answer": "XX-000"})
        assert out["decision"] == "нет"
        assert out["discount_code"] == ""

    @pytest.mark.asyncio
    async def test_check_callable_tuple_extraction(self):
        v = Validate(
            input_key="answer",
            strategy="check",
            check=lambda a: (a.lower() == "да", a.upper()),
            value_key="captured",
        )
        out = await _run(v, {"answer": "да"})
        assert out["decision"] == "да"
        assert out["captured"] == "ДА"

    @pytest.mark.asyncio
    async def test_verdict_dict_uses_ok_field(self):
        v = Validate(
            input_key="verdict",
            verdict_key="verdict",
            ok_field="ok",
            value_field="code",
            value_key="discount_code",
        )
        out = await _run(v, {"verdict": {"ok": True, "code": "AB-1234"}})
        assert out["decision"] == "да"
        assert out["discount_code"] == "AB-1234"
        out = await _run(v, {"verdict": {"ok": False}})
        assert out["decision"] == "нет"
        assert out["discount_code"] == ""

    @pytest.mark.asyncio
    async def test_unclear_verdict_writes_clarify_value(self):
        v = Validate(
            input_key="verdict",
            verdict_key="verdict",
            ok_field="ok",
            clear_field="clear",
            clarify_value="уточнить",
        )
        out = await _run(v, {"verdict": {"ok": False, "clear": False}})
        assert out["decision"] == "уточнить"
        out = await _run(v, {"verdict": {"ok": False, "clear": True}})
        assert out["decision"] == "нет"
        # an unclear reply vetoes an approval too — never auto-approve
        out = await _run(v, {"verdict": {"ok": True, "clear": False}})
        assert out["decision"] == "уточнить"


class TestAsk:
    def test_strategy_detection(self):
        assert Ask(equals="да").strategy == "equals"
        assert Ask(any_of=["да", "ок"]).strategy == "any_of"
        assert Ask(regex=r"\d+").strategy == "regex"
        assert Ask(system="s", schema={}).strategy == "model"

    def test_model_ask_needs_classifier(self):
        assert Ask.model(
            system="s", user="u", schema={}, model="m", provider="p"
        ).needs_classifier()
        assert not Ask.equals("да").needs_classifier()

    def test_validate_node_wiring(self):
        v = Ask.regex(r"^[A-Z]{2}-\d{4}$", decision_key="go", value_key="code").validate_node("answer")
        assert v.config["input_key"] == "answer"
        assert v.config["output_key"] == "go"
        assert v.config["value_key"] == "code"
        assert v.config["strategy"] == "regex"

    def test_model_ask_carries_clarify_config(self):
        v = Ask.model(
            system="s",
            user="u",
            schema={},
            model="m",
            provider="p",
            clear_field="clear",
            clarify_value="уточнить",
        ).validate_node("answer")
        assert v.config["clear_field"] == "clear"
        assert v.config["clarify_value"] == "уточнить"


class TestInterruptLoopAsk:
    @pytest.fixture(autouse=True)
    def _checkpoint(self, tmp_path):
        from draf.checkpoint import JSONFileCheckpointer

        self.cp = JSONFileCheckpointer(str(tmp_path))

    def _regex_flow(self) -> Flow:
        return (
            Flow("regex")
            .interrupt_loop(
                key="code",
                prompt="Введите промокод:",
                accept=Ask.regex(
                    r"^[A-Z]{2}-[0-9]{4}$",
                    decision_key="code_ok",
                    value_key="discount_code",
                ),
                body=Transform({"action": "value", "value": "неверный код", "output_key": "total"}),
                done=Transform({"action": "value", "value": "скидка применена", "output_key": "total"}),
                id="discount",
            )
        )

    def test_wires_interrupt_validate_loop(self):
        g = self._regex_flow().compile()
        edges = {(e.source_id, e.target_id, e.condition) for e in g.edges}
        assert ("discount-interrupt", "discount-validate", None) in edges
        assert any(
            src == "discount-validate" and cond == "code_ok=да"
            for src, _, cond in edges
        )
        assert any(
            src == "discount-validate" and cond == "code_ok!=да"
            for src, _, cond in edges
        )
        # the "нет" branch loops back through a fresh interrupt -> validate
        assert any(
            target == "discount-validate" for src, target, cond in edges
        )
        node_types = {n.type for nid, n in g.nodes.items()}
        assert {"interrupt", "validate", "transform"} <= node_types

    def test_rejects_invalid_code_and_captures_valid(self):
        g = self._regex_flow().compile()
        last = asyncio.run(_resume(g, self.cp, [{"code": "XX-000"}, {"code": "AB-1234"}]))
        assert last["total"] == "скидка применена"
        assert last["discount_code"] == "AB-1234"
        assert last["code_ok"] == "да"

    def test_value_key_written_from_loop_body(self):
        """The captured value survives the re-ask cycle."""
        g = self._regex_flow().compile()
        last = asyncio.run(_resume(g, self.cp, [{"code": "AB-1234"}]))
        assert last["discount_code"] == "AB-1234"


class TestInterruptLoopModel:
    @pytest.mark.asyncio
    async def test_wires_classifier_before_validate(self, monkeypatch, tmp_path):
        """A 'model' Ask inserts an LLM classifier between interrupt and validate."""
        from draf.checkpoint import JSONFileCheckpointer
        from draf.provider import ProviderRegistry

        mock_calls = []

        async def fake_execute(self, ctx, state):
            mock_calls.append(state.get("approved"))
            return {"verdict": {"ok": state.get("approved") == "да", "code": ""}}

        from draf.node.llm import LLM

        monkeypatch.setattr(LLM, "execute", fake_execute)

        flow = (
            Flow("model", providers=ProviderRegistry.from_presets("ollama"))
            .interrupt_loop(
                key="approved",
                prompt="Одобрить?",
                accept=Ask.model(
                    system="s",
                    user="Ответ: {approved}",
                    schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
                    model="m",
                    provider="ollama",
                    verdict_key="verdict",
                    decision_key="approved_ok",
                ),
                body=Transform({"action": "value", "value": "нет", "output_key": "total"}),
                done=Transform({"action": "value", "value": "да", "output_key": "total"}),
                id="approval",
            )
        )
        graph = flow.compile()
        edges = {(e.source_id, e.target_id, e.condition) for e in graph.edges}
        assert ("approval-interrupt", "approval-classifier", None) in edges
        assert ("approval-classifier", "approval-validate", None) in edges

        cp = JSONFileCheckpointer(str(tmp_path))
        last = await _resume(graph, cp, [{"approved": "да"}])
        assert last["total"] == "да"
        assert last["approved_ok"] == "да"
        assert "да" in mock_calls

    @pytest.mark.asyncio
    async def test_unclear_answer_reasks_without_body(self, monkeypatch, tmp_path):
        """A clear=false verdict routes back to the interrupt (re-ask) and the
        body chain is NOT re-run; a later pass completes the loop."""
        from draf.checkpoint import JSONFileCheckpointer
        from draf.provider import ProviderRegistry

        body_calls = []

        class _Body(Node):
            type = "body"

            async def execute(self, ctx, state):
                body_calls.append(1)
                return {}

        async def fake_execute(self, ctx, state):
            answer = state.get("approved")
            return {
                "verdict": {
                    "ok": answer == "да",
                    "clear": answer in ("да", "нет"),
                }
            }

        from draf.node.llm import LLM

        monkeypatch.setattr(LLM, "execute", fake_execute)

        flow = (
            Flow("model", providers=ProviderRegistry.from_presets("ollama"))
            .interrupt_loop(
                key="approved",
                prompt="Одобрить?",
                accept=Ask.model(
                    system="s",
                    user="Ответ: {approved}",
                    schema={
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                    },
                    model="m",
                    provider="ollama",
                    verdict_key="verdict",
                    decision_key="approved_ok",
                    clear_field="clear",
                    clarify_value="уточнить",
                ),
                body=_Body(),
                done=Transform({"action": "value", "value": "да", "output_key": "total"}),
                id="approval",
            )
        )
        graph = flow.compile()
        edges = {(e.source_id, e.target_id, e.condition) for e in graph.edges}
        assert (
            "approval-validate",
            "approval-interrupt",
            "approved_ok=уточнить",
        ) in edges

        cp = JSONFileCheckpointer(str(tmp_path))
        last = await _resume(
            graph,
            cp,
            [{"approved": "qhjrkjlkjsdgjdlksgj"}, {"approved": "да"}],
        )
        assert last["total"] == "да"
        assert body_calls == []  # the unclear reply only re-asked, no re-plan
