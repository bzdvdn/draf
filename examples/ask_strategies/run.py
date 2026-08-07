"""Ask strategies — validate an interrupt answer beyond a single word.

``interrupt_loop`` with an :class:`~teff.node.ask.Ask` strategy decides
whether the operator's answer passes, and can *capture a value* (a
discount code, a date, …) alongside the pass/fail decision.  This example
builds three variants of the same checkout flow:

1. ``regex`` gate — the operator enters a discount code.  Anything that
   doesn't match ``^[A-Z]{2}-[0-9]{4}$`` is rejected and re-asked; a valid
   code is captured into ``state["discount_code"]`` and applied.
2. ``equals`` approval — a bare "да" approves the total (or "нет"
   rejects it).
3. ``model`` approval — a tiny classifier LLM turns free-form answers
   ("конечно", "ок", "хорошо") into the verdict, so any phrasing counts
   as approval.

The graph is a plain ``Flow``; only the ``Ask`` strategy differs.  By
default the LLM classifier is mocked (no API key, no Ollama) so the
``model`` variant runs offline too.

Usage:
    python examples/ask_strategies/run.py
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from teff.checkpoint import JSONFileCheckpointer
from teff.flow import Flow
from teff.node import Ask, Transform
from teff.node.interrupt import GraphInterrupt
from teff.provider import ProviderRegistry

MODEL = "llama3.1:8b"
PROVIDER = "ollama"

VERDICT_SYSTEM = (
    "Ты классифицируешь ответ пользователя. Верни ТОЛЬКО JSON без пояснений "
    'с полем "ok" (boolean): true — если пользователь одобрил, false — если нет.'
)
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}

#: Free-form answers the mocked classifier counts as approval.
APPROVALS = {"да", "конечно", "ок", "хорошо", "согласен"}


def _mock_transport() -> None:
    """Scripted mock classifier: ``ok=true`` for any word in *APPROVALS*."""
    import httpx

    async def mock_post(*_args, **_kwargs):
        body = _kwargs.get("json") or {}
        messages = body.get("messages") or []
        user_text = next(
            (
                str(m.get("content", ""))
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        approved = any(word in user_text.lower() for word in APPROVALS)
        content = '{"ok": true}' if approved else '{"ok": false}'

        class _Reply:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "message": {"role": "assistant", "content": content},
                    "choices": [{"message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                }

        return _Reply()

    httpx.AsyncClient.post = mock_post  # type: ignore[method-assign]


class CheckoutState(TypedDict):
    discount_code: str
    total: str
    decision: str


def _regex_gate() -> Flow:
    """Discount code must match a pattern; the code is captured into state."""
    return Flow(
        "regex-gate", providers=ProviderRegistry.from_presets(PROVIDER)
    ).interrupt_loop(
        key="code",
        prompt="Введите промокод (формат XX-1234):",
        accept=Ask.regex(
            r"^[A-Z]{2}-[0-9]{4}$",
            decision_key="code_ok",
            value_key="discount_code",
        ),
        body=Transform(
            {"action": "value", "value": "неверный код", "output_key": "total"}
        ),
        done=Transform(
            {"action": "value", "value": "скидка применена", "output_key": "total"}
        ),
        id="discount",
    )


def _equals_gate() -> Flow:
    """Bare approval: the raw answer must normalize to "да"."""
    return Flow(
        "equals-gate", providers=ProviderRegistry.from_presets(PROVIDER)
    ).interrupt_loop(
        key="approved",
        prompt="Сумма 1000 руб. Одобрить? (да/нет)",
        accept=Ask.equals("да", decision_key="approved_ok"),
        body=Transform({"action": "value", "value": "отменено", "output_key": "total"}),
        done=Transform(
            {"action": "value", "value": "заказ оформлен", "output_key": "total"}
        ),
        id="approval",
    )


def _model_gate() -> Flow:
    """Free-form approval: a classifier LLM normalizes the answer."""
    return Flow(
        "model-gate", providers=ProviderRegistry.from_presets(PROVIDER)
    ).interrupt_loop(
        key="approved",
        prompt="Сумма 1000 руб. Одобрить? (можно своими словами)",
        accept=Ask.model(
            system=VERDICT_SYSTEM,
            user="Ответ пользователя:\n{approved}\n\nОдобрил ли пользователь?",
            schema=VERDICT_SCHEMA,
            model=MODEL,
            provider=PROVIDER,
            verdict_key="verdict",
            decision_key="approved_ok",
        ),
        body=Transform({"action": "value", "value": "отменено", "output_key": "total"}),
        done=Transform(
            {"action": "value", "value": "заказ оформлен", "output_key": "total"}
        ),
        id="approval",
    )


def _answers() -> dict[str, list[dict]]:
    """Simulated operator answers, keyed by flow name."""
    return {
        # invalid code first, then a valid one
        "regex-gate": [{"code": "XX-000"}, {"code": "AB-1234"}],
        # "нет" rejects, "да" approves
        "equals-gate": [{"approved": "нет"}, {"approved": "да"}],
        # free-form "конечно" still counts as approval via the classifier
        "model-gate": [{"approved": "нет"}, {"approved": "конечно"}],
    }


async def _resume(graph, cp: JSONFileCheckpointer, answers: list[dict]) -> dict:
    """Run the graph, feeding the simulated operator's answers to interrupts."""
    # First run triggers the opening interrupt (no resume yet).
    try:
        await graph.run(state={}, checkpointer=cp, checkpoint_id="ask-run")
    except GraphInterrupt as interrupt:
        print(f"  {interrupt.prompt}")
    last: dict = {}
    for answer in answers:
        try:
            last = await graph.run(
                state={}, checkpointer=cp, checkpoint_id="ask-run", resume=answer
            )
        except GraphInterrupt as interrupt:
            print(f"  {interrupt.prompt}")
        else:
            break
    return last


async def run_offline() -> None:
    """Run all three variants without any LLM (model variant is mocked)."""
    _mock_transport()

    for name, flow in [
        ("regex-gate", _regex_gate()),
        ("equals-gate", _equals_gate()),
        ("model-gate", _model_gate()),
    ]:
        print(f"\n=== {name} ===")
        cp = JSONFileCheckpointer(f"/tmp/ask-strategies-{name}")
        last = await _resume(flow.compile(), cp, _answers()[name])
        print(f"  discount_code={last.get('discount_code')!r}")
        print(f"  total={last['total']!r}")


async def main() -> None:
    await run_offline()


if __name__ == "__main__":
    asyncio.run(main())
