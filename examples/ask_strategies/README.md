# Ask strategies — validate interrupt answers beyond a single word

`interrupt_loop(key, accept=Ask(...))` replaces the old classifier + gate
plumbing with a single declarative strategy. The `Ask` decides whether the
operator's answer passes and can **capture an arbitrary value** (a discount
code, a date, …) alongside the pass/fail decision.

This example builds the same "approve the checkout" flow three times, only
swapping the `Ask` strategy:

1. **`regex` gate** — the operator enters a promo code. Anything not matching
   `^[A-Z]{2}-[0-9]{4}$` is rejected and re-asked; a valid code is captured
   into `state["discount_code"]` and applied.

2. **`equals` approval** — a bare `да` approves the total, `нет` rejects it.

3. **`model` approval** — a tiny classifier LLM turns free-form answers
   (`конечно`, `ок`, `хорошо`) into a `{ok: bool}` verdict, so any phrasing
   counts as approval.

## Run it

No API key and no Ollama needed — the `model` classifier is mocked. Only the
`regex` variant runs a real graph with no LLM at all:

```bash
uv run python examples/ask_strategies/run.py
```

Expected output (framework interrupt warnings go to stderr):

```
=== regex-gate ===
  Введите промокод (формат XX-1234):
  Введите промокод (формат XX-1234):
  discount_code='AB-1234'
  total='скидка применена'

=== equals-gate ===
  Сумма 1000 руб. Одобрить? (да/нет)
  Сумма 1000 руб. Одобрить? (да/нет)
  discount_code=None
  total='заказ оформлен'

=== model-gate ===
  Сумма 1000 руб. Одобрить? (можно своими словами)
  Сумма 1000 руб. Одобрить? (можно своими словами)
  discount_code=None
  total='заказ оформлен'
```

Each gate re-asks once (the first answer is rejected) and then passes. See
[docs/reference/nodes.md](../../docs/reference/nodes.md) for the full `Ask`
reference and [docs/guide/durable.md](../../docs/guide/durable.md) for the
checkpoint/resume mechanics behind interrupts.
