"""Prompts for the fraud-gate LLM nodes.

Both nodes use ``{key}`` placeholders rendered from top-level state keys (see
:func:`draf.prompt.render_template`).  The analyser reads the payment's free
``note`` and returns a structured :class:`~fraud_gate.graphs.state.RiskVerdict`;
the finalizer turns the routing decision into a short human-facing summary.
"""

from __future__ import annotations

ANALYSER_SYSTEM = (
    "Ты — аналитик платёжного шлюза. Оцени риск мошенничества по описанию "
    "транзакции и верни ровно один JSON-объект вида "
    '{{"risk": <число 0..1>, "signals": "<кратко, какие сигналы мошенничества>"}}. '
    "Только JSON, без пояснений."
)

ANALYSER_PROMPT = (
    "Транзакция: id={tx_id}, сумма={tx_amount}, торговец={tx_merchant}, "
    "страна={tx_country}, IP={tx_ip_country}.\n"
    'Комментарий плательщика: "{tx_note}"\n'
    "Оцени риск."
)

FINALIZER_SYSTEM = (
    "Ты — голос платёжного шлюза. Верни ровно один JSON-объект вида "
    '{{"decision": "approve или deny", "summary": "<одно предложение для клиента>"}}. '
    "Только JSON, без пояснений."
)

FINALIZER_PROMPT = (
    "Транзакция id={tx_id}, риск {risk}. Решение шлюза: {decision}. "
    "Сигналы: {reason}. Замечание аналитика: {review_decision}.\n"
    "Сформулируй итог в одно предложение."
)
