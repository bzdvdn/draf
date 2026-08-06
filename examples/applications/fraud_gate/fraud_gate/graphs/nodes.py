from __future__ import annotations

import json

from draf.node import Command, Node

#: Decision thresholds: score above ``review`` needs a human; above ``deny``
#: is blocked outright.
REVIEW_THRESHOLD = 0.6
DENY_THRESHOLD = 0.9


class Ingest(Node):
    """Entry — parse the JSON payment from the latest user message."""

    type = "ingest"

    async def execute(self, ctx, state: dict) -> dict:
        messages: list = state.get("messages") or []
        raw = "{}"
        for message in reversed(messages):
            if message.get("role") == "user":
                raw = message.get("content") or "{}"
                break
        tx: dict = json.loads(raw) if raw.startswith("{") else {}
        return {
            "tx_id": str(tx.get("id") or ""),
            "tx_amount": float(tx.get("amount") or 0.0),
            "tx_merchant": str(tx.get("merchant") or ""),
            "tx_country": str(tx.get("country") or ""),
            "tx_ip_country": str(tx.get("ip_country") or ""),
            "tx_note": str(tx.get("note") or ""),
            "analysis": {},
            "risk": 0.0,
            "decision": "",
            "reason": "",
            "review_decision": "",
            "final": "",
            "events": [{"role": "system", "content": "gate: transaction received"}],
        }


class Router(Node):
    """Dynamic step — pick the route from the analyser's risk score."""

    type = "router"

    async def execute(self, ctx, state: dict) -> dict | Command:
        analysis: dict = state.get("analysis") or {}
        risk = float(analysis.get("risk") or 0.0)
        signals = str(analysis.get("signals") or "") or "сигналы не указаны"

        log = [{"role": "system", "content": f"score={risk:.2f} ({signals})"}]

        if risk >= DENY_THRESHOLD:
            return Command(
                update={
                    "risk": round(risk, 2),
                    "decision": "deny",
                    "reason": signals,
                    "final": {"decision": "deny", "summary": f"Отказ: {signals}."},
                    "events": log,
                },
                goto=Command.STOP,
            )

        if risk >= REVIEW_THRESHOLD:
            return Command(
                update={
                    "risk": round(risk, 2),
                    "decision": "review",
                    "reason": signals,
                    "events": log,
                }
            )

        return Command(
            update={
                "risk": round(risk, 2),
                "decision": "approve",
                "reason": signals,
                "events": log,
            },
            goto="finalize",
        )
