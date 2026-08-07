"""Gate — deterministic loop decider with a retry budget.

A ``Gate`` turns a verdict (typically structured JSON from an ``LLM``)
into a discriminator value that a surrounding :meth:`teff.flow.Flow.loop`
or :meth:`teff.flow.Flow.branch` switches on — the "approve or fix" loop
behind QA / review cycles::

    flow.step(qa_llm)        # LLM(json_schema=...) -> state["verdict"]
    flow.step(Gate())        # -> state["decision"] = "yes" | "fix"
    flow.loop(
        key="decision", until="yes",
        done=finalize, body=[fix_agent, qa_llm],
    )

Each evaluation increments ``rounds_key``.  Once it reaches ``max_rounds``
the gate is forced to ``pass_value`` so the loop terminates
deterministically instead of raising a ``max_iterations`` error on the
enclosing graph.
"""

from __future__ import annotations

from teff.node.node import Node


class Gate(Node):
    """Translate a verdict object into a ``flow.loop`` decider value.

    Reads ``input_key`` (a dict), treats ``ok_field`` as the pass flag,
    and writes ``pass_value`` / ``fail_value`` to ``output_key``.  The
    verdict's ``message_field`` is copied to ``message_key`` when set
    (cleared on a pass).

    Parameters:
        input_key: State key holding the verdict object.
        ok_field: Field of the verdict object treated as the pass flag.
        output_key: State key receiving ``pass_value`` / ``fail_value``.
        pass_value: Value written when the verdict passes (or the budget is
            exhausted) — the value ``loop`` compares *until* against.
        fail_value: Value written when the verdict fails.
        rounds_key: State key with the evaluation counter (incremented here).
        max_rounds: After this many evaluations the gate is forced to
            ``pass_value``.
        message_field: Field of the verdict object with the remarks.
        message_key: State key receiving the remarks (cleared on a pass).
            Empty by default — no message is copied.
        missing_is_ok: Treat a missing / non-dict ``input_key`` value as a
            pass, so a dropped verdict never spins the loop forever.
    """

    type = "gate"

    def __init__(
        self,
        config: dict | None = None,
        *,
        input_key: str = "verdict",
        ok_field: str = "ok",
        output_key: str = "decision",
        pass_value: str = "yes",
        fail_value: str = "fix",
        rounds_key: str = "rounds",
        max_rounds: int = 3,
        message_field: str = "message",
        message_key: str = "",
        missing_is_ok: bool = True,
        **kwargs,
    ):
        merged = {
            "input_key": input_key,
            "ok_field": ok_field,
            "output_key": output_key,
            "pass_value": pass_value,
            "fail_value": fail_value,
            "rounds_key": rounds_key,
            "max_rounds": max_rounds,
            "message_field": message_field,
            "message_key": message_key,
            "missing_is_ok": missing_is_ok,
            **(config or {}),
            **kwargs,
        }
        super().__init__(**merged)

    async def execute(self, ctx, state: dict) -> dict:
        cfg = self.config
        rounds = int(state.get(cfg["rounds_key"], 0) or 0) + 1

        data = state.get(cfg["input_key"])
        if isinstance(data, dict):
            ok = bool(data.get(cfg["ok_field"], cfg["missing_is_ok"]))
            message = str(data.get(cfg["message_field"], "") or "")
        else:
            ok = bool(cfg["missing_is_ok"])
            message = ""

        forced = rounds >= int(cfg["max_rounds"])
        passed = bool(ok or forced)
        decision = cfg["pass_value"] if passed else cfg["fail_value"]

        out: dict = {
            cfg["rounds_key"]: rounds,
            cfg["output_key"]: decision,
        }
        message_key = cfg["message_key"]
        if message_key:
            out[message_key] = "" if passed else message.strip()
        return out
