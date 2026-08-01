"""Evaluation harness: run a workflow over a dataset and score the outputs.

A dataset is a list of examples; each example seeds the workflow state
and optionally carries an ``expected`` answer.  Outputs are scored by
exact match, by a similarity heuristic, or by an LLM judge.

    dataset.jsonl (one JSON object per line)::

        {"id": "q1", "query": "What is the mascot of Draf?",
         "expected": "a rocket", "input": "You are a helper."}

    every key except ``id`` / ``expected`` is merged into the workflow
    state as an initial override (on top of the workflow's own
    ``state.initial``).
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from typing import Any

from draf.harness import Harness
from draf.node.interrupt import GraphInterrupt

_OUTPUT_KEYS = ("answer", "final_answer", "output", "out", "result", "final")


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def extract_output(state: dict, output_key: str | None = None) -> str:
    """Pick the answer from the final state.

    Uses *output_key* when given; otherwise falls back to a heuristic
    list of common answer keys and finally the whole state as JSON.
    """
    if output_key:
        value = state.get(output_key)
        if value is not None and str(value).strip() != "":
            return str(value)
    for key in _OUTPUT_KEYS:
        value = state.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return json.dumps(state, ensure_ascii=False, default=str)


def load_dataset(path: str) -> list[dict]:
    """Load a dataset from a ``.json``, ``.jsonl``, or ``.csv`` file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset not found: {path}")
    if path.endswith(".jsonl"):
        return [_parse_line(line) for line in open(path) if line.strip()]
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of examples")
    return [
        item if isinstance(item, dict) else {"state": {"input": str(item)}}
        for item in data
    ]


def _parse_line(line: str) -> dict:
    item = json.loads(line)
    return item if isinstance(item, dict) else {"state": {"input": str(item)}}


def item_state(item: dict) -> dict:
    """The state overrides contributed by a dataset item."""
    return {k: v for k, v in item.items() if k not in ("id", "expected")}


def _question(item: dict) -> str:
    for key in ("query", "question", "input", "prompt"):
        if key in item:
            return str(item[key])
    return ""


async def judge_with_llm(
    harness: Harness,
    question: str,
    expected: str,
    actual: str,
) -> tuple[str, str]:
    """Ask an LLM whether *actual* satisfies *expected*.

    Returns ``(verdict, reason)`` where verdict is ``"PASS"`` or ``"FAIL"``.
    """
    prompt = (
        "You are a strict automatic evaluator for an AI workflow.\n"
        f"Input: {question}\n"
        f"Expected answer: {expected}\n"
        f"Actual output: {actual}\n"
        "Does the actual output satisfy the expected answer?\n"
        "Reply on a single line starting with exactly one word, PASS or FAIL, "
        "followed by a short reason."
    )
    reply = await harness.call([{"role": "user", "content": prompt}])
    content = (reply.content or "").strip()
    first = content.split("\n")[0].split(" ", 1)
    verdict = first[0].upper() if first else "FAIL"
    if verdict not in ("PASS", "FAIL"):
        verdict = "PASS" if "pass" in content.lower() else "FAIL"
    reason = first[1] if len(first) > 1 else ""
    return verdict, reason


async def run_eval(
    workflow: tuple[Any, list, dict, dict],
    dataset: list[dict],
    *,
    judge_model: str | None = None,
    judge_provider: str | None = None,
    exact: bool = False,
    max_examples: int | None = None,
    output_key: str | None = None,
) -> dict:
    """Evaluate *dataset* against a loaded workflow.

    Args:
        workflow: A ``(graph, tools, initial_state, reducers)`` tuple as
            returned by :func:`draf.yaml.load_workflow`.
        dataset: List of example dicts.
        judge_model: Model name for LLM judging (when *exact* is False).
        judge_provider: Provider key for the judge model.
        exact: Score by exact (normalised) string match instead of an LLM.
        max_examples: Cap the number of examples evaluated.
        output_key: State key holding the answer; defaults to a heuristic.

    Returns:
        A report dict: ``{total, passed, failed, errors, interrupted,
        cases, judge}``.
    """
    graph, tools, initial, reducers = workflow
    judge_harness: Harness | None = None
    if judge_model and not exact:
        judge_harness = Harness(model=judge_model, provider=judge_provider)

    cases: list[dict] = []
    for index, item in enumerate(dataset):
        if max_examples is not None and index >= max_examples:
            break
        state = {**initial, **item_state(item)}
        started = time.monotonic()
        status = "ok"
        output = ""
        error: str | None = None
        try:
            final = await graph.run(state, tools=tools, reducers=reducers)
            output = extract_output(final, output_key)
        except GraphInterrupt as exc:
            status = "interrupted"
            error = exc.prompt or exc.key
        except Exception as exc:  # noqa: BLE001 — collect per-case failures
            status = "error"
            error = str(exc)
        duration_ms = round((time.monotonic() - started) * 1000.0, 3)

        expected = item.get("expected")
        verdict: str | None = None
        reason: str | None = None
        if expected is not None and status == "ok":
            if exact:
                verdict = "PASS" if _norm(output) == _norm(str(expected)) else "FAIL"
            elif judge_harness is not None:
                verdict, reason = await judge_with_llm(
                    judge_harness, _question(item), str(expected), output
                )
            else:
                verdict = None

        cases.append(
            {
                "id": str(item.get("id") or index),
                "status": status,
                "duration_ms": duration_ms,
                "expected": expected,
                "output": output,
                "verdict": verdict,
                "reason": reason,
                "error": error,
            }
        )

    counted = [c for c in cases if c["verdict"] in ("PASS", "FAIL")]
    return {
        "total": len(cases),
        "passed": sum(c["verdict"] == "PASS" for c in counted),
        "failed": sum(c["verdict"] == "FAIL" for c in counted),
        "unscored": len(cases) - len(counted),
        "errors": sum(c["status"] == "error" for c in cases),
        "interrupted": sum(c["status"] == "interrupted" for c in cases),
        "judge": judge_model,
        "cases": cases,
    }


def format_report(report: dict) -> str:
    """Render an eval *report* as human-readable lines."""
    lines = [
        f"total={report['total']} passed={report['passed']} "
        f"failed={report['failed']} unscored={report['unscored']} "
        f"errors={report['errors']}",
    ]
    for case in report["cases"]:
        verdict = case["verdict"] or ("-" if case["status"] == "ok" else case["status"])
        mark = "ok" if verdict == "PASS" else ("!!" if verdict == "FAIL" else "..")
        lines.append(f"  [{mark}] {case['id']}: {verdict} ({case['duration_ms']}ms)")
    return "\n".join(lines)


def write_dataset_examples(dataset: list[dict], path: str) -> None:
    """Write *dataset* examples to a JSONL file (helper for authors)."""
    with open(path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
