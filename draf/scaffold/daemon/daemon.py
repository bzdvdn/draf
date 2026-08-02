"""Daemon entry point — process pending jobs from the queue directory.

A long-running worker with no HTTP surface: it polls ``data/queue/`` for
pending jobs, runs each as one durable conversation turn via the same
:class:`~src.service.assistant.Assistant` as the other templates, writes the
result to ``data/results/`` and removes the job file.

Usage::

    uv run python daemon.py                 # poll forever (Ctrl-C to stop)
    uv run python daemon.py --once          # drain the queue and exit
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.config import get_settings  # noqa: E402
from src.core import build_container  # noqa: E402
from src.queue import complete, load_job, pending  # noqa: E402
from src.service.assistant import Assistant  # noqa: E402


def _build_assistant():
    settings = get_settings()
    return build_container(settings), settings


async def _process_job(
    assistant: Assistant,
    job_id: str,
    *,
    queue_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict:
    job = load_job(job_id, queue_dir=queue_dir)
    try:
        result = await assistant.run_turn(job["session_id"], job["message"])
        payload: dict = {"ok": True, "job_id": job_id, "result": result}
    except Exception as exc:
        payload = {"ok": False, "job_id": job_id, "error": str(exc)}
    complete(job_id, payload, queue_dir=queue_dir, results_dir=results_dir)
    return payload


async def _drain_once(
    assistant: Assistant,
    *,
    max_jobs: int | None = None,
    queue_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> int:
    """Process all currently pending jobs; return how many were handled."""
    processed = 0
    for job_id in pending(queue_dir=queue_dir):
        if max_jobs is not None and processed >= max_jobs:
            break
        print(f"-- processing {job_id} --")
        result = await _process_job(
            assistant, job_id, queue_dir=queue_dir, results_dir=results_dir
        )
        print(
            f"== done {job_id}: ok={result.get('ok')} "
            f"error={result.get('error')!r}"
        )
        processed += 1
    return processed


async def _poll(assistant: Assistant, *, once: bool) -> None:
    while True:
        await _drain_once(assistant)
        if once:
            return
        await asyncio.sleep(get_settings().poll_interval)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the {{PROJECT_NAME}} daemon.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="drain the queue and exit instead of polling forever",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="process at most this many jobs per pass",
    )
    args = parser.parse_args()

    container, _ = _build_assistant()
    print(
        f"provider: {settings.provider}  model: {settings.model}  "
        f"(requires a running Ollama)"
    )
    if args.once:
        asyncio.run(_drain_once(container.assistant, max_jobs=args.max_jobs))
        return
    asyncio.run(_poll(container.assistant, once=False))


if __name__ == "__main__":
    main()
