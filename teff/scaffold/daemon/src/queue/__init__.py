"""File-backed job queue for the {{PROJECT_NAME}} daemon.

The daemon has no HTTP surface: producers drop a job (a JSON file) into the
queue directory, and the worker polls for pending jobs, runs each one as a
durable conversation turn, writes the result and deletes the job file.
Swap the directory for Redis/SQS by keeping the same function signatures.

Job file (``data/queue/<job_id>.json``)::

    {"session_id": "...", "message": "..."}

Result file (``data/results/<job_id>.json``): the final state dict, or
``{"error": "..."}`` when the turn failed.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from teff.checkpoint import DEFAULT_OWNER


def enqueue(
    message: str,
    *,
    session_id: str | None = None,
    owner: str = DEFAULT_OWNER,
    queue_dir: str | Path | None = None,
) -> str:
    """Write one pending job and return its job id."""
    job_id = uuid.uuid4().hex
    job = {
        "session_id": session_id or job_id,
        "message": message,
        "owner": owner,
    }
    path = _queue_path(queue_dir) / f"{job_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    return job_id


def pending(queue_dir: str | Path | None = None) -> list[str]:
    """Return the job ids currently waiting to be processed."""
    base = _queue_path(queue_dir)
    if not base.exists():
        return []
    return sorted(p.name[: -len(".json")] for p in base.glob("*.json"))


def load_job(job_id: str, *, queue_dir: str | Path | None = None) -> dict:
    """Read one pending job's payload."""
    return json.loads((_queue_path(queue_dir) / f"{job_id}.json").read_text("utf-8"))


def complete(
    job_id: str,
    result: dict,
    *,
    queue_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> None:
    """Write the result for *job_id* and remove the pending job file."""
    base = _results_path(results_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{job_id}.json").write_text(
        json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8"
    )
    job_path = _queue_path(queue_dir) / f"{job_id}.json"
    if job_path.exists():
        job_path.unlink()


def _queue_path(queue_dir: str | Path | None) -> Path:
    if queue_dir is not None:
        return Path(queue_dir)
    return Path(__file__).resolve().parents[2] / "data" / "queue"


def _results_path(results_dir: str | Path | None) -> Path:
    if results_dir is not None:
        return Path(results_dir)
    return Path(__file__).resolve().parents[2] / "data" / "results"
