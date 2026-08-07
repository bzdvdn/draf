"""Remote trace exporters: push a completed :class:`Run` to an HTTP endpoint.

None of these exporters need extra dependencies — they use ``urllib`` in a
background thread, so a slow or failing remote sink never blocks (or fails)
the workflow that produced the trace.  Errors are retried and then logged.

- :class:`HttpExporter` — POST the run JSON to any endpoint (e.g. our
  ``teff obs-server`` ingest, or a generic webhook).
- :class:`LangfuseExporter` — adapts a run to the Langfuse public API
  (``POST /api/public/traces``) with one span per node and one generation
  per LLM call.
- :class:`LangsmithExporter` — adapts a run to the LangSmith ``runs`` API
  (``POST /runs/batch``) as a chain with nested node runs and LLM runs.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from teff.observability.exporter import TraceExporter
from teff.observability.model import Run

logger = logging.getLogger(__name__)


def _post_json(
    url: str,
    payload: Any,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    retries: int = 3,
    backoff: float = 1.0,
) -> None:
    """POST *payload* to *url* with exponential backoff.

    Failures are logged and swallowed so the workflow never crashes on a
    flaky sink.
    """
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    delay = backoff
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                return
        except Exception as exc:
            if attempt >= retries:
                logger.warning("failed to export trace to %s: %s", url, exc)
                return
            time.sleep(delay)
            delay *= 2.0


def _iso_timestamp(base: float, ms: float) -> str:
    """ISO-8601 UTC timestamp for *base* (unix seconds) + *ms* offset."""
    return datetime.fromtimestamp(base + ms / 1000.0, tz=timezone.utc).isoformat()


class HttpExporter(TraceExporter):
    """POST each completed run as JSON to a remote ingest endpoint.

    The body is :meth:`Run.to_dict` plus ``created_at`` (seconds since the
    epoch) so the receiving side can order runs.  Sends are asynchronous:
    :meth:`export` returns immediately, :meth:`close` drains the queue.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
        retries: int = 3,
        backoff: float = 1.0,
    ):
        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._pool = ThreadPoolExecutor(max_workers=1)

    def export(self, run: Run) -> None:
        payload = run.to_dict()
        self._pool.submit(
            _post_json,
            self.url,
            payload,
            headers=self.headers,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def close(self) -> None:
        self._pool.shutdown(wait=True)


def _run_base(run: Run) -> float:
    return run.created_at or time.time()


class LangfuseExporter(TraceExporter):
    """Push runs to a Langfuse instance via the public traces API.

    Requires ``host`` plus a ``public_key``/``secret_key`` pair (Basic
    auth).  Each node becomes a ``span`` observation and every LLM call a
    ``generation`` observation attached to its node.
    """

    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        *,
        timeout: float = 10.0,
        retries: int = 3,
        backoff: float = 1.0,
    ):
        self.url = host.rstrip("/") + "/api/public/traces"
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}"}
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._pool = ThreadPoolExecutor(max_workers=1)

    def export(self, run: Run) -> None:
        base = _run_base(run)
        observations: list[dict[str, Any]] = []
        for node in run.nodes:
            observations.append(
                {
                    "id": node.node_id,
                    "type": "span",
                    "name": node.node_id,
                    "startTime": _iso_timestamp(base, node.start_ms),
                    "endTime": _iso_timestamp(base, node.end_ms or node.start_ms),
                    "level": "ERROR" if node.status == "error" else "DEFAULT",
                    "metadata": {"node_type": node.node_type, "error": node.error},
                }
            )
            for i, call in enumerate(node.llm_calls):
                observations.append(
                    {
                        "id": f"{node.node_id}:llm:{i}",
                        "type": "generation",
                        "parentObservationId": node.node_id,
                        "name": f"{node.node_id}.llm",
                        "model": call.model,
                        "input": call.messages,
                        "output": call.response,
                        "usage": {
                            "input": call.prompt_tokens,
                            "output": call.completion_tokens,
                        },
                        "startTime": _iso_timestamp(base, node.start_ms),
                        "endTime": _iso_timestamp(base, node.end_ms or node.start_ms),
                        "metadata": {"provider": call.provider, "cached": call.cached},
                    }
                )
        payload = {
            "name": run.name,
            "timestamp": _iso_timestamp(base, 0.0),
            "userId": run.owner,
            "sessionId": run.checkpoint_id,
            "metadata": {
                "status": run.status,
                "total_ms": round(run.total_ms, 3),
                "tags": run.tags,
                "notes": run.notes,
                "topology": run.topology.to_dict(),
            },
            "observations": observations,
        }
        self._pool.submit(
            _post_json,
            self.url,
            payload,
            headers=self.headers,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def close(self) -> None:
        self._pool.shutdown(wait=True)


class LangsmithExporter(TraceExporter):
    """Push runs to LangSmith via the ``/runs/batch`` endpoint.

    Requires an API key (``x-api-key`` header).  The run becomes a ``chain``
    run, each node a child ``chain`` run, and each LLM call an ``llm`` run.
    ``project`` is passed as extra metadata so the UI can group by project.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        project: str | None = None,
        timeout: float = 10.0,
        retries: int = 3,
        backoff: float = 1.0,
    ):
        self.url = api_url.rstrip("/") + "/runs/batch"
        headers = {"x-api-key": api_key}
        if project:
            headers["x-langchain-project"] = project
        self.headers = headers
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._pool = ThreadPoolExecutor(max_workers=1)

    def export(self, run: Run) -> None:
        base = _run_base(run)
        run_id = f"teff-{run.name}-{int(base * 1000)}"
        metadata = {
            "teff": True,
            "status": run.status,
            "owner": run.owner,
            "checkpoint_id": run.checkpoint_id,
            "tags": run.tags,
            "notes": run.notes,
            "topology": run.topology.to_dict(),
        }
        runs: list[dict[str, Any]] = [
            {
                "id": run_id,
                "name": run.name,
                "run_type": "chain",
                "inputs": {},
                "outputs": {},
                "start_time": _iso_timestamp(base, 0.0),
                "end_time": _iso_timestamp(base, run.total_ms),
                "extra": {"metadata": metadata},
                "error": None if run.status != "error" else run.notes or "error",
            }
        ]
        for node in run.nodes:
            node_id = f"{run_id}:{node.node_id}"
            runs.append(
                {
                    "id": node_id,
                    "name": node.node_id,
                    "run_type": "chain",
                    "parent_run_id": run_id,
                    "inputs": {},
                    "outputs": {},
                    "start_time": _iso_timestamp(base, node.start_ms),
                    "end_time": _iso_timestamp(base, node.end_ms or node.start_ms),
                    "extra": {
                        "metadata": {
                            "node_type": node.node_type,
                            "error": node.error,
                        }
                    },
                    "error": None if node.status != "error" else node.error,
                }
            )
            for i, call in enumerate(node.llm_calls):
                runs.append(
                    {
                        "id": f"{node_id}:llm:{i}",
                        "name": f"{node.node_id}.llm",
                        "run_type": "llm",
                        "parent_run_id": node_id,
                        "inputs": {"messages": call.messages},
                        "outputs": {"response": call.response},
                        "start_time": _iso_timestamp(base, node.start_ms),
                        "end_time": _iso_timestamp(base, node.end_ms or node.start_ms),
                        "extra": {
                            "metadata": {
                                "provider": call.provider,
                                "model": call.model,
                                "prompt_tokens": call.prompt_tokens,
                                "completion_tokens": call.completion_tokens,
                                "cached": call.cached,
                            }
                        },
                    }
                )
        self._pool.submit(
            _post_json,
            self.url,
            runs,
            headers=self.headers,
            timeout=self.timeout,
            retries=self.retries,
            backoff=self.backoff,
        )

    def close(self) -> None:
        self._pool.shutdown(wait=True)
