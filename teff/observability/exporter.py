"""Exporters for :class:`~teff.observability.model.Run` traces.

:class:`TraceExporter` is the stable interface every sink implements —
SQLite, JSONL, and (later) langfuse/langsmith adapters.  The collector
only talks to this interface, so backends are swappable.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from teff.observability.model import Run

logger = logging.getLogger(__name__)


class TraceExporter(ABC):
    """Persist a completed :class:`Run` to some backend."""

    @abstractmethod
    def export(self, run: Run) -> str | None:
        """Persist *run* and return its backend run id (``None`` if unknown).

        Idempotent when the store uses run ids.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the exporter."""


class JsonlExporter(TraceExporter):
    """Append each run as one JSON line to a newline-delimited file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def export(self, run: Run) -> None:
        self._handle.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class SQLiteExporter(TraceExporter):
    """Store runs in SQLite tables ``runs`` / ``nodes`` / ``llm_calls``.

    The ``runs`` table also carries the graph topology as JSON, and the
    metadata needed by a dashboard (owner, checkpoint id, status, tokens).
    """

    def __init__(self, path: str | Path = "./traces.db"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._create_schema()
        self._migrate()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                total_ms REAL NOT NULL,
                owner TEXT,
                checkpoint_id TEXT,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                topology TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS nodes (
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                node_id TEXT NOT NULL,
                node_type TEXT NOT NULL,
                start_ms REAL NOT NULL,
                end_ms REAL,
                status TEXT NOT NULL,
                error TEXT,
                tool_calls TEXT NOT NULL DEFAULT '[]',
                events TEXT NOT NULL DEFAULT '[]',
                UNIQUE(run_id, node_id)
            );
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                node_id TEXT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                messages TEXT NOT NULL,
                response TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                cached INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._conn.commit()

    def _migrate(self) -> None:
        """Upgrade databases created by older versions.

        Runs from the pre-uuid era used an ``INTEGER PRIMARY KEY
        AUTOINCREMENT``; the id column is now a uuid text key, which cannot
        be altered in place, so the trace tables are rebuilt empty (a
        ``runs`` table of the wrong type means the whole layout is old).
        """
        run_cols = {
            row[1]: row[2]
            for row in self._conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        if run_cols.get("run_id", "").upper() != "TEXT":
            self._conn.executescript("DROP TABLE IF EXISTS llm_calls;")
            self._conn.executescript("DROP TABLE IF EXISTS nodes;")
            self._conn.executescript("DROP TABLE IF EXISTS runs;")
            self._create_schema()
            run_cols = {
                row[1]: row[2]
                for row in self._conn.execute("PRAGMA table_info(runs)").fetchall()
            }
        if "tags" not in run_cols:
            self._conn.execute(
                "ALTER TABLE runs ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'"
            )
        if "notes" not in run_cols:
            self._conn.execute(
                "ALTER TABLE runs ADD COLUMN notes TEXT NOT NULL DEFAULT ''"
            )
        node_cols = {
            row[1] for row in self._conn.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "tool_calls" not in node_cols:
            self._conn.execute(
                "ALTER TABLE nodes ADD COLUMN tool_calls TEXT NOT NULL DEFAULT '[]'"
            )
        if "events" not in node_cols:
            self._conn.execute(
                "ALTER TABLE nodes ADD COLUMN events TEXT NOT NULL DEFAULT '[]'"
            )
        self._conn.commit()

    def export(self, run: Run) -> str:
        self._conn.execute(
            "INSERT INTO runs (run_id, name, status, total_ms, owner, checkpoint_id, "
            "prompt_tokens, completion_tokens, topology, tags, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run.run_id,
                run.name,
                run.status,
                run.total_ms,
                run.owner,
                run.checkpoint_id,
                run.prompt_tokens,
                run.completion_tokens,
                json.dumps(run.topology.to_dict()),
                json.dumps(list(run.tags)),
                run.notes,
                run.created_at or time.time(),
            ),
        )
        run_id = run.run_id
        for node in run.nodes:
            self._conn.execute(
                "INSERT INTO nodes (run_id, node_id, node_type, start_ms, end_ms, "
                "status, error, tool_calls, events) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    node.node_id,
                    node.node_type,
                    node.start_ms,
                    node.end_ms,
                    node.status,
                    node.error,
                    json.dumps([t.to_dict() for t in node.tool_calls]),
                    json.dumps([e.to_dict() for e in node.events]),
                ),
            )
        for call in run.llm_calls:
            self._conn.execute(
                "INSERT INTO llm_calls (run_id, node_id, provider, model, messages, "
                "response, prompt_tokens, completion_tokens, latency_ms, cached) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    call.node_id,
                    call.provider,
                    call.model,
                    json.dumps(call.messages),
                    call.response,
                    call.prompt_tokens,
                    call.completion_tokens,
                    call.latency_ms,
                    1 if call.cached else 0,
                ),
            )
        self._conn.commit()
        return run_id

    def list_runs(
        self,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
        name: str | None = None,
        owner: str | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Dashboard rows (no payloads) with filtering and pagination.

        Returns ``{"items": [...], "total": n}`` where *total* is the count
        before pagination, so the UI can render a page size / total.
        *name* / *owner* are case-insensitive substrings, *tag* matches an
        exact tag, *status* matches the run status exactly.
        """
        where: list[str] = []
        args: list[Any] = []
        if status:
            where.append("status = ?")
            args.append(status)
        if name:
            where.append("LOWER(name) LIKE ?")
            args.append(f"%{name.lower()}%")
        if owner:
            where.append("LOWER(COALESCE(owner, '')) LIKE ?")
            args.append(f"%{owner.lower()}%")
        if tag:
            where.append(
                "EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = ?)"
            )
            args.append(tag)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        total = self._conn.execute(
            f"SELECT COUNT(*) FROM runs{where_sql}", tuple(args)
        ).fetchone()[0]

        rows = self._conn.execute(
            f"SELECT run_id, name, status, total_ms, owner, checkpoint_id, "
            f"prompt_tokens, completion_tokens, tags, notes, created_at FROM runs"
            f"{where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*args, limit, offset),
        ).fetchall()
        cols = [
            "run_id",
            "name",
            "status",
            "total_ms",
            "owner",
            "checkpoint_id",
            "prompt_tokens",
            "completion_tokens",
            "tags",
            "notes",
            "created_at",
        ]
        items = []
        for row in rows:
            item = dict(zip(cols, row))
            item["tags"] = json.loads(item["tags"])
            items.append(item)
        return {"items": items, "total": total}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Full run payload: metadata, topology, node spans, LLM calls."""
        row = self._conn.execute(
            "SELECT name, status, total_ms, owner, checkpoint_id, prompt_tokens, "
            "completion_tokens, topology, tags, notes, created_at FROM runs "
            "WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        run = {
            "run_id": run_id,
            "name": row[0],
            "status": row[1],
            "total_ms": row[2],
            "owner": row[3],
            "checkpoint_id": row[4],
            "prompt_tokens": row[5],
            "completion_tokens": row[6],
            "tags": json.loads(row[8]),
            "notes": row[9],
            "created_at": row[10],
        }
        topology = json.loads(row[7])
        run["topology"] = topology

        node_rows = self._conn.execute(
            "SELECT node_id, node_type, start_ms, end_ms, status, error, "
            "tool_calls, events "
            "FROM nodes WHERE run_id = ? ORDER BY start_ms",
            (run_id,),
        ).fetchall()
        calls: dict[str, list[dict]] = {}
        for call in self._conn.execute(
            "SELECT node_id, provider, model, messages, response, prompt_tokens, "
            "completion_tokens, latency_ms, cached FROM llm_calls WHERE run_id = ? "
            "ORDER BY id",
            (run_id,),
        ).fetchall():
            payload = {
                "node_id": call[0],
                "provider": call[1],
                "model": call[2],
                "messages": json.loads(call[3]),
                "response": call[4],
                "prompt_tokens": call[5],
                "completion_tokens": call[6],
                "latency_ms": call[7],
                "cached": bool(call[8]),
            }
            calls.setdefault(call[0], []).append(payload)

        nodes = []
        for nid, ntype, start, end, status, error, tool_calls, events in node_rows:
            nodes.append(
                {
                    "node_id": nid,
                    "node_type": ntype,
                    "start_ms": start,
                    "end_ms": end,
                    "duration_ms": None if end is None else round(end - start, 3),
                    "status": status,
                    "error": error,
                    "llm_calls": calls.get(nid, []),
                    "tool_calls": json.loads(tool_calls),
                    "events": json.loads(events),
                }
            )
        run["nodes"] = nodes
        run["llm_calls"] = [call for node in nodes for call in node["llm_calls"]]
        return run

    def update_run(
        self,
        run_id: str,
        *,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> bool:
        """Patch a run's tags / notes. Returns False if the run doesn't exist."""
        fields: list[str] = []
        args: list[Any] = []
        if tags is not None:
            fields.append("tags = ?")
            args.append(json.dumps(list(tags)))
        if notes is not None:
            fields.append("notes = ?")
            args.append(notes)
        if not fields:
            return False
        args.append(run_id)
        cur = self._conn.execute(
            f"UPDATE runs SET {', '.join(fields)} WHERE run_id = ?", tuple(args)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()


class CompositeExporter(TraceExporter):
    """Fan one run out to several exporters (SQLite + remote sinks).

    A failure in one sink is logged and swallowed so the remaining sinks
    (and the workflow that produced the run) keep working.
    """

    def __init__(self, exporters: list[TraceExporter]):
        self.exporters = list(exporters)

    def export(self, run: Run) -> str | None:
        run_id: str | None = None
        for exporter in self.exporters:
            try:
                result = exporter.export(run)
            except Exception:
                logger.exception(
                    "composite exporter %s failed", type(exporter).__name__
                )
                continue
            if run_id is None and result is not None:
                run_id = result
        return run_id

    def close(self) -> None:
        for exporter in self.exporters:
            try:
                exporter.close()
            except Exception:
                logger.exception(
                    "composite exporter %s failed to close", type(exporter).__name__
                )
