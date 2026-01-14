"""Persistent run store for ConvoBench.

SQLite-first implementation used by the API and CLI.

Stores:
- runs
- traces
- evaluations
- metrics
- artifacts (file paths)

Design goals:
- dependency-light (sqlite3)
- append-only artifacts
- reproducible indexing via run_id and workflow_id
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_DB_PATH = "results/convobench.db"


@dataclass
class StoredRun:
    run_id: str
    created_at: str
    status: str
    config_json: dict[str, Any]
    manifest_json: Optional[dict[str, Any]]


class RunStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL,
                manifest_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                workflow_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                trace_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                evaluator_model TEXT NOT NULL,
                evaluation_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """
        )
        self._conn.commit()

    def upsert_run(
        self,
        run_id: str,
        created_at: str,
        status: str,
        config_json: dict[str, Any],
        manifest_json: Optional[dict[str, Any]] = None,
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO runs(run_id, created_at, status, config_json, manifest_json)
            VALUES(?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                status=excluded.status,
                config_json=excluded.config_json,
                manifest_json=excluded.manifest_json
            """,
            (
                run_id,
                created_at,
                status,
                json.dumps(config_json, default=str),
                json.dumps(manifest_json, default=str) if manifest_json is not None else None,
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> Optional[StoredRun]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        return StoredRun(
            run_id=row["run_id"],
            created_at=row["created_at"],
            status=row["status"],
            config_json=json.loads(row["config_json"]),
            manifest_json=json.loads(row["manifest_json"]) if row["manifest_json"] else None,
        )

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT run_id, created_at, status FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    def add_trace(self, run_id: str, workflow_id: str, scenario_id: str, trace_json: dict[str, Any]) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO traces(workflow_id, run_id, scenario_id, trace_json)
            VALUES(?,?,?,?)
            """,
            (workflow_id, run_id, scenario_id, json.dumps(trace_json, default=str)),
        )
        self._conn.commit()

    def list_traces(self, run_id: str) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT trace_json FROM traces WHERE run_id=?", (run_id,))
        return [json.loads(r[0]) for r in cur.fetchall()]

    def add_evaluation(
        self,
        run_id: str,
        workflow_id: str,
        scenario_id: str,
        evaluator_model: str,
        evaluation_json: dict[str, Any],
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO evaluations(run_id, workflow_id, scenario_id, evaluator_model, evaluation_json)
            VALUES(?,?,?,?,?)
            """,
            (run_id, workflow_id, scenario_id, evaluator_model, json.dumps(evaluation_json, default=str)),
        )
        self._conn.commit()

    def list_evaluations(self, run_id: str) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT evaluation_json FROM evaluations WHERE run_id=?", (run_id,))
        return [json.loads(r[0]) for r in cur.fetchall()]

    def add_metrics(self, run_id: str, workflow_id: str, scenario_id: str, metrics_json: dict[str, Any]) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO metrics(run_id, workflow_id, scenario_id, metrics_json)
            VALUES(?,?,?,?)
            """,
            (run_id, workflow_id, scenario_id, json.dumps(metrics_json, default=str)),
        )
        self._conn.commit()

    def list_metrics(self, run_id: str) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT metrics_json FROM metrics WHERE run_id=?", (run_id,))
        return [json.loads(r[0]) for r in cur.fetchall()]

    def add_artifact(self, run_id: str, kind: str, path: str, metadata: Optional[dict[str, Any]] = None) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO artifacts(run_id, kind, path, metadata_json)
            VALUES(?,?,?,?)
            """,
            (run_id, kind, path, json.dumps(metadata, default=str) if metadata else None),
        )
        self._conn.commit()
