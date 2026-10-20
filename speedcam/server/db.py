"""Local SQLite index of projects and job history.

Project folders (project.json, results, ...) stay the source of truth; this database only
indexes them for the project library (status, headline stats, last opened screen) and keeps
job history across restarts. It is rebuilt from the folders if deleted.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    created REAL NOT NULL,
    opened REAL,
    updated REAL,
    last_screen TEXT,
    source_exists INTEGER,
    width INTEGER, height INTEGER, capture_fps REAL, duration_s REAL,
    has_proxy INTEGER, has_results INTEGER, has_render INTEGER,
    calibrated INTEGER,
    n_vehicles INTEGER, n_valid INTEGER, mean_kmh REAL, p85_kmh REAL,
    analyzed REAL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    created REAL, started REAL, finished REAL,
    error TEXT,
    result TEXT
);
CREATE INDEX IF NOT EXISTS jobs_project ON jobs(project_id, created);
"""

PROJECT_COLS = ("id", "name", "source", "created", "opened", "updated", "last_screen", "source_exists",
                "width", "height", "capture_fps", "duration_s", "has_proxy", "has_results", "has_render",
                "calibrated", "n_vehicles", "n_valid", "mean_kmh", "p85_kmh", "analyzed")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _exec(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, args).fetchall()

    # -- projects --------------------------------------------------------------------
    def upsert_project(self, row: dict[str, Any]) -> None:
        """Insert or update the given columns; columns not in `row` keep their stored value."""
        cols = [c for c in PROJECT_COLS if c in row]
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "id")
        self._exec(f"INSERT INTO projects ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
                   f"ON CONFLICT(id) DO UPDATE SET {updates}", tuple(row[c] for c in cols))

    def update_project(self, pid: str, **fields: Any) -> None:
        if fields:
            sets = ", ".join(f"{k}=?" for k in fields)
            self._exec(f"UPDATE projects SET {sets} WHERE id=?", (*fields.values(), pid))

    def projects(self) -> list[dict]:
        rows = self._exec("SELECT * FROM projects ORDER BY COALESCE(opened, updated, created) DESC")
        return [dict(r) for r in rows]

    def project_ids(self) -> set[str]:
        return {r["id"] for r in self._exec("SELECT id FROM projects")}

    def delete_project(self, pid: str) -> None:
        self._exec("DELETE FROM projects WHERE id=?", (pid,))
        self._exec("DELETE FROM jobs WHERE project_id=?", (pid,))

    # -- jobs --------------------------------------------------------------------------
    def record_job(self, job: dict) -> None:
        self._exec(
            "INSERT INTO jobs (id, project_id, kind, status, created, started, finished, error, result) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
            "started=excluded.started, finished=excluded.finished, error=excluded.error, result=excluded.result",
            (job["id"], job["project_id"], job["kind"], job["status"], job.get("created"), job.get("started"),
             job.get("finished"), job.get("error"), json.dumps(_slim(job.get("result")))),
        )

    def jobs_for(self, pid: str, limit: int = 50) -> list[dict]:
        rows = self._exec("SELECT * FROM jobs WHERE project_id=? ORDER BY created DESC LIMIT ?", (pid, limit))
        out = []
        for r in rows:
            d = dict(r)
            d["result"] = json.loads(d["result"]) if d["result"] else None
            out.append(d)
        return out

    def all_jobs(self, limit: int = 500) -> list[dict]:
        """Job history across projects, newest first."""
        rows = self._exec("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d["result"] = json.loads(d["result"]) if d["result"] else None
            out.append(d)
        return out

    def job(self, jid: str) -> Optional[dict]:
        rows = self._exec("SELECT * FROM jobs WHERE id=?", (jid,))
        return dict(rows[0]) if rows else None

    def mark_interrupted(self) -> int:
        """Jobs left queued/running by a previous server process can never finish."""
        rows = self._exec("SELECT id FROM jobs WHERE status IN ('queued', 'running')")
        self._exec("UPDATE jobs SET status='interrupted', finished=? WHERE status IN ('queued', 'running')",
                   (time.time(),))
        return len(rows)

    def close(self) -> None:
        with self._lock:
            self.conn.close()


def _slim(result: Optional[Any]) -> Optional[Any]:
    """Drop bulky fields (e.g. preview images) before storing a job result."""
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if k not in ("jpeg", "vehicles")}
    return result
