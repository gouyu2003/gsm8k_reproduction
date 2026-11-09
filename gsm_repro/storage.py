from __future__ import annotations
import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from .common import canonical, utcnow

class Store:
    def __init__(self, directory, *, create=False):
        self.path = Path(directory) / "results.sqlite3"
        if not create and not self.path.is_file():
            raise ValueError("Run database is missing; refusing to reconstruct it")
        if create:
            with self.connection() as db:
                db.executescript("""
                CREATE TABLE tasks (
                    task_key TEXT PRIMARY KEY, qid TEXT NOT NULL, method TEXT NOT NULL,
                    repeat INTEGER NOT NULL, status TEXT NOT NULL, result_json TEXT, error TEXT);
                CREATE TABLE calls (
                    request_call_id TEXT PRIMARY KEY, task_key TEXT NOT NULL, logical_key TEXT NOT NULL,
                    agent INTEGER NOT NULL, round INTEGER NOT NULL, attempt INTEGER NOT NULL,
                    status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT, duration_ms INTEGER,
                    request_json TEXT NOT NULL, response_json TEXT, error_json TEXT,
                    UNIQUE(logical_key,attempt), FOREIGN KEY(task_key) REFERENCES tasks(task_key));
                """)

    @contextmanager
    def connection(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            yield db

    def initialize(self, records, methods, repeats):
        with self.connection() as db:
            db.executemany("INSERT INTO tasks VALUES (?,?,?,?,?,NULL,NULL)",
                [(f"{method}:{repeat}:{row['qid']}", row["qid"], method, repeat, "pending")
                 for repeat in range(repeats) for row in records for method in methods])

    def tasks(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute("SELECT * FROM tasks ORDER BY rowid")]

    def calls(self, task_key=None):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM calls" + (" WHERE task_key=?" if task_key else "") +
                              " ORDER BY rowid", (task_key,) if task_key else ()).fetchall()
        return [{**dict(r), "request": json.loads(r["request_json"]),
                 "response": json.loads(r["response_json"]) if r["response_json"] else None,
                 "error": json.loads(r["error_json"]) if r["error_json"] else None} for r in rows]

    def status(self, task_key, status, *, result=None, error=None):
        with self.connection() as db:
            current = db.execute("SELECT status FROM tasks WHERE task_key=?", (task_key,)).fetchone()
            if current is None:
                raise ValueError("Unknown task")
            if current[0] in ("completed", "failed", "interrupted") and status != current[0]:
                raise ValueError("Terminal task cannot be overwritten")
            db.execute("UPDATE tasks SET status=?,result_json=?,error=? WHERE task_key=?",
                       (status, canonical(result) if result is not None else None, error, task_key))

    def begin_call(self, request_id, task_key, logical_key, agent, round_, attempt, request):
        with self.connection() as db:
            db.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,'started',?,NULL,NULL,?,NULL,NULL)",
                       (request_id,task_key,logical_key,agent,round_,attempt,utcnow(),canonical(request)))

    def end_call(self, request_id, status, duration_ms, *, response=None, error=None):
        with self.connection() as db:
            changed = db.execute("""UPDATE calls SET status=?,finished_at=?,duration_ms=?,response_json=?,error_json=?
                WHERE request_call_id=? AND status='started'""",
                (status,utcnow(),duration_ms,canonical(response) if response is not None else None,
                 canonical(error) if error is not None else None,request_id)).rowcount
            if changed != 1:
                raise ValueError("Call already settled or missing")

    def mark_unknown_calls(self):
        with self.connection() as db:
            db.execute("UPDATE calls SET status='unknown',finished_at=? WHERE status='started'", (utcnow(),))
