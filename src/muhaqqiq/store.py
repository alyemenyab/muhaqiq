"""SQLite run store.

Two jobs: (1) keep every completed run so a report can be re-fetched by id
instead of re-researched, and (2) cache retrieval results so repeated queries
inside one session do not hammer the search provider.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .schemas import RunResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    question    TEXT NOT NULL,
    verdict     TEXT NOT NULL,
    coverage    REAL NOT NULL,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);

CREATE TABLE IF NOT EXISTS search_cache (
    query      TEXT NOT NULL,
    provider   TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    payload    TEXT NOT NULL,
    PRIMARY KEY (query, provider)
);
"""


class RunStore:
    def __init__(self, path: Path | str = ".muhaqqiq/runs.db") -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: sqlite3.Connection | None = None
        if str(self.path) == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            yield self._memory_conn
            self._memory_conn.commit()
            return
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- runs -------------------------------------------------------------- #
    def save_run(self, result: RunResult) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, created_at, question, verdict, coverage, duration_ms, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    result.meta.run_id,
                    result.meta.created_at,
                    result.brief.question,
                    result.verification.verdict.value,
                    result.verification.citation_coverage,
                    result.meta.duration_ms,
                    result.model_dump_json(),
                ),
            )

    def get_run(self, run_id: str) -> RunResult | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return RunResult.model_validate_json(row["payload"])

    def list_runs(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, created_at, question, verdict, coverage, duration_ms "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- search cache ------------------------------------------------------ #
    def cache_get(self, query: str, provider: str, max_age_s: float = 3600.0) -> list[dict] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT fetched_at, payload FROM search_cache WHERE query = ? AND provider = ?",
                (query, provider),
            ).fetchone()
        if row is None or (time.time() - row["fetched_at"]) > max_age_s:
            return None
        return json.loads(row["payload"])

    def cache_put(self, query: str, provider: str, payload: list[dict]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO search_cache (query, provider, fetched_at, payload) "
                "VALUES (?, ?, ?, ?)",
                (query, provider, time.time(), json.dumps(payload, ensure_ascii=False)),
            )
