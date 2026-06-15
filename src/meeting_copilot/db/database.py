"""Database connection and schema management."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

SCHEMA_VERSION = 1


class Database:
    """A thin wrapper around a SQLite connection.

    The wrapper applies the bundled schema on first use and exposes the raw
    connection for the repository layer. It can run fully in-memory which is
    convenient for tests.

    The connection is opened with ``check_same_thread=False`` so the background
    capture/processing thread can reuse it. This is safe under the app's
    single-writer model: the live pipeline thread is the only writer while
    running and is always stopped before any main-thread writes (stop/purge),
    and sqlite3 holds the GIL for the duration of each statement.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` lets the background processing thread reuse
        # this connection. Access is serialized by the app (the live capture
        # thread is the sole writer and is stopped before any main-thread writes),
        # and sqlite3 holds the GIL for the duration of each statement.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def _initialize(self) -> None:
        script = _load_schema_sql()
        self._conn.executescript(script)
        self._set_version_if_unset(SCHEMA_VERSION)
        self._conn.commit()

    def _set_version_if_unset(self, version: int) -> None:
        cur = self._conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        if row is None:
            self._conn.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))

    def schema_version(self) -> int:
        cur = self._conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _load_schema_sql() -> str:
    """Load the bundled schema SQL via importlib.resources."""
    try:
        return resources.files("meeting_copilot.db").joinpath("schema.sql").read_text(
            encoding="utf-8"
        )
    except (FileNotFoundError, ModuleNotFoundError):  # pragma: no cover - fallback
        return (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
