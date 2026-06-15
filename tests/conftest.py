"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from meeting_copilot.db import Database, Repository


@pytest.fixture
def db() -> Database:
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def repo(db: Database) -> Repository:
    return Repository(db)
