"""Application controller that wires configuration, secrets, and storage.

This is the headless-safe core of the desktop app. It owns the database and
exposes lifecycle operations (start/stop/pause/purge) that the UI layer calls.
The UI is optional and imported lazily.
"""

from __future__ import annotations

import logging

from .config import AppConfig, load_config
from .db import Database, Meeting, Repository
from .secrets_store import SecretStore

logger = logging.getLogger("meeting_copilot")


class AppController:
    """Owns long-lived application state independent of any UI."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        secrets: SecretStore | None = None,
        database: Database | None = None,
    ) -> None:
        self.config = config or load_config()
        self.secrets = secrets or SecretStore()
        self._database = database or Database(self.config.database_path())
        self.repository = Repository(self._database)
        self._active_meeting: Meeting | None = None
        self._paused = False

    @property
    def active_meeting(self) -> Meeting | None:
        return self._active_meeting

    @property
    def paused(self) -> bool:
        return self._paused

    def start_meeting(self, title: str | None = None) -> Meeting:
        """Begin a new meeting, becoming the active recording session."""
        if self._active_meeting is not None:
            return self._active_meeting
        self._active_meeting = self.repository.create_meeting(title=title)
        self._paused = False
        logger.info("Started meeting %s", self._active_meeting.id)
        return self._active_meeting

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop_meeting(self) -> None:
        """End the active meeting, if any."""
        if self._active_meeting is None:
            return
        self.repository.end_meeting(self._active_meeting.id)
        logger.info("Stopped meeting %s", self._active_meeting.id)
        self._active_meeting = None
        self._paused = False

    def purge_all(self) -> None:
        """Delete all locally stored data (no cloud copy ever exists)."""
        self.stop_meeting()
        self.repository.purge_all()
        logger.info("Purged all local data")

    def close(self) -> None:
        self._database.close()
