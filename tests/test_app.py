"""Tests for the app controller and screen-capture exclusion logic."""

from __future__ import annotations

from meeting_copilot.app import AppController
from meeting_copilot.config import AppConfig
from meeting_copilot.db import Database
from meeting_copilot.secrets_store import SecretStore
from meeting_copilot.ui.capture_exclusion import (
    current_platform,
    exclude_window_from_capture,
)


def make_controller() -> AppController:
    return AppController(
        config=AppConfig(),
        secrets=SecretStore(environ={}, use_keyring=False),
        database=Database(":memory:"),
    )


def test_meeting_lifecycle():
    controller = make_controller()
    meeting = controller.start_meeting(title="Sync")
    assert controller.active_meeting is meeting
    # starting again returns the same active meeting
    assert controller.start_meeting() is meeting

    controller.pause()
    assert controller.paused is True
    controller.resume()
    assert controller.paused is False

    controller.stop_meeting()
    assert controller.active_meeting is None
    assert controller.repository.get_meeting(meeting.id).ended_at is not None
    controller.close()


def test_purge_all_stops_meeting():
    controller = make_controller()
    controller.start_meeting()
    controller.purge_all()
    assert controller.active_meeting is None
    assert controller.repository.list_meetings() == []
    controller.close()


def test_capture_exclusion_without_handle():
    result = exclude_window_from_capture(None)
    assert result.excluded is False
    assert result.platform == current_platform()


def test_capture_exclusion_linux_not_guaranteed(monkeypatch):
    import meeting_copilot.ui.capture_exclusion as ce

    monkeypatch.setattr(ce, "current_platform", lambda: "linux")
    result = ce.exclude_window_from_capture(12345)
    assert result.platform == "linux"
    assert result.guaranteed is False
