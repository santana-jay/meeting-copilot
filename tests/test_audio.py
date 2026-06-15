"""Tests for the audio capture abstraction and factory."""

from __future__ import annotations

import pytest

from meeting_copilot.audio import (
    AudioChunk,
    CaptureUnavailable,
    NullAudioCapture,
    create_audio_capture,
)
from meeting_copilot.audio.factory import _platform_backend_name


def test_audio_chunk_properties():
    chunk = AudioChunk(samples=b"\x00\x00" * 1600, sample_rate=16_000, timestamp=0.0)
    assert chunk.num_samples == 1600
    assert chunk.duration == pytest.approx(0.1)


def test_null_capture_replays_chunks():
    chunks = [
        AudioChunk(samples=b"\x00\x00", sample_rate=16_000, timestamp=float(i))
        for i in range(3)
    ]
    capture = NullAudioCapture(chunks=chunks)
    assert capture.is_available()
    with capture:
        assert capture.running
        produced = list(capture.read())
    assert len(produced) == 3
    assert not capture.running


def test_factory_null_backend():
    capture = create_audio_capture("null")
    assert capture.name == "null"


def test_factory_unknown_backend():
    with pytest.raises(ValueError):
        create_audio_capture("does-not-exist")


def test_factory_auto_falls_back_to_null_when_unavailable(monkeypatch):
    # On CI the platform loopback backend's native dep is absent, so without a
    # fallback the factory should raise, and with fallback it returns null.
    import meeting_copilot.audio.factory as factory

    monkeypatch.setattr(factory, "_platform_backend_name", lambda: "linux")
    with pytest.raises(CaptureUnavailable):
        factory.create_audio_capture("auto")
    capture = factory.create_audio_capture("auto", fallback_to_null=True)
    assert capture.name == "null"


def test_platform_backend_name_is_known():
    assert _platform_backend_name() in {"windows", "macos", "linux", "null"}
