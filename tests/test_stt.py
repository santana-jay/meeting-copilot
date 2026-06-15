"""Tests for the STT interface and mock backend."""

from __future__ import annotations

from meeting_copilot.audio.base import AudioChunk
from meeting_copilot.stt import MockSTT, TranscriptEvent, create_stt


def _chunks(n: int) -> list[AudioChunk]:
    return [
        AudioChunk(samples=b"\x00\x00" * 16_000, sample_rate=16_000, timestamp=float(i))
        for i in range(n)
    ]


def test_mock_scripted_events():
    events = [TranscriptEvent(text="hello", start_time=0, end_time=1)]
    stt = MockSTT(events=events)
    assert list(stt.transcribe(_chunks(1))) == events


def test_mock_phrases_per_chunk():
    stt = MockSTT(phrases=["one", "", "three"])
    out = list(stt.transcribe(_chunks(3)))
    assert [e.text for e in out] == ["one", "three"]  # empty phrase skipped
    assert out[0].start_time == 0.0
    assert out[0].end_time == 1.0


def test_create_stt_mock():
    assert create_stt("mock").name == "mock"


def test_create_stt_unknown():
    try:
        create_stt("nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
