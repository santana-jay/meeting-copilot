"""Pluggable speech-to-text (STT) interfaces and backends."""

from __future__ import annotations

from .base import STTService, TranscriptEvent
from .mock import MockSTT

__all__ = ["STTService", "TranscriptEvent", "MockSTT", "create_stt"]


def create_stt(backend: str = "mock", **kwargs) -> STTService:
    """Create an STT backend by name (``"mock"`` or ``"whisper"``)."""
    if backend == "mock":
        return MockSTT(**kwargs)
    if backend == "whisper":
        from .whisper import WhisperSTT

        return WhisperSTT(**kwargs)
    raise ValueError(f"Unknown STT backend: {backend!r}")
