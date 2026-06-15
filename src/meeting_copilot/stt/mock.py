"""A deterministic mock STT backend for tests and offline development.

The mock can be seeded with scripted transcript events, or it can map each
incoming :class:`AudioChunk` to a caller-provided phrase. It performs no audio
decoding, which keeps the core test suite free of heavy dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from ..audio.base import AudioChunk
from .base import STTService, TranscriptEvent


class MockSTT(STTService):
    """Emit pre-scripted transcript events or per-chunk phrases."""

    def __init__(
        self,
        events: Sequence[TranscriptEvent] | None = None,
        *,
        phrases: Sequence[str] | None = None,
    ) -> None:
        self._events = list(events or [])
        self._phrases = list(phrases or [])

    @property
    def name(self) -> str:
        return "mock"

    def transcribe(self, chunks: Iterable[AudioChunk]) -> Iterator[TranscriptEvent]:
        if self._events:
            yield from self._events
            return

        for index, chunk in enumerate(chunks):
            text = self._phrases[index] if index < len(self._phrases) else ""
            if not text:
                continue
            yield TranscriptEvent(
                text=text,
                start_time=chunk.timestamp,
                end_time=chunk.timestamp + chunk.duration,
                is_final=True,
            )
