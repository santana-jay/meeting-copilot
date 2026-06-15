"""A no-op audio capture backend for headless/test environments.

It can optionally replay a pre-supplied list of :class:`AudioChunk` objects,
which makes it useful for deterministic pipeline tests without real hardware.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from .base import AudioCapture, AudioChunk


class NullAudioCapture(AudioCapture):
    """Audio capture that yields nothing (or a fixed, pre-seeded sequence)."""

    def __init__(
        self, sample_rate: int = 16_000, chunks: Sequence[AudioChunk] | None = None
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        self._chunks = list(chunks or [])

    @property
    def name(self) -> str:
        return "null"

    def is_available(self) -> bool:
        return True

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read(self) -> Iterator[AudioChunk]:
        for chunk in self._chunks:
            if not self._running:
                break
            yield chunk
