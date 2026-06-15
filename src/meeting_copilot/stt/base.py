"""STT interface and transcript event type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ..audio.base import AudioChunk


@dataclass(frozen=True)
class TranscriptEvent:
    """An incremental transcription result.

    Attributes:
        text: Recognized text for this segment.
        start_time: Segment start (seconds from capture start).
        end_time: Segment end (seconds from capture start).
        is_final: ``True`` for stable results, ``False`` for interim hypotheses.
        speaker: Best-effort speaker label, or ``None`` when unknown.
    """

    text: str
    start_time: float
    end_time: float
    is_final: bool = True
    speaker: str | None = None


class STTService(ABC):
    """Streaming speech-to-text interface.

    Implementations consume an iterable of :class:`AudioChunk` and yield
    :class:`TranscriptEvent` objects. Backends may emit interim
    (``is_final=False``) events followed by a final event for the same span.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def transcribe(self, chunks: Iterable[AudioChunk]) -> Iterator[TranscriptEvent]:
        """Yield transcript events for the given audio chunks."""

    def close(self) -> None:  # noqa: B027 - optional override, intentional no-op
        """Release any resources held by the backend (optional)."""
        return None
