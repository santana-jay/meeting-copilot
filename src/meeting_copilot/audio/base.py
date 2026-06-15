"""Audio capture interface and shared types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


class AudioCaptureError(Exception):
    """Base error for audio capture problems."""


class CaptureUnavailable(AudioCaptureError):
    """Raised when loopback capture is not available on this system.

    The message should include actionable, degraded-mode guidance so the user
    can enable a monitor source or install a virtual audio device.
    """


@dataclass(frozen=True)
class AudioChunk:
    """A chunk of mono PCM audio.

    Attributes:
        samples: Raw little-endian 16-bit PCM bytes.
        sample_rate: Samples per second.
        timestamp: Seconds since capture start for the first sample.
    """

    samples: bytes
    sample_rate: int
    timestamp: float

    @property
    def num_samples(self) -> int:
        return len(self.samples) // 2  # 16-bit mono

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.num_samples / self.sample_rate


class AudioCapture(ABC):
    """Abstract base class for loopback/system audio capture backends."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name (e.g. ``"wasapi-loopback"``)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if this backend can capture loopback audio."""

    @abstractmethod
    def start(self) -> None:
        """Begin capture. Raise :class:`CaptureUnavailable` if not possible."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capture and release resources."""

    @abstractmethod
    def read(self) -> Iterator[AudioChunk]:
        """Yield captured audio chunks until :meth:`stop` is called."""

    def degraded_mode_help(self) -> str:
        """Return platform guidance for enabling loopback capture."""
        return (
            "Loopback capture is unavailable. Enable a system audio monitor "
            "source or install a virtual loopback device, then restart capture."
        )

    def __enter__(self) -> AudioCapture:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
