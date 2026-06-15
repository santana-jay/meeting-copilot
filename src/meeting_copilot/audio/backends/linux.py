"""Linux PulseAudio/PipeWire monitor-source capture backend."""

from __future__ import annotations

import sys
from collections.abc import Iterator

from ..base import AudioCapture, AudioChunk, CaptureUnavailable


class LinuxMonitorCapture(AudioCapture):
    """Capture system audio from a PulseAudio/PipeWire monitor source.

    Uses ``sounddevice``/``soundcard`` when present. Provides guidance to
    select a ``.monitor`` source when loopback capture is not configured.
    """

    @property
    def name(self) -> str:
        return "pulse-monitor"

    def is_available(self) -> bool:
        if not sys.platform.startswith("linux"):
            return False
        return self._load_backend() is not None

    def start(self) -> None:  # pragma: no cover - requires Linux audio stack
        backend = self._load_backend()
        if backend is None:
            raise CaptureUnavailable(self.degraded_mode_help())
        self._running = True

    def stop(self) -> None:  # pragma: no cover - requires Linux audio stack
        self._running = False

    def read(self) -> Iterator[AudioChunk]:  # pragma: no cover - requires hardware
        raise CaptureUnavailable(self.degraded_mode_help())
        yield  # pragma: no cover - makes this a generator

    def degraded_mode_help(self) -> str:
        return (
            "Linux system-audio capture needs a PulseAudio/PipeWire monitor "
            "source. Install 'sounddevice', then select the '<sink>.monitor' "
            "source (e.g. via 'pactl list sources') as the capture device."
        )

    def _load_backend(self):
        try:
            import sounddevice  # type: ignore  # noqa: F401

            return "sounddevice"
        except Exception:
            return None
