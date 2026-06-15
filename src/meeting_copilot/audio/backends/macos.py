"""macOS ScreenCaptureKit audio capture backend."""

from __future__ import annotations

import sys
from collections.abc import Iterator

from ..base import AudioCapture, AudioChunk, CaptureUnavailable


class MacScreenCaptureKitCapture(AudioCapture):
    """Capture system audio via ScreenCaptureKit (macOS 13+).

    Falls back to guidance recommending a virtual audio device (e.g. an
    aggregate/loopback device) on older macOS where ScreenCaptureKit audio is
    unavailable.
    """

    @property
    def name(self) -> str:
        return "screencapturekit"

    def is_available(self) -> bool:
        if sys.platform != "darwin":
            return False
        return self._load_backend() is not None

    def start(self) -> None:  # pragma: no cover - requires macOS audio stack
        backend = self._load_backend()
        if backend is None:
            raise CaptureUnavailable(self.degraded_mode_help())
        self._running = True

    def stop(self) -> None:  # pragma: no cover - requires macOS audio stack
        self._running = False

    def read(self) -> Iterator[AudioChunk]:  # pragma: no cover - requires hardware
        raise CaptureUnavailable(self.degraded_mode_help())
        yield  # pragma: no cover - makes this a generator

    def degraded_mode_help(self) -> str:
        return (
            "macOS system-audio capture needs ScreenCaptureKit (macOS 13+) and "
            "Screen Recording permission, or a virtual loopback device such as "
            "an aggregate device. Grant permission in System Settings > Privacy "
            "& Security > Screen Recording, then restart capture."
        )

    def _load_backend(self):
        try:
            import ScreenCaptureKit  # type: ignore  # noqa: F401

            return "screencapturekit"
        except Exception:
            return None
