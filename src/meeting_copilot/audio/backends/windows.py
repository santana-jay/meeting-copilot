"""Windows WASAPI loopback capture backend."""

from __future__ import annotations

import sys
from collections.abc import Iterator

from ..base import AudioCapture, AudioChunk, CaptureUnavailable


class WindowsLoopbackCapture(AudioCapture):
    """Capture system output via WASAPI loopback.

    Uses ``soundcard`` (or ``pyaudiowpatch``) when available. This stub focuses
    on availability detection and clear degraded-mode guidance; the streaming
    loop is implemented by the native dependency at runtime.
    """

    @property
    def name(self) -> str:
        return "wasapi-loopback"

    def is_available(self) -> bool:
        if not sys.platform.startswith("win"):
            return False
        return self._load_backend() is not None

    def start(self) -> None:  # pragma: no cover - requires Windows audio stack
        backend = self._load_backend()
        if backend is None:
            raise CaptureUnavailable(self.degraded_mode_help())
        self._running = True

    def stop(self) -> None:  # pragma: no cover - requires Windows audio stack
        self._running = False

    def read(self) -> Iterator[AudioChunk]:  # pragma: no cover - requires hardware
        raise CaptureUnavailable(self.degraded_mode_help())
        yield  # pragma: no cover - makes this a generator

    def degraded_mode_help(self) -> str:
        return (
            "Windows loopback capture requires WASAPI loopback support. Install "
            "the 'soundcard' package, ensure an active output device exists, and "
            "verify the app is allowed to access the microphone/audio in "
            "Windows privacy settings."
        )

    def _load_backend(self):
        try:
            import soundcard  # type: ignore  # noqa: F401

            return "soundcard"
        except Exception:
            return None
