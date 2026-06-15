"""Runtime selection of an audio capture backend."""

from __future__ import annotations

import sys

from .backends import (
    LinuxMonitorCapture,
    MacScreenCaptureKitCapture,
    WindowsLoopbackCapture,
)
from .base import AudioCapture, CaptureUnavailable
from .null import NullAudioCapture

_BACKENDS = {
    "windows": WindowsLoopbackCapture,
    "macos": MacScreenCaptureKitCapture,
    "linux": LinuxMonitorCapture,
    "null": NullAudioCapture,
}


def _platform_backend_name() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return "null"


def available_backends() -> list[str]:
    """Return the names of backends that report availability on this system."""
    names: list[str] = []
    for name, cls in _BACKENDS.items():
        try:
            if cls().is_available():
                names.append(name)
        except Exception:
            continue
    return names


def create_audio_capture(
    backend: str = "auto",
    *,
    sample_rate: int = 16_000,
    fallback_to_null: bool = False,
) -> AudioCapture:
    """Create an audio capture backend.

    Args:
        backend: ``"auto"`` selects the platform default; otherwise one of
            ``"windows"``, ``"macos"``, ``"linux"``, ``"null"``.
        sample_rate: Desired capture sample rate.
        fallback_to_null: When ``True`` and the selected backend is
            unavailable, return a :class:`NullAudioCapture` instead of raising.

    Raises:
        CaptureUnavailable: If the requested backend is unavailable and
            ``fallback_to_null`` is ``False``.
        ValueError: If ``backend`` is not a known name.
    """
    name = _platform_backend_name() if backend == "auto" else backend
    if name not in _BACKENDS:
        raise ValueError(f"Unknown audio backend: {backend!r}")

    capture = _BACKENDS[name](sample_rate=sample_rate)
    if name != "null" and not capture.is_available():
        if fallback_to_null:
            return NullAudioCapture(sample_rate=sample_rate)
        raise CaptureUnavailable(capture.degraded_mode_help())
    return capture
