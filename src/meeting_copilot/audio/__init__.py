"""Cross-platform audio capture abstraction.

The :class:`AudioCapture` interface provides a uniform, backend-agnostic way to
stream system/loopback audio as PCM frames. Concrete backends are selected at
runtime by :func:`create_audio_capture`. Real OS backends import their heavy
dependencies lazily so importing this package never requires audio libraries.
"""

from __future__ import annotations

from .base import AudioCapture, AudioCaptureError, AudioChunk, CaptureUnavailable
from .factory import available_backends, create_audio_capture
from .null import NullAudioCapture

__all__ = [
    "AudioCapture",
    "AudioChunk",
    "AudioCaptureError",
    "CaptureUnavailable",
    "NullAudioCapture",
    "create_audio_capture",
    "available_backends",
]
