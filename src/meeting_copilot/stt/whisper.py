"""Local streaming Whisper STT backend (faster-whisper).

The heavy ``faster-whisper`` dependency is imported lazily so this module can
be imported without it installed. Audio chunks are accumulated and decoded
into transcript events. This is the default real STT implementation.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator

from ..audio.base import AudioChunk
from .base import STTService, TranscriptEvent


class WhisperSTT(STTService):
    """Transcribe audio with a local faster-whisper model."""

    def __init__(self, model: str = "base", *, language: str | None = None) -> None:
        self.model_name = model
        self.language = language
        self._model = None

    @property
    def name(self) -> str:
        return f"whisper:{self.model_name}"

    def _ensure_model(self):  # pragma: no cover - requires faster-whisper
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_name)
        return self._model

    def transcribe(  # pragma: no cover - requires faster-whisper + model
        self, chunks: Iterable[AudioChunk]
    ) -> Iterator[TranscriptEvent]:
        model = self._ensure_model()
        for chunk in chunks:
            audio = _pcm16_to_float32(chunk.samples)
            segments, _info = model.transcribe(audio, language=self.language)
            for segment in segments:
                yield TranscriptEvent(
                    text=segment.text.strip(),
                    start_time=chunk.timestamp + float(segment.start),
                    end_time=chunk.timestamp + float(segment.end),
                    is_final=True,
                )


def _pcm16_to_float32(samples: bytes):  # pragma: no cover - exercised with numpy present
    """Convert little-endian 16-bit PCM bytes to a float32 numpy array."""
    import numpy as np

    count = len(samples) // 2
    ints = struct.unpack(f"<{count}h", samples)
    return np.asarray(ints, dtype="float32") / 32768.0
