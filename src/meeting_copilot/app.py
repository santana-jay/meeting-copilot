"""Application controller that wires configuration, secrets, and storage.

This is the headless-safe core of the desktop app. It owns the database, the
processing pipeline, and the lifecycle operations (start/pause/stop/purge) that
the UI layer calls. Heavy/optional backends (real audio, Whisper, the Anthropic
SDK) are imported lazily, so the controller stays importable and testable in
headless environments.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from .ai import create_llm
from .ai.base import LLMClient
from .audio import AudioCapture, AudioChunk, create_audio_capture
from .config import AppConfig, load_config
from .db import Database, Meeting, Repository
from .intelligence import NoteExtractor, SuggestionEngine
from .pipeline import MeetingPipeline, PipelineSink
from .retrieval import EmbeddingModel, HashingEmbeddingModel, VectorStore
from .secrets_store import SecretStore
from .stt import STTService, create_stt

logger = logging.getLogger("meeting_copilot")


class AppController:
    """Owns long-lived application state independent of any UI."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        secrets: SecretStore | None = None,
        database: Database | None = None,
        sink: PipelineSink | None = None,
        llm: LLMClient | None = None,
        stt: STTService | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.config = config or load_config()
        self.secrets = secrets or SecretStore()
        self._database = database or Database(self.config.database_path())
        self.repository = Repository(self._database)
        self.sink = sink

        # Components are built lazily so the controller stays cheap to create
        # and so injected test doubles take precedence over config-driven ones.
        self._llm = llm
        self._stt = stt
        self._embedding_model = embedding_model
        self._pipeline: MeetingPipeline | None = None

        self._active_meeting: Meeting | None = None
        self._paused = False

    # -- lazy component builders -----------------------------------------
    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = create_llm(self.config, self.secrets)
        return self._llm

    @property
    def stt(self) -> STTService:
        if self._stt is None:
            self._stt = create_stt(self.config.stt_backend)
        return self._stt

    @property
    def embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = HashingEmbeddingModel()
        return self._embedding_model

    @property
    def pipeline(self) -> MeetingPipeline:
        """The processing pipeline, built once from the current components."""
        if self._pipeline is None:
            vector_store = VectorStore(self.repository, self.embedding_model)
            note_extractor = NoteExtractor(
                self.llm,
                max_tokens=self.config.anthropic_max_tokens,
                temperature=self.config.anthropic_temperature,
            )
            suggestion_engine = SuggestionEngine(
                self.llm,
                min_confidence=self.config.min_grounding_confidence,
                max_tokens=self.config.anthropic_max_tokens,
                temperature=self.config.anthropic_temperature,
            )
            self._pipeline = MeetingPipeline(
                repository=self.repository,
                vector_store=vector_store,
                stt=self.stt,
                note_extractor=note_extractor,
                suggestion_engine=suggestion_engine,
                sink=self.sink,
                window_seconds=float(self.config.transcript_window_seconds),
                retrieval_top_k=self.config.retrieval_top_k,
            )
        return self._pipeline

    @property
    def active_meeting(self) -> Meeting | None:
        return self._active_meeting

    @property
    def paused(self) -> bool:
        return self._paused

    def start_meeting(self, title: str | None = None) -> Meeting:
        """Begin a new meeting, becoming the active recording session."""
        if self._active_meeting is not None:
            return self._active_meeting
        self._active_meeting = self.repository.create_meeting(title=title)
        self._paused = False
        logger.info("Started meeting %s", self._active_meeting.id)
        return self._active_meeting

    def start_live_capture(self, capture: AudioCapture | None = None) -> Meeting:
        """Start a meeting (if needed) and begin live capture + processing.

        Args:
            capture: An audio capture backend. When ``None`` one is created from
                config, falling back to the null backend if loopback capture is
                unavailable so the app degrades gracefully instead of crashing.
        """
        meeting = self.start_meeting()
        if capture is None:
            capture = create_audio_capture(
                self.config.audio_backend,
                sample_rate=self.config.sample_rate,
                fallback_to_null=True,
            )
        logger.info("Live capture backend: %s", capture.name)
        self.pipeline.start(meeting.id, capture)
        return meeting

    def process_audio(self, chunks: Iterable[AudioChunk]) -> None:
        """Synchronously process a finite stream of audio chunks.

        Useful for headless/offline runs and tests; requires an active meeting.
        """
        if self._active_meeting is None:
            raise RuntimeError("No active meeting. Call start_meeting() first.")
        self.pipeline.process_chunks(self._active_meeting.id, chunks)

    def pause(self) -> None:
        self._paused = True
        if self._pipeline is not None:
            self._pipeline.pause()

    def resume(self) -> None:
        self._paused = False
        if self._pipeline is not None:
            self._pipeline.resume()

    def stop_meeting(self) -> None:
        """End the active meeting, if any, stopping live capture first."""
        if self._pipeline is not None:
            self._pipeline.stop()
        if self._active_meeting is None:
            return
        self.repository.end_meeting(self._active_meeting.id)
        logger.info("Stopped meeting %s", self._active_meeting.id)
        self._active_meeting = None
        self._paused = False

    def purge_all(self) -> None:
        """Delete all locally stored data (no cloud copy ever exists)."""
        self.stop_meeting()
        self.repository.purge_all()
        logger.info("Purged all local data")

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._database.close()
