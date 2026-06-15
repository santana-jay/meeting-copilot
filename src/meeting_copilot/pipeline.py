"""Live processing pipeline that turns captured audio into grounded output.

This module is the orchestration layer that was previously missing: it wires
the independently tested components together into a working application.

Flow for every meeting:

    audio capture -> STT -> persist segment -> index embedding
                 -> (per final segment) extract structured notes
                 -> retrieve relevant earlier snippets
                 -> generate a grounded, citation-checked suggestion
                 -> emit incremental updates to a :class:`PipelineSink`

The pipeline is import-safe and fully testable offline: with the null audio
backend, the mock STT, and the mock LLM client it exercises the real storage,
retrieval, validation, and grounding code without any heavy dependencies or
network I/O. A background-thread runner (:meth:`MeetingPipeline.start`) drives
the same logic for live capture in the desktop app.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .audio.base import AudioCapture
from .db.models import Note, Suggestion, TranscriptSegment
from .db.repository import Repository
from .intelligence.notes import NoteExtractor
from .intelligence.suggestions import GeneratedSuggestion, SuggestionEngine
from .retrieval.vector_store import VectorStore
from .stt.base import STTService, TranscriptEvent

logger = logging.getLogger("meeting_copilot.pipeline")


@runtime_checkable
class PipelineSink(Protocol):
    """Receiver of incremental pipeline updates (e.g. the overlay)."""

    def on_status(self, text: str) -> None:
        ...

    def on_transcript(self, segment: TranscriptSegment) -> None:
        ...

    def on_notes(self, notes: list[Note]) -> None:
        ...

    def on_suggestion(self, suggestion: Suggestion) -> None:
        ...


@dataclass
class CallbackSink:
    """A :class:`PipelineSink` built from optional callbacks.

    Any callback left as ``None`` is ignored, which makes this convenient for
    wiring just the parts of the UI that exist.
    """

    status: Callable[[str], None] | None = None
    transcript: Callable[[TranscriptSegment], None] | None = None
    notes: Callable[[list[Note]], None] | None = None
    suggestion: Callable[[Suggestion], None] | None = None

    def on_status(self, text: str) -> None:
        if self.status is not None:
            self.status(text)

    def on_transcript(self, segment: TranscriptSegment) -> None:
        if self.transcript is not None:
            self.transcript(segment)

    def on_notes(self, notes: list[Note]) -> None:
        if self.notes is not None:
            self.notes(notes)

    def on_suggestion(self, suggestion: Suggestion) -> None:
        if self.suggestion is not None:
            self.suggestion(suggestion)


@dataclass
class CollectingSink:
    """A :class:`PipelineSink` that records everything (handy for tests)."""

    statuses: list[str] = field(default_factory=list)
    transcripts: list[TranscriptSegment] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    suggestions: list[Suggestion] = field(default_factory=list)

    def on_status(self, text: str) -> None:
        self.statuses.append(text)

    def on_transcript(self, segment: TranscriptSegment) -> None:
        self.transcripts.append(segment)

    def on_notes(self, notes: list[Note]) -> None:
        self.notes.extend(notes)

    def on_suggestion(self, suggestion: Suggestion) -> None:
        self.suggestions.append(suggestion)


@dataclass(frozen=True)
class IntelligenceResult:
    """Outcome of one intelligence pass over the recent transcript window."""

    notes: list[Note]
    suggestion: GeneratedSuggestion


class MeetingPipeline:
    """Orchestrate capture, transcription, storage, and grounded intelligence.

    The pipeline does not own a meeting; callers pass a ``meeting_id`` so the
    same instance can be reused across meetings. It is safe to construct with
    mock/null backends for offline use.
    """

    def __init__(
        self,
        *,
        repository: Repository,
        vector_store: VectorStore,
        stt: STTService,
        note_extractor: NoteExtractor,
        suggestion_engine: SuggestionEngine,
        sink: PipelineSink | None = None,
        window_seconds: float = 120.0,
        retrieval_top_k: int = 5,
    ) -> None:
        self._repo = repository
        self._store = vector_store
        self._stt = stt
        self._notes = note_extractor
        self._suggestions = suggestion_engine
        self._sink = sink
        self._window_seconds = window_seconds
        self._retrieval_top_k = retrieval_top_k

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._capture: AudioCapture | None = None

    # -- single-step processing ------------------------------------------
    def ingest_event(self, meeting_id: int, event: TranscriptEvent) -> TranscriptSegment:
        """Persist and index one transcript event, returning the stored segment."""
        segment = self._repo.add_segment(
            TranscriptSegment(
                meeting_id=meeting_id,
                start_time=event.start_time,
                end_time=event.end_time,
                speaker=event.speaker,
                text=event.text,
                is_final=event.is_final,
            )
        )
        if event.is_final and segment.id is not None:
            try:
                self._store.index_segment(segment)
            except Exception:  # indexing must never break capture
                logger.exception("Failed to index segment %s", segment.id)
        self._emit("on_transcript", segment)
        return segment

    def run_intelligence(self, meeting_id: int) -> IntelligenceResult:
        """Extract notes and generate one grounded suggestion for the window.

        Uses the recent transcript window for note extraction and as the
        suggestion's live context, and retrieves earlier snippets (excluding the
        live window) for additional grounding. New notes and a non-abstained
        suggestion are persisted and emitted to the sink.
        """
        recent = self._repo.recent_segments(meeting_id, self._window_seconds)
        if not recent:
            return IntelligenceResult(notes=[], suggestion=GeneratedSuggestion(abstained=True))

        new_notes = self._extract_and_store_notes(meeting_id, recent)

        recent_ids = {seg.id for seg in recent if seg.id is not None}
        query = " ".join(seg.text for seg in recent)
        retrieved = self._store.search(
            query,
            meeting_id=meeting_id,
            top_k=self._retrieval_top_k,
            exclude_segment_ids=recent_ids,
        )

        generated = self._suggestions.generate(meeting_id, recent, retrieved)
        if not generated.abstained and generated.suggestion is not None:
            stored = self._repo.add_suggestion(generated.suggestion)
            self._emit("on_suggestion", stored)
        elif generated.reason:
            logger.debug("Suggestion abstained: %s", generated.reason)

        return IntelligenceResult(notes=new_notes, suggestion=generated)

    def _extract_and_store_notes(
        self, meeting_id: int, recent: list[TranscriptSegment]
    ) -> list[Note]:
        existing = {(n.kind, n.content) for n in self._repo.list_notes(meeting_id)}
        extracted = self._notes.extract(meeting_id, recent)
        new_notes: list[Note] = []
        for note in extracted:
            key = (note.kind, note.content)
            if key in existing:
                continue
            existing.add(key)
            new_notes.append(self._repo.add_note(note))
        if new_notes:
            self._emit("on_notes", new_notes)
        return new_notes

    def process_chunks(self, meeting_id: int, chunks: Iterable) -> None:
        """Run the full pipeline synchronously over an iterable of audio chunks.

        This is the deterministic path used by tests and by the threaded runner.
        After each final transcript event a fresh intelligence pass runs so notes
        and suggestions stay current with the conversation.
        """
        for event in self._stt.transcribe(chunks):
            self.ingest_event(meeting_id, event)
            if event.is_final:
                self.run_intelligence(meeting_id)

    # -- live threaded runner --------------------------------------------
    def start(self, meeting_id: int, capture: AudioCapture) -> None:
        """Begin live capture+processing on a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._capture = capture
        self._stop_event.clear()
        self._paused.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(meeting_id,),
            name="meeting-pipeline",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self, meeting_id: int) -> None:  # pragma: no cover - thread/live
        self._emit("on_status", "Listening…")
        try:
            with self._capture as capture:  # type: ignore[union-attr]
                for event in self._stt.transcribe(self._gated_chunks(capture)):
                    if self._stop_event.is_set():
                        break
                    self.ingest_event(meeting_id, event)
                    if event.is_final:
                        self.run_intelligence(meeting_id)
        except Exception as exc:
            logger.exception("Pipeline loop stopped on error")
            self._emit("on_status", f"Capture error: {exc}")
        finally:
            self._emit("on_status", "Stopped")

    def _gated_chunks(self, capture: AudioCapture):  # pragma: no cover - thread/live
        for chunk in capture.read():
            if self._stop_event.is_set():
                break
            if self._paused.is_set():
                continue
            yield chunk

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> None:
        """Signal the live loop to stop and wait briefly for it to finish."""
        self._stop_event.set()
        capture = self._capture
        if capture is not None:
            try:
                capture.stop()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Error stopping capture")
        thread = self._thread
        if thread is not None and thread.is_alive():  # pragma: no cover - thread/live
            thread.join(timeout=2.0)
        self._thread = None
        self._capture = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- helpers ----------------------------------------------------------
    def _emit(self, method: str, payload) -> None:
        if self._sink is None:
            return
        try:
            getattr(self._sink, method)(payload)
        except Exception:  # a faulty sink must never break processing
            logger.exception("Pipeline sink %s failed", method)
