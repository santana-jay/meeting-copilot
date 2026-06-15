"""End-to-end pipeline test: audio -> STT -> store -> notes + grounded suggestion.

Uses only mock/null backends so it runs offline and deterministically, while
exercising the real config, storage, retrieval, validation, and grounding code.
"""

from __future__ import annotations

import json

from meeting_copilot.ai import MockLLMClient
from meeting_copilot.audio import AudioChunk, NullAudioCapture
from meeting_copilot.db import Database, Repository, TranscriptSegment
from meeting_copilot.intelligence import NoteExtractor, SuggestionEngine
from meeting_copilot.retrieval import HashingEmbeddingModel, VectorStore
from meeting_copilot.stt import MockSTT


def test_full_pipeline_offline():
    repo = Repository(Database(":memory:"))
    meeting = repo.create_meeting(title="Planning")

    # 1. Capture (null backend replaying two chunks) -> STT (mock phrases).
    chunks = [
        AudioChunk(samples=b"\x00\x00" * 16_000, sample_rate=16_000, timestamp=float(i))
        for i in range(2)
    ]
    capture = NullAudioCapture(chunks=chunks)
    stt = MockSTT(
        phrases=[
            "We decided to migrate the database to Postgres next sprint.",
            "Alice will own the migration and report back Friday.",
        ]
    )

    store = VectorStore(repo, HashingEmbeddingModel(dimension=256))
    with capture:
        for event in stt.transcribe(capture.read()):
            seg = repo.add_segment(
                TranscriptSegment(
                    meeting_id=meeting.id,
                    start_time=event.start_time,
                    end_time=event.end_time,
                    text=event.text,
                )
            )
            store.index_segment(seg)

    segments = repo.list_segments(meeting.id)
    assert len(segments) == 2

    # 2. Note extraction (mock LLM returns structured JSON).
    notes_payload = json.dumps(
        {
            "notes": [
                {"kind": "decision", "content": "Migrate database to Postgres"},
                {"kind": "action_item", "content": "Alice owns the migration"},
            ],
            "unknowns": [],
        }
    )
    extractor = NoteExtractor(MockLLMClient([notes_payload]))
    notes = extractor.extract(meeting.id, segments)
    for note in notes:
        repo.add_note(note)
    assert len(repo.list_notes(meeting.id)) == 2

    # 3. Retrieval over stored segments for the suggestion context.
    retrieved = store.search("database migration owner", meeting_id=meeting.id, top_k=2)
    assert retrieved

    # 4. Grounded suggestion with a real quote from the transcript.
    suggestion_payload = json.dumps(
        {
            "abstain": False,
            "suggestion": "Confirm Friday as the migration status checkpoint.",
            "confidence": 0.82,
            "citations": [{"quote": "report back Friday"}],
        }
    )
    engine = SuggestionEngine(MockLLMClient([suggestion_payload]), min_confidence=0.55)
    result = engine.generate(meeting.id, segments, retrieved)
    assert result.abstained is False
    repo.add_suggestion(result.suggestion)

    stored = repo.list_suggestions(meeting.id)
    assert len(stored) == 1
    assert stored[0].citations[0].quote == "report back Friday"
    assert stored[0].citations[0].segment_id is not None
