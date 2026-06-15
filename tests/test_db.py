"""Tests for the SQLite schema, repository, and vector packing."""

from __future__ import annotations

import pytest

from meeting_copilot.db import (
    Citation,
    Database,
    Embedding,
    Note,
    NoteKind,
    Repository,
    Suggestion,
    TranscriptSegment,
)
from meeting_copilot.db.database import SCHEMA_VERSION
from meeting_copilot.db.repository import pack_vector, unpack_vector


def test_schema_version(db: Database):
    assert db.schema_version() == SCHEMA_VERSION


def test_meeting_lifecycle(repo: Repository):
    meeting = repo.create_meeting(title="Standup")
    assert meeting.id is not None
    assert repo.get_meeting(meeting.id).title == "Standup"

    repo.end_meeting(meeting.id, ended_at=123.0)
    assert repo.get_meeting(meeting.id).ended_at == 123.0
    assert len(repo.list_meetings()) == 1


def test_segments_and_recent_window(repo: Repository):
    meeting = repo.create_meeting()
    for start in (0.0, 50.0, 200.0):
        repo.add_segment(
            TranscriptSegment(
                meeting_id=meeting.id,
                start_time=start,
                end_time=start + 5,
                text=f"seg at {start}",
            )
        )
    assert len(repo.list_segments(meeting.id)) == 3
    recent = repo.recent_segments(meeting.id, window_seconds=120)
    # latest end_time is 205; threshold 85 -> only the segment ending at 205.
    assert [s.start_time for s in recent] == [200.0]


def test_notes(repo: Repository):
    meeting = repo.create_meeting()
    repo.add_note(Note(meeting_id=meeting.id, kind=NoteKind.ACTION_ITEM, content="Email Bob"))
    repo.add_note(Note(meeting_id=meeting.id, kind=NoteKind.TOPIC, content="Budget"))
    assert len(repo.list_notes(meeting.id)) == 2
    actions = repo.list_notes(meeting.id, kind=NoteKind.ACTION_ITEM)
    assert len(actions) == 1
    assert actions[0].content == "Email Bob"


def test_suggestions_with_citations(repo: Repository):
    meeting = repo.create_meeting()
    seg = repo.add_segment(
        TranscriptSegment(meeting_id=meeting.id, start_time=1, end_time=2, text="hello")
    )
    suggestion = Suggestion(
        meeting_id=meeting.id,
        content="Follow up on hello",
        confidence=0.8,
        citations=[Citation(segment_id=seg.id, start_time=1, end_time=2, quote="hello")],
    )
    repo.add_suggestion(suggestion)

    stored = repo.list_suggestions(meeting.id)
    assert len(stored) == 1
    assert stored[0].confidence == 0.8
    assert len(stored[0].citations) == 1
    assert stored[0].citations[0].quote == "hello"


def test_embeddings_roundtrip_and_upsert(repo: Repository):
    meeting = repo.create_meeting()
    seg = repo.add_segment(
        TranscriptSegment(meeting_id=meeting.id, start_time=0, end_time=1, text="x")
    )
    repo.upsert_embedding(Embedding(segment_id=seg.id, dim=3, vector=[0.1, 0.2, 0.3]))
    repo.upsert_embedding(Embedding(segment_id=seg.id, dim=3, vector=[1.0, 2.0, 3.0]))

    embeddings = list(repo.iter_embeddings(meeting.id))
    assert len(embeddings) == 1  # upsert replaced, not duplicated
    assert embeddings[0].vector == pytest.approx([1.0, 2.0, 3.0])


def test_cascade_purge(repo: Repository):
    meeting = repo.create_meeting()
    seg = repo.add_segment(
        TranscriptSegment(meeting_id=meeting.id, start_time=0, end_time=1, text="x")
    )
    repo.add_suggestion(
        Suggestion(
            meeting_id=meeting.id,
            content="s",
            confidence=0.5,
            citations=[Citation(segment_id=seg.id)],
        )
    )
    repo.purge_meeting(meeting.id)
    assert repo.get_meeting(meeting.id) is None
    assert repo.list_segments(meeting.id) == []
    assert repo.list_suggestions(meeting.id) == []


def test_purge_all(repo: Repository):
    m = repo.create_meeting()
    repo.add_segment(TranscriptSegment(meeting_id=m.id, start_time=0, end_time=1, text="x"))
    repo.purge_all()
    assert repo.list_meetings() == []


def test_vector_packing():
    vec = [0.0, -1.5, 3.25]
    assert unpack_vector(pack_vector(vec)) == pytest.approx(vec)
