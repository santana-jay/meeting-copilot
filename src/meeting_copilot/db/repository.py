"""Data-access layer mapping rows to dataclass models."""

from __future__ import annotations

import sqlite3
import struct
import time
from collections.abc import Iterable, Sequence

from .database import Database
from .models import (
    Citation,
    Embedding,
    Meeting,
    Note,
    NoteKind,
    Suggestion,
    TranscriptSegment,
)


def _now() -> float:
    return time.time()


def pack_vector(vector: Sequence[float]) -> bytes:
    """Serialize a float vector to little-endian float32 bytes."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    """Deserialize little-endian float32 bytes back to a list of floats."""
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


class Repository:
    """CRUD operations over the Meeting Copilot schema."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._conn = db.connection

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def commit(self) -> None:
        self._conn.commit()

    # -- meetings ---------------------------------------------------------
    def create_meeting(self, title: str | None = None, started_at: float | None = None) -> Meeting:
        now = _now()
        started = now if started_at is None else started_at
        cur = self._conn.execute(
            "INSERT INTO meetings(title, started_at, ended_at, created_at) VALUES (?, ?, ?, ?)",
            (title, started, None, now),
        )
        self._conn.commit()
        return Meeting(
            id=cur.lastrowid, title=title, started_at=started, ended_at=None, created_at=now
        )

    def end_meeting(self, meeting_id: int, ended_at: float | None = None) -> None:
        self._conn.execute(
            "UPDATE meetings SET ended_at = ? WHERE id = ?",
            (ended_at if ended_at is not None else _now(), meeting_id),
        )
        self._conn.commit()

    def get_meeting(self, meeting_id: int) -> Meeting | None:
        row = self._conn.execute(
            "SELECT * FROM meetings WHERE id = ?", (meeting_id,)
        ).fetchone()
        return _row_to_meeting(row) if row else None

    def list_meetings(self) -> list[Meeting]:
        rows = self._conn.execute(
            "SELECT * FROM meetings ORDER BY started_at DESC"
        ).fetchall()
        return [_row_to_meeting(r) for r in rows]

    # -- transcript segments ---------------------------------------------
    def add_segment(self, segment: TranscriptSegment) -> TranscriptSegment:
        now = segment.created_at or _now()
        cur = self._conn.execute(
            """INSERT INTO transcript_segments
               (meeting_id, start_time, end_time, speaker, text, is_final, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                segment.meeting_id,
                segment.start_time,
                segment.end_time,
                segment.speaker,
                segment.text,
                int(segment.is_final),
                now,
            ),
        )
        self._conn.commit()
        segment.id = cur.lastrowid
        segment.created_at = now
        return segment

    def list_segments(self, meeting_id: int) -> list[TranscriptSegment]:
        rows = self._conn.execute(
            "SELECT * FROM transcript_segments WHERE meeting_id = ? ORDER BY start_time",
            (meeting_id,),
        ).fetchall()
        return [_row_to_segment(r) for r in rows]

    def recent_segments(self, meeting_id: int, window_seconds: float) -> list[TranscriptSegment]:
        """Return final segments whose end_time falls within the last window."""
        row = self._conn.execute(
            "SELECT MAX(end_time) FROM transcript_segments WHERE meeting_id = ?",
            (meeting_id,),
        ).fetchone()
        latest = row[0] if row and row[0] is not None else 0.0
        threshold = latest - window_seconds
        rows = self._conn.execute(
            """SELECT * FROM transcript_segments
               WHERE meeting_id = ? AND end_time >= ? AND is_final = 1
               ORDER BY start_time""",
            (meeting_id, threshold),
        ).fetchall()
        return [_row_to_segment(r) for r in rows]

    # -- notes ------------------------------------------------------------
    def add_note(self, note: Note) -> Note:
        now = note.created_at or _now()
        kind = note.kind.value if isinstance(note.kind, NoteKind) else str(note.kind)
        cur = self._conn.execute(
            "INSERT INTO notes(meeting_id, kind, content, created_at) VALUES (?, ?, ?, ?)",
            (note.meeting_id, kind, note.content, now),
        )
        self._conn.commit()
        note.id = cur.lastrowid
        note.created_at = now
        return note

    def list_notes(self, meeting_id: int, kind: NoteKind | None = None) -> list[Note]:
        if kind is None:
            rows = self._conn.execute(
                "SELECT * FROM notes WHERE meeting_id = ? ORDER BY id", (meeting_id,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM notes WHERE meeting_id = ? AND kind = ? ORDER BY id",
                (meeting_id, kind.value),
            ).fetchall()
        return [_row_to_note(r) for r in rows]

    # -- suggestions + citations -----------------------------------------
    def add_suggestion(self, suggestion: Suggestion) -> Suggestion:
        now = suggestion.created_at or _now()
        cur = self._conn.execute(
            """INSERT INTO suggestions(meeting_id, content, confidence, abstained, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                suggestion.meeting_id,
                suggestion.content,
                suggestion.confidence,
                int(suggestion.abstained),
                now,
            ),
        )
        suggestion.id = cur.lastrowid
        suggestion.created_at = now
        for citation in suggestion.citations:
            citation.suggestion_id = suggestion.id
            self._insert_citation(citation)
        self._conn.commit()
        return suggestion

    def _insert_citation(self, citation: Citation) -> Citation:
        cur = self._conn.execute(
            """INSERT INTO citations(suggestion_id, segment_id, start_time, end_time, quote)
               VALUES (?, ?, ?, ?, ?)""",
            (
                citation.suggestion_id,
                citation.segment_id,
                citation.start_time,
                citation.end_time,
                citation.quote,
            ),
        )
        citation.id = cur.lastrowid
        return citation

    def list_suggestions(self, meeting_id: int) -> list[Suggestion]:
        rows = self._conn.execute(
            "SELECT * FROM suggestions WHERE meeting_id = ? ORDER BY id", (meeting_id,)
        ).fetchall()
        suggestions = [_row_to_suggestion(r) for r in rows]
        for suggestion in suggestions:
            suggestion.citations = self._list_citations(suggestion.id)
        return suggestions

    def _list_citations(self, suggestion_id: int | None) -> list[Citation]:
        if suggestion_id is None:
            return []
        rows = self._conn.execute(
            "SELECT * FROM citations WHERE suggestion_id = ? ORDER BY id", (suggestion_id,)
        ).fetchall()
        return [_row_to_citation(r) for r in rows]

    # -- embeddings -------------------------------------------------------
    def upsert_embedding(self, embedding: Embedding) -> Embedding:
        now = embedding.created_at or _now()
        blob = pack_vector(embedding.vector)
        self._conn.execute(
            """INSERT INTO embeddings(segment_id, dim, vector, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(segment_id) DO UPDATE SET
                   dim = excluded.dim,
                   vector = excluded.vector,
                   created_at = excluded.created_at""",
            (embedding.segment_id, embedding.dim, blob, now),
        )
        self._conn.commit()
        embedding.created_at = now
        return embedding

    def iter_embeddings(self, meeting_id: int | None = None) -> Iterable[Embedding]:
        if meeting_id is None:
            rows = self._conn.execute("SELECT * FROM embeddings").fetchall()
        else:
            rows = self._conn.execute(
                """SELECT e.* FROM embeddings e
                   JOIN transcript_segments s ON s.id = e.segment_id
                   WHERE s.meeting_id = ?""",
                (meeting_id,),
            ).fetchall()
        return [_row_to_embedding(r) for r in rows]

    def purge_meeting(self, meeting_id: int) -> None:
        """Delete a meeting and all dependent rows (cascade)."""
        self._conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
        self._conn.commit()

    def purge_all(self) -> None:
        """Delete all stored data."""
        for table in (
            "citations",
            "suggestions",
            "notes",
            "embeddings",
            "transcript_segments",
            "meetings",
        ):
            self._conn.execute(f"DELETE FROM {table}")
        self._conn.commit()


# -- row mappers ----------------------------------------------------------
def _row_to_meeting(row: sqlite3.Row) -> Meeting:
    return Meeting(
        id=row["id"],
        title=row["title"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        created_at=row["created_at"],
    )


def _row_to_segment(row: sqlite3.Row) -> TranscriptSegment:
    return TranscriptSegment(
        id=row["id"],
        meeting_id=row["meeting_id"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        speaker=row["speaker"],
        text=row["text"],
        is_final=bool(row["is_final"]),
        created_at=row["created_at"],
    )


def _row_to_note(row: sqlite3.Row) -> Note:
    return Note(
        id=row["id"],
        meeting_id=row["meeting_id"],
        kind=NoteKind(row["kind"]),
        content=row["content"],
        created_at=row["created_at"],
    )


def _row_to_suggestion(row: sqlite3.Row) -> Suggestion:
    return Suggestion(
        id=row["id"],
        meeting_id=row["meeting_id"],
        content=row["content"],
        confidence=row["confidence"],
        abstained=bool(row["abstained"]),
        created_at=row["created_at"],
    )


def _row_to_citation(row: sqlite3.Row) -> Citation:
    return Citation(
        id=row["id"],
        suggestion_id=row["suggestion_id"],
        segment_id=row["segment_id"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        quote=row["quote"],
    )


def _row_to_embedding(row: sqlite3.Row) -> Embedding:
    return Embedding(
        id=row["id"],
        segment_id=row["segment_id"],
        dim=row["dim"],
        vector=unpack_vector(row["vector"]),
        created_at=row["created_at"],
    )
