"""Typed data models for stored entities.

These are plain dataclasses decoupled from the database layer so the
intelligence and retrieval pipelines can pass them around without depending
on sqlite row objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class NoteKind(StrEnum):
    """Categories of structured notes extracted from a transcript."""

    TOPIC = "topic"
    DECISION = "decision"
    ACTION_ITEM = "action_item"
    OPEN_QUESTION = "open_question"


@dataclass
class Meeting:
    id: int | None = None
    title: str | None = None
    started_at: float = 0.0
    ended_at: float | None = None
    created_at: float = 0.0


@dataclass
class TranscriptSegment:
    id: int | None = None
    meeting_id: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    speaker: str | None = None
    text: str = ""
    is_final: bool = True
    created_at: float = 0.0


@dataclass
class Note:
    id: int | None = None
    meeting_id: int = 0
    kind: NoteKind = NoteKind.TOPIC
    content: str = ""
    created_at: float = 0.0


@dataclass
class Citation:
    id: int | None = None
    suggestion_id: int | None = None
    segment_id: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    quote: str | None = None


@dataclass
class Suggestion:
    id: int | None = None
    meeting_id: int = 0
    content: str = ""
    confidence: float = 0.0
    abstained: bool = False
    created_at: float = 0.0
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Embedding:
    id: int | None = None
    segment_id: int = 0
    dim: int = 0
    vector: list[float] = field(default_factory=list)
    created_at: float = 0.0
