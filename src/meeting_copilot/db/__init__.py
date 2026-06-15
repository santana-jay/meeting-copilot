"""Local SQLite storage for Meeting Copilot."""

from __future__ import annotations

from .database import SCHEMA_VERSION, Database
from .models import (
    Citation,
    Embedding,
    Meeting,
    Note,
    NoteKind,
    Suggestion,
    TranscriptSegment,
)
from .repository import Repository

__all__ = [
    "Database",
    "SCHEMA_VERSION",
    "Repository",
    "Meeting",
    "TranscriptSegment",
    "Note",
    "NoteKind",
    "Suggestion",
    "Citation",
    "Embedding",
]
