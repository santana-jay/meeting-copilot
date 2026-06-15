"""Structured note extraction through the LLM."""

from __future__ import annotations

import logging

from ..ai.base import LLMClient, Message
from ..db.models import Note, NoteKind, TranscriptSegment
from . import prompts
from .schemas import NOTES_SCHEMA
from .validation import ValidationError, parse_and_validate

logger = logging.getLogger("meeting_copilot.intelligence")


class NoteExtractor:
    """Extract topics, decisions, action items, and open questions.

    Notes are treated as an extraction of what was said (not AI proposals).
    Malformed responses are dropped and yield an empty list.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._temperature = temperature

    def extract(
        self, meeting_id: int, segments: list[TranscriptSegment]
    ) -> list[Note]:
        """Return structured notes for the given transcript window."""
        if not segments:
            return []

        user = prompts.build_notes_user_prompt(segments)
        response = self._client.complete(
            [Message(role="user", content=user)],
            system=prompts.NOTES_SYSTEM,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        try:
            data = parse_and_validate(response.text, NOTES_SCHEMA)
        except ValidationError as exc:
            logger.warning("Dropping malformed notes response: %s", exc)
            return []

        notes: list[Note] = []
        for item in data.get("notes", []):
            try:
                kind = NoteKind(item["kind"])
            except ValueError:
                continue
            content = item["content"].strip()
            if content:
                notes.append(Note(meeting_id=meeting_id, kind=kind, content=content))
        return notes
