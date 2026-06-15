"""Grounded, citation-aware suggestion generation with confidence gating."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..ai.base import LLMClient, Message
from ..db.models import Suggestion, TranscriptSegment
from ..retrieval.vector_store import ScoredSegment
from . import prompts
from .grounding import ground_suggestion
from .schemas import SUGGESTION_SCHEMA
from .validation import ValidationError, parse_and_validate

logger = logging.getLogger("meeting_copilot.intelligence")


@dataclass
class GeneratedSuggestion:
    """Result of a suggestion attempt.

    When ``abstained`` is ``True`` (model declined, weak grounding, low
    confidence, or a malformed response), ``suggestion`` is ``None`` and
    ``reason`` explains why. This makes "no confident suggestion" a first-class
    outcome rather than a forced guess.
    """

    abstained: bool
    suggestion: Suggestion | None = None
    reason: str | None = None


class SuggestionEngine:
    """Generate one grounded suggestion from recent + retrieved context.

    Suggestions are AI proposals. They are only surfaced when validated,
    grounded in cited transcript text, and above the confidence threshold.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        min_confidence: float = 0.55,
        min_grounding: float = 1.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._min_confidence = min_confidence
        self._min_grounding = min_grounding
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(
        self,
        meeting_id: int,
        recent: list[TranscriptSegment],
        retrieved: list[ScoredSegment] | None = None,
    ) -> GeneratedSuggestion:
        retrieved = retrieved or []
        if not recent and not retrieved:
            return GeneratedSuggestion(abstained=True, reason="No context available")

        user = prompts.build_suggestion_user_prompt(recent, retrieved)
        response = self._client.complete(
            [Message(role="user", content=user)],
            system=prompts.SUGGESTIONS_SYSTEM,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        try:
            data = parse_and_validate(response.text, SUGGESTION_SCHEMA)
        except ValidationError as exc:
            logger.warning("Dropping malformed suggestion response: %s", exc)
            return GeneratedSuggestion(abstained=True, reason=f"Malformed response: {exc}")

        if data.get("abstain"):
            return GeneratedSuggestion(abstained=True, reason="Model abstained")

        text = (data.get("suggestion") or "").strip()
        if not text:
            return GeneratedSuggestion(abstained=True, reason="Empty suggestion")

        confidence = float(data.get("confidence", 0.0))

        # Grounding context is the union of recent + retrieved segments.
        context_segments = list(recent) + [item.segment for item in retrieved]
        grounding = ground_suggestion(
            data.get("citations", []), context_segments, min_score=self._min_grounding
        )
        if not grounding.grounded:
            return GeneratedSuggestion(
                abstained=True,
                reason=f"Ungrounded (grounding score {grounding.score:.2f})",
            )

        if confidence < self._min_confidence:
            return GeneratedSuggestion(
                abstained=True,
                reason=f"Low confidence ({confidence:.2f} < {self._min_confidence:.2f})",
            )

        suggestion = Suggestion(
            meeting_id=meeting_id,
            content=text,
            confidence=confidence,
            abstained=False,
            citations=grounding.citations,
        )
        return GeneratedSuggestion(abstained=False, suggestion=suggestion)
