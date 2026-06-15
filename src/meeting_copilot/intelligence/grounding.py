"""Verify that suggestions are grounded in the transcript.

A suggestion is grounded only when its citations quote text that actually
appears in the provided transcript context. Ungrounded citations are dropped
and weakly grounded suggestions are rejected, in line with the README's
anti-hallucination requirements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..db.models import Citation, TranscriptSegment


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace for robust quote matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


@dataclass
class GroundingResult:
    """Outcome of grounding a suggestion's citations."""

    grounded: bool
    score: float
    citations: list[Citation] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)


def _match_segment(
    quote: str, segments: list[TranscriptSegment]
) -> TranscriptSegment | None:
    norm_quote = normalize(quote)
    if not norm_quote:
        return None
    for seg in segments:
        if norm_quote in normalize(seg.text):
            return seg
    return None


def ground_suggestion(
    raw_citations: list[dict],
    segments: list[TranscriptSegment],
    *,
    min_score: float = 1.0,
) -> GroundingResult:
    """Validate citation quotes against the transcript context.

    Args:
        raw_citations: Citation dicts from the validated LLM output.
        segments: The transcript segments that formed the prompt context.
        min_score: Minimum fraction of citations that must match for the
            suggestion to be considered grounded (default: all of them).

    Returns:
        A :class:`GroundingResult`. ``citations`` contains only verified
        citations, anchored to the matched segment's id and timestamps.
    """
    if not raw_citations:
        return GroundingResult(grounded=False, score=0.0)

    verified: list[Citation] = []
    dropped: list[dict] = []
    for raw in raw_citations:
        quote = raw.get("quote", "")
        match = _match_segment(quote, segments)
        if match is None:
            dropped.append(raw)
            continue
        verified.append(
            Citation(
                segment_id=match.id,
                start_time=match.start_time,
                end_time=match.end_time,
                quote=quote,
            )
        )

    score = len(verified) / len(raw_citations)
    grounded = bool(verified) and score >= min_score
    return GroundingResult(
        grounded=grounded, score=score, citations=verified, dropped=dropped
    )
