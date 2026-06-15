"""Low-temperature prompts for grounded extraction and suggestions.

The prompts encode the anti-hallucination requirements: ground every output in
the provided transcript, cite spans, prefer abstention over guessing, and mark
unknown names/numbers/quotes as unknown rather than inventing them.
"""

from __future__ import annotations

from ..db.models import TranscriptSegment
from ..retrieval.vector_store import ScoredSegment

NOTES_SYSTEM = (
    "You extract structured notes from a meeting transcript. Use ONLY the "
    "provided transcript. Do not infer facts that are not stated. If a name, "
    "number, or quote is unclear, add it to 'unknowns' instead of guessing. "
    "Return ONLY JSON matching this shape: "
    '{"notes": [{"kind": "topic|decision|action_item|open_question", '
    '"content": "..."}], "unknowns": ["..."]}. '
    "Notes must paraphrase or quote what was actually said."
)

SUGGESTIONS_SYSTEM = (
    "You are a meeting co-pilot that proposes a single, concise, helpful "
    "suggestion grounded ONLY in the provided transcript context. Never invent "
    "names, numbers, or quotes. Every claim must be supported by a citation "
    "quoting transcript text. If you cannot ground a confident suggestion, set "
    '"abstain" to true. Return ONLY JSON matching this shape: '
    '{"abstain": false, "suggestion": "...", "confidence": 0.0, '
    '"citations": [{"segment_id": 1, "start_time": 0.0, "end_time": 0.0, '
    '"quote": "exact transcript text"}]}. '
    "Set confidence between 0 and 1 reflecting how well-grounded the suggestion "
    "is. Prefer abstaining over a weakly grounded guess."
)


def format_segments(segments: list[TranscriptSegment]) -> str:
    """Render transcript segments as a numbered, timestamped block."""
    lines = []
    for seg in segments:
        speaker = seg.speaker or "unknown"
        lines.append(
            f"[id={seg.id} t={seg.start_time:.1f}-{seg.end_time:.1f} "
            f"speaker={speaker}] {seg.text}"
        )
    return "\n".join(lines)


def format_retrieved(snippets: list[ScoredSegment]) -> str:
    """Render retrieved past snippets with similarity scores."""
    lines = []
    for item in snippets:
        seg = item.segment
        lines.append(
            f"[id={seg.id} t={seg.start_time:.1f}-{seg.end_time:.1f} "
            f"score={item.score:.2f}] {seg.text}"
        )
    return "\n".join(lines)


def build_notes_user_prompt(segments: list[TranscriptSegment]) -> str:
    return (
        "Transcript window:\n"
        f"{format_segments(segments)}\n\n"
        "Extract notes as JSON now."
    )


def build_suggestion_user_prompt(
    recent: list[TranscriptSegment], retrieved: list[ScoredSegment]
) -> str:
    parts = ["Recent transcript window:", format_segments(recent)]
    if retrieved:
        parts += ["\nRelevant earlier snippets:", format_retrieved(retrieved)]
    parts.append("\nPropose one grounded suggestion as JSON now, or abstain.")
    return "\n".join(parts)
