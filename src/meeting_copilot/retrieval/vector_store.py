"""SQLite-backed vector store with brute-force cosine similarity search.

This is the dependency-free default backing described in the README
(SQLite-backed vector search). It persists one embedding per transcript
segment and ranks candidates by cosine similarity. For larger corpora a
``sqlite-vec``/FAISS backend can replace this while keeping the interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..db.models import Embedding, TranscriptSegment
from ..db.repository import Repository
from .embeddings import EmbeddingModel


@dataclass(frozen=True)
class ScoredSegment:
    """A transcript segment ranked by similarity to a query."""

    segment: TranscriptSegment
    score: float


class VectorStore:
    """Persist and search transcript-segment embeddings."""

    def __init__(self, repository: Repository, model: EmbeddingModel) -> None:
        self._repo = repository
        self._model = model

    def index_segment(self, segment: TranscriptSegment) -> None:
        """Compute and persist the embedding for a stored segment."""
        if segment.id is None:
            raise ValueError("Segment must be persisted (have an id) before indexing.")
        vector = self._model.embed(segment.text)
        self._repo.upsert_embedding(
            Embedding(segment_id=segment.id, dim=len(vector), vector=vector)
        )

    def search(
        self,
        query: str,
        *,
        meeting_id: int | None = None,
        top_k: int = 5,
        exclude_segment_ids: set[int] | None = None,
    ) -> list[ScoredSegment]:
        """Return the ``top_k`` most similar segments to ``query``.

        Args:
            query: Free-text query (usually the recent transcript window).
            meeting_id: Restrict search to a single meeting, or ``None`` for all.
            top_k: Maximum number of results.
            exclude_segment_ids: Segment ids to omit (e.g. the live window).
        """
        if top_k <= 0:
            return []
        query_vec = self._model.embed(query)
        exclude = exclude_segment_ids or set()

        # Map segment ids to segments for the candidate set.
        segments = self._segments_by_id(meeting_id)

        scored: list[ScoredSegment] = []
        for embedding in self._repo.iter_embeddings(meeting_id):
            if embedding.segment_id in exclude:
                continue
            segment = segments.get(embedding.segment_id)
            if segment is None:
                continue
            score = cosine_similarity(query_vec, embedding.vector)
            scored.append(ScoredSegment(segment=segment, score=score))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def _segments_by_id(self, meeting_id: int | None) -> dict[int, TranscriptSegment]:
        result: dict[int, TranscriptSegment] = {}
        meeting_ids = (
            [meeting_id]
            if meeting_id is not None
            else [m.id for m in self._repo.list_meetings()]
        )
        for mid in meeting_ids:
            if mid is None:
                continue
            for segment in self._repo.list_segments(mid):
                if segment.id is not None:
                    result[segment.id] = segment
        return result


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
