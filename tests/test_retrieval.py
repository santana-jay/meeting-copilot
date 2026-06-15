"""Tests for embeddings and the SQLite-backed vector store."""

from __future__ import annotations

from meeting_copilot.db import Repository, TranscriptSegment
from meeting_copilot.retrieval import HashingEmbeddingModel, VectorStore
from meeting_copilot.retrieval.embeddings import tokenize
from meeting_copilot.retrieval.vector_store import cosine_similarity


def test_hashing_embedding_is_deterministic_and_normalized():
    model = HashingEmbeddingModel(dimension=64)
    v1 = model.embed("the quick brown fox")
    v2 = model.embed("the quick brown fox")
    assert v1 == v2  # reproducible across calls (stable hashing)
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_empty_text_embeds_to_zero_vector():
    model = HashingEmbeddingModel(dimension=8)
    assert model.embed("   ") == [0.0] * 8


def test_tokenize():
    assert tokenize("Hello, World! it's 2024") == ["hello", "world", "it's", "2024"]


def test_cosine_similarity_bounds():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([], [1]) == 0.0


def test_vector_store_search_ranks_relevant(repo: Repository):
    meeting = repo.create_meeting()
    texts = [
        "We decided to migrate the database to Postgres next quarter.",
        "Lunch options were pizza and salad.",
        "The budget for marketing increased by ten percent.",
    ]
    segments = []
    for i, text in enumerate(texts):
        seg = repo.add_segment(
            TranscriptSegment(meeting_id=meeting.id, start_time=i, end_time=i + 1, text=text)
        )
        segments.append(seg)

    store = VectorStore(repo, HashingEmbeddingModel(dimension=512))
    for seg in segments:
        store.index_segment(seg)

    results = store.search("database migration postgres", meeting_id=meeting.id, top_k=2)
    assert len(results) == 2
    assert "Postgres" in results[0].segment.text


def test_vector_store_excludes_ids(repo: Repository):
    meeting = repo.create_meeting()
    seg = repo.add_segment(
        TranscriptSegment(meeting_id=meeting.id, start_time=0, end_time=1, text="hello world")
    )
    store = VectorStore(repo, HashingEmbeddingModel(dimension=32))
    store.index_segment(seg)
    results = store.search("hello", meeting_id=meeting.id, exclude_segment_ids={seg.id})
    assert results == []


def test_index_requires_persisted_segment(repo: Repository):
    store = VectorStore(repo, HashingEmbeddingModel(dimension=16))
    try:
        store.index_segment(TranscriptSegment(meeting_id=1, text="x"))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
