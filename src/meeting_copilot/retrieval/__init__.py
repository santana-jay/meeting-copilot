"""Embeddings and vector retrieval over transcript segments."""

from __future__ import annotations

from .embeddings import EmbeddingModel, HashingEmbeddingModel
from .vector_store import ScoredSegment, VectorStore

__all__ = [
    "EmbeddingModel",
    "HashingEmbeddingModel",
    "VectorStore",
    "ScoredSegment",
]
