"""Embedding models for transcript retrieval.

Provides an :class:`EmbeddingModel` interface and a dependency-free
:class:`HashingEmbeddingModel` default (a deterministic bag-of-words hashing
vectorizer). Heavier neural embedders can implement the same interface.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _stable_hash(value: str) -> int:
    """Return a process-independent hash so stored vectors stay reproducible."""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class EmbeddingModel(ABC):
    """Maps text to a fixed-length dense vector."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class HashingEmbeddingModel(EmbeddingModel):
    """Deterministic hashing vectorizer with L2-normalized output.

    This requires no model downloads and is suitable as a default and for
    tests. Similar texts map to similar vectors via shared token hash buckets.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in tokenize(text):
            bucket = _stable_hash(token) % self._dimension
            sign = 1.0 if (_stable_hash(token + "\0sign") & 1) else -1.0
            vector[bucket] += sign
        return _l2_normalize(vector)


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]
