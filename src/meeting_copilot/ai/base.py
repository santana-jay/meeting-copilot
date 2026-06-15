"""LLM client interface and shared message types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


class LLMError(Exception):
    """Raised when an LLM request fails."""


@dataclass(frozen=True)
class Message:
    """A single chat message."""

    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """A completion result from an LLM."""

    text: str
    model: str
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMClient(ABC):
    """Abstract chat-completion client.

    Implementations must be safe to construct without performing any network
    I/O so the rest of the system can be wired up and tested offline.
    """

    @property
    @abstractmethod
    def model(self) -> str:
        ...

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Return a completion for the given conversation."""
