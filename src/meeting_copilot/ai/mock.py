"""A scriptable mock LLM client for tests and offline development."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .base import LLMClient, LLMResponse, Message


class MockLLMClient(LLMClient):
    """Return canned responses or compute them from the input messages.

    Args:
        responses: A queue of response strings returned in order. After the
            queue is exhausted, the last response repeats.
        handler: Optional callable ``(messages, system) -> str`` used instead
            of ``responses`` to compute responses dynamically.
        model: Reported model id.
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        handler: Callable[[Sequence[Message], str | None], str] | None = None,
        model: str = "mock-claude",
    ) -> None:
        self._responses = list(responses or [])
        self._handler = handler
        self._model = model
        self._index = 0
        self.calls: list[dict] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self._handler is not None:
            text = self._handler(messages, system)
        elif self._responses:
            idx = min(self._index, len(self._responses) - 1)
            text = self._responses[idx]
            self._index += 1
        else:
            text = ""
        return LLMResponse(text=text, model=self._model, stop_reason="end_turn")
