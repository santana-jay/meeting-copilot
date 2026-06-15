"""Anthropic Claude Messages API client.

The ``anthropic`` SDK is imported lazily so importing this module never
requires the dependency. The model id is supplied by configuration rather than
hardcoded — model ids, limits, and pricing must be verified against
``docs.claude.com`` (see README).
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import LLMClient, LLMError, LLMResponse, Message


class AnthropicClient(LLMClient):
    """Chat-completion client backed by the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        if not api_key:
            raise LLMError("An Anthropic API key is required.")
        self._api_key = api_key
        self._model = model
        self._default_max_tokens = max_tokens
        self._default_temperature = temperature
        self._client = None

    @property
    def model(self) -> str:
        return self._model

    def _ensure_client(self):  # pragma: no cover - requires anthropic SDK
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(  # pragma: no cover - requires anthropic SDK + network
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        client = self._ensure_client()
        payload = [{"role": m.role, "content": m.content} for m in messages]
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=max_tokens or self._default_max_tokens,
                temperature=(
                    self._default_temperature if temperature is None else temperature
                ),
                system=system or "",
                messages=payload,
            )
        except Exception as exc:  # normalize SDK errors
            raise LLMError(str(exc)) from exc

        text = _extract_text(response)
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            model=self._model,
            stop_reason=getattr(response, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )


def _extract_text(response) -> str:  # pragma: no cover - requires SDK response objects
    """Concatenate text blocks from an Anthropic Messages response."""
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts)
