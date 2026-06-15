"""Runtime selection of an LLM client.

The factory builds a real :class:`AnthropicClient` when an API key is available
(via :class:`~meeting_copilot.secrets_store.SecretStore`) and otherwise falls
back to the scriptable :class:`MockLLMClient` so the application stays usable
offline and in tests without ever performing network I/O at import time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .anthropic_client import AnthropicClient
from .base import LLMClient
from .mock import MockLLMClient

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import AppConfig
    from ..secrets_store import SecretStore

logger = logging.getLogger("meeting_copilot.ai")


def create_llm(
    config: AppConfig,
    secrets: SecretStore | None = None,
    *,
    allow_mock_fallback: bool = True,
) -> LLMClient:
    """Create an LLM client from configuration and resolved secrets.

    Args:
        config: Application configuration (supplies the model id and limits).
        secrets: Secret store used to resolve the Anthropic API key. When the
            key is missing and ``allow_mock_fallback`` is ``True``, a
            :class:`MockLLMClient` is returned so the app degrades gracefully.
        allow_mock_fallback: When ``True`` (default) return a mock client if no
            API key is available; when ``False`` raise ``LLMError``.

    Returns:
        A ready-to-use :class:`LLMClient`. Construction never performs network
        I/O; the Anthropic SDK is only imported on the first completion.
    """
    api_key = secrets.get_anthropic_api_key() if secrets is not None else None
    if api_key:
        logger.info("Using Anthropic model %s", config.anthropic_model)
        return AnthropicClient(
            api_key,
            model=config.anthropic_model,
            max_tokens=config.anthropic_max_tokens,
            temperature=config.anthropic_temperature,
        )

    if not allow_mock_fallback:
        from .base import LLMError

        raise LLMError(
            "No Anthropic API key available. Set the ANTHROPIC_API_KEY "
            "environment variable or store it in the OS keychain."
        )

    logger.warning(
        "No Anthropic API key found; using the offline mock LLM client. "
        "Notes and suggestions will be unavailable until a key is configured."
    )
    return MockLLMClient(model="mock-claude")
