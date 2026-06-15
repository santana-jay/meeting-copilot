"""Tests for the LLM client interface, mock, and Anthropic client wiring."""

from __future__ import annotations

import pytest

from meeting_copilot.ai import LLMError, Message, MockLLMClient
from meeting_copilot.ai.anthropic_client import AnthropicClient


def test_mock_queue_responses():
    client = MockLLMClient(["a", "b"])
    assert client.complete([Message("user", "hi")]).text == "a"
    assert client.complete([Message("user", "hi")]).text == "b"
    # exhausted queue repeats the last response
    assert client.complete([Message("user", "hi")]).text == "b"


def test_mock_records_calls():
    client = MockLLMClient(["x"])
    client.complete([Message("user", "q")], system="sys", temperature=0.0)
    assert client.calls[0]["system"] == "sys"
    assert client.calls[0]["temperature"] == 0.0


def test_mock_handler():
    client = MockLLMClient(handler=lambda msgs, system: msgs[-1].content.upper())
    assert client.complete([Message("user", "hi")]).text == "HI"


def test_anthropic_requires_api_key():
    with pytest.raises(LLMError):
        AnthropicClient("", model="claude-x")


def test_anthropic_reports_model():
    client = AnthropicClient("key", model="claude-x")
    assert client.model == "claude-x"
