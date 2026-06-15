"""Tests for the live processing pipeline and its wiring into AppController.

Everything here runs offline with the null audio backend, the mock STT, and a
scripted mock LLM, while exercising the real storage, retrieval, validation, and
grounding code paths.
"""

from __future__ import annotations

import json
import time

import pytest

from meeting_copilot.ai import MockLLMClient, create_llm
from meeting_copilot.ai.anthropic_client import AnthropicClient
from meeting_copilot.app import AppController
from meeting_copilot.audio import AudioChunk, NullAudioCapture
from meeting_copilot.config import AppConfig
from meeting_copilot.db import Database, Repository
from meeting_copilot.intelligence import NoteExtractor, SuggestionEngine, prompts
from meeting_copilot.pipeline import CallbackSink, CollectingSink, MeetingPipeline
from meeting_copilot.retrieval import HashingEmbeddingModel, VectorStore
from meeting_copilot.secrets_store import SecretStore
from meeting_copilot.stt import MockSTT

NOTES_PAYLOAD = json.dumps(
    {
        "notes": [
            {"kind": "decision", "content": "Migrate database to Postgres"},
            {"kind": "action_item", "content": "Alice owns the migration"},
        ],
        "unknowns": [],
    }
)
SUGGESTION_PAYLOAD = json.dumps(
    {
        "abstain": False,
        "suggestion": "Confirm Friday as the migration status checkpoint.",
        "confidence": 0.82,
        "citations": [{"quote": "report back Friday"}],
    }
)
PHRASES = [
    "We decided to migrate the database to Postgres next sprint.",
    "Alice will own the migration and report back Friday.",
]


def _scripted_llm() -> MockLLMClient:
    """An LLM that answers note vs. suggestion prompts based on the system text."""

    def handler(messages, system):
        if system == prompts.NOTES_SYSTEM:
            return NOTES_PAYLOAD
        if system == prompts.SUGGESTIONS_SYSTEM:
            return SUGGESTION_PAYLOAD
        return ""

    return MockLLMClient(handler=handler)


def _make_pipeline(sink=None) -> tuple[MeetingPipeline, Repository]:
    repo = Repository(Database(":memory:"))
    store = VectorStore(repo, HashingEmbeddingModel(dimension=256))
    llm = _scripted_llm()
    pipeline = MeetingPipeline(
        repository=repo,
        vector_store=store,
        stt=MockSTT(phrases=PHRASES),
        note_extractor=NoteExtractor(llm),
        suggestion_engine=SuggestionEngine(llm, min_confidence=0.55),
        sink=sink,
    )
    return pipeline, repo


def _chunks(n: int = 2) -> list[AudioChunk]:
    return [
        AudioChunk(samples=b"\x00\x00" * 16_000, sample_rate=16_000, timestamp=float(i))
        for i in range(n)
    ]


def test_process_chunks_stores_transcript_notes_and_grounded_suggestion():
    sink = CollectingSink()
    pipeline, repo = _make_pipeline(sink)
    meeting = repo.create_meeting(title="Planning")

    pipeline.process_chunks(meeting.id, _chunks())

    segments = repo.list_segments(meeting.id)
    assert [s.text for s in segments] == PHRASES

    notes = repo.list_notes(meeting.id)
    assert {n.content for n in notes} == {
        "Migrate database to Postgres",
        "Alice owns the migration",
    }

    suggestions = repo.list_suggestions(meeting.id)
    assert len(suggestions) == 1
    assert suggestions[0].citations[0].quote == "report back Friday"
    assert suggestions[0].citations[0].segment_id is not None

    # The sink received incremental updates.
    assert len(sink.transcripts) == 2
    assert sink.notes  # at least one note emitted
    assert sink.suggestions and sink.suggestions[0].content


def test_notes_are_deduplicated_across_passes():
    pipeline, repo = _make_pipeline()
    meeting = repo.create_meeting()
    # Two passes over the same scripted notes must not duplicate rows.
    pipeline.process_chunks(meeting.id, _chunks())
    pipeline.run_intelligence(meeting.id)
    assert len(repo.list_notes(meeting.id)) == 2


def test_abstained_suggestion_is_not_stored():
    repo = Repository(Database(":memory:"))
    store = VectorStore(repo, HashingEmbeddingModel(dimension=256))
    abstain_payload = json.dumps({"abstain": True})

    def handler(messages, system):
        if system == prompts.NOTES_SYSTEM:
            return json.dumps({"notes": [], "unknowns": []})
        return abstain_payload

    llm = MockLLMClient(handler=handler)
    pipeline = MeetingPipeline(
        repository=repo,
        vector_store=store,
        stt=MockSTT(phrases=PHRASES),
        note_extractor=NoteExtractor(llm),
        suggestion_engine=SuggestionEngine(llm),
    )
    meeting = repo.create_meeting()
    pipeline.process_chunks(meeting.id, _chunks())
    assert repo.list_suggestions(meeting.id) == []


def test_run_intelligence_without_segments_abstains():
    pipeline, repo = _make_pipeline()
    meeting = repo.create_meeting()
    result = pipeline.run_intelligence(meeting.id)
    assert result.notes == []
    assert result.suggestion.abstained is True


def test_controller_process_audio_end_to_end():
    sink = CollectingSink()
    controller = AppController(
        config=AppConfig(),
        secrets=SecretStore(environ={}, use_keyring=False),
        database=Database(":memory:"),
        sink=sink,
        llm=_scripted_llm(),
        stt=MockSTT(phrases=PHRASES),
    )
    meeting = controller.start_meeting(title="Sync")
    controller.process_audio(_chunks())

    assert len(controller.repository.list_segments(meeting.id)) == 2
    assert len(controller.repository.list_notes(meeting.id)) == 2
    assert len(controller.repository.list_suggestions(meeting.id)) == 1
    assert sink.suggestions
    controller.close()


def test_controller_process_audio_requires_active_meeting():
    controller = AppController(
        config=AppConfig(),
        secrets=SecretStore(environ={}, use_keyring=False),
        database=Database(":memory:"),
        llm=_scripted_llm(),
        stt=MockSTT(phrases=PHRASES),
    )
    try:
        with pytest.raises(RuntimeError):
            controller.process_audio(_chunks())
    finally:
        controller.close()


def test_create_llm_falls_back_to_mock_without_key():
    client = create_llm(AppConfig(), SecretStore(environ={}, use_keyring=False))
    assert isinstance(client, MockLLMClient)


def test_create_llm_uses_anthropic_when_key_present():
    secrets = SecretStore(environ={"ANTHROPIC_API_KEY": "sk-test"}, use_keyring=False)
    client = create_llm(AppConfig(anthropic_model="claude-x"), secrets)
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-x"


def test_callback_sink_ignores_missing_callbacks():
    seen = []
    sink = CallbackSink(status=seen.append)
    sink.on_status("hi")
    sink.on_notes([])  # no callback configured -> no error
    assert seen == ["hi"]


def test_threaded_live_runner_processes_then_stops():
    sink = CollectingSink()
    pipeline, repo = _make_pipeline(sink)
    meeting = repo.create_meeting()

    pipeline.start(meeting.id, NullAudioCapture(chunks=_chunks()))
    # Wait for the background thread to drain the seeded chunks.
    deadline = time.time() + 5.0
    while pipeline.running and len(repo.list_segments(meeting.id)) < 2:
        if time.time() > deadline:
            break
        time.sleep(0.01)
    pipeline.stop()

    assert pipeline.running is False
    assert len(repo.list_segments(meeting.id)) == 2
    assert len(repo.list_suggestions(meeting.id)) == 1
    assert sink.statuses[0] == "Listening…"
    assert sink.statuses[-1] == "Stopped"
