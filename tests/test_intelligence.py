"""Tests for validation, grounding, note extraction, and suggestions."""

from __future__ import annotations

import json

import pytest

from meeting_copilot.ai import MockLLMClient
from meeting_copilot.db.models import NoteKind, TranscriptSegment
from meeting_copilot.intelligence import (
    NoteExtractor,
    SuggestionEngine,
    ValidationError,
    ground_suggestion,
    parse_and_validate,
)
from meeting_copilot.intelligence.schemas import NOTES_SCHEMA, SUGGESTION_SCHEMA


def seg(id_, text, start=0.0, end=1.0):
    return TranscriptSegment(id=id_, meeting_id=1, start_time=start, end_time=end, text=text)


# -- validation ----------------------------------------------------------
def test_extract_json_from_fence():
    text = 'Here you go:\n```json\n{"abstain": true}\n```'
    data = parse_and_validate(text, SUGGESTION_SCHEMA)
    assert data["abstain"] is True


def test_extract_json_from_prose():
    text = 'Sure! {"abstain": false, "suggestion": "x", "confidence": 0.9} done'
    data = parse_and_validate(text, SUGGESTION_SCHEMA)
    assert data["suggestion"] == "x"


def test_invalid_json_raises():
    with pytest.raises(ValidationError):
        parse_and_validate("not json at all", SUGGESTION_SCHEMA)


def test_schema_violation_raises():
    # confidence out of range
    bad = json.dumps({"abstain": False, "confidence": 2.0})
    with pytest.raises(ValidationError):
        parse_and_validate(bad, SUGGESTION_SCHEMA)


def test_notes_schema_requires_notes():
    with pytest.raises(ValidationError):
        parse_and_validate("{}", NOTES_SCHEMA)


# -- grounding -----------------------------------------------------------
def test_grounding_matches_quote():
    segments = [seg(1, "We will ship on Friday")]
    result = ground_suggestion([{"quote": "ship on Friday"}], segments)
    assert result.grounded is True
    assert result.score == 1.0
    assert result.citations[0].segment_id == 1
    assert result.citations[0].start_time == 0.0


def test_grounding_drops_unmatched_quote():
    segments = [seg(1, "We will ship on Friday")]
    result = ground_suggestion(
        [{"quote": "ship on Friday"}, {"quote": "launch a rocket"}], segments
    )
    # default min_score=1.0 means a single unmatched citation fails grounding
    assert result.grounded is False
    assert len(result.citations) == 1
    assert len(result.dropped) == 1


def test_grounding_empty_citations():
    result = ground_suggestion([], [seg(1, "anything")])
    assert result.grounded is False
    assert result.score == 0.0


# -- note extraction -----------------------------------------------------
def test_note_extractor_parses_notes():
    payload = json.dumps(
        {
            "notes": [
                {"kind": "action_item", "content": "Email the report"},
                {"kind": "decision", "content": "Adopt Postgres"},
            ],
            "unknowns": [],
        }
    )
    extractor = NoteExtractor(MockLLMClient([payload]))
    notes = extractor.extract(1, [seg(1, "Email the report; we adopt Postgres")])
    assert {n.kind for n in notes} == {NoteKind.ACTION_ITEM, NoteKind.DECISION}


def test_note_extractor_drops_malformed():
    extractor = NoteExtractor(MockLLMClient(["garbage not json"]))
    assert extractor.extract(1, [seg(1, "x")]) == []


def test_note_extractor_empty_segments():
    extractor = NoteExtractor(MockLLMClient(["{}"]))
    assert extractor.extract(1, []) == []


# -- suggestions ---------------------------------------------------------
def _suggestion_payload(suggestion, confidence, quote, abstain=False):
    return json.dumps(
        {
            "abstain": abstain,
            "suggestion": suggestion,
            "confidence": confidence,
            "citations": [{"quote": quote}],
        }
    )


def test_suggestion_accepted_when_grounded_and_confident():
    segments = [seg(1, "Let's revisit the pricing model next week")]
    payload = _suggestion_payload(
        "Schedule a pricing review next week", 0.8, "revisit the pricing model"
    )
    engine = SuggestionEngine(MockLLMClient([payload]), min_confidence=0.55)
    result = engine.generate(1, segments)
    assert result.abstained is False
    assert result.suggestion is not None
    assert result.suggestion.confidence == 0.8
    assert result.suggestion.citations[0].segment_id == 1


def test_suggestion_abstains_when_model_declines():
    payload = json.dumps({"abstain": True})
    engine = SuggestionEngine(MockLLMClient([payload]))
    result = engine.generate(1, [seg(1, "hello")])
    assert result.abstained is True
    assert "abstain" in result.reason.lower()


def test_suggestion_abstains_when_ungrounded():
    segments = [seg(1, "We discussed the weather")]
    payload = _suggestion_payload("Buy more servers", 0.95, "provision more servers")
    engine = SuggestionEngine(MockLLMClient([payload]))
    result = engine.generate(1, segments)
    assert result.abstained is True
    assert "ungrounded" in result.reason.lower()


def test_suggestion_abstains_on_low_confidence():
    segments = [seg(1, "We will ship on Friday")]
    payload = _suggestion_payload("Confirm the Friday ship date", 0.30, "ship on Friday")
    engine = SuggestionEngine(MockLLMClient([payload]), min_confidence=0.55)
    result = engine.generate(1, segments)
    assert result.abstained is True
    assert "confidence" in result.reason.lower()


def test_suggestion_abstains_on_malformed():
    engine = SuggestionEngine(MockLLMClient(["not json"]))
    result = engine.generate(1, [seg(1, "x")])
    assert result.abstained is True
    assert "malformed" in result.reason.lower()


def test_suggestion_abstains_without_context():
    engine = SuggestionEngine(MockLLMClient(["{}"]))
    result = engine.generate(1, [])
    assert result.abstained is True
