"""JSON schemas for structured LLM outputs.

All AI outputs are validated against these schemas before display or storage.
Malformed or ungrounded responses are dropped (see :mod:`validation` and
:mod:`grounding`).
"""

from __future__ import annotations

NOTE_KINDS = ["topic", "decision", "action_item", "open_question"]

# Output of the note-extraction prompt: structured notes grounded in the
# transcript window. ``unknowns`` lists names/numbers/quotes the model could
# not resolve, which must be marked unknown rather than invented.
NOTES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["notes"],
    "properties": {
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "content"],
                "properties": {
                    "kind": {"type": "string", "enum": NOTE_KINDS},
                    "content": {"type": "string", "minLength": 1},
                },
            },
        },
        "unknowns": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

# Output of the suggestion prompt. ``abstain`` lets the model decline when it
# cannot ground a confident suggestion. ``citations`` must reference transcript
# spans; grounding is verified separately.
SUGGESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["abstain"],
    "properties": {
        "abstain": {"type": "boolean"},
        "suggestion": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["quote"],
                "properties": {
                    "segment_id": {"type": ["integer", "null"]},
                    "start_time": {"type": ["number", "null"]},
                    "end_time": {"type": ["number", "null"]},
                    "quote": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}
