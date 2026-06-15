"""Parse and validate structured JSON from LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class ValidationError(Exception):
    """Raised when an LLM response is not valid JSON or violates the schema."""


def extract_json(text: str) -> str:
    """Extract a JSON document from raw model text.

    Handles responses wrapped in Markdown code fences and responses with
    surrounding prose by locating the outermost ``{...}`` object.
    """
    if not text or not text.strip():
        raise ValidationError("Empty response")

    fenced = _FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text.strip()


def parse_and_validate(text: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Parse model output and validate it against ``schema``.

    Returns the parsed object on success. Raises :class:`ValidationError` for
    malformed JSON or schema violations so callers can drop the response.
    """
    raw = extract_json(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON: {exc}") from exc

    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ValidationError(f"Schema violation: {exc.message}") from exc
    return data
