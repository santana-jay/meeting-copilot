"""Application configuration.

Configuration is layered, in increasing order of precedence:

1. Built-in defaults (see :class:`AppConfig`).
2. A TOML config file (``MEETING_COPILOT_CONFIG`` or the platform config dir).
3. Environment variables prefixed with ``MEETING_COPILOT_``.

Secrets (such as the Anthropic API key) are intentionally *not* part of the
config file; they are resolved separately via :mod:`meeting_copilot.secrets_store`.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints

ENV_PREFIX = "MEETING_COPILOT_"
CONFIG_ENV_VAR = "MEETING_COPILOT_CONFIG"
APP_DIR_NAME = "meeting-copilot"


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration.

    Attributes are intentionally primitive so the whole config round-trips
    cleanly to and from TOML/JSON and environment variables.
    """

    # Storage
    data_dir: str = ""
    database_filename: str = "meeting_copilot.sqlite3"

    # AI / Anthropic. The model id is configurable rather than hardcoded; see
    # README — model ids/limits/pricing must be verified against docs.claude.com.
    anthropic_model: str = "claude-3-5-sonnet-latest"
    anthropic_max_tokens: int = 1024
    anthropic_temperature: float = 0.0

    # Retrieval / grounding
    transcript_window_seconds: int = 120
    retrieval_top_k: int = 5
    min_grounding_confidence: float = 0.55

    # Speech-to-text
    stt_backend: str = "mock"  # "mock" | "whisper"
    whisper_model: str = "base"

    # Audio capture
    audio_backend: str = "auto"  # "auto" | "windows" | "macos" | "linux" | "null"
    sample_rate: int = 16_000

    # UI
    hotkey_toggle_overlay: str = "ctrl+alt+space"
    overlay_opacity: float = 0.92

    def database_path(self) -> Path:
        """Return the absolute path to the SQLite database file."""
        return self.resolved_data_dir() / self.database_filename

    def resolved_data_dir(self) -> Path:
        """Return the data directory, falling back to the platform default."""
        if self.data_dir:
            return Path(self.data_dir).expanduser()
        return default_data_dir()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_config_dir() -> Path:
    """Return the platform-appropriate configuration directory."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    elif _is_macos():
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_DIR_NAME


def default_data_dir() -> Path:
    """Return the platform-appropriate data directory."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    elif _is_macos():
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_DIR_NAME


def _is_macos() -> bool:
    import sys

    return sys.platform == "darwin"


def default_config_path() -> Path:
    return default_config_dir() / "config.toml"


def _coerce(field_type: Any, raw: Any) -> Any:
    """Coerce a raw value (often a string from env) to the field's type."""
    if field_type is bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    return str(raw)


def _field_types() -> dict[str, Any]:
    """Resolve real field types (annotations are strings under PEP 563)."""
    hints = get_type_hints(AppConfig)
    return {f.name: hints[f.name] for f in fields(AppConfig)}


def _from_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Filter and coerce a mapping to known config fields."""
    known = _field_types()
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = key.lower()
        if name in known:
            out[name] = _coerce(known[name], value)
    return out


def _load_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _load_env(environ: dict[str, str]) -> dict[str, Any]:
    out: dict[str, str] = {}
    for key, value in environ.items():
        if key.startswith(ENV_PREFIX) and key != CONFIG_ENV_VAR:
            out[key[len(ENV_PREFIX) :].lower()] = value
    return out


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> AppConfig:
    """Load configuration from defaults, file, and environment.

    Args:
        config_path: Explicit path to a TOML config file. When ``None`` the
            ``MEETING_COPILOT_CONFIG`` env var or the platform default is used.
        environ: Environment mapping to read from (defaults to ``os.environ``).
    """
    environ = dict(os.environ if environ is None else environ)

    if config_path is None:
        env_path = environ.get(CONFIG_ENV_VAR)
        path = Path(env_path) if env_path else default_config_path()
    else:
        path = Path(config_path)

    merged: dict[str, Any] = {}
    merged.update(_from_mapping(_load_file(path)))
    merged.update(_from_mapping(_load_env(environ)))

    return AppConfig(**merged)
