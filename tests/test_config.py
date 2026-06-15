"""Tests for configuration loading and precedence."""

from __future__ import annotations

from meeting_copilot.config import AppConfig, load_config


def test_defaults():
    cfg = load_config(environ={})
    assert cfg.database_filename == "meeting_copilot.sqlite3"
    assert cfg.anthropic_temperature == 0.0
    assert cfg.retrieval_top_k == 5


def test_file_loading(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'anthropic_model = "claude-test"\nretrieval_top_k = 9\noverlay_opacity = 0.5\n',
        encoding="utf-8",
    )
    cfg = load_config(path, environ={})
    assert cfg.anthropic_model == "claude-test"
    assert cfg.retrieval_top_k == 9
    assert cfg.overlay_opacity == 0.5


def test_env_overrides_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('anthropic_model = "from-file"\n', encoding="utf-8")
    cfg = load_config(
        path,
        environ={
            "MEETING_COPILOT_ANTHROPIC_MODEL": "from-env",
            "MEETING_COPILOT_RETRIEVAL_TOP_K": "3",
        },
    )
    assert cfg.anthropic_model == "from-env"
    assert cfg.retrieval_top_k == 3  # coerced from string


def test_env_type_coercion():
    cfg = load_config(
        environ={
            "MEETING_COPILOT_ANTHROPIC_TEMPERATURE": "0.7",
            "MEETING_COPILOT_SAMPLE_RATE": "48000",
        }
    )
    assert cfg.anthropic_temperature == 0.7
    assert cfg.sample_rate == 48000


def test_unknown_keys_ignored(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('totally_unknown = 1\nanthropic_model = "ok"\n', encoding="utf-8")
    cfg = load_config(path, environ={})
    assert cfg.anthropic_model == "ok"
    assert not hasattr(cfg, "totally_unknown")


def test_database_path_uses_data_dir(tmp_path):
    cfg = AppConfig(data_dir=str(tmp_path), database_filename="x.sqlite3")
    assert cfg.database_path() == tmp_path / "x.sqlite3"
