"""Tests for the secret store."""

from __future__ import annotations

from meeting_copilot.secrets_store import (
    ANTHROPIC_SECRET_KEY,
    SecretStore,
)


class FakeKeyring:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service, name):
        return self.store.get((service, name))

    def set_password(self, service, name, value):
        self.store[(service, name)] = value

    def delete_password(self, service, name):
        del self.store[(service, name)]


def test_env_takes_precedence():
    store = SecretStore(environ={"ANTHROPIC_API_KEY": "env-key"}, use_keyring=False)
    assert store.get_anthropic_api_key() == "env-key"


def test_missing_returns_none():
    store = SecretStore(environ={}, use_keyring=False)
    assert store.get_anthropic_api_key() is None


def test_require_raises():
    store = SecretStore(environ={}, use_keyring=False)
    try:
        store.require(ANTHROPIC_SECRET_KEY)
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)


def test_keyring_fallback(monkeypatch):
    fake = FakeKeyring()
    store = SecretStore(environ={})
    monkeypatch.setattr(store, "_keyring", lambda: fake)

    assert store.get_anthropic_api_key() is None
    assert store.set(ANTHROPIC_SECRET_KEY, "kc-key") is True
    assert store.get_anthropic_api_key() == "kc-key"
    assert store.delete(ANTHROPIC_SECRET_KEY) is True
    assert store.get_anthropic_api_key() is None


def test_env_overrides_keyring(monkeypatch):
    fake = FakeKeyring()
    fake.set_password("meeting-copilot", ANTHROPIC_SECRET_KEY, "kc-key")
    store = SecretStore(environ={"ANTHROPIC_API_KEY": "env-key"})
    monkeypatch.setattr(store, "_keyring", lambda: fake)
    assert store.get_anthropic_api_key() == "env-key"


def test_set_without_keyring_returns_false():
    store = SecretStore(environ={}, use_keyring=False)
    assert store.set(ANTHROPIC_SECRET_KEY, "x") is False
