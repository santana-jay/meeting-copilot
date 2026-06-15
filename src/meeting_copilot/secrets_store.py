"""Secret handling.

Secrets are resolved from, in order of precedence:

1. Process environment variables (e.g. ``ANTHROPIC_API_KEY``).
2. The OS keychain via the optional ``keyring`` package.

Secrets are *never* written to the config file or the SQLite database. The
keychain is only touched when ``keyring`` is installed; otherwise the store
degrades to an environment-only resolver so the core remains importable in
headless/test environments.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

KEYCHAIN_SERVICE = "meeting-copilot"
ANTHROPIC_ENV_VAR = "ANTHROPIC_API_KEY"
ANTHROPIC_SECRET_KEY = "anthropic_api_key"

# Maps logical secret names to the environment variable that may hold them.
_ENV_ALIASES = {
    ANTHROPIC_SECRET_KEY: ANTHROPIC_ENV_VAR,
}


class SecretStore:
    """Resolve secrets from the environment and the OS keychain.

    The keychain is accessed lazily so importing this module never requires
    the optional ``keyring`` dependency.
    """

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        service: str = KEYCHAIN_SERVICE,
        use_keyring: bool = True,
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._service = service
        self._use_keyring = use_keyring

    def get(self, name: str) -> str | None:
        """Return the secret value for ``name`` or ``None`` if unset.

        Environment variables take precedence over the keychain so that
        deployments and tests can override stored credentials.
        """
        env_var = _ENV_ALIASES.get(name, name.upper())
        value = self._environ.get(env_var)
        if value:
            return value
        return self._keyring_get(name)

    def require(self, name: str) -> str:
        """Return the secret value for ``name`` or raise ``KeyError``."""
        value = self.get(name)
        if not value:
            raise KeyError(
                f"Missing required secret '{name}'. Set the "
                f"{_ENV_ALIASES.get(name, name.upper())} environment variable "
                f"or store it in the OS keychain."
            )
        return value

    def set(self, name: str, value: str) -> bool:
        """Persist a secret to the OS keychain.

        Returns ``True`` if the secret was stored, ``False`` when keychain
        support is unavailable.
        """
        backend = self._keyring()
        if backend is None:
            return False
        backend.set_password(self._service, name, value)
        return True

    def delete(self, name: str) -> bool:
        """Remove a secret from the OS keychain, if present."""
        backend = self._keyring()
        if backend is None:
            return False
        try:
            backend.delete_password(self._service, name)
            return True
        except Exception:
            return False

    def get_anthropic_api_key(self) -> str | None:
        return self.get(ANTHROPIC_SECRET_KEY)

    # -- internal helpers -------------------------------------------------
    def _keyring(self):  # pragma: no cover - exercised via monkeypatch in tests
        if not self._use_keyring:
            return None
        try:
            import keyring
        except Exception:
            return None
        return keyring

    def _keyring_get(self, name: str) -> str | None:
        backend = self._keyring()
        if backend is None:
            return None
        try:
            return backend.get_password(self._service, name)
        except Exception:
            return None
