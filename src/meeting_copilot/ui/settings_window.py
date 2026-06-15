"""Settings window (PySide6).

Presents editable application settings and the Anthropic API key field. The
API key is written to the OS keychain via :class:`SecretStore`, never to the
config file. Qt is imported lazily.
"""

from __future__ import annotations

from ..config import AppConfig
from ..secrets_store import ANTHROPIC_SECRET_KEY, SecretStore


class SettingsWindow:
    """A minimal settings dialog backed by config + secret store."""

    def __init__(self, config: AppConfig, secrets: SecretStore) -> None:
        self.config = config
        self.secrets = secrets
        self._widget = None

    def save_api_key(self, api_key: str) -> bool:
        """Persist the Anthropic API key to the OS keychain."""
        return self.secrets.set(ANTHROPIC_SECRET_KEY, api_key)

    def build(self):  # pragma: no cover - requires Qt + display
        from PySide6.QtWidgets import (
            QFormLayout,
            QLineEdit,
            QPushButton,
            QWidget,
        )

        widget = QWidget(None)
        layout = QFormLayout(widget)

        model_edit = QLineEdit(self.config.anthropic_model)
        api_key_edit = QLineEdit()
        api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        existing = self.secrets.get_anthropic_api_key()
        if existing:
            api_key_edit.setPlaceholderText("•••• stored in keychain")

        hotkey_edit = QLineEdit(self.config.hotkey_toggle_overlay)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(lambda: self.save_api_key(api_key_edit.text()))

        layout.addRow("Anthropic model", model_edit)
        layout.addRow("Anthropic API key", api_key_edit)
        layout.addRow("Toggle overlay hotkey", hotkey_edit)
        layout.addRow(save_btn)

        self._widget = widget
        return widget
