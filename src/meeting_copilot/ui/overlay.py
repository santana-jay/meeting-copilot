"""Private always-on-top suggestions overlay (PySide6).

The overlay is created only while the app is running and is excluded from
screen capture so other meeting participants never see it. Qt is imported
lazily; importing this module without PySide6 installed is safe.
"""

from __future__ import annotations

from .capture_exclusion import ExclusionResult, exclude_window_from_capture


class Overlay:
    """A small, frameless, always-on-top overlay window.

    Use :meth:`build` to construct the underlying Qt widget. Keeping the Qt
    object creation in a method (rather than ``__init__``) lets the class be
    imported and reasoned about without a running ``QApplication``.
    """

    def __init__(self, opacity: float = 0.92) -> None:
        self.opacity = opacity
        self._widget = None
        self._label = None

    def build(self):  # pragma: no cover - requires Qt + display
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

        widget = QWidget(None)
        widget.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        widget.setWindowOpacity(self.opacity)

        layout = QVBoxLayout(widget)
        label = QLabel("Meeting Copilot — listening…", widget)
        label.setWordWrap(True)
        layout.addWidget(label)

        self._widget = widget
        self._label = label
        return widget

    def apply_capture_exclusion(self) -> ExclusionResult:  # pragma: no cover - requires Qt
        """Exclude the overlay from screen capture once it has a handle."""
        handle = None
        if self._widget is not None:
            handle = int(self._widget.winId())
        return exclude_window_from_capture(handle)

    def set_text(self, text: str) -> None:  # pragma: no cover - requires Qt
        if self._label is not None:
            self._label.setText(text)

    def show(self) -> None:  # pragma: no cover - requires Qt
        if self._widget is not None:
            self._widget.show()

    def hide(self) -> None:  # pragma: no cover - requires Qt
        if self._widget is not None:
            self._widget.hide()

    def toggle(self) -> None:  # pragma: no cover - requires Qt
        if self._widget is None:
            return
        self._widget.hide() if self._widget.isVisible() else self._widget.show()
