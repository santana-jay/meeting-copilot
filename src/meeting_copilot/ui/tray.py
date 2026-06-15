"""System tray entry point and controls (PySide6).

Provides a tray icon with pause/stop/purge actions, a recording indicator,
and a hotkey-triggered overlay toggle. Qt is imported lazily.
"""

from __future__ import annotations

from collections.abc import Callable


class TrayApp:
    """Wires together the tray icon, overlay, and settings window.

    The class stores the callbacks and config it needs; the actual Qt objects
    are only created in :meth:`run`, keeping the module import-safe.
    """

    def __init__(
        self,
        *,
        on_pause: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        on_purge: Callable[[], None] | None = None,
        on_toggle_overlay: Callable[[], None] | None = None,
    ) -> None:
        self.on_pause = on_pause or (lambda: None)
        self.on_stop = on_stop or (lambda: None)
        self.on_purge = on_purge or (lambda: None)
        self.on_toggle_overlay = on_toggle_overlay or (lambda: None)
        self._recording = False

    @property
    def recording(self) -> bool:
        return self._recording

    def set_recording(self, value: bool) -> None:
        self._recording = value

    def run(self) -> int:  # pragma: no cover - requires Qt + display
        from PySide6.QtGui import QAction, QIcon
        from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

        app = QApplication.instance() or QApplication([])
        tray = QSystemTrayIcon(QIcon())
        menu = QMenu()

        toggle = QAction("Toggle overlay")
        toggle.triggered.connect(lambda: self.on_toggle_overlay())
        pause = QAction("Pause")
        pause.triggered.connect(lambda: self.on_pause())
        stop = QAction("Stop")
        stop.triggered.connect(lambda: self.on_stop())
        purge = QAction("Purge all data")
        purge.triggered.connect(lambda: self.on_purge())
        quit_action = QAction("Quit")
        quit_action.triggered.connect(app.quit)

        for action in (toggle, pause, stop, purge, quit_action):
            menu.addAction(action)
        tray.setContextMenu(menu)
        tray.setToolTip("Meeting Copilot")
        tray.show()
        return app.exec()
