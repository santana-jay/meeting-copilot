"""Command-line entry point for Meeting Copilot.

Running ``meeting-copilot`` launches the desktop tray app when PySide6 is
available, otherwise it prints a headless status summary so the core remains
usable for diagnostics in environments without a GUI.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .app import AppController
from .config import load_config
from .ui import is_ui_available


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meeting-copilot", description=__doc__)
    parser.add_argument("--config", help="Path to a TOML config file")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print configuration/status and exit (no GUI)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = load_config(args.config)
    controller = AppController(config=config)

    if args.status or not is_ui_available():
        _print_status(controller)
        controller.close()
        return 0

    return _run_gui(controller)  # pragma: no cover - requires Qt + display


def _print_status(controller: AppController) -> None:
    from .audio import available_backends

    config = controller.config
    has_key = bool(controller.secrets.get_anthropic_api_key())
    print("Meeting Copilot — headless status")
    print(f"  data dir:        {config.resolved_data_dir()}")
    print(f"  database:        {config.database_path()}")
    print(f"  anthropic model: {config.anthropic_model}")
    print(f"  anthropic key:   {'present' if has_key else 'MISSING'}")
    print(f"  llm mode:        {'anthropic' if has_key else 'mock (offline)'}")
    print(f"  stt backend:     {config.stt_backend}")
    print(f"  audio backend:   {config.audio_backend}")
    print(f"  audio available: {', '.join(available_backends()) or 'none'}")
    print(f"  ui available:    {is_ui_available()}")


def _run_gui(controller: AppController) -> int:  # pragma: no cover - requires Qt
    from .pipeline import CallbackSink
    from .ui.overlay import Overlay
    from .ui.tray import TrayApp

    overlay = Overlay(opacity=controller.config.overlay_opacity)
    overlay.build()
    overlay.apply_capture_exclusion()
    overlay.show()

    view = _OverlayView(overlay)
    controller.sink = CallbackSink(
        status=view.set_status,
        transcript=view.add_transcript,
        notes=view.add_notes,
        suggestion=view.set_suggestion,
    )

    tray = TrayApp(
        on_stop=controller.stop_meeting,
        on_pause=controller.pause,
        on_purge=controller.purge_all,
        on_toggle_overlay=overlay.toggle,
    )
    controller.start_live_capture()
    return tray.run()


class _OverlayView:  # pragma: no cover - requires Qt + display
    """Render incremental pipeline updates into the overlay label."""

    def __init__(self, overlay) -> None:
        self._overlay = overlay
        self._status = "Listening…"
        self._last_line = ""
        self._suggestion = ""

    def _render(self) -> None:
        parts = [self._status]
        if self._last_line:
            parts.append(f"❝ {self._last_line}")
        if self._suggestion:
            parts.append(f"💡 {self._suggestion}")
        self._overlay.set_text("\n".join(parts))

    def set_status(self, text: str) -> None:
        self._status = text
        self._render()

    def add_transcript(self, segment) -> None:
        self._last_line = segment.text
        self._render()

    def add_notes(self, notes) -> None:
        self._render()

    def set_suggestion(self, suggestion) -> None:
        cite = ""
        if suggestion.citations and suggestion.citations[0].quote:
            cite = f"  (cited: “{suggestion.citations[0].quote}”)"
        self._suggestion = f"{suggestion.content}{cite}"
        self._render()


if __name__ == "__main__":
    sys.exit(main())
