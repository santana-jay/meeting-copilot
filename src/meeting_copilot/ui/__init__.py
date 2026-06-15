"""Thin desktop UI shell.

All Qt (PySide6) imports happen lazily inside functions/classes so that the
non-UI core remains importable in headless and test environments where Qt is
not installed. Import :func:`is_ui_available` to check availability first.
"""

from __future__ import annotations


def is_ui_available() -> bool:
    """Return ``True`` if the PySide6 GUI toolkit can be imported."""
    try:
        import PySide6  # noqa: F401
    except Exception:
        return False
    return True


__all__ = ["is_ui_available"]
