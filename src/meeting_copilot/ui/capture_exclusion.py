"""Platform screen-capture exclusion for the private overlay.

The suggestions overlay must be visible only to the local user. While the app
is running it is excluded from screen sharing/recording using each platform's
capture-exclusion API:

* Windows: ``SetWindowDisplayAffinity`` with ``WDA_EXCLUDEFROMCAPTURE``
* macOS: ``NSWindow.sharingType = NSWindowSharingNone``
* Linux: best-effort hint; X11/Wayland cannot universally guarantee exclusion

This module contains no Qt dependency: it operates on a native window handle
and returns a structured result so callers can degrade gracefully and inform
the user when exclusion cannot be guaranteed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

# Windows constant from winuser.h. WDA_EXCLUDEFROMCAPTURE (0x11) requires
# Windows 10 version 2004+, otherwise WDA_MONITOR (0x01) is used as a fallback.
WDA_EXCLUDEFROMCAPTURE = 0x11
WDA_MONITOR = 0x01


@dataclass(frozen=True)
class ExclusionResult:
    """Outcome of attempting to exclude a window from screen capture."""

    excluded: bool
    platform: str
    detail: str

    @property
    def guaranteed(self) -> bool:
        """Whether the platform can guarantee capture exclusion."""
        return self.excluded and self.platform in {"windows", "macos"}


def current_platform() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def exclude_window_from_capture(window_handle: int | None) -> ExclusionResult:
    """Attempt to exclude a native window from screen capture.

    Args:
        window_handle: Native window id (HWND on Windows, NSView/NSWindow ptr on
            macOS). ``None`` means the handle is not yet available.

    Returns:
        An :class:`ExclusionResult` describing whether exclusion succeeded and
        whether it is guaranteed on this platform.
    """
    platform = current_platform()
    if window_handle is None:
        return ExclusionResult(False, platform, "No native window handle available yet.")

    if platform == "windows":
        return _exclude_windows(window_handle)
    if platform == "macos":
        return _exclude_macos(window_handle)
    return _exclude_linux(window_handle)


def _exclude_windows(hwnd: int) -> ExclusionResult:  # pragma: no cover - OS specific
    try:
        import ctypes

        user32 = ctypes.windll.user32
        if user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
            return ExclusionResult(True, "windows", "WDA_EXCLUDEFROMCAPTURE applied.")
        if user32.SetWindowDisplayAffinity(hwnd, WDA_MONITOR):
            return ExclusionResult(
                True, "windows", "Fell back to WDA_MONITOR (older Windows build)."
            )
        return ExclusionResult(False, "windows", "SetWindowDisplayAffinity failed.")
    except Exception as exc:
        return ExclusionResult(False, "windows", f"Exclusion error: {exc}")


def _exclude_macos(handle: int) -> ExclusionResult:  # pragma: no cover - OS specific
    try:
        import objc  # type: ignore
        from AppKit import NSWindowSharingNone  # type: ignore

        window = objc.objc_object(c_void_p=handle)
        window.setSharingType_(NSWindowSharingNone)
        return ExclusionResult(True, "macos", "NSWindowSharingNone applied.")
    except Exception as exc:
        return ExclusionResult(False, "macos", f"Exclusion error: {exc}")


def _exclude_linux(handle: int) -> ExclusionResult:  # pragma: no cover - OS specific
    # Neither X11 nor Wayland exposes a universally honored capture-exclusion
    # flag. We mark this as not guaranteed so the UI can warn the user.
    return ExclusionResult(
        False,
        "linux",
        "Capture exclusion is not guaranteed on Linux; warn the user and "
        "consider a separate virtual display for screen sharing.",
    )
