"""Platform-specific audio capture backends.

Each backend wraps a platform loopback API and imports its native dependency
lazily inside :meth:`start`/`is_available`, so importing this package is safe
on any OS and in headless/test environments.
"""

from __future__ import annotations

from .linux import LinuxMonitorCapture
from .macos import MacScreenCaptureKitCapture
from .windows import WindowsLoopbackCapture

__all__ = [
    "WindowsLoopbackCapture",
    "MacScreenCaptureKitCapture",
    "LinuxMonitorCapture",
]
