"""Meeting Copilot: a local-first background meeting co-pilot.

The package is organised so that the non-UI core (configuration, storage,
audio/STT abstractions, the Anthropic client, retrieval, and the grounded
intelligence pipeline) can be imported and unit-tested without any GUI,
audio, or network dependencies. Heavy/platform-specific integrations are
imported lazily by their respective backends.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
