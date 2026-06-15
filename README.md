# Meeting Copilot

## Implementation status

The application is now wired end-to-end. The non-UI core, the live processing
pipeline that connects every component, and a thin desktop shell are implemented
as an installable Python package under `src/meeting_copilot/`. The original
architecture proposal follows below and remains the source of truth for the
roadmap.

### What is built

- **Skeleton application (Milestone 2):** layered configuration, secret handling
  (env + OS keychain), SQLite schema/repository (meetings, transcript segments,
  notes, suggestions, citations, embeddings), an app controller with
  start/pause/stop/purge, and a headless-safe Qt UI shell (tray, private overlay
  with screen-capture exclusion, settings).
- **Live processing pipeline (`pipeline.py`):** the orchestration layer that ties
  the components together. For each meeting it runs
  audio capture → STT → persist segment → index embedding → grounded note
  extraction → retrieval over earlier snippets → citation-checked suggestion,
  emitting incremental updates to a sink (the overlay). It runs synchronously for
  offline/headless use and on a background thread for live capture. The
  `AppController` builds the pipeline from config (STT, audio, embeddings, and
  LLM backends), and `create_llm` selects the Anthropic client when an API key is
  present or a deterministic offline mock otherwise.
- **Non-UI core (Milestone-spanning interfaces):**
  - `audio/` — `AudioCapture` abstraction with Windows/macOS/Linux backends
    (lazy native deps, clear degraded-mode guidance) and a null backend.
  - `stt/` — pluggable `STTService` with a deterministic mock and a lazy
    `faster-whisper` backend.
  - `ai/` — `LLMClient` interface, an Anthropic Claude Messages client (model id
    supplied by config, not hardcoded), a scriptable mock, and a `create_llm`
    factory.
  - `retrieval/` — embedding interface with a dependency-free hashing embedder
    and a SQLite-backed cosine vector store.
  - `intelligence/` — JSON-schema validation, grounded note extraction, and
    citation-aware suggestions with grounding checks, confidence gating, and
    first-class abstention.

Real audio capture, streaming STT decoding, and the live Qt event loop require
their optional dependencies and a real desktop session; the rest of the core —
including the full pipeline orchestration — is fully unit-tested offline (see
`tests/`).

### Getting started

```bash
pip install -e ".[dev]"      # core + pytest/ruff
ruff check .                 # lint
pytest                       # run the offline test suite
python -m meeting_copilot --status   # headless config/status summary
```

Launching `python -m meeting_copilot` with PySide6 installed starts the tray
app, shows the private overlay, and begins live capture + processing; without a
GUI it prints the headless status summary instead.

Optional extras: `.[ui]` (PySide6), `.[ai]` (anthropic), `.[stt]`
(faster-whisper), `.[keyring]` (OS keychain).

Provide the Anthropic API key via the `ANTHROPIC_API_KEY` environment variable
or the OS keychain — it is never written to config or the database.

---

## Architecture proposal and stack confirmation

This repository is currently in the **plan-first** stage for a cross-platform
background meeting co-pilot. The initial implementation should proceed only
after this architecture and stack are explicitly approved.


### Proposed stack

- **Language:** Python 3.11+
- **Desktop UI / tray / overlay:** PySide6 (Qt)
- **Local storage:** SQLite
- **Embeddings / retrieval:** SQLite-backed vector search (`sqlite-vec`) or FAISS
- **Speech-to-text:** pluggable STT interface with local streaming Whisper
  (`faster-whisper`) as the default implementation
- **AI integration:** Anthropic Claude Messages API with streaming
- **Secrets:** environment variables and OS keychain only

### Architecture overview

1. **Desktop shell**
   - PySide6 system tray app
   - Global hotkey to toggle a small always-on-top overlay
   - Settings, pause/stop/purge controls, and a visible recording indicator
   - The overlay is **private to the local user only**: while running it must be
     excluded from screen sharing and screen recording so other meeting
     participants never see it, using each platform's screen-capture exclusion
     API (Windows `SetWindowDisplayAffinity` with `WDA_EXCLUDEFROMCAPTURE`,
     macOS `NSWindow.sharingType = .none`, and the equivalent capture-exclusion
     hint on Linux), and never appears when the app is not running

2. **Audio capture layer**
   - `AudioCapture` abstraction with runtime-selected platform backends
   - Windows: WASAPI loopback
   - macOS: ScreenCaptureKit with virtual-device fallback guidance
   - Linux: PulseAudio/PipeWire monitor source
   - Clear degraded-mode instructions when loopback capture is unavailable

3. **Streaming transcript pipeline**
   - Voice activity detection to chunk live audio
   - Pluggable streaming STT service
   - Incremental transcript persistence with timestamps and speaker turns where feasible

4. **Grounded intelligence pipeline**
   - Recent live transcript window + retrieved past transcript snippets only
   - Structured note extraction for topics, decisions, action items, and open questions
   - Suggestion generation with citations, confidence scores, and abstention when weakly grounded
   - JSON schema validation before UI display

5. **Local-first data model**
   - SQLite for meetings, transcript segments, notes, suggestions, citations, and embeddings
   - Audio never stored in the cloud
   - Only relevant transcript text sent to Anthropic

### Privacy and visibility requirements

- The suggestions overlay must be visible **only to the local user** while the
  app is running, and must be excluded from screen sharing/recording so other
  meeting participants cannot see it
- Use platform screen-capture exclusion APIs to enforce this (Windows
  `WDA_EXCLUDEFROMCAPTURE`, macOS `NSWindowSharingNone`, Linux capture-exclusion
  hints) and degrade gracefully where the platform cannot guarantee exclusion
- The overlay and any windows must not appear when the app is not running
- Audio never leaves the machine; only relevant transcript text is sent to Anthropic

### Anti-hallucination requirements

- Never claim zero hallucination
- Ground every suggestion in retrieved transcript context
- Require citation spans/timestamps in suggestion output
- Prefer “no confident suggestion” over guessing
- Validate structured JSON outputs and drop malformed or ungrounded responses
- Treat notes as extraction of what was said and suggestions as AI proposals
- Mark unknown names, numbers, and quotes as unknown instead of inventing them

### Milestone plan

1. **Plan and approval**
   - Confirm architecture, stack, and privacy boundaries
2. **Skeleton application**
   - Tray app, overlay, settings window, config loading, secret handling, SQLite schema
3. **Cross-platform audio capture**
   - Prototype and document per-OS loopback capture and graceful fallback behavior
4. **Streaming transcription**
   - STT pipeline into a live transcript view and local persistence
5. **Structured notes**
   - Incremental note extraction through Claude with low-temperature prompts
6. **Grounded suggestions**
   - Retrieval over past transcripts, citation-aware suggestion generation, confidence gating
7. **Polish**
   - Hotkeys, packaging, setup docs, and tests for non-UI core components

### Implementation notes

- The highest-risk milestone is cross-platform loopback audio capture, so it should be
  prototyped before deeper AI/UI work.
- Anthropic model IDs, limits, and pricing must be verified against
  `docs.claude.com` during implementation rather than hardcoded from memory.
- Tests should focus on non-UI core behavior: platform audio abstraction,
  retrieval and grounding, structured output validation, and the Anthropic client.
