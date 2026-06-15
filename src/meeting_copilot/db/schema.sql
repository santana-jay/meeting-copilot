-- Meeting Copilot local-first schema.
-- All audio stays on-device; only relevant transcript text is ever sent to AI.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meetings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT,
    started_at    REAL NOT NULL,
    ended_at      REAL,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id    INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    start_time    REAL NOT NULL,           -- seconds from meeting start
    end_time      REAL NOT NULL,
    speaker       TEXT,                    -- best-effort, may be NULL/unknown
    text          TEXT NOT NULL,
    is_final      INTEGER NOT NULL DEFAULT 1,
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_meeting
    ON transcript_segments(meeting_id, start_time);

CREATE TABLE IF NOT EXISTS notes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id    INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,           -- topic | decision | action_item | open_question
    content       TEXT NOT NULL,
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_notes_meeting ON notes(meeting_id, kind);

CREATE TABLE IF NOT EXISTS suggestions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id    INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    content       TEXT NOT NULL,
    confidence    REAL NOT NULL,
    abstained     INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_suggestions_meeting ON suggestions(meeting_id);

CREATE TABLE IF NOT EXISTS citations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_id INTEGER NOT NULL REFERENCES suggestions(id) ON DELETE CASCADE,
    segment_id    INTEGER REFERENCES transcript_segments(id) ON DELETE SET NULL,
    start_time    REAL,
    end_time      REAL,
    quote         TEXT
);

CREATE INDEX IF NOT EXISTS idx_citations_suggestion ON citations(suggestion_id);

CREATE TABLE IF NOT EXISTS embeddings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id    INTEGER NOT NULL REFERENCES transcript_segments(id) ON DELETE CASCADE,
    dim           INTEGER NOT NULL,
    vector        BLOB NOT NULL,           -- little-endian float32 array
    created_at    REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_segment ON embeddings(segment_id);
