CREATE TABLE IF NOT EXISTS meetings (
    meeting_id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    audio_object_key TEXT NOT NULL,
    audio_checksum TEXT,
    audio_duration_seconds INT,
    status TEXT NOT NULL DEFAULT 'scheduled',
    asr_status TEXT NOT NULL DEFAULT 'not_requested',
    asr_job_id TEXT,
    asr_last_error TEXT,
    asr_requested_at TIMESTAMPTZ,
    asr_completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT meetings_status_check CHECK (status IN ('scheduled', 'in_progress', 'completed', 'failed', 'canceled')),
    CONSTRAINT meetings_asr_status_check CHECK (asr_status IN ('not_requested', 'queued', 'processing', 'completed', 'failed'))
);

CREATE TABLE IF NOT EXISTS transcripts (
    transcript_id UUID PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(meeting_id),
    transcript_text TEXT,
    transcript_object_key TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS summaries (
    summary_id UUID PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(meeting_id),
    model_version TEXT,
    summary_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS action_items (
    action_item_id UUID PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(meeting_id),
    item_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id UUID PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(meeting_id),
    reviewer_id TEXT NOT NULL,
    rating INT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    approved BOOLEAN NOT NULL,
    correction_label TEXT NOT NULL CHECK (correction_label IN ('none', 'minor', 'major', 'rewrite')),
    edited_summary TEXT NOT NULL,
    edited_action_items TEXT NOT NULL,
    review_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reviews_meeting_id ON reviews (meeting_id);
CREATE INDEX IF NOT EXISTS idx_reviews_approved_created ON reviews (approved, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_transcripts_meeting_created ON transcripts (meeting_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_meeting_created ON summaries (meeting_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_action_items_meeting_created ON action_items (meeting_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_meetings_asr_status ON meetings (asr_status, created_at ASC);

CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_version TEXT PRIMARY KEY,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
