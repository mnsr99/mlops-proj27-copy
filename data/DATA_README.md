# Data Platform Contract & Operations Guide

## 1) Contract Freeze (canonical truth)

### Feedback record contract (`reviews`)

**Policy**: exactly **one canonical review row per meeting** (`reviews.meeting_id` is `UNIQUE`).

Required fields:
- `review_id` (UUID, PK)
- `meeting_id` (UUID, FK, UNIQUE)
- `reviewer_id` (TEXT, NOT NULL)
- `rating` (INT, 1..5, NOT NULL)
- `approved` (BOOLEAN, NOT NULL)
- `correction_label` (`none | minor | major | rewrite`, NOT NULL)
- `edited_summary` (TEXT, NOT NULL)
- `edited_action_items` (TEXT, NOT NULL)
- `created_at` (TIMESTAMPTZ)
- `updated_at` (TIMESTAMPTZ)

Optional:
- `review_notes` (TEXT, nullable)

### Meeting lifecycle enums

`meetings.status`:
- `scheduled`
- `in_progress`
- `completed`
- `failed`
- `canceled`

`meetings.asr_status`:
- `not_requested`
- `queued`
- `processing`
- `completed`
- `failed`

Both are enforced by database `CHECK` constraints.

### Approved feedback export duplicate policy

Default export policy is **`latest-per-meeting`** (latest approved review chosen deterministically).
Alternative policy: `all-approved`.

---

## 2) Environment variables and compose wiring

### API / pipelines DB config
All DB clients share `data/common/db.py`:
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`

`data/docker-compose.yml` wires these explicitly for the API service.

### Export delivery controls
- `FEEDBACK_EXPORT_PATH`
- `FEEDBACK_EXPORT_POLICY` (`latest-per-meeting` or `all-approved`)
- `TRAINING_HOST`
- `TRAINING_USER`
- `TRAINING_SSH_KEY_PATH`

### Transitional dataset mode
- `REQUIRE_TRANSCRIPT` (default true)
- `APPROVED_ONLY` (default true)

### ASR handoff controls
- `ASR_CLAIM_BATCH_SIZE`

---

## 3) Schema and migration order

Fresh bootstrap schema: `data/sql/schema.sql`.

Migration-safe path for existing DBs:
1. Apply `data/sql/migrations/001_feedback_audio_contract.sql`
2. Deploy API
3. Run pipelines in dry-run/manual mode
4. Enable scheduled exporter + ASR handoff

Migration details include:
- Reviews contract columns + constraints + indexes
- Meeting audio/ASR columns + enum checks
- Backfill of review text fields from latest summaries/action_items where missing
- Backfill of `asr_status` from transcript presence

---

## 4) API contracts

### Write endpoints
- `POST /meetings`
- `POST /transcripts`
- `POST /summaries`
- `POST /reviews`

`POST /reviews` hardening:
- Required retraining fields are enforced at write-time.
- Transcript/summary fallback is **enrichment only** for missing edited fields.
- If required fields are still missing after fallback, request fails with 4xx (422).

### Retrieval endpoints (stable)
- `GET /transcripts/{transcript_id}`
- `GET /transcripts/by_meeting/{meeting_id}`
- `GET /meetings/{meeting_id}`
- `GET /meetings/{meeting_id}/audio`

All preserve explicit `404` behavior when records are absent.

---

## 5) Export policy and manifest semantics

`data/pipelines/export_feedback_jsonl.py` exports approved feedback using deterministic dedupe and emits manifest metadata:
- policy used
- `records_exported`
- `malformed_records_skipped`
- `duplicate_records_skipped`
- `multi_review_meetings`
- delivery checksum/size
- transfer target (if SCP configured)

---

## 6) ASR handoff lifecycle and idempotency

`data/pipelines/handoff_asr_jobs.py` uses atomic claim semantics:
- `FOR UPDATE SKIP LOCKED`
- `UPDATE ... RETURNING`

This prevents concurrent runners from queueing the same meeting twice.

State transitions:
- `not_requested -> queued -> processing -> completed`
- `not_requested -> queued -> processing -> failed`
- `failed -> queued` (retry)

---

## 7) Transitional dataset build behavior

`data/pipelines/build_dataset.py` now uses deterministic `latest-*` CTEs to avoid fan-out joins.
Manifest includes skipped counters:
- `missing_transcript`
- `not_approved`
- `missing_targets`

---

## 8) Validators and milestone observability

Validators in `data/pipelines/validators/`:
- `validate_feedback_contract.py`
- `validate_split_leakage.py`

Milestone metrics to track operationally:
- transcript completeness
- approval/edit/rating distributions
- ASR queue lag/failure counts
- periodic data-drift snapshot across rating/correction-label distributions
