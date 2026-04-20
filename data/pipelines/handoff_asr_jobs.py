import json
import sys
from pathlib import Path
import os
import uuid
from datetime import UTC, datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import db_cursor


# State transitions:
# not_requested -> queued -> processing -> completed
# not_requested -> queued -> processing -> failed
# failed -> queued (retry)
# Idempotency rule: queue operation only claims rows with asr_status in (not_requested, failed)
# and no active asr_job_id. Atomic UPDATE ... FROM (FOR UPDATE SKIP LOCKED) prevents duplicate queueing.


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def main():
    batch_size = _env_int("ASR_CLAIM_BATCH_SIZE", 50)
    now = datetime.now(UTC)

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            WITH claimable AS (
                SELECT m.meeting_id
                FROM meetings m
                WHERE m.audio_object_key IS NOT NULL
                  AND m.asr_status IN ('not_requested', 'failed')
                  AND (m.asr_job_id IS NULL OR m.asr_status = 'failed')
                ORDER BY m.created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT %s
            )
            UPDATE meetings m
            SET asr_status = 'queued',
                asr_job_id = ('asr-' || md5(m.meeting_id::text || clock_timestamp()::text)),
                asr_requested_at = %s,
                asr_last_error = NULL
            FROM claimable c
            WHERE m.meeting_id = c.meeting_id
            RETURNING m.meeting_id, m.asr_job_id, m.audio_object_key, m.asr_requested_at
            """,
            (batch_size, now),
        )
        rows = cur.fetchall()

    jobs = [
        {
            "meeting_id": str(meeting_id),
            "asr_job_id": asr_job_id,
            "audio_object_key": audio_object_key,
            "queued_at": queued_at.isoformat() if queued_at else None,
        }
        for meeting_id, asr_job_id, audio_object_key, queued_at in rows
    ]

    payload = {
        "claimed_jobs": len(jobs),
        "jobs": jobs,
        "run_id": str(uuid.uuid4()),
        "claimed_at": now.isoformat(),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
