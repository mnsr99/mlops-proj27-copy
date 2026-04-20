import sys
import uuid
from pathlib import Path
from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import db_cursor


class MeetingStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class AsrStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CorrectionLabel(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"
    REWRITE = "rewrite"


app = FastAPI()


class MeetingCreate(BaseModel):
    source: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    audio_object_key: str
    status: MeetingStatus = MeetingStatus.SCHEDULED
    asr_status: AsrStatus = AsrStatus.NOT_REQUESTED


class TranscriptCreate(BaseModel):
    meeting_id: str
    transcript_text: str
    transcript_object_key: str


class SummaryCreate(BaseModel):
    meeting_id: str
    model_version: str
    summary_text: str
    action_item_text: str


class ReviewCreate(BaseModel):
    meeting_id: str
    reviewer_id: str
    rating: int = Field(ge=1, le=5)
    approved: bool
    correction_label: CorrectionLabel
    edited_summary: Optional[str] = None
    edited_action_items: Optional[str] = None
    review_notes: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/meetings")
def create_meeting(payload: MeetingCreate):
    meeting_id = str(uuid.uuid4())
    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            INSERT INTO meetings (
                meeting_id, source, started_at, ended_at, audio_object_key, status, asr_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                meeting_id,
                payload.source,
                payload.started_at,
                payload.ended_at,
                payload.audio_object_key,
                payload.status.value,
                payload.asr_status.value,
            ),
        )
    return {"meeting_id": meeting_id}


@app.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: str):
    with db_cursor() as (_, cur):
        cur.execute(
            """
            SELECT meeting_id, source, started_at, ended_at, audio_object_key, status, asr_status,
                   asr_job_id, asr_last_error, created_at
            FROM meetings
            WHERE meeting_id = %s
            """,
            (meeting_id,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="meeting not found")

    return {
        "meeting_id": str(row[0]),
        "source": row[1],
        "started_at": row[2],
        "ended_at": row[3],
        "audio_object_key": row[4],
        "status": row[5],
        "asr_status": row[6],
        "asr_job_id": row[7],
        "asr_last_error": row[8],
        "created_at": row[9],
    }


@app.get("/meetings/{meeting_id}/audio")
def get_meeting_audio(meeting_id: str):
    with db_cursor() as (_, cur):
        cur.execute(
            """
            SELECT meeting_id, audio_object_key, audio_checksum, audio_duration_seconds
            FROM meetings
            WHERE meeting_id = %s
            """,
            (meeting_id,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="meeting not found")
    return {
        "meeting_id": str(row[0]),
        "audio_object_key": row[1],
        "audio_checksum": row[2],
        "audio_duration_seconds": row[3],
    }


@app.post("/transcripts")
def create_transcript(payload: TranscriptCreate):
    transcript_id = str(uuid.uuid4())
    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            INSERT INTO transcripts (
                transcript_id, meeting_id, transcript_text, transcript_object_key
            ) VALUES (%s, %s, %s, %s)
            """,
            (
                transcript_id,
                payload.meeting_id,
                payload.transcript_text,
                payload.transcript_object_key,
            ),
        )
    return {"transcript_id": transcript_id}


@app.get("/transcripts/{transcript_id}")
def get_transcript(transcript_id: str):
    with db_cursor() as (_, cur):
        cur.execute(
            """
            SELECT transcript_id, meeting_id, transcript_text, transcript_object_key, created_at
            FROM transcripts
            WHERE transcript_id = %s
            """,
            (transcript_id,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="transcript not found")

    return {
        "transcript_id": str(row[0]),
        "meeting_id": str(row[1]),
        "transcript_text": row[2],
        "transcript_object_key": row[3],
        "created_at": row[4],
    }


@app.get("/transcripts/by_meeting/{meeting_id}")
def get_transcript_by_meeting(meeting_id: str):
    with db_cursor() as (_, cur):
        cur.execute(
            """
            SELECT transcript_id, meeting_id, transcript_text, transcript_object_key, created_at
            FROM transcripts
            WHERE meeting_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (meeting_id,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="transcript not found")
    return {
        "transcript_id": str(row[0]),
        "meeting_id": str(row[1]),
        "transcript_text": row[2],
        "transcript_object_key": row[3],
        "created_at": row[4],
    }


@app.post("/summaries")
def create_summary(payload: SummaryCreate):
    summary_id = str(uuid.uuid4())
    action_item_id = str(uuid.uuid4())
    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            INSERT INTO summaries (
                summary_id, meeting_id, model_version, summary_text
            ) VALUES (%s, %s, %s, %s)
            """,
            (
                summary_id,
                payload.meeting_id,
                payload.model_version,
                payload.summary_text,
            ),
        )
        cur.execute(
            """
            INSERT INTO action_items (
                action_item_id, meeting_id, item_text
            ) VALUES (%s, %s, %s)
            """,
            (
                action_item_id,
                payload.meeting_id,
                payload.action_item_text,
            ),
        )
    return {"summary_id": summary_id, "action_item_id": action_item_id}


@app.post("/reviews")
def create_review(payload: ReviewCreate):
    # Enrichment fallback only: use latest model summary/action items if missing.
    edited_summary = payload.edited_summary
    edited_action_items = payload.edited_action_items

    with db_cursor() as (_, cur):
        if not edited_summary:
            cur.execute(
                """
                SELECT summary_text
                FROM summaries
                WHERE meeting_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (payload.meeting_id,),
            )
            row = cur.fetchone()
            edited_summary = row[0] if row else None

        if not edited_action_items:
            cur.execute(
                """
                SELECT item_text
                FROM action_items
                WHERE meeting_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (payload.meeting_id,),
            )
            row = cur.fetchone()
            edited_action_items = row[0] if row else None

    if not edited_summary or not edited_action_items:
        raise HTTPException(
            status_code=422,
            detail="edited_summary and edited_action_items are required (directly or via fallback artifacts)",
        )

    review_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            INSERT INTO reviews (
                review_id,
                meeting_id,
                reviewer_id,
                rating,
                approved,
                correction_label,
                edited_summary,
                edited_action_items,
                review_notes,
                created_at,
                updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (meeting_id)
            DO UPDATE SET
                reviewer_id = EXCLUDED.reviewer_id,
                rating = EXCLUDED.rating,
                approved = EXCLUDED.approved,
                correction_label = EXCLUDED.correction_label,
                edited_summary = EXCLUDED.edited_summary,
                edited_action_items = EXCLUDED.edited_action_items,
                review_notes = EXCLUDED.review_notes,
                updated_at = EXCLUDED.updated_at
            RETURNING review_id
            """,
            (
                review_id,
                payload.meeting_id,
                payload.reviewer_id,
                payload.rating,
                payload.approved,
                payload.correction_label.value,
                edited_summary,
                edited_action_items,
                payload.review_notes,
                now,
                now,
            ),
        )
        stored_review_id = cur.fetchone()[0]
    return {"review_id": str(stored_review_id)}
