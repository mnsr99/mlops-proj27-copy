import json
import sys
from pathlib import Path
import os
from datetime import UTC, datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import db_cursor


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    dataset_version = datetime.now(UTC).strftime("v%Y%m%d_%H%M%S")
    out_dir = f"output/{dataset_version}"
    os.makedirs(out_dir, exist_ok=True)

    require_transcript = _truthy("REQUIRE_TRANSCRIPT", "1")
    approved_only = _truthy("APPROVED_ONLY", "1")


    skipped = {
        "missing_transcript": 0,
        "not_approved": 0,
        "missing_targets": 0,
    }

    with db_cursor() as (_, cur):
        cur.execute(
            f"""
            WITH latest_transcript AS (
                SELECT DISTINCT ON (meeting_id)
                    meeting_id,
                    transcript_text,
                    transcript_object_key,
                    created_at
                FROM transcripts
                ORDER BY meeting_id, created_at DESC, transcript_id DESC
            ),
            latest_summary AS (
                SELECT DISTINCT ON (meeting_id)
                    meeting_id,
                    summary_text,
                    created_at
                FROM summaries
                ORDER BY meeting_id, created_at DESC, summary_id DESC
            ),
            latest_action_items AS (
                SELECT DISTINCT ON (meeting_id)
                    meeting_id,
                    item_text,
                    created_at
                FROM action_items
                ORDER BY meeting_id, created_at DESC, action_item_id DESC
            ),
            latest_review AS (
                SELECT DISTINCT ON (meeting_id)
                    review_id,
                    meeting_id,
                    rating,
                    approved,
                    edited_summary,
                    edited_action_items,
                    correction_label,
                    reviewer_id,
                    created_at
                FROM reviews
                ORDER BY meeting_id, created_at DESC, review_id DESC
            )
            SELECT
                m.meeting_id,
                t.transcript_text,
                s.summary_text,
                a.item_text,
                r.review_id,
                r.edited_summary,
                r.edited_action_items,
                r.rating,
                r.approved,
                r.correction_label,
                r.reviewer_id
            FROM meetings m
            LEFT JOIN latest_transcript t ON m.meeting_id = t.meeting_id
            LEFT JOIN latest_summary s ON m.meeting_id = s.meeting_id
            LEFT JOIN latest_action_items a ON m.meeting_id = a.meeting_id
            LEFT JOIN latest_review r ON m.meeting_id = r.meeting_id
            ORDER BY m.created_at ASC
            """
        )
        rows = cur.fetchall()

    examples = []
    for (
        meeting_id,
        transcript_text,
        model_summary,
        model_action_items,
        review_id,
        edited_summary,
        edited_action_items,
        rating,
        approved,
        correction_label,
        reviewer_id,
    ) in rows:
        if require_transcript and not transcript_text:
            skipped["missing_transcript"] += 1
            continue
        if approved_only and not approved:
            skipped["not_approved"] += 1
            continue

        target_summary = edited_summary or model_summary
        target_action_items = edited_action_items or model_action_items
        if not target_summary or not target_action_items:
            skipped["missing_targets"] += 1
            continue

        examples.append(
            {
                "dataset_version": dataset_version,
                "meeting_id": str(meeting_id),
                "review_id": str(review_id) if review_id else None,
                "input_transcript": transcript_text,
                "target_summary": target_summary,
                "target_action_items": target_action_items,
                "model_summary": model_summary,
                "model_action_items": model_action_items,
                "rating": rating,
                "approved": approved,
                "correction_label": correction_label,
                "reviewer_id": reviewer_id,
                "source": "production_feedback",
            }
        )

    n = len(examples)
    train_end = max(1, int(n * 0.7)) if n else 0
    val_end = max(train_end + 1, int(n * 0.85)) if n > 2 else n

    train = examples[:train_end]
    val = examples[train_end:val_end]
    test = examples[val_end:]

    write_jsonl(f"{out_dir}/train.jsonl", train)
    write_jsonl(f"{out_dir}/val.jsonl", val)
    write_jsonl(f"{out_dir}/test.jsonl", test)

    with open(f"{out_dir}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_version": dataset_version,
                "created_at": datetime.now(UTC).isoformat(),
                "num_examples": n,
                "train_count": len(train),
                "val_count": len(val),
                "test_count": len(test),
                "require_transcript": require_transcript,
                "approved_only": approved_only,
                "skipped": skipped,
            },
            f,
            indent=2,
        )

    with db_cursor(commit=True) as (_, cur):
        cur.execute(
            "INSERT INTO dataset_versions (dataset_version, source) VALUES (%s, %s)",
            (dataset_version, "production_feedback"),
        )

    print(dataset_version)


if __name__ == "__main__":
    main()
