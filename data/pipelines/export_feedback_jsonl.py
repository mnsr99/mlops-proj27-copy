import hashlib
import json
import sys
from pathlib import Path
import os
import subprocess
from datetime import UTC, datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import db_cursor


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_transfer(path: str):
    host = _env("TRAINING_HOST")
    user = _env("TRAINING_USER")
    key_path = _env("TRAINING_SSH_KEY_PATH")
    if not host or not user or not key_path:
        return None

    target = f"{user}@{host}:{os.path.basename(path)}"
    subprocess.run(["scp", "-i", key_path, path, target], check=True)
    return target


def main():
    export_policy = _env("FEEDBACK_EXPORT_POLICY", "latest-per-meeting")
    export_path = _env(
        "FEEDBACK_EXPORT_PATH",
        f"output/feedback_export_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.jsonl",
    )
    os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)

    malformed = 0
    duplicates = 0
    multi_review_meetings = 0

    with db_cursor() as (_, cur):
        cur.execute(
            """
            SELECT meeting_id, COUNT(*)
            FROM reviews
            GROUP BY meeting_id
            HAVING COUNT(*) > 1
            """
        )
        multi_review_meetings = len(cur.fetchall())

        if export_policy == "latest-per-meeting":
            cur.execute(
                """
                WITH ranked AS (
                    SELECT
                        r.review_id,
                        r.meeting_id,
                        r.rating,
                        r.approved,
                        r.correction_label,
                        r.edited_summary,
                        r.edited_action_items,
                        r.reviewer_id,
                        r.review_notes,
                        r.created_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY r.meeting_id
                            ORDER BY r.created_at DESC, r.review_id DESC
                        ) AS rn
                    FROM reviews r
                    WHERE r.approved = TRUE
                )
                SELECT
                    ranked.review_id,
                    ranked.meeting_id,
                    ranked.rating,
                    ranked.approved,
                    ranked.correction_label,
                    ranked.edited_summary,
                    ranked.edited_action_items,
                    ranked.reviewer_id,
                    ranked.review_notes,
                    ranked.created_at,
                    t.transcript_text
                FROM ranked
                LEFT JOIN LATERAL (
                    SELECT transcript_text
                    FROM transcripts
                    WHERE meeting_id = ranked.meeting_id
                    ORDER BY created_at DESC
                    LIMIT 1
                ) t ON TRUE
                WHERE ranked.rn = 1
                ORDER BY ranked.created_at ASC, ranked.review_id ASC
                """
            )
        elif export_policy == "all-approved":
            cur.execute(
                """
                SELECT
                    r.review_id,
                    r.meeting_id,
                    r.rating,
                    r.approved,
                    r.correction_label,
                    r.edited_summary,
                    r.edited_action_items,
                    r.reviewer_id,
                    r.review_notes,
                    r.created_at,
                    t.transcript_text
                FROM reviews r
                LEFT JOIN LATERAL (
                    SELECT transcript_text
                    FROM transcripts
                    WHERE meeting_id = r.meeting_id
                    ORDER BY created_at DESC
                    LIMIT 1
                ) t ON TRUE
                WHERE r.approved = TRUE
                ORDER BY r.created_at ASC, r.review_id ASC
                """
            )
        else:
            raise ValueError(f"unsupported FEEDBACK_EXPORT_POLICY: {export_policy}")
        rows = cur.fetchall()

    seen_review_ids = set()
    exported = []
    for row in rows:
        (
            review_id,
            meeting_id,
            rating,
            approved,
            correction_label,
            edited_summary,
            edited_action_items,
            reviewer_id,
            review_notes,
            created_at,
            transcript_text,
        ) = row

        if str(review_id) in seen_review_ids:
            duplicates += 1
            continue
        seen_review_ids.add(str(review_id))

        if not edited_summary or not edited_action_items:
            malformed += 1
            continue

        exported.append(
            {
                "review_id": str(review_id),
                "meeting_id": str(meeting_id),
                "rating": rating,
                "approved": approved,
                "correction_label": correction_label,
                "edited_summary": edited_summary,
                "edited_action_items": edited_action_items,
                "reviewer_id": reviewer_id,
                "review_notes": review_notes,
                "transcript_text": transcript_text,
                "review_created_at": created_at.isoformat() if created_at else None,
            }
        )

    with open(export_path, "w", encoding="utf-8") as f:
        for row in exported:
            f.write(json.dumps(row) + "\n")

    export_size_bytes = os.path.getsize(export_path)
    export_sha256 = sha256_file(export_path)
    transfer_target = maybe_transfer(export_path)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "export_path": export_path,
        "export_policy": export_policy,
        "records_exported": len(exported),
        "malformed_records_skipped": malformed,
        "duplicate_records_skipped": duplicates,
        "multi_review_meetings": multi_review_meetings,
        "export_size_bytes": export_size_bytes,
        "export_sha256": export_sha256,
        "transfer_target": transfer_target,
        "delivery_controls": {
            "FEEDBACK_EXPORT_PATH": _env("FEEDBACK_EXPORT_PATH"),
            "TRAINING_HOST": _env("TRAINING_HOST"),
            "TRAINING_USER": _env("TRAINING_USER"),
            "TRAINING_SSH_KEY_PATH": _env("TRAINING_SSH_KEY_PATH"),
        },
    }

    manifest_path = f"{export_path}.manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({"export_path": export_path, "manifest_path": manifest_path}))


if __name__ == "__main__":
    main()
