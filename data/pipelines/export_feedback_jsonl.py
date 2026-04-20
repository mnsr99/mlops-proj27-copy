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


def _truthy(name: str, default: str = "0") -> bool:
    return _env(name, default).strip().lower() in {"1", "true", "yes", "on"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_transfer(local_path: str):
    host = _env("TRAINING_HOST")
    user = _env("TRAINING_USER")
    key_path = _env("TRAINING_SSH_KEY_PATH")
    remote_path = _env(
        "FEEDBACK_EXPORT_PATH",
        "/home/cc/mlops-proj27/training/data/feedback_records.jsonl",
    )

    if not host or not user or not key_path:
        return None

    remote_dir = str(Path(remote_path).parent)
    subprocess.run(
        [
            "ssh",
            "-i",
            key_path,
            f"{user}@{host}",
            f"mkdir -p {remote_dir}",
        ],
        check=True,
    )
    target = f"{user}@{host}:{remote_path}"
    subprocess.run(["scp", "-i", key_path, local_path, target], check=True)
    return target


def main():
    export_policy = _env("FEEDBACK_EXPORT_POLICY", "all-approved")
    local_export_path = _env(
        "LOCAL_FEEDBACK_EXPORT_PATH",
        f"output/feedback_export_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.jsonl",
    )
    os.makedirs(os.path.dirname(local_export_path) or ".", exist_ok=True)

    malformed = 0
    duplicates = 0
    multi_review_meetings = 0
    require_transcript = _truthy("REQUIRE_TRANSCRIPT", "1")

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
                        r.edited_summary,
                        r.edited_action_items,
                        r.created_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY r.meeting_id
                            ORDER BY r.created_at DESC, r.review_id DESC
                        ) AS rn
                    FROM reviews r
                    WHERE r.approved = TRUE
                ),
                latest_transcript AS (
                    SELECT DISTINCT ON (meeting_id)
                        meeting_id,
                        transcript_text,
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
                )
                SELECT
                    ranked.review_id,
                    ranked.meeting_id,
                    lt.transcript_text,
                    ls.summary_text AS original_summary,
                    la.item_text AS original_action_items,
                    ranked.edited_summary,
                    ranked.edited_action_items,
                    ranked.rating,
                    ranked.approved,
                    ranked.created_at
                FROM ranked
                LEFT JOIN latest_transcript lt ON lt.meeting_id = ranked.meeting_id
                LEFT JOIN latest_summary ls ON ls.meeting_id = ranked.meeting_id
                LEFT JOIN latest_action_items la ON la.meeting_id = ranked.meeting_id
                WHERE ranked.rn = 1
                ORDER BY ranked.created_at ASC, ranked.review_id ASC
                """
            )
        elif export_policy == "all-approved":
            cur.execute(
                """
                WITH latest_transcript AS (
                    SELECT DISTINCT ON (meeting_id)
                        meeting_id,
                        transcript_text,
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
                )
                SELECT
                    r.review_id,
                    r.meeting_id,
                    lt.transcript_text,
                    ls.summary_text AS original_summary,
                    la.item_text AS original_action_items,
                    r.edited_summary,
                    r.edited_action_items,
                    r.rating,
                    r.approved,
                    r.created_at
                FROM reviews r
                LEFT JOIN latest_transcript lt ON lt.meeting_id = r.meeting_id
                LEFT JOIN latest_summary ls ON ls.meeting_id = r.meeting_id
                LEFT JOIN latest_action_items la ON la.meeting_id = r.meeting_id
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
            transcript,
            original_summary,
            original_action_items,
            edited_summary,
            edited_action_items,
            rating,
            approved,
            created_at,
        ) = row

        if str(review_id) in seen_review_ids:
            duplicates += 1
            continue
        seen_review_ids.add(str(review_id))

        if require_transcript and not transcript:
            malformed += 1
            continue
        if not original_summary or not edited_summary:
            malformed += 1
            continue

        edited_flag = bool(edited_summary and edited_summary.strip())

        exported.append(
            {
                "meeting_id": str(meeting_id),
                "transcript": transcript,
                "original_summary": original_summary,
                "original_action_items": original_action_items,
                "edited_summary": edited_summary,
                "edited_action_items": edited_action_items,
                "rating": rating,
                "edited_flag": edited_flag,
                "approved": approved,
                "created_at": created_at.isoformat() if created_at else None,
            }
        )

    with open(local_export_path, "w", encoding="utf-8") as f:
        for row in exported:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    export_size_bytes = os.path.getsize(local_export_path)
    export_sha256 = sha256_file(local_export_path)
    transfer_target = maybe_transfer(local_export_path)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "local_export_path": local_export_path,
        "remote_export_path": _env(
            "FEEDBACK_EXPORT_PATH",
            "/home/cc/mlops-proj27/training/data/feedback_records.jsonl",
        ),
        "export_policy": export_policy,
        "records_exported": len(exported),
        "malformed_records_skipped": malformed,
        "duplicate_records_skipped": duplicates,
        "multi_review_meetings": multi_review_meetings,
        "export_size_bytes": export_size_bytes,
        "export_sha256": export_sha256,
        "transfer_target": transfer_target,
        "delivery_controls": {
            "TRAINING_HOST": _env("TRAINING_HOST"),
            "TRAINING_USER": _env("TRAINING_USER"),
            "TRAINING_SSH_KEY_PATH": _env("TRAINING_SSH_KEY_PATH"),
            "FEEDBACK_EXPORT_PATH": _env(
                "FEEDBACK_EXPORT_PATH",
                "/home/cc/mlops-proj27/training/data/feedback_records.jsonl",
            ),
            "LOCAL_FEEDBACK_EXPORT_PATH": _env("LOCAL_FEEDBACK_EXPORT_PATH"),
            "REQUIRE_TRANSCRIPT": require_transcript,
        },
    }

    manifest_path = f"{local_export_path}.manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({"export_path": local_export_path, "manifest_path": manifest_path}))


if __name__ == "__main__":
    main()
