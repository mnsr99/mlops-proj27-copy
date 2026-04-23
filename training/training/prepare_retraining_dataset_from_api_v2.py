import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


DATA_API_BASE = os.environ.get("DATA_API_BASE", "http://129.114.27.10:30800")
MEETING_MANIFEST_PATH = Path(os.environ.get("MEETING_MANIFEST_PATH", "data/meeting_manifest.jsonl"))

OUTPUT_DIR = Path(os.environ.get("RETRAIN_DATA_DIR", "data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = OUTPUT_DIR / "retraining_ready_train.jsonl"
VAL_PATH = OUTPUT_DIR / "retraining_ready_val.jsonl"
TEST_PATH = OUTPUT_DIR / "retraining_ready_test.jsonl"
STATS_PATH = OUTPUT_DIR / "retraining_stats.json"

SEED = int(os.environ.get("DATASET_SPLIT_SEED", "42"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.2"))
TEST_RATIO = float(os.environ.get("TEST_RATIO", "0.1"))

MIN_RATING = int(os.environ.get("MIN_RATING", "4"))
REQUIRE_APPROVED = os.environ.get("REQUIRE_APPROVED", "true").lower() == "true"
REQUIRE_EDITED_SUMMARY = os.environ.get("REQUIRE_EDITED_SUMMARY", "true").lower() == "true"
MIN_SUMMARY_CHARS = int(os.environ.get("MIN_SUMMARY_CHARS", "10"))

REQUEST_TIMEOUT = (10, 120)


def get_json(url: str) -> Any:
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    if not resp.ok:
        raise RuntimeError(f"GET failed: {url} | status={resp.status_code} | body={resp.text}")
    try:
        return resp.json()
    except Exception:
        return resp.text


def normalize_payload_to_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return {"raw_text": payload}


def normalize_payload_to_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("items", "results", "data", "reviews"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [payload]
    return []


def read_meeting_ids_from_manifest(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Meeting manifest not found: {path}. "
            f"You need run_online_pipeline.py to append meeting_ids first."
        )

    meeting_ids: List[str] = []
    seen = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meeting_id = str(row.get("meeting_id", "")).strip()
            if meeting_id and meeting_id not in seen:
                seen.add(meeting_id)
                meeting_ids.append(meeting_id)

    return meeting_ids


def get_meeting(meeting_id: str) -> Dict[str, Any]:
    return normalize_payload_to_dict(get_json(f"{DATA_API_BASE}/meetings/{meeting_id}"))


def get_reviews_by_meeting(meeting_id: str) -> List[Dict[str, Any]]:
    return normalize_payload_to_list(get_json(f"{DATA_API_BASE}/reviews/by_meeting/{meeting_id}"))


def get_transcript_by_meeting(meeting_id: str) -> Dict[str, Any]:
    return normalize_payload_to_dict(get_json(f"{DATA_API_BASE}/transcripts/by_meeting/{meeting_id}"))


def get_summary_by_meeting(meeting_id: str) -> Dict[str, Any]:
    return normalize_payload_to_dict(get_json(f"{DATA_API_BASE}/summaries/by_meeting/{meeting_id}"))


def normalize_action_items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]


def review_fingerprint(meeting_id: str, review: Dict[str, Any]) -> str:
    payload = {
        "meeting_id": meeting_id,
        "review_id": str(review.get("review_id", "") or ""),
        "reviewer_id": str(review.get("reviewer_id", "") or ""),
        "rating": int(review.get("rating", 0) or 0),
        "approved": bool(review.get("approved", False)),
        "edited_summary": str(review.get("edited_summary", "") or "").strip(),
        "edited_action_items": normalize_action_items(review.get("edited_action_items")),
        "review_notes": str(review.get("review_notes", "") or "").strip(),
        "correction_label": str(review.get("correction_label", "") or "").strip(),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_example(
    meeting_id: str,
    review: Dict[str, Any],
    transcript_record: Dict[str, Any],
    summary_record: Optional[Dict[str, Any]] = None,
    meeting_record: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    approved = bool(review.get("approved", False))
    rating = int(review.get("rating", 0) or 0)
    edited_summary = str(review.get("edited_summary", "") or "").strip()
    correction_label = str(review.get("correction_label", "") or "").strip()
    reviewer_id = str(review.get("reviewer_id", "") or "").strip()
    review_notes = str(review.get("review_notes", "") or "").strip()
    edited_action_items = normalize_action_items(review.get("edited_action_items"))

    if REQUIRE_APPROVED and not approved:
        return None
    if rating < MIN_RATING:
        return None
    if REQUIRE_EDITED_SUMMARY and not edited_summary:
        return None
    if len(edited_summary) < MIN_SUMMARY_CHARS:
        return None

    transcript_text = str(
        transcript_record.get("transcript_text")
        or transcript_record.get("transcript")
        or transcript_record.get("text")
        or ""
    ).strip()
    if not transcript_text:
        return None

    original_summary = ""
    if summary_record:
        original_summary = str(
            summary_record.get("summary_text")
            or summary_record.get("summary")
            or ""
        ).strip()

    audio_object_key = ""
    if meeting_record:
        audio_object_key = str(meeting_record.get("audio_object_key", "") or "").strip()

    fp = review_fingerprint(meeting_id, review)

    return {
        "meeting_id": meeting_id,
        "input_transcript": transcript_text,
        "target_summary": edited_summary,
        "original_summary": original_summary,
        "rating": rating,
        "approved": approved,
        "correction_label": correction_label,
        "reviewer_id": reviewer_id,
        "review_notes": review_notes,
        "edited_action_items": edited_action_items,
        "audio_object_key": audio_object_key,
        "source": "reviews_api",
        "source_review_fingerprint": fp,
    }


def split_examples(examples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ex in examples:
        grouped.setdefault(ex["meeting_id"], []).append(ex)

    meeting_ids = list(grouped.keys())
    rng = random.Random(SEED)
    rng.shuffle(meeting_ids)

    # Preferred: meeting-level split.
    if len(meeting_ids) >= 2:
        n = len(meeting_ids)
        n_test = int(round(n * TEST_RATIO))
        n_val = int(round(n * VAL_RATIO))

        if n >= 3:
            if TEST_RATIO > 0:
                n_test = max(1, n_test)
            if VAL_RATIO > 0:
                n_val = max(1, n_val)

        if n_test + n_val >= n:
            n_test = 0
            n_val = max(1, min(n - 1, n_val))

        test_ids = set(meeting_ids[:n_test])
        val_ids = set(meeting_ids[n_test:n_test + n_val])
        train_ids = set(meeting_ids[n_test + n_val:])

        if not train_ids and meeting_ids:
            last_id = meeting_ids[-1]
            train_ids.add(last_id)
            val_ids.discard(last_id)
            test_ids.discard(last_id)

        train_rows, val_rows, test_rows = [], [], []
        for meeting_id, rows in grouped.items():
            if meeting_id in test_ids:
                test_rows.extend(rows)
            elif meeting_id in val_ids:
                val_rows.extend(rows)
            else:
                train_rows.extend(rows)

        # Safety fallback if val accidentally empty.
        if not val_rows and train_rows:
            val_rows = train_rows[:1]
        return train_rows, val_rows, test_rows, "meeting_level"

    # Fallback for testing when only one meeting has feedback.
    rows = examples[:]
    rng.shuffle(rows)
    if len(rows) == 1:
        return rows, rows, [], "single_example_fallback"
    val_count = max(1, int(round(len(rows) * VAL_RATIO)))
    if val_count >= len(rows):
        val_count = 1
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]
    if not train_rows:
        train_rows = rows[1:]
        val_rows = rows[:1]
    return train_rows, val_rows, [], "single_meeting_example_fallback"


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_fingerprint(examples: List[Dict[str, Any]]) -> str:
    fps = sorted(str(x.get("source_review_fingerprint", "")) for x in examples)
    raw = "\n".join(fps)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-empty", action="store_true", help="Write empty output files instead of failing when no eligible examples exist.")
    args = parser.parse_args()

    meeting_ids = read_meeting_ids_from_manifest(MEETING_MANIFEST_PATH)
    print(f"[INFO] Loaded {len(meeting_ids)} meeting_id(s) from manifest")

    examples: List[Dict[str, Any]] = []
    skipped_meetings = 0
    total_reviews_seen = 0
    errors: List[str] = []

    for meeting_id in meeting_ids:
        try:
            meeting_record = get_meeting(meeting_id)
            transcript_record = get_transcript_by_meeting(meeting_id)

            try:
                summary_record = get_summary_by_meeting(meeting_id)
            except Exception:
                summary_record = {}

            reviews = get_reviews_by_meeting(meeting_id)
            total_reviews_seen += len(reviews)

            for review in reviews:
                ex = build_example(
                    meeting_id=meeting_id,
                    review=review,
                    transcript_record=transcript_record,
                    summary_record=summary_record,
                    meeting_record=meeting_record,
                )
                if ex is not None:
                    examples.append(ex)

        except Exception as e:
            skipped_meetings += 1
            errors.append(f"{meeting_id}: {e}")
            print(f"[WARN] Skipping meeting_id={meeting_id} due to error: {e}")

    # de-dupe by review fingerprint
    deduped: Dict[str, Dict[str, Any]] = {}
    for ex in examples:
        deduped[ex["source_review_fingerprint"]] = ex
    examples = list(deduped.values())

    if not examples and not args.write_empty:
        raise RuntimeError(
            "No eligible retraining examples were built from API data. "
            "Check manifest, reviews, transcripts, and filtering thresholds."
        )

    if examples:
        train_rows, val_rows, test_rows, split_mode = split_examples(examples)
    else:
        train_rows, val_rows, test_rows, split_mode = [], [], [], "empty"

    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(VAL_PATH, val_rows)
    write_jsonl(TEST_PATH, test_rows)

    stats = {
        "meeting_manifest_path": str(MEETING_MANIFEST_PATH),
        "meeting_ids_in_manifest": len(meeting_ids),
        "skipped_meetings": skipped_meetings,
        "total_reviews_seen": total_reviews_seen,
        "eligible_examples": len(examples),
        "unique_meetings_used": len({x["meeting_id"] for x in examples}),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "test_examples": len(test_rows),
        "dataset_fingerprint": dataset_fingerprint(examples) if examples else "",
        "split_mode": split_mode,
        "train_path": str(TRAIN_PATH),
        "val_path": str(VAL_PATH),
        "test_path": str(TEST_PATH),
        "errors": errors,
    }

    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[INFO] Retraining dataset generation finished.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
