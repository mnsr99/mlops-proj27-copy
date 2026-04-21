import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_INPUT = "data/feedback_records.jsonl"
DEFAULT_OUTPUT_ALL = "data/sample_retraining_data.jsonl"
DEFAULT_OUTPUT_TRAIN = "data/retraining_ready_train.jsonl"
DEFAULT_OUTPUT_VAL = "data/retraining_ready_val.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skip invalid JSON on line {i}: {e}")
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_action_items(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [clean_text(v) for v in x if clean_text(v)]
    if isinstance(x, str):
        s = clean_text(x)
        if not s:
            return []
        return [s]
    return []


def choose_target_summary(rec: Dict[str, Any]) -> str:
    edited_summary = clean_text(rec.get("edited_summary"))
    if edited_summary:
        return edited_summary

    approved = bool(rec.get("approved", False))
    if approved:
        for key in ("original_summary", "summary"):
            candidate = clean_text(rec.get(key))
            if candidate:
                return candidate

    return ""


def choose_input_transcript(rec: Dict[str, Any]) -> str:
    for key in ("transcript", "input_transcript", "transcript_text"):
        value = clean_text(rec.get(key))
        if value:
            return value
    return ""


def build_training_examples(records: List[Dict[str, Any]], min_rating: int) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    seen = set()

    for rec in records:
        rating = rec.get("rating", 0)
        try:
            rating_num = int(rating)
        except Exception:
            rating_num = 0

        if rating_num < min_rating:
            continue

        transcript = choose_input_transcript(rec)
        target_summary = choose_target_summary(rec)

        if not transcript or not target_summary:
            continue

        meeting_id = clean_text(rec.get("meeting_id")) or "unknown_meeting"

        original_summary = clean_text(rec.get("original_summary")) or clean_text(rec.get("summary"))
        original_action_items = normalize_action_items(rec.get("original_action_items"))
        edited_action_items = normalize_action_items(rec.get("edited_action_items"))

        row = {
            "meeting_id": meeting_id,
            "input_transcript": transcript,
            "target_summary": target_summary,
            "original_summary": original_summary,
            "rating": rating_num,
            "approved": bool(rec.get("approved", False)),
            "edited_flag": bool(rec.get("edited_flag", False)),
            "original_action_items": original_action_items,
            "edited_action_items": edited_action_items,
        }

        dedupe_key = (
            row["meeting_id"],
            row["input_transcript"],
            row["target_summary"],
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        examples.append(row)

    return examples


def split_examples(
    rows: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    if not rows:
        return [], []

    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)

    if n == 1:
        # 为了先把训练流程跑通，只有一条样本时 train/val 都放同一条
        return shuffled, shuffled

    val_count = max(1, int(round(n * val_ratio)))
    if val_count >= n:
        val_count = 1

    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]

    if not train_rows:
        train_rows = shuffled[1:]
        val_rows = shuffled[:1]

    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare retraining dataset from feedback JSONL")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input feedback JSONL")
    parser.add_argument("--output-all", type=str, default=DEFAULT_OUTPUT_ALL, help="Normalized all examples JSONL")
    parser.add_argument("--output-train", type=str, default=DEFAULT_OUTPUT_TRAIN, help="Train JSONL")
    parser.add_argument("--output-val", type=str, default=DEFAULT_OUTPUT_VAL, help="Validation JSONL")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-rating", type=int, default=0, help="Minimum rating to keep a feedback record")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_all = Path(args.output_all)
    output_train = Path(args.output_train)
    output_val = Path(args.output_val)

    print(f"[INFO] Reading feedback from: {input_path}")
    records = read_jsonl(input_path)
    print(f"[INFO] Raw feedback records: {len(records)}")

    examples = build_training_examples(records, min_rating=args.min_rating)
    print(f"[INFO] Usable training examples: {len(examples)}")

    if not examples:
        raise ValueError(
            "No usable retraining examples were created. "
            "Please check whether feedback_records.jsonl contains transcript and edited_summary "
            "(or approved original summary)."
        )

    train_rows, val_rows = split_examples(
        examples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    write_jsonl(output_all, examples)
    write_jsonl(output_train, train_rows)
    write_jsonl(output_val, val_rows)

    print(f"[INFO] Wrote normalized examples to: {output_all}")
    print(f"[INFO] Wrote train split to: {output_train} ({len(train_rows)} rows)")
    print(f"[INFO] Wrote val split to: {output_val} ({len(val_rows)} rows)")

    if examples:
        print("[INFO] Example output row:")
        print(json.dumps(examples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
