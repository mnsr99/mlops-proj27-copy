import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if p.suffix.lower() == ".jsonl":
        records = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError("JSON file must contain either one object or a list of objects.")

    raise ValueError("Supported input file types: .json, .jsonl")


def choose_target_summary(record: Dict[str, Any], min_rating_for_unedited: int) -> Tuple[bool, str]:
    transcript = str(record.get("transcript", "")).strip()
    original_summary = str(record.get("original_summary", "")).strip()
    edited_summary = str(record.get("edited_summary", "")).strip()

    rating = record.get("rating")
    edited_flag = bool(record.get("edited_flag", False))
    approved = bool(record.get("approved", True))

    if not approved:
        return False, ""

    if not transcript:
        return False, ""

    if edited_flag:
        if edited_summary:
            return True, edited_summary
        return False, ""

    # 没编辑但评分高，也可以作为弱监督样本
    if rating is not None and int(rating) >= min_rating_for_unedited and original_summary:
        return True, original_summary

    return False, ""


def convert_records(
    records: List[Dict[str, Any]],
    min_rating_for_unedited: int,
    min_transcript_chars: int,
    min_summary_chars: int,
) -> List[Dict[str, Any]]:
    output = []

    for r in records:
        transcript = str(r.get("transcript", "")).strip()
        keep, target_summary = choose_target_summary(r, min_rating_for_unedited)

        if not keep:
            continue

        if len(transcript) < min_transcript_chars:
            continue

        if len(target_summary.strip()) < min_summary_chars:
            continue

        output.append(
            {
                "meeting_id": r.get("meeting_id"),
                "input_transcript": transcript,
                "target_summary": target_summary.strip(),
                "target_action_items": r.get("edited_action_items") or r.get("original_action_items") or [],
                "rating": r.get("rating"),
                "edited_flag": bool(r.get("edited_flag", False)),
                "approved": bool(r.get("approved", True)),
                "created_at": r.get("created_at"),
                "data_source": "user_feedback",
            }
        )

    return output


def sort_records_for_time_split(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 没有 created_at 的会被排到前面；有的话按字符串排序
    return sorted(records, key=lambda x: str(x.get("created_at", "")))


def split_train_val(records: List[Dict[str, Any]], val_ratio: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not records:
        return [], []

    records = sort_records_for_time_split(records)
    n_total = len(records)
    n_val = max(1, int(n_total * val_ratio)) if n_total >= 5 else 0

    if n_val == 0:
        return records, []

    train_records = records[:-n_val]
    val_records = records[-n_val:]
    return train_records, val_records


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare retraining-ready dataset from user feedback.")
    parser.add_argument("--input", type=str, required=True, help="Path to feedback JSON/JSONL file")
    parser.add_argument("--train-output", type=str, required=True, help="Path to output retraining train JSONL")
    parser.add_argument("--val-output", type=str, default="", help="Optional path to validation JSONL")
    parser.add_argument("--min-rating-for-unedited", type=int, default=4)
    parser.add_argument("--min-transcript-chars", type=int, default=20)
    parser.add_argument("--min-summary-chars", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)

    args = parser.parse_args()

    records = load_json_records(args.input)
    converted = convert_records(
        records=records,
        min_rating_for_unedited=args.min_rating_for_unedited,
        min_transcript_chars=args.min_transcript_chars,
        min_summary_chars=args.min_summary_chars,
    )

    train_records, val_records = split_train_val(converted, args.val_ratio)

    write_jsonl(train_records, args.train_output)

    if args.val_output:
        write_jsonl(val_records, args.val_output)

    print(f"Loaded feedback records: {len(records)}")
    print(f"Converted retraining samples: {len(converted)}")
    print(f"Train samples written: {len(train_records)} -> {args.train_output}")

    if args.val_output:
        print(f"Validation samples written: {len(val_records)} -> {args.val_output}")


if __name__ == "__main__":
    main()
