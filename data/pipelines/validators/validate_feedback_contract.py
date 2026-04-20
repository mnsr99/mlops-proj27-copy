import argparse
import json
from collections import Counter

REQUIRED_FIELDS = {
    "review_id",
    "meeting_id",
    "approved",
    "rating",
    "correction_label",
    "edited_summary",
    "edited_action_items",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    seen = set()
    counters = Counter()

    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            counters["rows"] += 1
            row = json.loads(line)

            missing = [k for k in REQUIRED_FIELDS if k not in row or row[k] in (None, "")]
            if missing:
                counters["malformed"] += 1
                continue

            rid = row["review_id"]
            if rid in seen:
                counters["duplicate_review_id"] += 1
                continue
            seen.add(rid)

            counters["valid"] += 1

    print(json.dumps(counters, indent=2))


if __name__ == "__main__":
    main()
