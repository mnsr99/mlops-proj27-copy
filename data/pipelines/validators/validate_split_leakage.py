import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()

    seen = defaultdict(set)
    for split, path in (("train", args.train), ("val", args.val), ("test", args.test)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                meeting_id = row.get("meeting_id")
                if meeting_id:
                    seen[meeting_id].add(split)

    leaked = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    report = {
        "meetings_checked": len(seen),
        "split_leakage_count": len(leaked),
        "split_leakage": leaked,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
