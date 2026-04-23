import json
import os
import subprocess
import sys
from pathlib import Path


STATS_PATH = Path(os.environ.get("RETRAIN_STATS_PATH", "data/retraining_stats.json"))
STATE_PATH = Path(os.environ.get("RETRAIN_STATE_PATH", "data/retraining_last_fingerprint.txt"))
MIN_RETRAIN_EXAMPLES = int(os.environ.get("MIN_RETRAIN_EXAMPLES", "5"))
CONFIG_PATH = os.environ.get("TRAIN_CONFIG_PATH", "config.yaml")


def run_cmd(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def read_last_fingerprint() -> str:
    if not STATE_PATH.exists():
        return ""
    return STATE_PATH.read_text(encoding="utf-8").strip()


def write_last_fingerprint(fp: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(fp, encoding="utf-8")


def main():
    run_cmd([sys.executable, "prepare_retraining_dataset_from_api.py", "--write-empty"])

    if not STATS_PATH.exists():
        raise RuntimeError(f"Stats file not found: {STATS_PATH}")

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    eligible_examples = int(stats.get("eligible_examples", 0))
    train_examples = int(stats.get("train_examples", 0))
    val_examples = int(stats.get("val_examples", 0))
    current_fingerprint = str(stats.get("dataset_fingerprint", "") or "")
    previous_fingerprint = read_last_fingerprint()

    print(f"[INFO] eligible_examples={eligible_examples}")
    print(f"[INFO] train_examples={train_examples}")
    print(f"[INFO] val_examples={val_examples}")
    print(f"[INFO] current_fingerprint={current_fingerprint}")
    print(f"[INFO] previous_fingerprint={previous_fingerprint}")

    if eligible_examples < MIN_RETRAIN_EXAMPLES:
        print(
            f"[SKIP] Not enough eligible retraining examples. "
            f"Need at least {MIN_RETRAIN_EXAMPLES}, got {eligible_examples}."
        )
        return

    if not train_examples or not val_examples:
        print("[SKIP] Train/validation split is empty. Need more feedback from at least one more usable example.")
        return

    if current_fingerprint and current_fingerprint == previous_fingerprint:
        print("[SKIP] No new eligible API review data since the last successful retraining.")
        return

    run_cmd([sys.executable, "train.py", "--config", CONFIG_PATH])

    if current_fingerprint:
        write_last_fingerprint(current_fingerprint)
        print(f"[INFO] Updated retraining state file: {STATE_PATH}")

    print("[INFO] API-driven retraining finished successfully.")


if __name__ == "__main__":
    main()
