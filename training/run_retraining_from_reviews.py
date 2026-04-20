import json
import os
import subprocess
import sys
from pathlib import Path


STATS_PATH = Path(os.environ.get("RETRAIN_STATS_PATH", "data/retraining_stats.json"))
MIN_RETRAIN_EXAMPLES = int(os.environ.get("MIN_RETRAIN_EXAMPLES", "5"))
CONFIG_PATH = os.environ.get("TRAIN_CONFIG_PATH", "config.yaml")


def run_cmd(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    run_cmd([sys.executable, "prepare_retraining_dataset_from_api.py"])

    if not STATS_PATH.exists():
        raise RuntimeError(f"Stats file not found: {STATS_PATH}")

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    eligible_examples = int(stats.get("eligible_examples", 0))
    train_examples = int(stats.get("train_examples", 0))
    val_examples = int(stats.get("val_examples", 0))

    print(f"[INFO] eligible_examples={eligible_examples}")
    print(f"[INFO] train_examples={train_examples}")
    print(f"[INFO] val_examples={val_examples}")

    if eligible_examples < MIN_RETRAIN_EXAMPLES:
        print(
            f"[SKIP] Not enough eligible retraining examples. "
            f"Need at least {MIN_RETRAIN_EXAMPLES}, got {eligible_examples}."
        )
        return

    run_cmd([sys.executable, "train.py", "--config", CONFIG_PATH])


if __name__ == "__main__":
    main()
