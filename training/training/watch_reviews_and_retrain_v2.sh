#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/retraining_runner.log"
POLL_SECONDS="${POLL_SECONDS:-30}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$LOG_DIR"

echo "Polling API-based reviews for retraining"
echo "Base config: ${TRAIN_CONFIG_PATH:-config.yaml}"
echo "Polling every ${POLL_SECONDS}s"
echo "Log file: $LOG_FILE"

while true; do
  TS="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$TS] Polling reviews API..." | tee -a "$LOG_FILE"

  if $PYTHON_BIN run_retraining_from_reviews.py >> "$LOG_FILE" 2>&1; then
    echo "[$TS] Poll cycle finished." | tee -a "$LOG_FILE"
  else
    echo "[$TS] Poll cycle failed." | tee -a "$LOG_FILE"
  fi

  sleep "$POLL_SECONDS"
done
