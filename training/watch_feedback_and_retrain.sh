#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WATCH_FILE="data/feedback_records.jsonl"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/retraining_runner.log"
POLL_SECONDS=30

mkdir -p "$LOG_DIR"

if [[ ! -f "$WATCH_FILE" ]]; then
  touch "$WATCH_FILE"
fi

echo "Watching: $WATCH_FILE"
echo "Base config: config.yaml"
echo "Polling every ${POLL_SECONDS}s"
echo "Log file: $LOG_FILE"

LAST_MTIME=$(stat -c %Y "$WATCH_FILE" 2>/dev/null || echo 0)

while true; do
  CURRENT_MTIME=$(stat -c %Y "$WATCH_FILE" 2>/dev/null || echo 0)

  if [[ "$CURRENT_MTIME" != "$LAST_MTIME" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Change detected. Starting retraining..." | tee -a "$LOG_FILE"

    ./run_retraining.sh >> "$LOG_FILE" 2>&1 || true

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Retraining finished." | tee -a "$LOG_FILE"

    LAST_MTIME="$CURRENT_MTIME"
  fi

  sleep "$POLL_SECONDS"
done
