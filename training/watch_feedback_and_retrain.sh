#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -d ".venv" ]]; then
  source .venv/bin/activate
else
  echo "Missing .venv in $SCRIPT_DIR"
  exit 1
fi

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://129.114.26.182:30500}"
export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-http://129.114.26.182:30900}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minio}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minio123}"
export MLFLOW_REGISTERED_MODEL_NAME="${MLFLOW_REGISTERED_MODEL_NAME:-jitsi-summarizer}"

FEEDBACK_FILE="${1:-data/feedback_records.jsonl}"
BASE_CONFIG="${2:-config.yaml}"
POLL_SECONDS="${POLL_SECONDS:-30}"

STATE_DIR=".retraining_state"
HASH_FILE="$STATE_DIR/feedback.sha256"
LOCK_FILE="$STATE_DIR/retraining.lock"
LOG_DIR="logs"
RUN_LOG="$LOG_DIR/retraining_runner.log"

mkdir -p "$STATE_DIR" "$LOG_DIR"

if [[ ! -f "$FEEDBACK_FILE" ]]; then
  echo "Feedback file not found: $FEEDBACK_FILE"
  exit 1
fi

hash_file() {
  sha256sum "$1" | awk '{print $1}'
}

is_nonempty_feedback() {
  [[ -s "$FEEDBACK_FILE" ]]
}

run_retraining() {
  echo "[$(date '+%F %T')] Change detected. Starting retraining..." | tee -a "$RUN_LOG"

  if [[ -x "./run_retraining.sh" ]]; then
    ./run_retraining.sh "$FEEDBACK_FILE" "$BASE_CONFIG" >> "$RUN_LOG" 2>&1
  else
    bash ./run_retraining.sh "$FEEDBACK_FILE" "$BASE_CONFIG" >> "$RUN_LOG" 2>&1
  fi

  echo "[$(date '+%F %T')] Retraining finished." | tee -a "$RUN_LOG"
}

CURRENT_HASH="$(hash_file "$FEEDBACK_FILE")"
echo "$CURRENT_HASH" > "$HASH_FILE"

echo "Watching: $FEEDBACK_FILE"
echo "Base config: $BASE_CONFIG"
echo "Polling every ${POLL_SECONDS}s"
echo "Log file: $RUN_LOG"

while true; do
  sleep "$POLL_SECONDS"

  if [[ ! -f "$FEEDBACK_FILE" ]]; then
    echo "[$(date '+%F %T')] Feedback file missing, waiting..." | tee -a "$RUN_LOG"
    continue
  fi

  if ! is_nonempty_feedback; then
    echo "[$(date '+%F %T')] Feedback file is empty, waiting..." >> "$RUN_LOG"
    continue
  fi

  NEW_HASH="$(hash_file "$FEEDBACK_FILE")"
  OLD_HASH="$(cat "$HASH_FILE" 2>/dev/null || echo '')"

  if [[ "$NEW_HASH" != "$OLD_HASH" ]]; then
    if [[ -f "$LOCK_FILE" ]]; then
      echo "[$(date '+%F %T')] Retraining already in progress, skipping this change." | tee -a "$RUN_LOG"
      echo "$NEW_HASH" > "$HASH_FILE"
      continue
    fi

    touch "$LOCK_FILE"
    trap 'rm -f "$LOCK_FILE"' EXIT

    if run_retraining; then
      echo "$NEW_HASH" > "$HASH_FILE"
    else
      echo "[$(date '+%F %T')] Retraining failed." | tee -a "$RUN_LOG"
    fi

    rm -f "$LOCK_FILE"
    trap - EXIT
  fi
done
