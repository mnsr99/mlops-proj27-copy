#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export DATA_API_BASE="${DATA_API_BASE:-http://129.114.27.10:30800}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://129.114.27.10:30500}"
export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-http://129.114.27.10:30900}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minio}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minio123}"
export MLFLOW_REGISTERED_MODEL_NAME="${MLFLOW_REGISTERED_MODEL_NAME:-jitsi-summarizer}"

echo "Starting API-driven retraining pipeline..."
echo "Working directory: $SCRIPT_DIR"
echo "Using Python: $PYTHON_BIN"
echo "DATA_API_BASE=$DATA_API_BASE"

$PYTHON_BIN run_retraining_from_reviews.py
