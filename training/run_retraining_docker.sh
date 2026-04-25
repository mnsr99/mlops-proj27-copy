#!/bin/bash
set -e

cd /home/cc/mlops-proj27/training

mkdir -p data outputs logs

echo "========== Retraining started at $(date) =========="

sudo docker run --rm \
  -e DATA_API_BASE="http://129.114.27.10:30800" \
  -e RETRAIN_DATA_DIR="/app/training/data" \
  -v "$PWD/data:/app/training/data" \
  -v "$PWD/outputs:/app/training/outputs" \
  proj27-training \
  python3 run_retraining_from_reviews_v2.py

echo "========== Retraining finished at $(date) =========="
