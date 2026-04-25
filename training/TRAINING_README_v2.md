# TRAINING README / RUNBOOK

## Purpose

This document explains the official training and retraining workflow.

## Official mainline

Use only these files as the current training mainline:

- `train.py`
- `prepare_retraining_dataset_from_api_v2.py`
- `run_retraining_from_reviews_v2.py`
- `run_retraining_v2.sh`
- `watch_reviews_and_retrain_v2.sh`

## What each official file does

### `train.py`
Normal summarization training.

### `prepare_retraining_dataset_from_api_v2.py`
Reads feedback from API `/reviews` and creates retraining dataset files.

### `run_retraining_from_reviews_v2.py`
Runs end-to-end retraining from API reviews.

### `run_retraining_v2.sh`
Shell wrapper for retraining.

### `watch_reviews_and_retrain_v2.sh`
Polls for review updates and triggers retraining.

## Environment variables

```bash
export DATA_API_BASE="http://129.114.27.10:30800"
export MLFLOW_TRACKING_URI="http://129.114.27.10:30500"
export MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:30900"
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
export MIN_RETRAIN_EXAMPLES="1"
export TRAIN_SAVE_STRATEGY="no"
unset MLFLOW_LOG_HF_MODEL_FILES
export CLEAN_LOCAL_OUTPUT_DIR_AFTER_REGISTER="1"
```

## Normal training

```bash
cd ~/mlops-proj27/training
python3 train.py --config config.yaml
```

## Build retraining dataset from API

```bash
cd ~/mlops-proj27/training
python3 prepare_retraining_dataset_from_api_v2.py --write-empty
cat data/retraining_stats.json
```

## Run retraining

```bash
cd ~/mlops-proj27/training
python3 run_retraining_from_reviews_v2.py
```

## Run watcher

```bash
cd ~/mlops-proj27/training
mkdir -p logs
pkill -f watch_reviews_and_retrain_v2.sh || true
nohup bash ./watch_reviews_and_retrain_v2.sh > logs/watch_reviews_v2.out 2>&1 &
tail -f logs/watch_reviews_v2.out
tail -f logs/retraining_runner.log
```

## Evaluation summary

The summarization model is evaluated with transcript-summary pairs.
The main metric is ROUGE, especially ROUGE-L.
Validation happens during training.
The best model logic is based on `eval_rougeL`.
Model registration depends on the quality-gate logic in `train.py`.

## Storage control

To reduce storage pressure:

- do not save many checkpoints by default
- do not log the whole output directory by default
- keep only the final model for normal runs
- clean local output after registration when possible

## Success checklist

Training / retraining is successful if:

- no runtime error appears
- dataset files are generated correctly
- `eligible_examples` is greater than zero when retraining is expected
- a new MLflow registered model version appears
