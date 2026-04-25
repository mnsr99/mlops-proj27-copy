# Training

This folder owns the summarization training and retraining workflow for our project.

The goal of this part is to:

- train the summarization model
- evaluate model quality
- build retraining datasets from user feedback
- retrain the model from API reviews
- register new model versions in MLflow

## Official training mainline

Use only these files as the current training mainline:

- `train.py` — normal summarization training
- `prepare_retraining_dataset_from_api_v2.py` — build retraining dataset from API `/reviews`
- `run_retraining_from_reviews_v2.py` — end-to-end API-driven retraining
- `run_retraining_v2.sh` — shell entry for retraining
- `watch_reviews_and_retrain_v2.sh` — watcher for review-driven retraining

Everything else should be treated as old, archived, testing-only, or example material unless explicitly needed.

## What this folder is responsible for

This folder is responsible for:

1. training the summarization model
2. evaluating the summarization model with ROUGE-based metrics
3. creating retraining datasets from user feedback
4. retraining from API reviews
5. registering new summarization model versions in MLflow

## What each official file does

### `train.py`
Runs normal summarization training.

### `prepare_retraining_dataset_from_api_v2.py`
Reads feedback from API `/reviews` and creates retraining dataset files.

### `run_retraining_from_reviews_v2.py`
Runs end-to-end retraining from API reviews.

### `run_retraining_v2.sh`
Shell wrapper for retraining.

### `watch_reviews_and_retrain_v2.sh`
Polls for review updates and triggers retraining automatically.

## Summarization evaluation

Current evaluation design:

- input: transcript-summary pairs
- validation happens during training
- main metric: ROUGE, especially ROUGE-L
- best model logic is based on `eval_rougeL`
- model registration depends on the quality-gate logic in `train.py`

## Storage policy

To avoid filling MinIO / MLflow artifacts with unnecessary checkpoints:

- do not enable large checkpoint saving unless needed
- do not log the whole output directory by default
- keep only the final model for normal runs
- use environment variables to control optional artifact logging
- clean local output after registration when possible

Recommended runtime settings:

```bash
export TRAIN_SAVE_STRATEGY="no"
unset MLFLOW_LOG_HF_MODEL_FILES
export CLEAN_LOCAL_OUTPUT_DIR_AFTER_REGISTER="1"
```

## Required environment variables

```bash
export DATA_API_BASE="http://129.114.27.10:30800"
export MLFLOW_TRACKING_URI="http://129.114.27.10:30500"
export MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:30900"
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
export MIN_RETRAIN_EXAMPLES="1"
```

## Normal training

```bash
cd ~/mlops-proj27/training

python3 train.py --config config.yaml
```

A more explicit example:

```bash
cd ~/mlops-proj27/training

export MLFLOW_TRACKING_URI="http://129.114.27.10:30500"
export MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:30900"
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
export TRAIN_SAVE_STRATEGY="no"
unset MLFLOW_LOG_HF_MODEL_FILES
export CLEAN_LOCAL_OUTPUT_DIR_AFTER_REGISTER="1"

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

A more explicit example:

```bash
cd ~/mlops-proj27/training

export DATA_API_BASE="http://129.114.27.10:30800"
export MLFLOW_TRACKING_URI="http://129.114.27.10:30500"
export MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:30900"
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
export MIN_RETRAIN_EXAMPLES="1"
export TRAIN_SAVE_STRATEGY="no"
unset MLFLOW_LOG_HF_MODEL_FILES
export CLEAN_LOCAL_OUTPUT_DIR_AFTER_REGISTER="1"

python3 prepare_retraining_dataset_from_api_v2.py --write-empty
cat data/retraining_stats.json
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

## How to tell retraining succeeded

A successful retraining run should show:

- `data/retraining_stats.json` exists
- `eligible_examples > 0`
- retraining finishes without error
- the training process finishes without failure
- MLflow shows a new registered model version

## Suggested cleanup policy

Keep only the current official mainline files in the main directory.
Move older retraining files, testing scripts, and examples into `archive/` or `examples/` so the main workflow stays clear.

Suggested examples:

Move older files into `archive/`, for example:

- `prepare_retraining_dataset_from_api.py`
- `run_retraining_from_reviews.py`
- `run_retraining.sh`
- `watch_feedback_and_retrain.sh`
- `test_asr_from_minio_all.py`

Move example files into `examples/`, for example:

- `ASR-output_example.json`
