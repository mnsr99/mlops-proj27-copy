# training

This folder owns the summarization training and retraining workflow for the project.

## Official entry points

These are the current official training-side entry points:

- `train.py` — normal summarization training
- `prepare_retraining_dataset_from_api_v2.py` — build retraining dataset from API `/reviews`
- `run_retraining_from_reviews_v2.py` — end-to-end API-driven retraining
- `run_retraining_v2.sh` — shell entry for retraining
- `watch_reviews_and_retrain_v2.sh` — watcher for review-driven retraining

Everything else should be treated as old, archived, testing-only, or example material unless explicitly needed.

## What this folder does

This folder is responsible for:

1. training the summarization model
2. evaluating the model with ROUGE-based metrics
3. building retraining datasets from user feedback
4. retraining from API reviews
5. registering new model versions in MLflow

## Summarization evaluation

Current evaluation logic:

- input: transcript-summary pairs
- validation: run during training
- metrics: ROUGE, especially ROUGE-L
- best model selection: based on `eval_rougeL`
- registration gate: quality-gate logic in `train.py`

## Storage policy

To avoid filling MinIO / MLflow artifacts with unnecessary checkpoints:

- do not enable large checkpoint saving unless needed
- do not log the whole output directory by default
- keep only the final model for normal runs
- use environment variables to control optional artifact logging

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
```

## Normal training

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

## Retraining from API reviews

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

## Suggested folder cleanup

Keep in the main training directory:

- `train.py`
- `prepare_retraining_dataset_from_api_v2.py`
- `run_retraining_from_reviews_v2.py`
- `run_retraining_v2.sh`
- `watch_reviews_and_retrain_v2.sh`
- `register_asr_model.py`
- `run_online_pipeline.py`
- `config.yaml`
- `requirements.txt`
- `Dockerfile`
- `TRAINING_README.md`

Move older files into `archive/`, for example:

- `prepare_retraining_dataset_from_api.py`
- `run_retraining_from_reviews.py`
- `run_retraining.sh`
- `watch_feedback_and_retrain.sh`
- `test_asr_from_minio_all.py`

Move examples into `examples/`, for example:

- `ASR-output_example.json`

## How to tell retraining succeeded

A successful retraining run should show:

- `data/retraining_stats.json` exists
- `eligible_examples > 0`
- retraining finishes without error
- MLflow shows a new registered model version
