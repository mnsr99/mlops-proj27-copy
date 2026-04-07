# MLOps Project: Data Platform for Meeting Summarization

## Overview

This project implements an end-to-end MLOps data platform for training and evaluating meeting summarization models.

The system ingests both:

* **Production-style data** (simulated meeting outputs)
* **External datasets** (QMSum)

All data is:

* normalized into a consistent schema
* versioned
* stored in S3-compatible object storage (MinIO)

---

## System Architecture

The platform is deployed on a cloud VM (Chameleon Cloud) and uses Docker Compose to orchestrate services:

* **API service** (FastAPI)
* **PostgreSQL** (metadata / relational storage)
* **MinIO** (S3-compatible object storage)
* **Adminer** (database UI)

---

## Setup Instructions

### 1. Start Infrastructure

From `data/`:

```bash
docker compose up -d --build
```

Verify:

```bash
docker ps
```

---

### 2. Access Services

* API: http://<VM_IP>:8000
* Adminer: http://<VM_IP>:5050
* MinIO Console: http://<VM_IP>:9001

Default MinIO credentials:

```
user: minio
password: minio123
```

---

## Data Pipelines

### 1. Production Data Pipeline

Generates synthetic meeting data and uploads it to MinIO.

```bash
cd data
source .venv/bin/activate
python generator/generator.py
```

Output:

```
s3://jitsi-data/v<timestamp>/
```

Example:

```
v20260407_030840/
  train.jsonl
  val.jsonl
  test.jsonl
  manifest.json
```

---

### 2. External Data Pipeline (QMSum)

Ingests and normalizes the QMSum dataset.

#### Step 1: Download dataset

```bash
git clone https://github.com/Yale-LILY/QMSum.git
```

#### Step 2: Copy raw data

```bash
mkdir -p external_data/qmsum_raw
find QMSum/data/ALL -name "*.json" -exec cp {} external_data/qmsum_raw/ \;
```

#### Step 3: Run ingestion

```bash
python pipelines/ingest_qmsum.py
```

Output:

```
s3://jitsi-data/external/qmsum/qmsum_v<timestamp>/
```

Example:

```
external/qmsum/qmsum_v20260407_032624/
  qmsum_train.jsonl
  manifest.json
```

---

## Data Schema

All datasets are normalized into a unified JSONL format:

```json
{
  "dataset_version": "string",
  "source": "QMSum | production",
  "source_meeting_id": "string",
  "query_id": "string",
  "query_text": "string",
  "input_transcript": "string",
  "target_summary": "string",
  "target_action_items": [],
  "split": "train | val | test"
}
```

---

## Object Storage Structure

```
jitsi-data/
  v<timestamp>/                      # production data
  external/
    qmsum/
      qmsum_v<timestamp>/            # external data
```

---

## Key Design Decisions

### 1. Dataset Versioning

Each pipeline produces a timestamped dataset version:

* enables reproducibility
* allows comparison across training runs

### 2. Unified Schema

External and production data are transformed into a consistent format to:

* simplify training pipelines
* enable dataset mixing

### 3. Object Storage (MinIO)

Used instead of local files because:

* scalable
* cloud-compatible (S3 API)
* standard in production ML systems

### 4. Separation of Pipelines

* `generator/` → production data
* `pipelines/` → external data ingestion

---

## Reproducibility

To fully reproduce the system:

```bash
git clone <repo>
cd mlops-proj27/data
docker compose up -d --build

# run pipelines
python generator/generator.py
python pipelines/ingest_qmsum.py
```

---

## Results

* Production dataset version: `v20260407_030840`
* External dataset version: `qmsum_v20260407_032624`
* Data successfully stored in MinIO

---

## Future Work

* Add data validation and quality checks
* Introduce training pipeline (model fine-tuning)
* Add experiment tracking (MLflow)
* Automate pipelines with orchestration (Airflow)

---
