# ---------------------------------------------------------------------------
# baseline-mlflow — FastAPI summarizer that loads the jitsi-summarizer model
# from the MLflow Model Registry (not from HuggingFace hub).
#
# Why this fork exists:
#   serving/baseline/app.py pins MODEL_PATH to a public HF checkpoint, which
#   means retraining the model has no effect on what production serves. This
#   variant pulls models:/jitsi-summarizer@<alias> on startup so that
#   promoting a new version in MLflow is enough to roll out a new model
#   (kubectl rollout restart picks it up).
#
# Env vars:
#   MLFLOW_TRACKING_URI     - http://mlflow.platform.svc.cluster.local:5000
#   MLFLOW_S3_ENDPOINT_URL  - http://minio.platform.svc.cluster.local:9000
#   AWS_ACCESS_KEY_ID       - MinIO access key
#   AWS_SECRET_ACCESS_KEY   - MinIO secret key
#   MODEL_NAME              - registered model name (default: jitsi-summarizer)
#   MODEL_ALIAS             - registry alias (default: production)
# ---------------------------------------------------------------------------

import logging
import os
import re
from typing import List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serving")

MODEL_NAME = os.environ.get("MODEL_NAME", "jitsi-summarizer")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

app = FastAPI(title="jitsi-summarizer (MLflow-backed)")

# Load once at startup. If this fails, the pod crashes and k8s reschedules —
# which is what we want for a broken model promotion.
log.info("Loading model from MLflow: %s", MODEL_URI)
_model = mlflow.pyfunc.load_model(MODEL_URI)
log.info("Model loaded.")


class PredictRequest(BaseModel):
    meeting_id: str
    transcript: str


class PredictResponse(BaseModel):
    meeting_id: str
    summary: str
    action_items: List[str]
    model_name: str = MODEL_NAME
    model_alias: str = MODEL_ALIAS


def _extract_text(result) -> str:
    """Best-effort extraction of the summary string out of whatever the
    PyFunc predict() happens to return. Handles DataFrame / list / dict /
    plain string — so we don't break if training swaps output flavors."""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return ""
        # Take the first (and usually only) column of the first row.
        return str(result.iloc[0, 0])
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            # Common keys seen in summarization PyFuncs.
            for k in ("summary", "summary_text", "output", "generated_text", "text"):
                if k in first:
                    return str(first[k])
            # Fall back to first value.
            return str(next(iter(first.values())))
        return str(first)
    if isinstance(result, dict):
        for k in ("summary", "summary_text", "output", "generated_text", "text"):
            if k in result:
                return str(result[k])
    return str(result)


def _split_action_items(generated: str) -> (str, List[str]):
    """Preserve the same output contract as serving/baseline/app.py:
    split the generated text on the 'Action Items:' marker (case-insensitive)
    into (summary, [action items])."""
    match = re.search(r"action items:", generated, flags=re.IGNORECASE)
    if not match:
        return generated.strip(), []
    summary = generated[: match.start()].strip()
    tail = generated[match.end():]
    items = [s.strip() for s in tail.split(".") if s.strip()]
    return summary, items


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame([{"text": req.transcript}])
        raw = _model.predict(df)
        generated = _extract_text(raw)
        summary, action_items = _split_action_items(generated)
        return PredictResponse(
            meeting_id=req.meeting_id,
            summary=summary,
            action_items=action_items,
        )
    except Exception as e:
        log.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(e))
