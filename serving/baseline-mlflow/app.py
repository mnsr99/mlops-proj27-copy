# ---------------------------------------------------------------------------
# baseline-mlflow — FastAPI summarizer that loads the jitsi-summarizer model
# from the MLflow Model Registry via the pyfunc flavor.
#
# Why pyfunc (again):
#   Earlier versions of this file bypassed pyfunc and loaded the raw HF
#   artifact directly, because v1 of the registered model was pickled with
#   pipeline("summarization") — a task removed from transformers 5.x's
#   pipeline registry, so load_context() would blow up at startup.
#   Training fixed the bug (pipeline("text2text-generation") in train.py)
#   and registered v7; production alias now points at v7, so the pyfunc
#   path actually loads again. Using it here keeps serving "MLflow-native":
#   promoting a new alias in the registry is the only step needed to ship
#   a new model.
#
# Generation params are no longer controlled here:
#   SummarizationPyFuncModel.predict() hardcodes max_new_tokens=128 and
#   uses pipeline defaults for beams / no_repeat_ngram_size. If summaries
#   come out too short or too repetitive, fix it in train.py and retrain —
#   not here. (Env vars MAX_NEW_TOKENS / NUM_BEAMS from the old workaround
#   are intentionally removed so nobody thinks they still do anything.)
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
from typing import List, Tuple

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serving")

MODEL_NAME = os.environ.get("MODEL_NAME", "jitsi-summarizer")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

app = FastAPI(title="jitsi-summarizer (MLflow-backed)")


def _load_pyfunc_model():
    """Resolve the alias -> version for logging, then load via pyfunc.

    The MlflowClient call is purely for observability — if it fails we
    still try the pyfunc load, which is the thing that actually matters.
    """
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        log.info("Alias %s -> version %s (run_id=%s)", MODEL_ALIAS, mv.version, mv.run_id)
    except Exception as e:
        log.warning("Could not resolve alias metadata (non-fatal): %s", e)

    log.info("Loading pyfunc model: %s", MODEL_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)
    log.info("Model loaded; ready to serve.")
    return model


# Load once at startup. If this fails, the pod crashes and k8s reschedules —
# which is what we want for a broken model promotion.
_model = _load_pyfunc_model()


class PredictRequest(BaseModel):
    meeting_id: str
    transcript: str


class PredictResponse(BaseModel):
    meeting_id: str
    summary: str
    action_items: List[str]
    model_name: str = MODEL_NAME
    model_alias: str = MODEL_ALIAS


def _generate(text: str) -> str:
    """Call the pyfunc. Training-side predict() expects a DataFrame with a
    'text' column and returns a DataFrame with a 'summary' column
    (see training/train.py:SummarizationPyFuncModel.predict)."""
    result_df = _model.predict(pd.DataFrame({"text": [text]}))
    return str(result_df["summary"].iloc[0])


def _split_action_items(generated: str) -> Tuple[str, List[str]]:
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
        generated = _generate(req.transcript)
        summary, action_items = _split_action_items(generated)
        return PredictResponse(
            meeting_id=req.meeting_id,
            summary=summary,
            action_items=action_items,
        )
    except Exception as e:
        log.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(e))
