import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import mlflow
import mlflow.pyfunc
import pandas as pd
import requests


# ----------------------------
# Shared platform defaults
# ----------------------------
DATA_API_BASE = os.environ.get("DATA_API_BASE", "http://129.114.26.182:30800")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.26.182:30500")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://129.114.26.182:30900")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://129.114.26.182:30900")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")

ASR_MODEL_URI = os.environ.get("ASR_MODEL_URI", "models:/jitsi-asr/1")

# Served summarizer endpoint
SUMMARIZER_PREDICT_URL = os.environ.get(
    "SUMMARIZER_PREDICT_URL",
    "http://129.114.26.182:30810/predict",
)

DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "en")
TRANSCRIPT_BUCKET = os.environ.get("TRANSCRIPT_BUCKET", "jitsi-data")
DEFAULT_ACTION_ITEMS = os.environ.get(
    "DEFAULT_ACTION_ITEMS",
    "No action items identified.",
)

REQUEST_CONNECT_TIMEOUT = int(os.environ.get("REQUEST_CONNECT_TIMEOUT", "10"))
REQUEST_READ_TIMEOUT = int(os.environ.get("REQUEST_READ_TIMEOUT", "120"))

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm", ".mp4"}

MEETING_MANIFEST_PATH = Path(
    os.environ.get("MEETING_MANIFEST_PATH", "data/meeting_manifest.jsonl")
)
MEETING_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

# MLflow env only needed for ASR now
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
try:
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
except Exception:
    pass

_ASR_MODEL = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append_meeting_manifest(meeting_id: str, object_key: str) -> None:
    row = {
        "meeting_id": meeting_id,
        "audio_object_key": object_key,
        "created_at": now_iso(),
    }
    with MEETING_MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def is_audio_file(object_key: str) -> bool:
    return Path(object_key).suffix.lower() in AUDIO_EXTENSIONS


def list_audio_objects(bucket: str, prefix: str = "") -> List[str]:
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    keys: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if is_audio_file(key):
                keys.append(key)

    return sorted(keys)


def download_from_minio(bucket: str, object_key: str) -> str:
    s3 = get_s3_client()

    suffix = Path(object_key).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        local_path = tmp.name

    s3.download_file(bucket, object_key, local_path)
    return local_path


def upload_text_to_minio(bucket: str, object_key: str, text: str) -> None:
    s3 = get_s3_client()
    s3.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=text.encode("utf-8"),
        ContentType="text/plain",
    )


def get_asr_model():
    global _ASR_MODEL
    if _ASR_MODEL is None:
        print(f"[MLflow] tracking_uri={mlflow.get_tracking_uri()}", flush=True)
        print(
            f"[MLflow] MLFLOW_S3_ENDPOINT_URL={os.environ.get('MLFLOW_S3_ENDPOINT_URL')}",
            flush=True,
        )
        print(f"[Model] Loading ASR model from {ASR_MODEL_URI}", flush=True)
        _ASR_MODEL = mlflow.pyfunc.load_model(ASR_MODEL_URI)
        print("[Model] ASR model loaded", flush=True)
    return _ASR_MODEL


def run_asr(local_audio_path: str, meeting_id: str, language: str) -> Dict[str, Any]:
    model = get_asr_model()

    df = pd.DataFrame(
        [
            {
                "meeting_id": meeting_id,
                "audio_path": local_audio_path,
                "language": language,
                "source": "minio_audio",
            }
        ]
    )

    print(f"[ASR] Start predict for meeting_id={meeting_id}", flush=True)
    result = model.predict(df)
    print("[ASR] Predict finished", flush=True)

    if isinstance(result, pd.DataFrame):
        records = result.to_dict(orient="records")
    elif isinstance(result, list):
        records = result
    else:
        raise RuntimeError(f"Unexpected ASR output type: {type(result)}")

    if not records:
        raise RuntimeError("ASR model returned empty result.")

    if not isinstance(records[0], dict):
        raise RuntimeError(f"Unexpected ASR record type: {type(records[0])}")

    return records[0]


def post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        url,
        json=payload,
        timeout=(REQUEST_CONNECT_TIMEOUT, REQUEST_READ_TIMEOUT),
    )

    if not resp.ok:
        print("\nPOST failed", flush=True)
        print("URL:", url, flush=True)
        print("Payload:", flush=True)
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        print("Status code:", resp.status_code, flush=True)
        print("Response text:", flush=True)
        print(resp.text, flush=True)
        resp.raise_for_status()

    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}


def get_json_or_text(url: str) -> Any:
    resp = requests.get(
        url,
        timeout=(REQUEST_CONNECT_TIMEOUT, REQUEST_READ_TIMEOUT),
    )

    if not resp.ok:
        print("\nGET failed", flush=True)
        print("URL:", url, flush=True)
        print("Status code:", resp.status_code, flush=True)
        print("Response text:", flush=True)
        print(resp.text, flush=True)
        resp.raise_for_status()

    try:
        return resp.json()
    except Exception:
        return resp.text


def create_meeting(audio_object_key: str, source: str = "minio_audio") -> Any:
    payload = {
        "source": source,
        "started_at": now_iso(),
        "ended_at": now_iso(),
        "audio_object_key": audio_object_key,
        "status": "scheduled",
        "asr_status": "not_requested",
    }
    return post_json(f"{DATA_API_BASE}/meetings", payload)


def create_transcript(
    meeting_id: str,
    transcript_text: str,
    transcript_object_key: str,
) -> Any:
    payload = {
        "meeting_id": meeting_id,
        "transcript_text": transcript_text,
        "transcript_object_key": transcript_object_key,
    }
    return post_json(f"{DATA_API_BASE}/transcripts", payload)


def get_transcript_by_meeting(meeting_id: str) -> Any:
    return get_json_or_text(f"{DATA_API_BASE}/transcripts/by_meeting/{meeting_id}")


def create_summary(
    meeting_id: str,
    summary_text: str,
    action_item_text: str,
    model_version: str,
) -> Any:
    safe_summary = (summary_text or "").strip()
    safe_actions = (action_item_text or "").strip() or DEFAULT_ACTION_ITEMS

    payload = {
        "meeting_id": meeting_id,
        "model_version": model_version,
        "summary_text": safe_summary,
        "action_item_text": safe_actions,
    }
    return post_json(f"{DATA_API_BASE}/summaries", payload)


def extract_meeting_id(meeting_resp: Any) -> str:
    if isinstance(meeting_resp, dict):
        for key in ("meeting_id", "id", "uuid"):
            value = meeting_resp.get(key)
            if value:
                return str(value)
    raise RuntimeError(
        f"Could not find meeting_id in create_meeting response: {meeting_resp}"
    )


def call_served_summarizer(meeting_id: str, transcript_text: str) -> Dict[str, Any]:
    payload = {
        "meeting_id": meeting_id,
        "transcript": transcript_text,
    }

    print(f"[SUM API] POST {SUMMARIZER_PREDICT_URL}", flush=True)
    resp = requests.post(
        SUMMARIZER_PREDICT_URL,
        json=payload,
        timeout=(REQUEST_CONNECT_TIMEOUT, REQUEST_READ_TIMEOUT),
    )

    if not resp.ok:
        print("\nSummarizer /predict failed", flush=True)
        print("URL:", SUMMARIZER_PREDICT_URL, flush=True)
        print("Payload:", flush=True)
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        print("Status code:", resp.status_code, flush=True)
        print("Response text:", flush=True)
        print(resp.text, flush=True)
        resp.raise_for_status()

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"Summarizer /predict returned non-JSON response: {resp.text}")

    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected /predict response type: {type(data)}")

    return data


def extract_summary_and_actions(predict_resp: Dict[str, Any]) -> Dict[str, str]:
    summary_text = str(predict_resp.get("summary", "")).strip()

    action_items = predict_resp.get("action_items", [])
    if isinstance(action_items, list):
        clean_items = [str(x).strip() for x in action_items if str(x).strip()]
        action_item_text = "\n".join(clean_items) if clean_items else DEFAULT_ACTION_ITEMS
    elif isinstance(action_items, str):
        action_item_text = action_items.strip() or DEFAULT_ACTION_ITEMS
    else:
        action_item_text = DEFAULT_ACTION_ITEMS

    model_name = str(predict_resp.get("model_name", "jitsi-summarizer")).strip() or "jitsi-summarizer"
    model_alias = str(predict_resp.get("model_alias", "production")).strip() or "production"
    model_version = f"{model_name}@{model_alias}"

    if not summary_text:
        raise RuntimeError(f"Empty summary returned from /predict: {predict_resp}")

    return {
        "summary_text": summary_text,
        "action_item_text": action_item_text,
        "model_version": model_version,
    }


def process_single_audio(bucket: str, object_key: str, language: str) -> Dict[str, Any]:
    print("\n" + "=" * 80, flush=True)
    print(f"Processing audio object: {bucket}/{object_key}", flush=True)
    print("=" * 80, flush=True)

    local_audio_path: Optional[str] = None

    print("=== Step 0: Create meeting record ===", flush=True)
    meeting_resp = create_meeting(audio_object_key=object_key, source="minio_audio")
    print("Meeting API response:", flush=True)
    print(meeting_resp, flush=True)

    meeting_id = extract_meeting_id(meeting_resp)
    print(f"Resolved meeting_id: {meeting_id}", flush=True)
    append_meeting_manifest(meeting_id, object_key)

    try:
        print("\n=== Step 1: Download audio from MinIO ===", flush=True)
        local_audio_path = download_from_minio(bucket, object_key)
        print(f"Downloaded to: {local_audio_path}", flush=True)

        print("\n=== Step 2: Run ASR model ===", flush=True)
        asr_output = run_asr(local_audio_path, meeting_id, language)
        print(json.dumps(asr_output, indent=2, ensure_ascii=False), flush=True)

        transcript_text = (
            asr_output.get("transcript")
            or asr_output.get("text")
            or asr_output.get("transcript_text")
            or ""
        ).strip()

        if not transcript_text:
            raise RuntimeError(f"No transcript text found in ASR output: {asr_output}")

        transcript_object_key = f"transcripts/{meeting_id}.txt"

        print("\n=== Step 3: Upload transcript text to MinIO ===", flush=True)
        upload_text_to_minio(TRANSCRIPT_BUCKET, transcript_object_key, transcript_text)
        print(
            f"Uploaded transcript to minio://{TRANSCRIPT_BUCKET}/{transcript_object_key}",
            flush=True,
        )

        print("\n=== Step 4: Store ASR output into /transcripts ===", flush=True)
        transcript_resp = create_transcript(
            meeting_id=meeting_id,
            transcript_text=transcript_text,
            transcript_object_key=transcript_object_key,
        )
        print("Transcript API response:", flush=True)
        print(transcript_resp, flush=True)

        print("\n=== Step 5: Read transcript back from API ===", flush=True)
        transcript_record = get_transcript_by_meeting(meeting_id)
        print("Transcript fetched by meeting:", flush=True)
        print(transcript_record, flush=True)

        if isinstance(transcript_record, dict):
            summarization_input = (
                transcript_record.get("transcript_text")
                or transcript_record.get("transcript")
                or transcript_record.get("text")
                or transcript_text
            )
        else:
            summarization_input = str(transcript_record)

        print("\n=== Step 6: Call served summarization /predict API ===", flush=True)
        predict_resp = call_served_summarizer(meeting_id, summarization_input)
        print("Summarizer /predict response:", flush=True)
        print(json.dumps(predict_resp, indent=2, ensure_ascii=False), flush=True)

        parsed = extract_summary_and_actions(predict_resp)
        summary_text = parsed["summary_text"]
        action_item_text = parsed["action_item_text"]
        model_version = parsed["model_version"]

        print("\n=== Step 7: Store summary into /summaries ===", flush=True)
        summary_resp = create_summary(
            meeting_id=meeting_id,
            summary_text=summary_text,
            action_item_text=action_item_text,
            model_version=model_version,
        )
        print("Summary API response:", flush=True)
        print(summary_resp, flush=True)

        print("\n=== Pipeline finished successfully for this file ===", flush=True)

        return {
            "meeting_id": meeting_id,
            "object_key": object_key,
            "status": "success",
            "summary_stored": True,
        }

    finally:
        if local_audio_path:
            try:
                os.remove(local_audio_path)
            except Exception:
                pass


def main():
    bucket = os.environ.get("MINIO_BUCKET", "audio-files")
    object_key = os.environ.get("MINIO_OBJECT_KEY")
    prefix = os.environ.get("MINIO_PREFIX", "")
    language = os.environ.get("ASR_LANGUAGE", DEFAULT_LANGUAGE)
    max_files = int(os.environ.get("MAX_FILES", "0"))
    fail_fast = os.environ.get("FAIL_FAST", "false").lower() == "true"

    if not bucket:
        raise ValueError("MINIO_BUCKET is required.")

    print(f"[Config] DATA_API_BASE={DATA_API_BASE}", flush=True)
    print(f"[Config] MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}", flush=True)
    print(f"[Config] MLFLOW_S3_ENDPOINT_URL={MLFLOW_S3_ENDPOINT_URL}", flush=True)
    print(f"[Config] MINIO_ENDPOINT={MINIO_ENDPOINT}", flush=True)
    print(f"[Config] ASR_MODEL_URI={ASR_MODEL_URI}", flush=True)
    print(f"[Config] SUMMARIZER_PREDICT_URL={SUMMARIZER_PREDICT_URL}", flush=True)
    print(f"[Config] MEETING_MANIFEST_PATH={MEETING_MANIFEST_PATH}", flush=True)

    if object_key:
        object_keys = [object_key]
        print(f"[Config] Single-file mode. MINIO_OBJECT_KEY={object_key}", flush=True)
    else:
        print(
            f"[Config] Batch mode. Listing audio files from bucket='{bucket}', prefix='{prefix}'",
            flush=True,
        )
        object_keys = list_audio_objects(bucket, prefix=prefix)

        if not object_keys:
            raise ValueError(
                f"No audio files found in bucket '{bucket}' with prefix '{prefix}'."
            )

    if max_files > 0:
        object_keys = object_keys[:max_files]
        print(
            f"[Config] MAX_FILES applied. Will process first {len(object_keys)} file(s).",
            flush=True,
        )

    print(f"[Config] Total audio files to process: {len(object_keys)}", flush=True)
    for idx, key in enumerate(object_keys, start=1):
        print(f"  {idx}. {key}", flush=True)

    success_count = 0
    failed: List[Dict[str, str]] = []

    for idx, key in enumerate(object_keys, start=1):
        print(f"\n##### [{idx}/{len(object_keys)}] Start #####", flush=True)
        try:
            result = process_single_audio(bucket=bucket, object_key=key, language=language)
            success_count += 1
            print(f"##### [{idx}/{len(object_keys)}] Success: {result} #####", flush=True)
        except Exception as e:
            failed_item = {"object_key": key, "error": str(e)}
            failed.append(failed_item)
            print(f"##### [{idx}/{len(object_keys)}] Failed #####", flush=True)
            print(json.dumps(failed_item, indent=2, ensure_ascii=False), flush=True)

            if fail_fast:
                raise

    print("\n" + "=" * 80, flush=True)
    print("Batch processing finished.", flush=True)
    print(f"Success count: {success_count}", flush=True)
    print(f"Failure count: {len(failed)}", flush=True)
    if failed:
        print("Failed items:", flush=True)
        print(json.dumps(failed, indent=2, ensure_ascii=False), flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
