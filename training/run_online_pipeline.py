import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import mlflow.pyfunc
import pandas as pd
import requests


# =========================
# Configuration
# =========================

DATA_API_BASE = os.environ.get("DATA_API_BASE", "http://129.114.26.182:30800")
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://129.114.26.182:30900")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")

ASR_MODEL_URI = os.environ.get("ASR_MODEL_URI", "models:/jitsi-asr/1")
SUMMARIZATION_MODEL_URI = os.environ.get("SUMMARIZATION_MODEL_URI", "models:/jitsi-summarizer/6")

DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "en")


# =========================
# Helpers
# =========================

def download_from_minio(bucket: str, object_key: str) -> str:
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    suffix = Path(object_key).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        local_path = tmp.name

    s3.download_file(bucket, object_key, local_path)
    return local_path


def run_asr(local_audio_path: str, meeting_id: str, language: str) -> Dict[str, Any]:
    model = mlflow.pyfunc.load_model(ASR_MODEL_URI)

    df = pd.DataFrame([
        {
            "meeting_id": meeting_id,
            "audio_path": local_audio_path,
            "language": language,
            "source": "minio_audio",
        }
    ])

    result = model.predict(df)
    records = result.to_dict(orient="records")

    if not records:
        raise RuntimeError("ASR model returned empty result.")

    return records[0]


def run_summarization(transcript_text: str) -> str:
    model = mlflow.pyfunc.load_model(SUMMARIZATION_MODEL_URI)

    # 你的 summarization pyfunc 之前是吃 'text' 列
    df = pd.DataFrame([{"text": transcript_text}])

    result = model.predict(df)

    if isinstance(result, pd.DataFrame):
        if "summary" in result.columns:
            return str(result.iloc[0]["summary"])
        return str(result.iloc[0, 0])

    if isinstance(result, list) and result:
        return str(result[0])

    return str(result)


def post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}


def get_json(url: str) -> Dict[str, Any]:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}


# =========================
# API Wrappers
# =========================

def create_meeting(meeting_id: str, bucket: str, object_key: str) -> Dict[str, Any]:
    """
    注意：
    这里的 payload 字段名需要和你们 /meetings 的真实 schema 对齐。
    先给你一个很常见的写法。
    """
    payload = {
        "meeting_id": meeting_id,
        "audio_bucket": bucket,
        "audio_object_key": object_key,
    }
    return post_json(f"{DATA_API_BASE}/meetings", payload)


def create_transcript(
    meeting_id: str,
    language: str,
    transcript_text: str,
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    注意：
    这里的 payload 字段名也需要按你们 /transcripts 实际定义微调。
    """
    payload = {
        "meeting_id": meeting_id,
        "language": language,
        "transcript": transcript_text,
        "segments": segments,
    }
    return post_json(f"{DATA_API_BASE}/transcripts", payload)


def get_transcript_by_meeting(meeting_id: str) -> Dict[str, Any]:
    return get_json(f"{DATA_API_BASE}/transcripts/by_meeting/{meeting_id}")


def create_summary(
    meeting_id: str,
    transcript_id: Optional[str],
    summary_text: str,
) -> Dict[str, Any]:
    """
    注意：
    transcript_id 是否需要，取决于你们 summaries API 的 schema。
    """
    payload = {
        "meeting_id": meeting_id,
        "transcript_id": transcript_id,
        "summary": summary_text,
    }
    return post_json(f"{DATA_API_BASE}/summaries", payload)


def create_review(
    meeting_id: str,
    summary_id: Optional[str],
    rating: int,
    approved: bool,
    edited_summary: Optional[str] = None,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    这是把 summary output 喂给 review API 的示例。
    retraining 最后一般会从 reviews / approved feedback 导出到 feedback_records.jsonl
    """
    payload = {
        "meeting_id": meeting_id,
        "summary_id": summary_id,
        "rating": rating,
        "approved": approved,
        "edited_summary": edited_summary,
        "comment": comment,
    }
    return post_json(f"{DATA_API_BASE}/reviews", payload)


# =========================
# Main Pipeline
# =========================

def main():
    meeting_id = os.environ.get("MEETING_ID", "demo_001")
    bucket = os.environ.get("MINIO_BUCKET", "audio-files")
    object_key = os.environ.get("MINIO_OBJECT_KEY")

    if not object_key:
        raise ValueError("MINIO_OBJECT_KEY is required.")

    language = os.environ.get("ASR_LANGUAGE", DEFAULT_LANGUAGE)

    print(f"=== Step 0: Create / register meeting ===")
    try:
        meeting_resp = create_meeting(meeting_id, bucket, object_key)
        print("Meeting API response:")
        print(json.dumps(meeting_resp, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Create meeting failed (can still continue if meeting already exists): {e}")

    print(f"\n=== Step 1: Download audio from MinIO ===")
    local_audio_path = download_from_minio(bucket, object_key)
    print(f"Downloaded to: {local_audio_path}")

    try:
        print(f"\n=== Step 2: Run ASR model ===")
        asr_output = run_asr(local_audio_path, meeting_id, language)
        print("ASR output:")
        print(json.dumps(asr_output, indent=2, ensure_ascii=False))

        transcript_text = asr_output["transcript"]
        segments = asr_output.get("segments", [])
        asr_language = asr_output.get("language", language)

        print(f"\n=== Step 3: Store ASR output into /transcripts ===")
        transcript_resp = create_transcript(
            meeting_id=meeting_id,
            language=asr_language,
            transcript_text=transcript_text,
            segments=segments,
        )
        print("Transcript API response:")
        print(json.dumps(transcript_resp, indent=2, ensure_ascii=False))

        transcript_id = (
            transcript_resp.get("transcript_id")
            or transcript_resp.get("id")
            or transcript_resp.get("transcript", {}).get("id")
            if isinstance(transcript_resp, dict) else None
        )

        print(f"\n=== Step 4: Read transcript back from API ===")
        transcript_record = get_transcript_by_meeting(meeting_id)
        print("Transcript fetched by meeting:")
        print(json.dumps(transcript_record, indent=2, ensure_ascii=False))

        if isinstance(transcript_record, dict):
            if "transcript" in transcript_record and isinstance(transcript_record["transcript"], str):
                summarization_input = transcript_record["transcript"]
            elif "data" in transcript_record and isinstance(transcript_record["data"], dict):
                summarization_input = transcript_record["data"].get("transcript", transcript_text)
            else:
                summarization_input = transcript_text
        else:
            summarization_input = transcript_text

        print(f"\n=== Step 5: Run summarization model ===")
        summary_text = run_summarization(summarization_input)
        print("Summary output:")
        print(summary_text)

        print(f"\n=== Step 6: Store summary into /summaries ===")
        summary_resp = create_summary(
            meeting_id=meeting_id,
            transcript_id=transcript_id,
            summary_text=summary_text,
        )
        print("Summary API response:")
        print(json.dumps(summary_resp, indent=2, ensure_ascii=False))

        summary_id = (
            summary_resp.get("summary_id")
            or summary_resp.get("id")
            or summary_resp.get("summary", {}).get("id")
            if isinstance(summary_resp, dict) else None
        )

        print(f"\n=== Step 7: Create a review record ===")
        review_resp = create_review(
            meeting_id=meeting_id,
            summary_id=summary_id,
            rating=5,
            approved=True,
            edited_summary=summary_text,
            comment="auto test review",
        )
        print("Review API response:")
        print(json.dumps(review_resp, indent=2, ensure_ascii=False))

        print(f"\n=== Pipeline finished successfully ===")

    finally:
        try:
            os.remove(local_audio_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
