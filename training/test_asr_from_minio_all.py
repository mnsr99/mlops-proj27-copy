import os
import tempfile
from pathlib import Path

import boto3
import mlflow.pyfunc
import pandas as pd


def is_audio_file(key: str) -> bool:
    audio_exts = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".aac", ".ogg"}
    return Path(key).suffix.lower() in audio_exts


def main():
    minio_endpoint = os.environ.get("MINIO_ENDPOINT", "http://129.114.26.182:30900")
    minio_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
    minio_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")

    bucket_name = os.environ.get("MINIO_BUCKET", "audio-files")
    language = os.environ.get("ASR_LANGUAGE", "en")
    model_uri = os.environ.get("ASR_MODEL_URI", "models:/jitsi-asr/1")

    s3 = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
    )

    print(f"Listing objects in bucket: {bucket_name}")
    resp = s3.list_objects_v2(Bucket=bucket_name)

    if "Contents" not in resp:
        print("No objects found in bucket.")
        return

    audio_keys = [obj["Key"] for obj in resp["Contents"] if is_audio_file(obj["Key"])]

    if not audio_keys:
        print("No audio/video files found in bucket.")
        return

    print("Audio objects found:")
    for key in audio_keys:
        print(f" - {key}")

    print(f"\nLoading ASR model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    all_results = []

    for idx, object_key in enumerate(audio_keys, start=1):
        suffix = Path(object_key).suffix or ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            local_audio_path = tmp.name

        try:
            print(f"\n[{idx}/{len(audio_keys)}] Downloading {object_key} -> {local_audio_path}")
            s3.download_file(bucket_name, object_key, local_audio_path)

            meeting_id = Path(object_key).stem

            df = pd.DataFrame([
                {
                    "meeting_id": meeting_id,
                    "audio_path": local_audio_path,
                    "language": language,
                    "source": "minio_audio",
                }
            ])

            print(f"[{idx}/{len(audio_keys)}] Running ASR inference for {object_key} ...")
            result = model.predict(df)

            records = result.to_dict(orient="records")
            all_results.extend(records)

            print(f"[{idx}/{len(audio_keys)}] Result:")
            print(records)

        except Exception as e:
            print(f"[{idx}/{len(audio_keys)}] Failed on {object_key}: {e}")

        finally:
            try:
                os.remove(local_audio_path)
            except OSError:
                pass

    print("\nAll finished.")
    print("\nCollected results:")
    for item in all_results:
        print(item)


if __name__ == "__main__":
    main()
