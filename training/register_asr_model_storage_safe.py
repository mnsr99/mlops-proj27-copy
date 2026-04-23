import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import mlflow.pyfunc
import pandas as pd
from faster_whisper import WhisperModel
from mlflow.tracking import MlflowClient


class FasterWhisperPyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None
        self.model_size_or_path = "small"
        self.device = "cpu"
        self.compute_type = "int8"
        self.beam_size = 5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def load_context(self, context):
        model_config = getattr(context, "model_config", {}) or {}

        self.model_size_or_path = model_config.get("model_size_or_path", "small")
        self.device = model_config.get("device", "cpu")
        self.compute_type = model_config.get("compute_type", "int8")
        self.beam_size = int(model_config.get("beam_size", 5))

        self.model = WhisperModel(
            self.model_size_or_path,
            device=self.device,
            compute_type=self.compute_type,
        )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        required_cols = {"meeting_id", "audio_path"}
        missing = required_cols.difference(model_input.columns)
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")

        results: List[Dict[str, Any]] = []

        for _, row in model_input.iterrows():
            meeting_id = str(row["meeting_id"])
            audio_path = str(row["audio_path"])

            language = None
            if "language" in model_input.columns and pd.notna(row.get("language")):
                language = str(row["language"])

            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=self.beam_size,
            )

            segment_items = []
            transcript_parts = []

            for seg in segments:
                text = seg.text.strip()
                transcript_parts.append(text)
                segment_items.append(
                    {
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": text,
                    }
                )

            results.append(
                {
                    "meeting_id": meeting_id,
                    "language": info.language,
                    "transcript": " ".join(transcript_parts).strip(),
                    "segments": segment_items,
                }
            )

        return pd.DataFrame(results)


def register_asr_model() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.27.10:30500")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "jitsi-asr")
    registered_model_name = os.environ.get("MLFLOW_ASR_REGISTERED_MODEL_NAME", "jitsi-asr")
    registered_model_alias = os.environ.get("MLFLOW_ASR_REGISTERED_MODEL_ALIAS", "production")

    model_size_or_path = os.environ.get("ASR_MODEL_SIZE_OR_PATH", "small")
    device = os.environ.get("ASR_DEVICE", "cpu")
    compute_type = os.environ.get("ASR_COMPUTE_TYPE", "int8")
    beam_size = int(os.environ.get("ASR_BEAM_SIZE", "5"))

    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_registry_uri(tracking_uri)
    except Exception:
        pass
    mlflow.set_experiment(experiment_name)

    input_example = pd.DataFrame(
        [
            {
                "meeting_id": "demo_001",
                "audio_path": "/data/recordings/demo_001.wav",
                "language": "en",
                "source": "jitsi_recording",
            }
        ]
    )

    with mlflow.start_run(run_name="faster_whisper_small_register") as active_run:
        run_id = active_run.info.run_id

        mlflow.log_param("asr_backend", "faster-whisper")
        mlflow.log_param("model_size_or_path", model_size_or_path)
        mlflow.log_param("device", device)
        mlflow.log_param("compute_type", compute_type)
        mlflow.log_param("beam_size", beam_size)
        mlflow.log_param("registered_model_name", registered_model_name)
        mlflow.log_param("registered_model_alias", registered_model_alias)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_model_dir = Path(tmpdir) / "registered_asr_model"

            pyfunc_model = FasterWhisperPyFuncModel()

            mlflow.pyfunc.save_model(
                path=str(local_model_dir),
                python_model=pyfunc_model,
                input_example=input_example,
                pip_requirements=[
                    "mlflow==2.19.0",
                    "faster-whisper",
                    "pandas",
                ],
                model_config={
                    "model_size_or_path": model_size_or_path,
                    "device": device,
                    "compute_type": compute_type,
                    "beam_size": beam_size,
                },
            )

            mlflow.log_artifacts(str(local_model_dir), artifact_path="registered_asr_model")

        model_uri = f"runs:/{run_id}/registered_asr_model"
        registration = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )

        client = MlflowClient()
        try:
            client.set_registered_model_alias(
                name=registered_model_name,
                alias=registered_model_alias,
                version=registration.version,
            )
        except Exception as e:
            print(f"Warning: failed to set alias '{registered_model_alias}': {e}")

        print("ASR model registered successfully.")
        print(f"MLflow tracking URI: {tracking_uri}")
        print(f"MLflow experiment: {experiment_name}")
        print(f"Model URI: {model_uri}")
        print(f"Registered model name: {registered_model_name}")
        print(f"Registered version: {registration.version}")
        print(f"Registered alias: {registered_model_alias}")


if __name__ == "__main__":
    register_asr_model()
