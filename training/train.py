import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import evaluate
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline,
)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
    except Exception:
        return "unknown"


def get_device_info() -> Dict[str, Any]:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
    else:
        info["gpu_name"] = "cpu"
        info["gpu_total_memory_gb"] = 0.0

    return info


def ensure_output_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_meeting_dataset(data_cfg: Dict[str, Any]) -> DatasetDict:
    dataset_name = data_cfg.get("dataset_name")
    dataset_config = data_cfg.get("dataset_config")
    text_column = data_cfg.get("text_column", "transcript")
    summary_column = data_cfg.get("summary_column", "summary")

    if dataset_name:
        ds = load_dataset(dataset_name, dataset_config)
    else:
        train_path = data_cfg.get("train_file")
        validation_path = data_cfg.get("validation_file")
        test_path = data_cfg.get("test_file")

        if not train_path:
            raise ValueError("No train_file provided in config.yaml")

        data_files = {"train": train_path}
        if validation_path:
            data_files["validation"] = validation_path
        if test_path:
            data_files["test"] = test_path

        extension = Path(train_path).suffix.lower()
        if extension not in [".json", ".jsonl", ".csv"]:
            raise ValueError("Supported local file types: json, jsonl, csv")

        loader_name = "json" if extension in [".json", ".jsonl"] else "csv"
        ds = load_dataset(loader_name, data_files=data_files)

    if "validation" not in ds:
        split = ds["train"].train_test_split(
            test_size=data_cfg.get("validation_split", 0.1),
            seed=data_cfg.get("seed", 42),
        )
        ds = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
        })

    if "test" not in ds and data_cfg.get("make_test_from_validation", False):
        split = ds["validation"].train_test_split(
            test_size=0.5,
            seed=data_cfg.get("seed", 42),
        )
        ds = DatasetDict({
            "train": ds["train"],
            "validation": split["train"],
            "test": split["test"],
        })

    required_columns = {text_column, summary_column}
    for split_name in ds.keys():
        missing = required_columns.difference(ds[split_name].column_names)
        if missing:
            raise ValueError(
                f"Split '{split_name}' is missing required columns: {missing}"
            )

    return ds


def maybe_limit_dataset(
    ds: DatasetDict,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
) -> DatasetDict:
    out = {}
    for split_name, split_ds in ds.items():
        limit = None
        if split_name == "train":
            limit = max_train
        elif split_name == "validation":
            limit = max_val
        elif split_name == "test":
            limit = max_test

        if limit is not None:
            limit = min(limit, len(split_ds))
            out[split_name] = split_ds.select(range(limit))
        else:
            out[split_name] = split_ds

    return DatasetDict(out)


def build_preprocess_fn(tokenizer, cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    text_column = data_cfg["text_column"]
    summary_column = data_cfg["summary_column"]
    source_prefix = model_cfg.get("source_prefix", "")
    max_source_length = model_cfg.get("max_source_length", 1024)
    max_target_length = model_cfg.get("max_target_length", 256)

    def preprocess_fn(examples):
        inputs = [source_prefix + str(x) for x in examples[text_column]]
        targets = [str(x) for x in examples[summary_column]]

        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_fn


def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    return preds, labels


def compute_metrics_builder(tokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        result = {k: round(v * 100, 4) for k, v in result.items()}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = round(float(np.mean(prediction_lens)), 4)

        return result

    return compute_metrics


class SummarizationPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_dir = context.artifacts["hf_model_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipe = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,
        )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        if isinstance(model_input, pd.DataFrame):
            if "text" in model_input.columns:
                texts = model_input["text"].astype(str).tolist()
            else:
                texts = model_input.iloc[:, 0].astype(str).tolist()
        elif isinstance(model_input, list):
            texts = [str(x) for x in model_input]
        else:
            texts = [str(model_input)]

        outputs = self.pipe(texts, truncation=True, max_new_tokens=128)

        summaries = []
        for out in outputs:
            if isinstance(out, list):
                out = out[0]
            if isinstance(out, dict) and "summary_text" in out:
                summaries.append(out["summary_text"])
            else:
                summaries.append(str(out))

        return pd.DataFrame({"summary": summaries})


def log_and_optionally_register_model(
    output_dir: str,
    run_id: str,
    cfg: Dict[str, Any],
    eval_metrics: Dict[str, Any],
):
    registered_model_name = (
        cfg.get("mlflow", {}).get("registered_model_name")
        or os.environ.get("MLFLOW_REGISTERED_MODEL_NAME")
    )

    quality_gate_metric = cfg["train"].get("quality_gate_metric", "eval_rougeL")
    quality_gate_threshold = float(cfg["train"].get("quality_gate_threshold", 32.0))

    metric_value = eval_metrics.get(quality_gate_metric)
    metric_value = float(metric_value) if metric_value is not None else None
    passed_gate = metric_value is not None and metric_value >= quality_gate_threshold

    mlflow.log_param("quality_gate_metric", quality_gate_metric)
    mlflow.log_param("quality_gate_threshold", quality_gate_threshold)
    mlflow.log_param("quality_gate_passed", passed_gate)
    if metric_value is not None:
        mlflow.log_metric("quality_gate_metric_value", metric_value)

    input_example = pd.DataFrame(
        {
            "text": [
                "Alice: We need to finish the UI this week. Bob: I will handle the backend API."
            ]
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        local_model_dir = Path(tmpdir) / "registered_model"

        mlflow.pyfunc.save_model(
            path=str(local_model_dir),
            python_model=SummarizationPyFuncModel(),
            artifacts={"hf_model_dir": str(Path(output_dir).resolve())},
            input_example=input_example,
            pip_requirements=[
                "mlflow==2.19.0",
                "transformers",
                "torch",
                "sentencepiece",
                "pandas",
            ],
        )

        mlflow.log_artifacts(str(local_model_dir), artifact_path="registered_model")

    model_uri = f"runs:/{run_id}/registered_model"

    if registered_model_name and passed_gate:
        registration = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )
        return model_uri, registered_model_name, registration, passed_gate

    return model_uri, None, None, passed_gate


def train(cfg: Dict[str, Any]) -> None:
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tracking_uri = cfg["mlflow"].get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
    experiment_name = cfg["mlflow"].get("experiment_name", "jitsi-summarization")
    run_name = cfg["mlflow"].get(
        "run_name",
        f"{cfg['candidate_name']}_{int(time.time())}"
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    device_info = get_device_info()
    git_sha = get_git_sha()

    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id

        mlflow.log_param("candidate_name", cfg["candidate_name"])
        mlflow.log_param("git_sha", git_sha)

        flat_params = {
            "model_name": cfg["model"]["model_name"],
            "max_source_length": cfg["model"].get("max_source_length", 1024),
            "max_target_length": cfg["model"].get("max_target_length", 256),
            "learning_rate": cfg["train"]["learning_rate"],
            "per_device_train_batch_size": cfg["train"]["per_device_train_batch_size"],
            "per_device_eval_batch_size": cfg["train"]["per_device_eval_batch_size"],
            "num_train_epochs": cfg["train"]["num_train_epochs"],
            "weight_decay": cfg["train"].get("weight_decay", 0.0),
            "warmup_ratio": cfg["train"].get("warmup_ratio", 0.0),
            "gradient_accumulation_steps": cfg["train"].get("gradient_accumulation_steps", 1),
            "fp16": cfg["train"].get("fp16", False),
            "generation_num_beams": cfg["train"].get("generation_num_beams", 4),
            "dataset_name": cfg["data"].get("dataset_name", "local_files"),
            "text_column": cfg["data"]["text_column"],
            "summary_column": cfg["data"]["summary_column"],
            "seed": seed,
        }

        for k, v in flat_params.items():
            mlflow.log_param(k, v)

        for k, v in device_info.items():
            mlflow.log_param(k, v)

        model_name = cfg["model"]["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        raw_ds = load_meeting_dataset(cfg["data"])
        raw_ds = maybe_limit_dataset(
            raw_ds,
            cfg["data"].get("max_train_samples"),
            cfg["data"].get("max_eval_samples"),
            cfg["data"].get("max_test_samples"),
        )

        mlflow.log_param("train_size", len(raw_ds["train"]))
        mlflow.log_param("validation_size", len(raw_ds["validation"]))
        if "test" in raw_ds:
            mlflow.log_param("test_size", len(raw_ds["test"]))

        preprocess_fn = build_preprocess_fn(tokenizer, cfg)
        tokenized_ds = raw_ds.map(
            preprocess_fn,
            batched=True,
            remove_columns=raw_ds["train"].column_names,
            desc="Tokenizing dataset",
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="longest",
        )

        output_dir = cfg["train"]["output_dir"]
        ensure_output_dir(output_dir)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            eval_strategy=cfg["train"].get("eval_strategy", "epoch"),
            save_strategy=cfg["train"].get("save_strategy", "epoch"),
            logging_strategy=cfg["train"].get("logging_strategy", "steps"),
            logging_steps=cfg["train"].get("logging_steps", 10),
            learning_rate=cfg["train"]["learning_rate"],
            per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg["train"]["per_device_eval_batch_size"],
            weight_decay=cfg["train"].get("weight_decay", 0.0),
            num_train_epochs=cfg["train"]["num_train_epochs"],
            predict_with_generate=True,
            fp16=cfg["train"].get("fp16", False) and torch.cuda.is_available(),
            gradient_accumulation_steps=cfg["train"].get("gradient_accumulation_steps", 1),
            warmup_ratio=cfg["train"].get("warmup_ratio", 0.0),
            load_best_model_at_end=True,
            metric_for_best_model=cfg["train"].get("metric_for_best_model", "eval_rougeL"),
            greater_is_better=cfg["train"].get("greater_is_better", True),
            save_total_limit=cfg["train"].get("save_total_limit", 2),
            report_to=[],
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["validation"],
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_builder(tokenizer),
        )

        start_time = time.time()

        train_result = trainer.train()
        train_metrics = train_result.metrics

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"train_{k}", float(v))

        eval_metrics = trainer.evaluate(
            eval_dataset=tokenized_ds["validation"],
            metric_key_prefix="eval",
        )
        for k, v in eval_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        if "test" in tokenized_ds:
            test_metrics = trainer.evaluate(
                eval_dataset=tokenized_ds["test"],
                metric_key_prefix="test",
            )
            for k, v in test_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

        total_train_sec = time.time() - start_time
        mlflow.log_metric("total_train_time_sec", total_train_sec)

        if cfg["train"]["num_train_epochs"] > 0:
            mlflow.log_metric(
                "time_per_epoch_sec",
                total_train_sec / cfg["train"]["num_train_epochs"],
            )

        if torch.cuda.is_available():
            peak_vram_bytes = torch.cuda.max_memory_allocated()
            mlflow.log_metric("peak_vram_gb", peak_vram_bytes / (1024 ** 3))

        config_dump_path = Path(output_dir) / "resolved_config.json"
        with open(config_dump_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        mlflow.log_artifact(str(config_dump_path))

        mlflow.log_artifacts(output_dir, artifact_path="hf_model_files")

        model_uri, registered_model_name, _, passed_gate = log_and_optionally_register_model(
            output_dir=output_dir,
            run_id=run_id,
            cfg=cfg,
            eval_metrics=eval_metrics,
        )

        print("\nTraining finished successfully.")
        print(f"Model saved to: {output_dir}")
        print(f"MLflow experiment: {experiment_name}")
        print(f"MLflow model URI: {model_uri}")
        print(f"Quality gate passed: {passed_gate}")

        if registered_model_name:
            print(f"Registered model name: {registered_model_name}")
        else:
            print("Model was NOT registered because it did not pass the quality gate.")

        if tracking_uri:
            print(f"MLflow tracking URI: {tracking_uri}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a meeting summarization model with MLflow tracking."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
