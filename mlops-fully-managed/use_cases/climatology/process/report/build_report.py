#!/usr/bin/env python

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from glob import glob

import boto3
import numpy as np
import pandas as pd
import mlflow
import sagemaker_mlflow  # noqa: F401

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MLFLOW_PARENT_RUN_ID = os.environ.get("MLFLOW_PARENT_RUN_ID") or None

# Batch transform joined each prediction back onto its input record and the
# output_filter kept only the true value and the prediction
PREDICTION_COLUMNS = ["actual", "predicted"]


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="/opt/ml/processing/raw")
    parser.add_argument("--report_s3_uri", type=str, required=True)
    parser.add_argument("--mlflow_arn", type=str, default=None)
    parser.add_argument("--mlflow_experiment_name", type=str, default="Default")

    return parser.parse_known_args()


def split_s3_uri(s3_uri):
    _, _, rest = s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def read_predictions(input_dir):
    """
    Read every batch transform output file the job produced
    """
    paths = sorted(
        path
        for pattern in ("**/*.out", "**/*.csv")
        for path in glob(os.path.join(input_dir, pattern), recursive=True)
    )
    if not paths:
        raise FileNotFoundError(f"No batch transform output found under {input_dir}")

    logger.info(f"Reading predictions from {len(paths)} file(s)")
    predictions = pd.concat(
        [pd.read_csv(path, header=None, names=PREDICTION_COLUMNS) for path in paths],
        ignore_index=True,
    )
    logger.info(f"Read {len(predictions):,} scored records")

    return predictions.apply(pd.to_numeric, errors="coerce").dropna()


def latest_run(experiment_name, step):
    """
    The newest run a given pipeline step logged - inferred by its step tag
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named '{experiment_name}'")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.step = '{step}'",
        order_by=["attributes.start_time DESC"],
        output_format="pandas",
    )
    if not isinstance(runs, pd.DataFrame):
        raise TypeError(f"Expected a run DataFrame from MLflow, got {type(runs)}")
    if runs.empty:
        raise ValueError(f"No {step} run in experiment '{experiment_name}'")

    return runs.iloc[0]


def field(run, name, default="unknown"):
    value = run.get(name)
    return default if value is None or pd.isna(value) else value


def comparison_table(run, decision_metric, best_model):
    """
    One row per candidate model type as scored by the selection step
    """
    candidates = str(field(run, "params.candidates", "")).split(", ")
    rows = []
    for model_type in sorted(name for name in candidates if name):
        score = run.get(f"metrics.{decision_metric}_{model_type}")
        marker = " **(selected)**" if model_type == best_model else ""
        score_text = "n/a" if score is None or pd.isna(score) else f"{score:.4f}"
        rows.append(f"| `{model_type}`{marker} | {score_text} |")

    if not rows:
        return "_No candidate scores were recorded._"

    return "\n".join(
        [
            f"| Model | {decision_metric} |",
            "| --- | --- |",
            *rows,
        ]
    )


def best_model_metrics(run):
    """
    Every metric the winning training run logged
    """
    prefix = "metrics.best_model_"
    metrics = {
        name.removeprefix(prefix): run[name]
        for name in run.index
        if name.startswith(prefix) and pd.notna(run[name])
    }
    if not metrics:
        return "_No metrics were recorded for the winning model._"

    return "\n".join(
        [
            "| Metric | Value |",
            "| --- | --- |",
            *(f"| {name} | {value:.4f} |" for name, value in sorted(metrics.items())),
        ]
    )


def batch_metrics(predictions):
    """
    Score the transform output against the labels it was joined back onto
    """
    residuals = np.asarray(
        predictions["actual"] - predictions["predicted"], dtype=float
    )
    return {
        "samples_scored": len(predictions),
        "batch_rmse": float(np.sqrt(np.mean(residuals**2))),
        "batch_mae": float(np.mean(np.abs(residuals))),
    }


def build_markdown(predictions, run, test_run, metrics):
    """
    Assemble the report

    Model selection comes from the selection run, test metrics from the evaluation
    run, and the batch numbers are recomputed from the transform output
    """
    decision_metric = str(field(run, "params.decision_metric", "validation_rmse"))
    best_model = str(field(run, "params.best_model"))

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""# Climatology Pipeline Report

Generated {generated_at}

## Batch inference

| | |
| --- | --- |
| Samples scored | **{metrics["samples_scored"]:,}** |
| Batch RMSE (deg C) | {metrics["batch_rmse"]:.4f} |
| Batch MAE (deg C) | {metrics["batch_mae"]:.4f} |
| Mean actual high (deg C) | {predictions["actual"].mean():.2f} |
| Mean predicted high (deg C) | {predictions["predicted"].mean():.2f} |

## Test set

Scored in-process by the evaluation step on the full test split, using the
registered model package.

| | |
| --- | --- |
| Rows | **{int(field(test_run, "metrics.test_rows", 0)):,}** |
| Test RMSE (deg C) | {field(test_run, "metrics.test_rmse", float("nan")):.4f} |
| Test MAE (deg C) | {field(test_run, "metrics.test_mae", float("nan")):.4f} |
| Test R2 | {field(test_run, "metrics.test_r2", float("nan")):.4f} |

## Best performing model

Selected on `{decision_metric}` across {len(str(field(run, "params.candidates", "")).split(", "))} candidate model types.

{comparison_table(run, decision_metric, best_model)}

The winner was **`{best_model}`**, with the following validation metrics:

{best_model_metrics(run)}

## Provenance

| | |
| --- | --- |
| Training dataset | `{field(run, "params.dataset_s3_uri")}` |
| Training job | `{field(run, "params.best_model_training_job")}` |
| Model artifact | `{field(run, "params.best_model_artifact")}` |
| Scored artifact | `{field(run, "params.best_model_staged_artifact")}` |
| Model package | `{field(test_run, "params.model_package_arn")}` |
| Select MLflow run | `{field(run, "run_id")}` |
| Evaluation MLflow run | `{field(test_run, "run_id")}` |
"""


def start(args):
    """
    Summarise the batch inference run as a markdown file in S3
    """
    predictions = read_predictions(args.input_dir)
    run = latest_run(args.mlflow_experiment_name, "selection")
    test_run = latest_run(args.mlflow_experiment_name, "test_evaluation")
    metrics = batch_metrics(predictions)
    markdown = build_markdown(predictions, run, test_run, metrics)

    bucket, key = split_s3_uri(args.report_s3_uri)
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=markdown.encode("utf-8"),
        ContentType="text/markdown",
    )
    logger.info(f"Wrote report to {args.report_s3_uri}")
    print(markdown)

    mlflow.log_params(
        {
            "report_s3_uri": args.report_s3_uri,
            "best_model": field(run, "params.best_model"),
            "dataset_s3_uri": field(run, "params.dataset_s3_uri"),
            "selection_run_id": field(run, "run_id"),
            "evaluation_run_id": field(test_run, "run_id"),
            "model_package_arn": field(test_run, "params.model_package_arn"),
        }
    )
    mlflow.log_metrics(metrics)
    mlflow.log_text(markdown, "report.md")


if __name__ == "__main__":
    args, _ = parse_args()

    mlflow.set_tracking_uri(args.mlflow_arn)
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(
        run_name="climatology-reporting", parent_run_id=MLFLOW_PARENT_RUN_ID
    ):
        mlflow.set_tag("step", "reporting")
        start(args)
