#!/usr/bin/env python
"""
Pick the better of the two trained models and set up the steps that follow
"""

import argparse
import json
import logging
import os
import sys

import boto3
import pandas as pd
import mlflow
import sagemaker_mlflow  # noqa: F401

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Set by the engine when this runs as a pipeline step, so this run nests under the
# one named after the execution. Empty when the script is run on its own
MLFLOW_PARENT_RUN_ID = os.environ.get("MLFLOW_PARENT_RUN_ID") or None


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    # Where the winning model.tar.gz is staged, which is the fixed artifact uri
    # the batch transform model points at
    # Destination, not an input: whichever model wins is copied here
    parser.add_argument("--best_model_output_s3_uri", type=str, required=True)
    # The winner's metrics, in the shape the model package expects. Also a
    # destination: the register step points at this uri before it is written
    parser.add_argument("--best_model_metrics_s3_uri", type=str, required=True)
    # Channel the engine turns into a property file, so later steps can read the
    # winning training job name out of this run
    parser.add_argument(
        "--metadata_dir", type=str, default="/opt/ml/processing/metadata"
    )
    parser.add_argument("--decision_metric", type=str, default="validation_rmse")
    # Lower is better for error metrics like RMSE, higher for R2
    parser.add_argument(
        "--decision_mode", type=str, choices=["min", "max"], default="min"
    )
    parser.add_argument("--mlflow_arn", type=str, default=None)
    parser.add_argument("--mlflow_experiment_name", type=str, default="Default")

    return parser.parse_known_args()


def split_s3_uri(s3_uri):
    _, _, rest = s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def latest_training_runs(experiment_name, decision_metric):
    """
    The most recent finished training run per model type

    Each training script tags its run with its model type, so this is what makes
    the comparison generic: adding a third model type needs no change here
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named '{experiment_name}'")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.step = 'training' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        output_format="pandas",
    )
    if not isinstance(runs, pd.DataFrame):
        raise TypeError(f"Expected a run DataFrame from MLflow, got {type(runs)}")
    if runs.empty:
        raise ValueError(f"No finished training runs in experiment '{experiment_name}'")

    metric_column = f"metrics.{decision_metric}"
    if metric_column not in runs.columns:
        raise ValueError(
            f"No training run logged '{decision_metric}'; found "
            f"{[c for c in runs.columns if c.startswith('metrics.')]}"
        )

    # search_runs is already newest-first, so the first row per model type wins
    latest = runs.dropna(subset=[metric_column]).drop_duplicates(
        subset=["tags.model_type"], keep="first"
    )
    logger.info(f"Comparing {len(latest)} model type(s) on {decision_metric}")

    return latest


def describe_model(training_job_name):
    """
    Artifact and training-data locations for one training job

    The S3 uri of an input channel is deliberately not exposed inside the
    training container, so provenance for the report comes from the job itself
    """
    job = boto3.client("sagemaker").describe_training_job(
        TrainingJobName=training_job_name
    )
    dataset_s3_uri = next(
        (
            channel["DataSource"]["S3DataSource"]["S3Uri"]
            for channel in job["InputDataConfig"]
            if channel["ChannelName"] == "train"
        ),
        "",
    )
    return job["ModelArtifacts"]["S3ModelArtifacts"], dataset_s3_uri


def stage_best_model(model_artifact, best_model_output_s3_uri):
    """
    Copy the winning artifact to a fixed location

    The transform job's model is defined before this step runs, so it has to
    point at a uri that does not change with the winner or the job name
    """
    source_bucket, source_key = split_s3_uri(model_artifact)
    destination_bucket, destination_key = split_s3_uri(best_model_output_s3_uri)

    boto3.client("s3").copy_object(
        Bucket=destination_bucket,
        Key=destination_key,
        CopySource={"Bucket": source_bucket, "Key": source_key},
    )
    logger.info(f"Staged {model_artifact} -> {best_model_output_s3_uri}")


def write_model_metrics(metrics, best_model_metrics_s3_uri):
    """
    Write the winner's metrics in the shape a model package reads

    The register step cannot build this itself: a model package's metrics come
    from one training job's MLflow run, and which job that is only becomes known
    here. standard_deviation is required by the schema and unknown from a single
    run, so it goes out as NaN
    """
    bucket, key = split_s3_uri(best_model_metrics_s3_uri)
    body = {
        name: {"value": value, "standard_deviation": "NaN"}
        for name, value in metrics.items()
    }
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(body).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(f"Wrote {len(body)} package metric(s) to {best_model_metrics_s3_uri}")


def write_metadata(metadata_dir, metadata):
    """
    Publish runtime facts on the metadata channel for later steps to read

    The engine exposes this as a property file which is the only way a value
    discovered here can be substituted into a step whose arguments were fixed
    when the pipeline was defined. The name is part of that contract - it has to
    match METADATA_FILE_NAME in src/process.py
    """
    os.makedirs(metadata_dir, exist_ok=True)
    path = os.path.join(metadata_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f)
    logger.info(f"Wrote metadata to {path}: {metadata}")


def start(args):
    """
    Pick the better of the trained models
    """
    runs = latest_training_runs(args.mlflow_experiment_name, args.decision_metric)
    metric_column = f"metrics.{args.decision_metric}"

    scores = dict(zip(runs["tags.model_type"], runs[metric_column]))
    logger.info(f"Scores: {scores}")

    chooser = min if args.decision_mode == "min" else max
    best_model_type = chooser(scores, key=lambda name: scores[name])
    best_run = runs[runs["tags.model_type"] == best_model_type].iloc[0]

    training_job_name = best_run["tags.sm_training_job_name"]
    model_artifact, dataset_s3_uri = describe_model(training_job_name)
    best_metrics = {
        column.removeprefix("metrics."): float(best_run[column])
        for column in runs.columns
        if column.startswith("metrics.") and pd.notna(best_run[column])
    }

    stage_best_model(model_artifact, args.best_model_output_s3_uri)
    write_model_metrics(best_metrics, args.best_model_metrics_s3_uri)
    write_metadata(
        args.metadata_dir,
        {
            "training_job_name": training_job_name,
            "model_type": best_model_type,
            "model_artifact": model_artifact,
            "staged_model_artifact": args.best_model_output_s3_uri,
            "metrics_s3_uri": args.best_model_metrics_s3_uri,
            "dataset_s3_uri": dataset_s3_uri,
            "mlflow_run_id": best_run["run_id"],
        },
    )
    logger.info(
        f"Best model: {best_model_type} "
        f"({args.decision_metric}={scores[best_model_type]:.4f})"
    )

    # The reporting step reads every one of these back off this run
    mlflow.log_params(
        {
            "best_model": best_model_type,
            "best_model_run_id": best_run["run_id"],
            "best_model_training_job": training_job_name,
            "best_model_artifact": model_artifact,
            "best_model_staged_artifact": args.best_model_output_s3_uri,
            "best_model_metrics": args.best_model_metrics_s3_uri,
            "dataset_s3_uri": dataset_s3_uri,
            "decision_metric": args.decision_metric,
            "decision_mode": args.decision_mode,
            "candidates": ", ".join(sorted(scores)),
        }
    )
    mlflow.log_metrics(
        {
            **{
                f"{args.decision_metric}_{model_type}": float(score)
                for model_type, score in scores.items()
            },
            f"best_{args.decision_metric}": float(scores[best_model_type]),
        }
    )

    # Carry the winner's full metric set onto this run so the report does not
    # have to go back to the training runs
    mlflow.log_metrics(
        {f"best_model_{name}": value for name, value in best_metrics.items()}
    )


if __name__ == "__main__":
    args, _ = parse_args()

    mlflow.set_tracking_uri(args.mlflow_arn)
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(
        run_name="climatology-select-best-model", parent_run_id=MLFLOW_PARENT_RUN_ID
    ):
        mlflow.set_tag("step", "selection")
        start(args)
