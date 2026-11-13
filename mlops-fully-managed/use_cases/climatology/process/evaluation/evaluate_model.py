#!/usr/bin/env python
"""
Score the registered model on the held-out test split

Two tasks:
  1. Pull the model package the register step created, load its artifact, and
     compute test metrics over the whole test split
  2. Publish the metrics on the metadata channel which the Model Card Lambda
     reads to fill in the card's evaluation details
"""

import argparse
import json
import logging
import os
import sys
import tarfile
import tempfile
from glob import glob

import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import sagemaker_mlflow  # noqa: F401

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MLFLOW_PARENT_RUN_ID = os.environ.get("MLFLOW_PARENT_RUN_ID") or None

# Written by SageMaker into every processing container
PROCESSING_JOB_CONFIG = "/opt/ml/config/processingjobconfig.json"


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="/opt/ml/processing/raw")
    # Appended by the engine from the parent register step at execution time
    parser.add_argument("--model_package_arn", type=str, default=None)
    # Fallback when run on its own: the newest version in this group
    parser.add_argument("--model_package_group_name", type=str, default=None)
    # Channel the engine uploads, so the card Lambda can read the evaluation back
    parser.add_argument(
        "--metadata_dir", type=str, default="/opt/ml/processing/metadata"
    )
    parser.add_argument("--mlflow_arn", type=str, default=None)
    parser.add_argument("--mlflow_experiment_name", type=str, default="Default")

    return parser.parse_known_args()


def split_s3_uri(s3_uri):
    _, _, rest = s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def resolve_model_package(sm_client, model_package_arn, model_package_group_name):
    if model_package_arn:
        return model_package_arn
    if not model_package_group_name:
        raise ValueError("Pass --model_package_arn or --model_package_group_name")

    packages = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["ModelPackageSummaryList"]
    if not packages:
        raise ValueError(f"No model packages in group '{model_package_group_name}'")
    return packages[0]["ModelPackageArn"]


def model_data_s3_uri(sm_client, model_package_arn):
    """
    The artifact a model package serves, whichever field it was registered with
    """
    container = sm_client.describe_model_package(ModelPackageName=model_package_arn)[
        "InferenceSpecification"
    ]["Containers"][0]
    if container.get("ModelDataUrl"):
        return container["ModelDataUrl"]
    return container["ModelDataSource"]["S3DataSource"]["S3Uri"]


def load_model(model_s3_uri):
    """
    Download and unpack model.tar.gz, returning the estimator and its feature names
    """
    bucket, key = split_s3_uri(model_s3_uri)
    model_dir = tempfile.mkdtemp()
    tar_path = os.path.join(model_dir, "model.tar.gz")
    boto3.client("s3").download_file(bucket, key, tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(model_dir, filter="data")

    bundle = joblib.load(os.path.join(model_dir, "model.joblib"))
    if isinstance(bundle, dict):
        return bundle["model"], bundle.get("feature_names")
    return bundle, None


def read_test_split(input_dir):
    paths = sorted(glob(os.path.join(input_dir, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")

    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    logger.info(f"Read {len(df):,} test rows from {len(paths)} file(s)")
    return df


def test_metrics(estimator, feature_names, df):
    """
    Target is the first column, as the processing step writes it
    """
    y = df.iloc[:, 0]
    X = df[feature_names] if feature_names is not None else df.iloc[:, 1:]
    predictions = estimator.predict(X)
    return {
        "test_rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "test_mae": float(mean_absolute_error(y, predictions)),
        "test_r2": float(r2_score(y, predictions)),
        "test_rows": len(df),
    }


def processing_job_info():
    """
    This job's ARN and the S3 uri of its test split input, or blanks outside SageMaker
    """
    try:
        with open(PROCESSING_JOB_CONFIG) as f:
            config = json.load(f)
    except FileNotFoundError:
        return None, None
    dataset_s3_uri = next(
        (
            channel["S3Input"]["S3Uri"]
            for channel in config.get("ProcessingInputs", [])
            if channel["InputName"] == "raw"
        ),
        None,
    )
    return config.get("ProcessingJobArn"), dataset_s3_uri


def write_metadata(metadata_dir, metadata):
    """
    The name has to match METADATA_FILE_NAME in src/process.py
    """
    os.makedirs(metadata_dir, exist_ok=True)
    path = os.path.join(metadata_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f)
    logger.info(f"Wrote metadata to {path}: {metadata}")


def start(args):
    sm_client = boto3.client("sagemaker")
    model_package_arn = resolve_model_package(
        sm_client, args.model_package_arn, args.model_package_group_name
    )
    model_s3_uri = model_data_s3_uri(sm_client, model_package_arn)
    logger.info(f"Evaluating {model_package_arn} ({model_s3_uri})")

    estimator, feature_names = load_model(model_s3_uri)
    df = read_test_split(args.input_dir)

    metrics = test_metrics(estimator, feature_names, df)
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")

    job_arn, dataset_s3_uri = processing_job_info()
    # The `evaluation` block is the contract with the Model Card Lambda
    write_metadata(
        args.metadata_dir,
        {
            "model_package_arn": model_package_arn,
            "evaluation": {
                "name": "Evaluation metrics",
                "observation": (
                    f"Scored in-process on all {metrics['test_rows']:,} rows of the "
                    "held-out test split"
                ),
                "job_arn": job_arn,
                "datasets": [dataset_s3_uri] if dataset_s3_uri else [],
                "metrics": metrics,
                "metadata": {
                    "mlflow_run_id": mlflow.active_run().info.run_id,
                    "model_data_s3_uri": model_s3_uri,
                },
            },
        },
    )

    # The reporting step reads these back off this run
    mlflow.log_params(
        {
            "model_package_arn": model_package_arn,
            "model_data_s3_uri": model_s3_uri,
        }
    )
    mlflow.log_metrics(metrics)


if __name__ == "__main__":
    args, _ = parse_args()

    mlflow.set_tracking_uri(args.mlflow_arn)
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(
        run_name="climatology-test-evaluation", parent_run_id=MLFLOW_PARENT_RUN_ID
    ):
        mlflow.set_tag("step", "test_evaluation")
        start(args)
