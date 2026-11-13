#!/usr/bin/env python

import argparse
import logging
import os
import sys
from glob import glob

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import mlflow
import sagemaker_mlflow  # noqa: F401

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Set by the engine when this runs as a pipeline step, so this run nests under the
# one named after the execution. Empty when the script is run on its own
MLFLOW_PARENT_RUN_ID = os.environ.get("MLFLOW_PARENT_RUN_ID") or None

# Compared against the other model type by the evaluation step, so the name has
# to match the one random_forest_train.py logs
DECISION_METRIC = "validation_rmse"
MODEL_TYPE = "xgboost"


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    num_cpus = os.environ.get("SM_NUM_CPUS", 4)
    parser.add_argument("--n_jobs", type=int, default=num_cpus)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    # Both model types subsample identically so the comparison stays fair
    parser.add_argument("--max_train_rows", type=int, default=1000000)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--mlflow_arn", type=str, default=None)
    parser.add_argument("--mlflow_experiment_name", type=str, default="Default")

    return parser.parse_known_args()


def read_channel(channel):
    """
    Read the CSVs the processing job wrote to one input channel

    SageMaker mounts each channel at /opt/ml/input/data/<channel>. The processing
    job writes one file per instance with the target in the first column
    """
    channel_dir = os.environ.get(
        f"SM_CHANNEL_{channel.upper()}", f"/opt/ml/input/data/{channel}"
    )
    paths = sorted(glob(os.path.join(channel_dir, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {channel_dir}")

    logger.info(f"Reading {channel} data from {len(paths)} file(s)")
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    logger.info(f"{channel} shape: {df.shape}")

    return df.iloc[:, 1:], df.iloc[:, 0]


def subsample(X, y, max_rows, random_state):
    """
    Cap the training set so the demo finishes in minutes
    """
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y

    sampled = X.sample(n=max_rows, random_state=random_state)
    logger.info(f"Sampled {max_rows:,} of {len(X):,} training rows")
    return sampled, y.loc[sampled.index]


def start(args):
    """
    Train an XGBoost regressor on daily high temperature
    """
    X_train, y_train = read_channel("train")
    X_validation, y_validation = read_channel("validation")
    X_train, y_train = subsample(X_train, y_train, args.max_train_rows, args.random_state)

    hyperparameters = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "n_jobs": args.n_jobs,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": args.random_state,
        "verbosity": 1,
    }

    model = XGBRegressor(**hyperparameters)
    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)
    metrics = {
        DECISION_METRIC: float(
            np.sqrt(mean_squared_error(y_validation, predictions))
        ),
        "validation_mae": float(mean_absolute_error(y_validation, predictions)),
        "validation_r2": float(r2_score(y_validation, predictions)),
    }
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")

    if mlflow.active_run() is not None:
        mlflow.log_params(
            {**hyperparameters, "max_train_rows": args.max_train_rows}
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metric("train_rows", len(X_train))

    # The handler rebuilds a named DataFrame from these, so batch transform can
    # score a bare CSV without the feature order being hardcoded anywhere
    joblib.dump(
        {"model": model, "feature_names": list(X_train.columns)},
        os.path.join(os.environ["SM_MODEL_DIR"], "model.joblib"),
    )


if __name__ == "__main__":
    args, _ = parse_args()

    mlflow_arn = (args.mlflow_arn or "").strip()
    if mlflow_arn and mlflow_arn.lower() != "none":
        mlflow.set_tracking_uri(mlflow_arn)
        mlflow.set_experiment(args.mlflow_experiment_name)

        with mlflow.start_run(
            run_name=f"climatology-{MODEL_TYPE}", parent_run_id=MLFLOW_PARENT_RUN_ID
        ):
            # The evaluation step selects on model_type and resolves the model
            # artifact by describing the tagged training job
            mlflow.set_tag("model_type", MODEL_TYPE)
            mlflow.set_tag("step", "training")
            mlflow.set_tag("sm_training_job_name", os.environ.get("TRAINING_JOB_NAME"))
            start(args)
    else:
        logger.info("No MLflow ARN provided; skipping experiment tracking")
        start(args)
