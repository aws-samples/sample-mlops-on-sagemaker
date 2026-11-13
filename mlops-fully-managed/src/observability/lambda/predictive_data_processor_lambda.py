"""
Predictive ML monitoring Lambda - processes batches of DataCapture JSONL records
dispatched by Step Functions Distributed Map.

Receives input shaped like:
{
  "BatchInput": { "s3_bucket": "...", "s3_key": "..." },
  "Items": [ {captureData record}, ... ]
}

Runs evidently data drift and model quality reports, logs to MLflow. Model quality
is binary classification or regression, picked by TASK_TYPE.
"""

import json
import logging
import os
import re
import base64
from datetime import datetime

import boto3
import pandas as pd
import mlflow
from evidently import Report, Dataset, DataDefinition, BinaryClassification, Regression
from evidently.presets import (
    DataDriftPreset,
    DataSummaryPreset,
    ClassificationPreset,
    RegressionPreset,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
sns_client = boto3.client("sns")

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SAGEMAKER_ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
DATA_CAPTURE_BUCKET = os.environ["DATA_CAPTURE_BUCKET"]
REFERENCE_BUCKET = os.environ.get("REFERENCE_BUCKET", DATA_CAPTURE_BUCKET)
BASELINE_S3_KEY = os.environ["BASELINE_S3_KEY"]
GROUND_TRUTH_S3_KEY = os.environ.get("GROUND_TRUTH_S3_KEY", "")
FEATURE_COLUMNS = os.environ["FEATURE_COLUMNS"].split(",")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.1"))
TASK_TYPE = os.environ.get("TASK_TYPE", "binary_classification")
if TASK_TYPE == "regression":
    # Evidently metric name -> limit, e.g. {"MAE": 3.0, "R2Score": 0.8}
    QUALITY_THRESHOLDS = {
        name: float(limit)
        for name, limit in json.loads(
            os.environ.get("REGRESSION_THRESHOLDS") or "{}"
        ).items()
    }
else:
    QUALITY_THRESHOLDS = {
        "F1Score": float(os.environ.get("THRESHOLD_F1", "0.7")),
        "Accuracy": float(os.environ.get("THRESHOLD_ACCURACY", "0.8")),
        "RocAuc": float(os.environ.get("THRESHOLD_ROC_AUC", "0.75")),
    }

# Breached when below the limit
HIGHER_IS_BETTER = {"F1Score", "Accuracy", "RocAuc", "Precision", "Recall", "R2Score"}


def decode(payload):
    raw = payload.get("data", "")
    if payload.get("encoding") == "BASE64":
        raw = base64.b64decode(raw).decode("utf-8")
    return raw


def parse_record(record):
    """
    One row per CSV line, since a single request can carry several records

    Rows after the first get a :<n> suffix on the inference id, so ground truth for a
    multi-row request is keyed <id>, <id>:1, <id>:2, ...
    """
    event_metadata = record.get("eventMetadata", {})
    inference_id = event_metadata.get("inferenceId", event_metadata.get("eventId", ""))
    capture = record.get("captureData", {})
    input_rows = decode(capture.get("endpointInput", {})).strip().splitlines()
    output_rows = decode(capture.get("endpointOutput", {})).strip().splitlines()

    if len(input_rows) != len(output_rows):
        raise ValueError(
            f"{len(input_rows)} input rows but {len(output_rows)} predictions in {inference_id}"
        )

    rows = []
    for index, (row_in, row_out) in enumerate(zip(input_rows, output_rows)):
        try:
            prediction = float(row_out.strip())
        except (ValueError, TypeError):
            prediction = float("nan")

        row = {
            "inference_id": inference_id if index == 0 else f"{inference_id}:{index}",
            "prediction": prediction,
        }
        for col, val in zip(FEATURE_COLUMNS, row_in.split(",")):
            row[col] = val
        rows.append(row)
    return rows


def log_drift_metrics(report_dict):
    metrics = report_dict.get("metrics", []) or []
    drifted_columns_logged = False

    for metric in metrics:
        # Evidently names carry their arguments, e.g. ValueDrift(column=tmax_lag_1,method=...)
        full_name = metric.get("metric_name", "")
        metric_name = full_name.split("(")[0].strip()
        value = metric.get("value")

        if metric_name == "DriftedColumnsCount":
            drifted_columns_logged = True
            if isinstance(value, dict):
                count = float(value.get("count", 0))
                share = float(value.get("share", 0))
            else:
                count = float(value) if value is not None else 0.0
                share = 0.0
            mlflow.log_metric("DriftedColumnsCount.count", count)
            mlflow.log_metric("DriftedColumnsCount.share", share)
            continue

        if metric_name == "ValueDrift":
            match = re.search(r"column=([^,)]+)", full_name)
            column = match.group(1) if match else "unknown"
            try:
                numeric_val = float(value)
            except (TypeError, ValueError):
                continue
            if numeric_val > DRIFT_THRESHOLD:
                mlflow.log_metric(f"ValueDrift:{column}", numeric_val)

    if not drifted_columns_logged:
        mlflow.log_metric("DriftedColumnsCount.count", 0.0)
        mlflow.log_metric("DriftedColumnsCount.share", 0.0)


def metric_value(value):
    """
    A metric's headline number; MAE, MAPE and MeanError report {mean, std}
    """
    if isinstance(value, dict) and "mean" in value:
        value = value["mean"]
    return float(value)


def log_quality_metrics(report_dict):
    for metric in report_dict.get("metrics", []):
        metric_name = metric.get("metric_name", "").split("(")[0].strip()
        value = metric.get("value")

        if isinstance(value, dict) and "mean" not in value:
            for label, val in value.items():
                try:
                    mlflow.log_metric(f"{metric_name}_label_{label}", float(val))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                mlflow.log_metric(metric_name, metric_value(value))
            except (TypeError, ValueError):
                continue


def check_thresholds(report_dict, thresholds):
    breached = {}
    for metric in report_dict.get("metrics", []):
        name = metric.get("metric_name", "").split("(")[0].strip()
        if name in thresholds:
            try:
                val = metric_value(metric.get("value"))
            except (TypeError, ValueError):
                continue
            if name in HIGHER_IS_BETTER:
                if val < thresholds[name]:
                    breached[name] = val
            elif val > thresholds[name]:
                breached[name] = val
    return breached


def send_alert(breached, run_name):
    if not SNS_TOPIC_ARN:
        return
    lines = "\n".join(
        f"  - {m}: {v:.4f} (threshold={QUALITY_THRESHOLDS[m]:.4f})"
        for m, v in breached.items()
    )
    message = (
        f"MODEL QUALITY DEGRADATION DETECTED\n"
        f"{'=' * 50}\n\n"
        f"MLflow Experiment : {MLFLOW_EXPERIMENT_NAME}\n"
        f"MLflow Run        : {run_name}\n"
        f"Endpoint          : {SAGEMAKER_ENDPOINT_NAME}\n\n"
        f"Metrics past threshold:\n{lines}\n\n"
        f"Review the full Evidently report in the MLflow artifacts.\n"
    )
    sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"Model Quality Alert - {MLFLOW_EXPERIMENT_NAME}",
        Message=message,
    )


def run_data_drift(production_data, reference_data, timestamp):
    data_definition = DataDefinition()
    reference_dataset = Dataset.from_pandas(
        reference_data, data_definition=data_definition
    )
    production_dataset = Dataset.from_pandas(
        production_data, data_definition=data_definition
    )

    with mlflow.start_run(run_name=f"data_drift_{timestamp}"):
        mlflow.log_params(
            {
                "endpoint_name": SAGEMAKER_ENDPOINT_NAME,
                "reference_data_size": len(reference_data),
                "current_data_size": len(production_data),
                "monitoring_timestamp": datetime.now().isoformat(),
            }
        )

        # Pin the numeric test so DRIFT_THRESHOLD is always a Wasserstein distance, not a p-value
        drift_report = Report(metrics=[DataDriftPreset(num_method="wasserstein", num_threshold=DRIFT_THRESHOLD)])
        drift_snapshot = drift_report.run(
            reference_data=reference_dataset, current_data=production_dataset
        )

        drift_html = f"/tmp/data_drift_{timestamp}.html"
        drift_json = f"/tmp/data_drift_{timestamp}.json"
        drift_snapshot.save_html(drift_html)
        drift_snapshot.save_json(drift_json)
        mlflow.log_artifact(drift_html, "evidently_report_data_drift_html")
        mlflow.log_artifact(drift_json, "evidently_report_data_drift_json")
        log_drift_metrics(drift_snapshot.dict())

        summary_report = Report(metrics=[DataSummaryPreset()])
        summary_snapshot = summary_report.run(
            reference_data=reference_dataset, current_data=production_dataset
        )
        summary_html = f"/tmp/data_summary_{timestamp}.html"
        summary_snapshot.save_html(summary_html)
        mlflow.log_artifact(summary_html, "evidently_report_data_quality")


def run_model_quality(production_data, timestamp):
    if not GROUND_TRUTH_S3_KEY:
        logger.info("No ground truth key configured, skipping model quality")
        return

    captured_df = production_data.copy()
    if TASK_TYPE == "regression":
        data_definition = DataDefinition(
            regression=[Regression(target="target", prediction="prediction")]
        )
        preset = RegressionPreset()
    else:
        # The endpoint returns a probability for the positive class
        captured_df["prediction"] = (captured_df["prediction"] > 0.5).astype(int)
        data_definition = DataDefinition(
            classification=[
                BinaryClassification(
                    target="target", prediction_labels="prediction", pos_label=1
                )
            ]
        )
        preset = ClassificationPreset()

    gt_obj = s3_client.get_object(Bucket=REFERENCE_BUCKET, Key=GROUND_TRUTH_S3_KEY)
    gt_df = pd.read_csv(gt_obj["Body"], dtype={"inference_id": str})
    eval_data = captured_df.merge(gt_df, on="inference_id", how="inner").dropna(
        subset=["prediction", "target"]
    )

    if eval_data.empty:
        logger.warning("No matching ground truth data found")
        return

    eval_dataset = Dataset.from_pandas(eval_data, data_definition=data_definition)

    run_name = f"model_quality_{timestamp}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "endpoint_name": SAGEMAKER_ENDPOINT_NAME,
                "eval_data_size": len(eval_data),
                "captured_data_size": len(captured_df),
                "ground_truth_size": len(gt_df),
                "monitoring_timestamp": datetime.now().isoformat(),
                "task_type": TASK_TYPE,
            }
        )

        report = Report(metrics=[preset])
        snapshot = report.run(reference_data=None, current_data=eval_dataset)

        kind = "regression" if TASK_TYPE == "regression" else "classification"
        html_path = f"/tmp/{kind}_report_{timestamp}.html"
        json_path = f"/tmp/{kind}_report_{timestamp}.json"
        snapshot.save_html(html_path)
        snapshot.save_json(json_path)
        mlflow.log_artifact(html_path, f"evidently_{kind}_report_html")
        mlflow.log_artifact(json_path, f"evidently_{kind}_report_json")

        report_dict = snapshot.dict()
        log_quality_metrics(report_dict)

        breached = check_thresholds(report_dict, QUALITY_THRESHOLDS)
        if breached:
            send_alert(breached, run_name)


def handler(event, context):
    """Process a batch of JSONL records from Distributed Map."""
    logger.info(f"Received batch event with {len(event.get('Items', []))} items")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    batch_input = event.get("BatchInput", {})
    s3_key = batch_input.get("s3_key", "")
    items = event.get("Items", [])

    if not items:
        return {"statusCode": 200, "body": "No records in batch"}

    raw_records = []
    for record in items:
        try:
            raw_records.extend(parse_record(record))
        except Exception as e:
            logger.error(f"Error parsing record: {e}", exc_info=True)
            continue

    if not raw_records:
        return {"statusCode": 200, "body": "No parseable records"}

    production_data = pd.DataFrame(raw_records)
    for col in FEATURE_COLUMNS:
        if col in production_data.columns:
            production_data[col] = pd.to_numeric(production_data[col], errors="coerce")

    baseline_obj = s3_client.get_object(Bucket=REFERENCE_BUCKET, Key=BASELINE_S3_KEY)
    reference_data = pd.read_csv(baseline_obj["Body"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_data_drift(production_data[FEATURE_COLUMNS], reference_data, timestamp)
    run_model_quality(production_data, timestamp)

    return {
        "statusCode": 200,
        "records_processed": len(raw_records),
        "s3_key": s3_key,
    }
