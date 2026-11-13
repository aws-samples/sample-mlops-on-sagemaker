import argparse
import json
import pathlib
import sys

import boto3
import yaml

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy observability infrastructure")
    parser.add_argument(
        "--env",
        required=True,
        help="YAML config file in observability/ (e.g. observability.predictive.yaml)",
    )
    return parser.parse_args()


def get_account_and_region():
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    session = boto3.session.Session()
    region = session.region_name
    return account_id, region


def get_data_capture_bucket(endpoint_arn):
    sm_client = boto3.client("sagemaker")
    endpoint_name = endpoint_arn.split("/")[-1]
    endpoint_config_name = sm_client.describe_endpoint(EndpointName=endpoint_name)[
        "EndpointConfigName"
    ]
    endpoint_config = sm_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name
    )
    data_capture = endpoint_config.get("DataCaptureConfig", {})
    if not data_capture.get("EnableCapture"):
        print("WARNING: DataCapture is not enabled on this endpoint")
    destination = data_capture.get("DestinationS3Uri", "")
    if destination.startswith("s3://"):
        parts = destination[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix
    return "", ""


def resolve_image_uris(config):
    monitoring_type = config["monitoring_type"]
    image_uris = {}
    if monitoring_type == "llm":
        image_uris["llm"] = config.get("obs_llm_image_uri", "")
        image_uris["evaluate"] = config.get("obs_evaluate_image_uri", "")
    else:
        image_uris["predictive"] = config.get("obs_predictive_image_uri", "")
    return image_uris


def enable_s3_eventbridge(bucket_name):
    """Enable EventBridge notifications on the S3 bucket, preserving existing config."""
    s3 = boto3.client("s3")
    try:
        current_config = s3.get_bucket_notification_configuration(Bucket=bucket_name)
    except Exception:
        current_config = {}

    current_config.pop("ResponseMetadata", None)

    if "EventBridgeConfiguration" in current_config:
        print(f"EventBridge already enabled on bucket {bucket_name}")
        return

    current_config["EventBridgeConfiguration"] = {}
    s3.put_bucket_notification_configuration(
        Bucket=bucket_name,
        NotificationConfiguration=current_config,
        # The existing config is written back too, and one stale destination (a
        # deleted Lambda, say) would otherwise fail the whole put
        SkipDestinationValidation=True,
    )
    print(f"Enabled EventBridge notifications on bucket {bucket_name}")


# Config keys for regression thresholds, mapped to the Evidently metric names they limit
REGRESSION_THRESHOLD_METRICS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "mape": "MAPE",
    "r2_score": "R2Score",
}


def regression_thresholds(predictive_config):
    if predictive_config.get("task_type") != "regression":
        return {}
    thresholds = predictive_config.get("thresholds", {})
    return {
        metric: float(thresholds[key])
        for key, metric in REGRESSION_THRESHOLD_METRICS.items()
        if thresholds.get(key) is not None
    }


def deploy_stack(config, image_uris, data_capture_bucket):
    stack_name = config.get("stack_name", "sagemaker-observability")
    monitoring_type = config["monitoring_type"]

    scorers_config = {}
    alert_thresholds = {}
    if monitoring_type == "llm":
        llm_config = config.get("llm", {})
        scorers = llm_config.get("scorers", {})
        scorers_config = {
            "builtin": scorers.get("builtin", []),
            "guidelines": scorers.get("guidelines", []),
            "custom_judges": scorers.get("custom_judges", []),
            "alert_fail_values": config.get("alert_fail_values", {}),
        }
        alert_thresholds = config.get("alert_thresholds", {})

    eval_params = config.get("evaluation_model_parameters", {})
    predictive_config = config.get("predictive_ml", {})

    parameters = [
        {"ParameterKey": "StackPrefix", "ParameterValue": stack_name},
        {"ParameterKey": "MonitoringType", "ParameterValue": monitoring_type},
        {
            "ParameterKey": "DataCaptureBucketName",
            "ParameterValue": data_capture_bucket,
        },
        {"ParameterKey": "MlflowAppArn", "ParameterValue": config["mlflow_app_arn"]},
        {
            "ParameterKey": "MlflowExperimentName",
            "ParameterValue": config["mlflow_experiment_name"],
        },
        {
            "ParameterKey": "SagemakerEndpointName",
            "ParameterValue": config["sagemaker_endpoint_arn"].split("/")[-1],
        },
        {
            "ParameterKey": "EvaluationModelId",
            "ParameterValue": config.get(
                "evaluation_model_id",
                "bedrock:/global.anthropic.claude-sonnet-4-20250514-v1:0",
            ),
        },
        {
            "ParameterKey": "EvaluationModelParameters",
            "ParameterValue": json.dumps(eval_params),
        },
        {"ParameterKey": "ScorersConfig", "ParameterValue": json.dumps(scorers_config)},
        {
            "ParameterKey": "LambdaMemorySize",
            "ParameterValue": str(config.get("lambda_memory_size", 3008)),
        },
        {
            "ParameterKey": "LambdaTimeout",
            "ParameterValue": str(config.get("lambda_timeout_seconds", 900)),
        },
        {
            "ParameterKey": "LogRetentionDays",
            "ParameterValue": str(config.get("log_retention_days", 7)),
        },
        {
            "ParameterKey": "StepFunctionsTimeoutSeconds",
            "ParameterValue": str(config.get("step_functions_timeout_hours", 1) * 3600),
        },
        {
            "ParameterKey": "MaxItemsPerBatch",
            "ParameterValue": str(config.get("max_items_per_batch", 10)),
        },
        {
            "ParameterKey": "MaxConcurrency",
            "ParameterValue": str(config.get("max_concurrency", 100)),
        },
        {
            "ParameterKey": "LlmLambdaImageUri",
            "ParameterValue": image_uris.get("llm", ""),
        },
        {
            "ParameterKey": "EvaluateLambdaImageUri",
            "ParameterValue": image_uris.get("evaluate", ""),
        },
        {
            "ParameterKey": "PredictiveLambdaImageUri",
            "ParameterValue": image_uris.get("predictive", ""),
        },
        {
            "ParameterKey": "SnsTopicArn",
            "ParameterValue": config.get("sns_topic_arn", ""),
        },
        {
            "ParameterKey": "AlertThresholds",
            "ParameterValue": json.dumps(alert_thresholds),
        },
        {
            "ParameterKey": "ReferenceBucket",
            "ParameterValue": predictive_config.get("reference_bucket", ""),
        },
        {
            "ParameterKey": "BaselineS3Key",
            "ParameterValue": predictive_config.get("baseline_s3_key", ""),
        },
        {
            "ParameterKey": "GroundTruthS3Key",
            "ParameterValue": predictive_config.get("ground_truth_s3_key", ""),
        },
        {
            "ParameterKey": "FeatureColumns",
            "ParameterValue": ",".join(predictive_config.get("feature_columns", [])),
        },
        {
            "ParameterKey": "TaskType",
            "ParameterValue": predictive_config.get("task_type", "binary_classification"),
        },
        {
            "ParameterKey": "RegressionThresholds",
            "ParameterValue": json.dumps(regression_thresholds(predictive_config)),
        },
        {
            "ParameterKey": "ThresholdF1",
            "ParameterValue": str(
                predictive_config.get("thresholds", {}).get("f1_score", 0.7)
            ),
        },
        {
            "ParameterKey": "ThresholdAccuracy",
            "ParameterValue": str(
                predictive_config.get("thresholds", {}).get("accuracy", 0.8)
            ),
        },
        {
            "ParameterKey": "ThresholdRocAuc",
            "ParameterValue": str(
                predictive_config.get("thresholds", {}).get("roc_auc", 0.75)
            ),
        },
        {
            "ParameterKey": "DriftThreshold",
            "ParameterValue": str(predictive_config.get("drift_threshold", 0.1)),
        },
    ]

    cfn_client = boto3.client("cloudformation")

    try:
        cfn_client.describe_stacks(StackName=stack_name)
        action = "update_stack"
    except cfn_client.exceptions.ClientError:
        action = "create_stack"

    template_path = f"{PROJECT_ROOT}/observability/template.yaml"
    with open(template_path) as f:
        template_body = f.read()

    cfn_args = {
        "StackName": stack_name,
        "TemplateBody": template_body,
        "Parameters": parameters,
        "Capabilities": ["CAPABILITY_NAMED_IAM"],
    }

    if action == "create_stack":
        cfn_client.create_stack(**cfn_args)
        print(f"Creating stack: {stack_name}")
        waiter = cfn_client.get_waiter("stack_create_complete")
    else:
        try:
            cfn_client.update_stack(**cfn_args)
            print(f"Updating stack: {stack_name}")
            waiter = cfn_client.get_waiter("stack_update_complete")
        except cfn_client.exceptions.ClientError as e:
            if "No updates are to be performed" in str(e):
                print("No stack updates needed")
                return
            raise

    print("Waiting for stack operation to complete...")
    waiter.wait(StackName=stack_name)
    print(f"Stack {action.replace('_', ' ')} complete: {stack_name}")


def deploy(env=False):
    args = parse_args()
    if env:
        args.env = env

    # The config belongs to the use case, so it resolves against the directory the
    # launcher was run from -- unlike template.yaml, which ships with this module and
    # is found under PROJECT_ROOT.
    config_path = f"observability/{args.env}"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config.get("sagemaker_endpoint_arn"):
        print("ERROR: sagemaker_endpoint_arn is required in config")
        sys.exit(1)

    if not config.get("mlflow_app_arn"):
        print("ERROR: mlflow_app_arn is required in config")
        sys.exit(1)

    if config.get("monitoring_type") not in ("llm", "predictive_ml"):
        print("ERROR: monitoring_type must be 'llm' or 'predictive_ml'")
        sys.exit(1)

    account_id, region = get_account_and_region()
    print(f"Deploying to account {account_id} in {region}")

    data_capture_bucket, data_capture_prefix = get_data_capture_bucket(
        config["sagemaker_endpoint_arn"]
    )
    if not data_capture_bucket:
        print("ERROR: Could not determine DataCapture S3 bucket from endpoint")
        sys.exit(1)
    print(f"DataCapture bucket: {data_capture_bucket} (prefix: {data_capture_prefix})")

    image_uris = resolve_image_uris(config)

    deploy_stack(config, image_uris, data_capture_bucket)
    enable_s3_eventbridge(data_capture_bucket)


if __name__ == "__main__":
    deploy()
