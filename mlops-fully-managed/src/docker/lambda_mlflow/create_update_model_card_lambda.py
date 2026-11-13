import json
import os
import logging
import boto3
import mlflow
import sagemaker_mlflow  # noqa: F401

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_training_job_info(sm_client, training_job_name):
    response = sm_client.describe_training_job(TrainingJobName=training_job_name)
    training_datasets_s3_uris = []
    for input_data in response["InputDataConfig"]:
        if input_data["ChannelName"] not in ["code", "sm_drivers"]:
            training_datasets_s3_uris.append(
                input_data["DataSource"]["S3DataSource"]["S3Uri"]
            )
    return {
        "sm_training_job_arn": response["TrainingJobArn"],
        "sm_training_job_name": training_job_name,
        "model_artifact_s3_uri": response["ModelArtifacts"]["S3ModelArtifacts"],
        "image_uri": response["AlgorithmSpecification"]["TrainingImage"],
        "training_datasets_s3_uris": training_datasets_s3_uris,
    }


def get_mlflow_run_id_for_training_job(
    sm_training_job_name, mlflow_arn, mlflow_experiment_name
):
    mlflow.set_tracking_uri(mlflow_arn)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(mlflow_experiment_name)
    if not experiment:
        raise ValueError(f"MLflow experiment not found: {mlflow_experiment_name}")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.sm_training_job_name = '{sm_training_job_name}'",
    )
    if not runs:
        raise ValueError(
            f"MLflow run id not found for SM training job name: {sm_training_job_name}"
        )
    return runs[0].info.run_id


def get_mlflow_run_metadata(run_id, mlflow_arn):
    mlflow.set_tracking_uri(mlflow_arn)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    params = run.data.params
    return metrics, params


def read_s3_json(s3_uri):
    bucket, _, key = s3_uri.removeprefix("s3://").partition("/")
    response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read())


def evaluation_details_item(evaluation):
    """
    Get Model Card evaluation_details entry from the
    evaluation step which publishes details in its metadata.json
    """
    item = {
        "name": evaluation["name"],
        "metric_groups": [
            {
                "name": evaluation["name"],
                "metric_data": [
                    {"name": name, "type": "number", "value": float(value)}
                    for name, value in evaluation["metrics"].items()
                ],
            }
        ],
    }
    if evaluation.get("observation"):
        item["evaluation_observation"] = evaluation["observation"]
    if evaluation.get("job_arn"):
        item["evaluation_job_arn"] = evaluation["job_arn"]
    if evaluation.get("datasets"):
        item["datasets"] = evaluation["datasets"]
    if evaluation.get("metadata"):
        # The card schema only takes string values here
        item["metadata"] = {k: str(v) for k, v in evaluation["metadata"].items()}
    return item


def create_update_model_card(
    sm_client,
    sm_training_job_info,
    model_name,
    model_description,
    metrics,
    params,
    evaluation=None,
):
    training_metrics = [
        {"name": name, "value": value} for name, value in metrics.items()
    ]
    hyper_parameters = [
        {"name": name, "value": value} for name, value in params.items()
    ]

    model_card_content = {
        "model_overview": {
            "model_creator": "SageMaker Pipeline",
            "model_description": model_description,
            "model_artifact": [sm_training_job_info["model_artifact_s3_uri"]],
        },
        "training_details": {
            "training_job_details": {
                "training_arn": sm_training_job_info["sm_training_job_arn"],
                "training_environment": {
                    "container_image": [sm_training_job_info["image_uri"]]
                },
                "training_datasets": sm_training_job_info["training_datasets_s3_uris"],
                "training_metrics": training_metrics,
                "hyper_parameters": hyper_parameters,
            },
        },
    }

    if evaluation:
        model_card_content["evaluation_details"] = [evaluation_details_item(evaluation)]

    model_card_content_json = json.dumps(model_card_content)

    try:
        response = sm_client.create_model_card(
            ModelCardName=model_name,
            Content=model_card_content_json,
            ModelCardStatus="Approved",
        )
        status = "created"
    except sm_client.exceptions.ConflictException:
        logger.info(
            f"Model Card '{model_name}' already exists - updating existing model card..."
        )
        response = sm_client.update_model_card(
            ModelCardName=model_name,
            Content=model_card_content_json,
        )
        status = "updated"

    return {
        "model_card_name": model_name,
        "model_card_arn": response["ModelCardArn"],
        "model_card_content_json": model_card_content_json,
        "status": status,
    }


def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")

    training_job_name = event["training_job_name"]
    model_package_arn = event.get("model_package_arn")
    evaluation_metadata_s3_uri = event.get("evaluation_metadata_s3_uri")
    mlflow_arn = os.environ["MLFLOW_ARN"]
    mlflow_experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
    model_name = os.environ["MODEL_NAME"]
    model_description = os.environ["MODEL_DESCRIPTION"]

    sm_client = boto3.client("sagemaker")

    training_job_info = get_training_job_info(sm_client, training_job_name)

    run_id = get_mlflow_run_id_for_training_job(
        training_job_name, mlflow_arn, mlflow_experiment_name
    )
    metrics, params = get_mlflow_run_metadata(run_id, mlflow_arn)

    evaluation = None
    if evaluation_metadata_s3_uri:
        evaluation = read_s3_json(evaluation_metadata_s3_uri).get("evaluation")
        if not evaluation:
            logger.warning(f"No 'evaluation' block in {evaluation_metadata_s3_uri}")

    result = create_update_model_card(
        sm_client,
        training_job_info,
        model_name,
        model_description,
        metrics,
        params,
        evaluation=evaluation,
    )

    logger.info(f"Associating model card with model package: {model_package_arn}")
    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelCard={
            "ModelCardContent": result["model_card_content_json"],
            "ModelCardStatus": "Approved",
        },
    )
    result["model_package_arn"] = model_package_arn

    logger.info(f"Model card result: {json.dumps(result)}")
    return result
