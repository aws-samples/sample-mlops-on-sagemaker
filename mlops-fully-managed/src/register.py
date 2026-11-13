import argparse
import json
import os
import subprocess
import tempfile

import yaml
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.utils.utils import SageMakerClient
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.metadata_properties import MetadataProperties
from sagemaker.core.model_metrics import ModelMetrics, MetricsSource
from sagemaker.core.resources import ModelCard
from sagemaker.core.model_life_cycle import ModelLifeCycle
import mlflow
import sagemaker_mlflow  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Register trained model")
    parser.add_argument(
        "--env",
        default="register.yaml",
        help="YAML config file name in register/ (default: register.yaml)",
    )
    return parser.parse_args()


def get_git_info():
    user_name = None
    git_remote_url = None

    try:
        result = subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True, check=True
        )
        user_name = result.stdout.strip()
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_remote_url = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return user_name, git_remote_url


def get_mlflow_run_id_for_training_job(sm_training_job_name, mlflow_arn):
    mlflow.set_tracking_uri(mlflow_arn)
    experiments = mlflow.search_experiments()
    run_id = None
    experiment_name_of_run = None
    for experiment in experiments:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.sm_training_job_name = '{sm_training_job_name}'",
        )
        if not runs.empty:
            experiment_name_of_run = experiment.name
            run_id = runs.iloc[0]["run_id"]
        if run_id:
            break
    if not run_id:
        raise ValueError(
            f"MLflow run id not found for SM training job name: {sm_training_job_name}"
        )
    return run_id, experiment_name_of_run


def get_mlflow_run_metadata(run_id, mlflow_experimen_name, mlflow_arn):
    mlflow.set_tracking_uri(mlflow_arn)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    params = run.data.params

    return metrics, params


def create_update_model_card(
    sm_training_job_info,
    register_env_config,
    user_name,
):
    # Get experiment metadata from MLflow
    mlflow_run_id, mlflow_experimen_name = get_mlflow_run_id_for_training_job(
        sm_training_job_info["sm_training_job_name"], register_env_config["mlflow_arn"]
    )
    metrics, params = get_mlflow_run_metadata(
        mlflow_run_id, mlflow_experimen_name, register_env_config["mlflow_arn"]
    )

    # Convert MLflow metrics to model card format
    training_metrics = []
    for metric_name, metric_value in metrics.items():
        training_metrics.append(
            {
                "name": metric_name,
                "value": metric_value,
            }
        )

    # Convert MLflow params to model card format
    hyper_parameters = []
    for param_name, param_value in params.items():
        hyper_parameters.append(
            {
                "name": param_name,
                "value": param_value,
            }
        )

    model_card_content = {
        "model_overview": {
            "model_creator": user_name or "Unknown",
            "model_description": register_env_config["model_description"],
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

    model_card_content_json = json.dumps(model_card_content)
    try:
        model_card_name = register_env_config["model_name"]
        model_card = ModelCard.create(
            model_card_name=model_card_name,
            content=model_card_content_json,
            model_card_status="Approved",
        )
    except:
        print(
            f"Model Card '{model_card_name}' already exists - getting existing model card..."
        )
        model_card = ModelCard.get(model_card_name)
        model_card = model_card.update(content=model_card_content_json)

    return model_card


def get_model_test_metrics(sm_training_job_name, mlflow_arn, model_artifact_s3_uri):
    # Get experiment metadata from MLflow
    mlflow_run_id, mlflow_experimen_name = get_mlflow_run_id_for_training_job(
        sm_training_job_name, mlflow_arn
    )
    metrics, params = get_mlflow_run_metadata(
        mlflow_run_id, mlflow_experimen_name, mlflow_arn
    )

    # Transform metrics to the required format
    transformed_metrics = {}
    for metric_name, metric_value in metrics.items():
        transformed_metrics[metric_name] = {
            "value": metric_value,
            "standard_deviation": "NaN",
        }

    # Write experiment metadata to temp file and upload to S3
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(transformed_metrics, f)
        f.close()
        metrics_file = f.name

    # Upload metrics to same location as model artifact
    sm_session = Session()
    prefix_partial = "/".join(model_artifact_s3_uri.split("/")[3:-2])
    bucket = model_artifact_s3_uri.split("/")[2]
    key_prefix = f"{prefix_partial}/metrics"
    s3_metrics_uri = sm_session.upload_data(
        path=metrics_file,
        bucket=bucket,
        key_prefix=key_prefix,
    )
    os.unlink(metrics_file)

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            content_type="application/json",
            s3_uri=s3_metrics_uri,
        )
    )
    return model_metrics


def make_model_builder(
    env=False,
    model_card_name=None,
    skip_model_card=False,
    pipeline_session=None,
    train_step=None,
    train_step_image_uri=None,
):
    args = parse_args()
    if env:
        args.env = env

    with open(f"register/{args.env}") as f:
        register_env_config = yaml.safe_load(f)

    # Get training job info
    sm_training_job_name = None
    if pipeline_session and train_step is not None and train_step_image_uri is not None:
        model_artifact_s3_uri = train_step.properties.ModelArtifacts.S3ModelArtifacts
        image_uri = train_step_image_uri
    elif register_env_config.get("model_data_s3_uri"):
        model_artifact_s3_uri = register_env_config["model_data_s3_uri"]
        image_uri = register_env_config.get("inference_image_uri")
        if not image_uri:
            raise ValueError(
                "Set inference_image_uri alongside model_data_s3_uri in the register config"
            )
    else:
        sm_client = SageMakerClient().sagemaker_client
        sm_training_job_name = register_env_config["training_job_name"]
        response = sm_client.describe_training_job(TrainingJobName=sm_training_job_name)
        sm_training_job_arn = response["TrainingJobArn"]
        model_artifact_s3_uri = response["ModelArtifacts"]["S3ModelArtifacts"]
        image_uri = response["AlgorithmSpecification"]["TrainingImage"]
        training_datasets_s3_uris = []
        for input_data in response["InputDataConfig"]:
            if input_data["ChannelName"] not in ["code", "sm_drivers"]:
                training_datasets_s3_uris.append(
                    input_data["DataSource"]["S3DataSource"]["S3Uri"]
                )
        sm_training_job_info = {
            "sm_training_job_arn": sm_training_job_arn,
            "sm_training_job_name": sm_training_job_name,
            "model_artifact_s3_uri": model_artifact_s3_uri,
            "image_uri": image_uri,
            "training_datasets_s3_uris": training_datasets_s3_uris,
        }

    # Get inference image if specified, otherwise default to training image
    image_uri = register_env_config.get("inference_image_uri") or image_uri

    if (
        register_env_config.get("mlflow_arn")
        and sm_training_job_name
        and model_card_name is None
        and skip_model_card is False
    ):
        user_name, git_remote_url = get_git_info()
        model_card = create_update_model_card(
            sm_training_job_info,
            register_env_config,
            user_name,
        )
    elif model_card_name is not None and skip_model_card is not False:
        model_card = ModelCard.get(model_card_name)
    else:
        model_card = None

    # Model metrics to store in model package
    model_metrics = None
    if register_env_config.get("model_metrics_s3_uri"):
        # Get model test metrics from a previous job (like a gated quality step)
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                content_type="application/json",
                s3_uri=register_env_config["model_metrics_s3_uri"],
            )
        )
    elif (
        not pipeline_session
        and sm_training_job_name
        and register_env_config.get("mlflow_arn")
    ):
        model_metrics = get_model_test_metrics(
            sm_training_job_name,
            register_env_config["mlflow_arn"],
            model_artifact_s3_uri,
        )

    # Create ModelBuilder
    if pipeline_session:
        session = pipeline_session
    else:
        session = Session()
    model_builder = ModelBuilder(
        s3_model_data_url={
            "S3DataSource": {
                "S3Uri": model_artifact_s3_uri,
                "S3DataType": "S3Object",
                "CompressionType": "Gzip",
            }
        },
        image_uri=image_uri,
        sagemaker_session=session,
        role_arn=register_env_config["sagemaker_execution_role"],
    )

    # Create a SageMaker AI Model if not being used in a SageMaker AI Pipeline
    if not pipeline_session:
        model_builder.build(model_name=sm_training_job_name)

    return model_builder, model_card, register_env_config, model_metrics


def register_model(
    model_builder,
    model_card,
    register_env_config,
    model_metrics=None,
    customer_metadata=None,
):
    # Get repo metadata or drop the empty keys and omit both fields entirely
    # when there is no git metadata to attach
    user_name, git_remote_url = get_git_info()
    metadata_properites = (
        MetadataProperties(repository=git_remote_url, generated_by=user_name)
        if (user_name or git_remote_url)
        else None
    )
    customer_metadata_properties = {
        key: value
        for key, value in {
            "user": user_name,
            "git_remote_url": git_remote_url,
            **(customer_metadata or {}),
        }.items()
        if value
    } or None

    model_status = register_env_config["model_status"]
    model_life_cycle = ModelLifeCycle(stage="Development", stage_status=model_status)

    # Register the model
    return model_builder.register(
        model_package_group_name=register_env_config["model_package_group_name"],
        model_metrics=model_metrics,
        metadata_properties=metadata_properites,
        customer_metadata_properties=customer_metadata_properties,
        description=register_env_config["model_description"],
        # source_uri=model_artifact_s3_uri,
        inference_instances=register_env_config["inference_instances"],
        content_types=["application/json"],
        response_types=["application/json"],
        model_card=model_card,
        model_life_cycle=model_life_cycle,
        approval_status=model_status,
    )


if __name__ == "__main__":
    model_builder, model_card, register_env_config, model_metrics = make_model_builder()
    register_model(model_builder, model_card, register_env_config, model_metrics)
