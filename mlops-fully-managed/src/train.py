import argparse
import os
import platform
from urllib.parse import quote

import yaml
from job_secrets import secrets_env
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.utils.utils import SageMakerClient
from sagemaker.core.training.configs import (
    SourceCode,
    Compute,
    InputData,
    OutputDataConfig,
    StoppingCondition,
)
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.model_trainer import Mode
from sagemaker.core.workflow.pipeline_context import PipelineSession


def parse_args():
    parser = argparse.ArgumentParser(description="Training job launcher")
    parser.add_argument(
        "--env",
        default="train.yaml",
        help="YAML config file name in train/ (default: train.yaml)",
    )
    parser.add_argument(
        "--local",
        default="false",
        help="Local training; Docker required (default: false)",
    )
    return parser.parse_args()


def get_console_link(url, text):
    """Return hyperlink to be printed in the console"""
    # URL encode the URL to handle special characters in auth tokens
    encoded_url = quote(url, safe=":/")
    link_str = f"\x1b]8;;{encoded_url}\x1b\\{text}\x1b]8;;\x1b\\"
    return link_str


def get_mlflow_ui_presigned_url(mlflow_arn):
    sm_client = SageMakerClient().sagemaker_client
    if "mlflow-app" in mlflow_arn:
        response = sm_client.create_presigned_mlflow_app_url(
            Arn=mlflow_arn,
            ExpiresInSeconds=300,
            SessionExpirationDurationInSeconds=3600,
        )
        mlflow_ui_url = response["AuthorizedUrl"]
    else:
        mlflow_tracking_server_name = mlflow_arn.split("/")[1]
        response = sm_client.create_presigned_mlflow_tracking_server_url(
            TrackingServerName=mlflow_tracking_server_name,
            ExpiresInSeconds=300,
            SessionExpirationDurationInSeconds=3600,
        )
        mlflow_ui_url = response["AuthorizedUrl"]
    return mlflow_ui_url


def create_input_config(train_env_config):
    input_config = []
    if train_env_config.get("training_data_s3_path"):
        input_config.append(
            InputData(
                channel_name="train",
                data_source=train_env_config["training_data_s3_path"],
            )
        )
    if train_env_config.get("validation_data_s3_path"):
        input_config.append(
            InputData(
                channel_name="validation",
                data_source=train_env_config["validation_data_s3_path"],
            )
        )
    if train_env_config.get("test_data_s3_path"):
        input_config.append(
            InputData(
                channel_name="test",
                data_source=train_env_config["test_data_s3_path"],
            )
        )
    return input_config


def enable_local():
    os.environ["PATH"] = (
        "/usr/local/bin:/Applications/Docker.app/Contents/Resources/bin:/Applications/Docker.app/Contents/Resources/cli-plugins:"
        + os.environ["PATH"]
    )
    # Set Docker platform for Apple Silicon compatibility
    if platform.machine() == "arm64":
        os.environ["DOCKER_DEFAULT_PLATFORM"] = "linux/amd64"
    instance_type = "local_cpu"
    training_mode = Mode.LOCAL_CONTAINER
    return instance_type, training_mode


def run_model_trainer(model_trainer, input_config, train_env_config):
    model_trainer_train = model_trainer.train(input_data_config=input_config, wait=True)

    if not isinstance(model_trainer.sagemaker_session, PipelineSession):
        print("\nTraining complete!")
        if model_trainer.training_mode == Mode.SAGEMAKER_TRAINING_JOB:
            training_job_name = model_trainer._latest_training_job.training_job_name
            print(f"Training job: {training_job_name}")

        if train_env_config.get("mlflow_arn"):
            mlflow_ui_url = get_mlflow_ui_presigned_url(train_env_config["mlflow_arn"])
            # print(get_console_link(mlflow_ui_url, "MLflow UI URL"))

    return model_trainer_train


def make_model_trainer(env=False):
    args = parse_args()
    if env:
        args.env = env

    with open(f"train/{args.env}") as f:
        train_env_config = yaml.safe_load(f)

    secret_config = secrets_env(train_env_config)

    instance_type = train_env_config["instance_type"]
    training_mode = Mode.SAGEMAKER_TRAINING_JOB
    if args.local == "true":
        instance_type, training_mode = enable_local()

    input_config = create_input_config(train_env_config)

    command = train_env_config["source_code_entry_point_command"]
    if train_env_config.get("mlflow_arn"):
        command += f" --mlflow_arn {train_env_config["mlflow_arn"]}"
        command += (
            f" --mlflow_experiment_name {train_env_config["mlflow_experiment_name"]}"
        )

    source_code_dir = train_env_config["source_code_dir"]
    requirements = None
    if os.path.isfile(os.path.join(source_code_dir, "requirements.txt")):
        requirements = "requirements.txt"

    source_code_config = SourceCode(
        source_dir=source_code_dir,
        command=command,
        requirements=requirements,
    )

    compute_config = Compute(
        instance_type=instance_type,
        instance_count=train_env_config["instance_count"],
        volume_size_in_gb=train_env_config["instance_volume_size"],
        keep_alive_period_in_seconds=train_env_config["keep_alive_hours"] * 3600,
        # Reserved capacity from a flexible training plan, if the config names one
        training_plan_arn=train_env_config.get("flexible_training_plan_arn"),
    )

    stopping_condition = StoppingCondition(
        max_runtime_in_seconds=train_env_config["max_runtime_hours"] * 3600
    )

    output_config = OutputDataConfig(
        s3_output_path=train_env_config["model_output_s3_path"],
    )

    model_trainer = ModelTrainer(
        training_image=train_env_config["image_uri"],
        source_code=source_code_config,
        compute=compute_config,
        output_data_config=output_config,
        stopping_condition=stopping_condition,
        role=train_env_config["sagemaker_execution_role"],
        environment=secret_config,
        base_job_name=train_env_config["job_name"],
        training_mode=training_mode,
        sagemaker_session=Session(),
    )
    return model_trainer, input_config, train_env_config


if __name__ == "__main__":
    model_trainer, input_config, train_env_config = make_model_trainer()
    run_model_trainer(model_trainer, input_config, train_env_config)
