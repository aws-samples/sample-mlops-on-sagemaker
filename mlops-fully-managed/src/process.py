import argparse
import shlex

import yaml
from job_secrets import secrets_env
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.processing import FrameworkProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingS3Input,
    ProcessingOutput,
    ProcessingS3Output,
)

# Property file to be shared between pipeline steps
METADATA_OUTPUT_NAME = "metadata"
METADATA_LOCAL_PATH = "/opt/ml/processing/metadata"
METADATA_FILE_NAME = "metadata.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Processing job launcher")
    parser.add_argument(
        "--env",
        default="process.yaml",
        help="YAML config file name in process/ (default: process.yaml)",
    )
    return parser.parse_args()


def create_output_config(process_env_config):
    output_config = []
    if process_env_config.get("train_data_output_s3_path"):
        output_config.append(
            ProcessingOutput(
                output_name="train",
                s3_output=ProcessingS3Output(
                    s3_uri=process_env_config["train_data_output_s3_path"],
                    local_path="/opt/ml/processing/train",
                    s3_upload_mode="EndOfJob",
                ),
            )
        )
    if process_env_config.get("validation_data_output_s3_path"):
        output_config.append(
            ProcessingOutput(
                output_name="validation",
                s3_output=ProcessingS3Output(
                    s3_uri=process_env_config["validation_data_output_s3_path"],
                    local_path="/opt/ml/processing/validation",
                    s3_upload_mode="EndOfJob",
                ),
            )
        )
    if process_env_config.get("test_data_output_s3_path"):
        output_config.append(
            ProcessingOutput(
                output_name="test",
                s3_output=ProcessingS3Output(
                    s3_uri=process_env_config["test_data_output_s3_path"],
                    local_path="/opt/ml/processing/test",
                    s3_upload_mode="EndOfJob",
                ),
            )
        )
    if process_env_config.get("metadata_output_s3_path"):
        output_config.append(
            ProcessingOutput(
                output_name=METADATA_OUTPUT_NAME,
                s3_output=ProcessingS3Output(
                    s3_uri=process_env_config["metadata_output_s3_path"],
                    local_path=METADATA_LOCAL_PATH,
                    s3_upload_mode="EndOfJob",
                ),
            )
        )
    return output_config


def parse_entry_point_command(entry_point_commmand):
    parts = shlex.split(entry_point_commmand)
    if "." not in parts[0]:
        command = [parts[0]]
        code = parts[1]
        arguments = parts[2:]
    else:
        command = None
        code = parts[0]
        arguments = parts[1:]

    return command, code, arguments or None


def run_processor(processor, processor_args):
    return processor.run(**processor_args)


def create_processor(env=False):
    args = parse_args()
    if env:
        args.env = env

    with open(f"process/{args.env}") as f:
        process_env_config = yaml.safe_load(f)

    session = Session()
    env_config = {
        "AWS_DEFAULT_REGION": session.boto_region_name,
        **secrets_env(process_env_config),
    }

    entry_point_command = process_env_config["source_code_entry_point_command"]
    if process_env_config.get("mlflow_arn"):
        entry_point_command += f" --mlflow_arn {process_env_config['mlflow_arn']}"
        entry_point_command += (
            " --mlflow_experiment_name "
            f"{process_env_config.get('mlflow_experiment_name', 'Default')}"
        )

    command, code, arguments = parse_entry_point_command(entry_point_command)

    processor = FrameworkProcessor(
        image_uri=process_env_config["image_uri"],
        command=command,
        instance_type=process_env_config["instance_type"],
        instance_count=int(process_env_config["instance_count"]),
        volume_size_in_gb=int(process_env_config["instance_volume_size"]),
        role=process_env_config["sagemaker_execution_role"],
        env=env_config,
        base_job_name=process_env_config["job_name"],
        sagemaker_session=session,
    )

    input_config = []
    if process_env_config.get("raw_data_s3_path"):
        input_config.append(
            ProcessingInput(
                input_name="raw",
                s3_input=ProcessingS3Input(
                    s3_uri=process_env_config["raw_data_s3_path"],
                    local_path="/opt/ml/processing/raw",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    s3_data_distribution_type="ShardedByS3Key",
                ),
            )
        )

    processor_args = {
        "source_dir": process_env_config["source_code_dir"],
        "code": code,
        "arguments": arguments,
        "inputs": input_config,
        "outputs": create_output_config(process_env_config),
    }

    return processor, processor_args, process_env_config


if __name__ == "__main__":
    processor, processor_args, _ = create_processor()
    run_processor(processor, processor_args)
