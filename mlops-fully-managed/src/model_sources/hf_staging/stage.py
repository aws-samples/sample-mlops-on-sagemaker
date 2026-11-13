"""
Download a HuggingFace repo into a processing job's output directory

Runs inside the staging job huggingface.py launches, not locally. SageMaker uploads
whatever lands in the output directory to the staged prefix once this exits, so the
layout here is the layout the serving container mounts at /opt/ml/model
"""

import argparse
import json
import os
import shutil

import boto3
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

OUTPUT_DIR = "/opt/ml/processing/weights"

# Matches job_secrets.SECRETS_ARN_ENV and huggingface.SECRET_KEY_ENV
SECRETS_ARN_ENV = "SECRETS_ARN"
SECRET_KEY_ENV = "HF_SECRET_KEY"
DEFAULT_SECRET_KEY = "HF_TOKEN"


def parse_args():
    parser = argparse.ArgumentParser(description="HuggingFace weight staging")
    parser.add_argument("--repo-id", required=True, help="Hub repo to download")
    parser.add_argument("--revision", required=True, help="Commit sha to pin to")
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Glob to leave out of the download; repeatable",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return parser.parse_args()


def hub_token():
    """
    The Hub token out of Secrets Manager, or None for a public repo
    """
    secrets_arn = os.environ.get(SECRETS_ARN_ENV)
    if not secrets_arn:
        return None

    key = os.environ.get(SECRET_KEY_ENV, DEFAULT_SECRET_KEY)
    secret = boto3.client("secretsmanager").get_secret_value(SecretId=secrets_arn)
    body = secret["SecretString"]

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()

    if not isinstance(parsed, dict):
        return body.strip()
    if key not in parsed:
        raise SystemExit(
            f"Secret {secrets_arn} holds JSON with no '{key}' key, only "
            f"{sorted(parsed)}. Point huggingface.secret_key at the right one"
        )

    return str(parsed[key]).strip()


def main():
    args = parse_args()

    print(f"Downloading {args.repo_id} at {args.revision} to {args.output_dir}")

    try:
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=args.output_dir,
            token=hub_token(),
            ignore_patterns=args.ignore_pattern or None,
            max_workers=8,
        )
    except GatedRepoError as error:
        raise SystemExit(
            f"{args.repo_id} is gated and the token was refused. The account the "
            f"token belongs to has to accept the terms on the model's Hub page; "
            f"this is not an IAM problem"
        ) from error
    except RepositoryNotFoundError as error:
        raise SystemExit(
            f"The Hub has no '{args.repo_id}' the token can see. A private repo and "
            f"a typo fail the same way"
        ) from error

    # Serving container has no use for .cache
    cache_dir = os.path.join(args.output_dir, ".cache")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)

    staged = sorted(os.listdir(args.output_dir))
    print(f"Downloaded {len(staged)} entries: {staged}")


if __name__ == "__main__":
    main()
