#!/bin/bash
set -euo pipefail

# Override environment file if user provides one
ENV_FILE=".env.docker"
if [[ "${1:-}" == "--env" ]]; then
    ENV_FILE="$2"
fi

# Platform images (lambda_mlflow, obs-*) keep their configs in src/docker/; use-case images
# keep theirs in use_cases/<name>/docker/. Prefer the caller's directory so a use-case folder
# picks up its own configs, and fall back to this script's directory so the repo root finds the
# platform ones. Whichever wins becomes the build context, and both DOCKERFILE_PATH and every
# COPY in the Dockerfiles are relative to it.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "docker/$ENV_FILE" ]]; then
    CONTEXT="$PWD"
elif [[ -f "$SCRIPT_DIR/docker/$ENV_FILE" ]]; then
    CONTEXT="$SCRIPT_DIR"
else
    echo "✗ Could not find docker/$ENV_FILE in $PWD or $SCRIPT_DIR" >&2
    exit 1
fi

# Load Docker configuration from .env.docker
set -a
source "$CONTEXT/docker/$ENV_FILE"
set +a

echo "Context: $CONTEXT"
echo "Repository: $ECR_REPOSITORY_NAME"
echo "Tag: $IMAGE_TAG"
echo ""

# Build the image (Dockerfile is in docker/ folder)
echo "Building Docker image..."
docker build -f "$CONTEXT/$DOCKERFILE_PATH" --platform linux/amd64 \
    -t "$ECR_REPOSITORY_NAME:$IMAGE_TAG" "$CONTEXT"
echo "✓ Image built successfully"
