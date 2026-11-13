#!/bin/bash
set -euo pipefail

# Override environment file if user provides one
ENV_FILE=".env.docker"
if [[ "${1:-}" == "--env" ]]; then
    ENV_FILE="$2"
fi

# Prefer the caller's directory (a use-case folder) and fall
# back to this script's directory (src/, for the platform images), so both resolve the same
# config from the same command the build used
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

# If AWS_ACCOUNT_ID is empty, fetch it
if [ -z "${AWS_ACCOUNT_ID:-}" ]; then
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    export AWS_ACCOUNT_ID
fi

echo "Repository: $ECR_REPOSITORY_NAME"
echo "Tag: $IMAGE_TAG"
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"
echo ""

# Check if repository exists
echo "Checking if ECR repository exists..."
if aws ecr describe-repositories --repository-names "$ECR_REPOSITORY_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "✓ Repository '$ECR_REPOSITORY_NAME' exists"
else
    echo "✗ Repository '$ECR_REPOSITORY_NAME' does not exist"
    echo "Creating repository..."
    aws ecr create-repository \
        --repository-name "$ECR_REPOSITORY_NAME" \
        --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo "✓ Repository created successfully"
fi

# Get ECR login token and authenticate Docker
echo ""
echo "Authenticating with ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
echo "✓ Authentication successful"

# Tag the image for ECR
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:$IMAGE_TAG"
echo ""
echo "Tagging image for ECR..."
docker tag "$ECR_REPOSITORY_NAME:$IMAGE_TAG" "$ECR_URI"
echo "✓ Image tagged: $ECR_URI"

# Push to ECR
echo ""
echo "Pushing image to ECR..."
docker push "$ECR_URI"
echo "✓ Image pushed successfully"

echo ""
echo "Image URI: $ECR_URI"
