#!/bin/bash
set -e

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <state-machine-arn> <s3-bucket> [s3-key]"
    echo ""
    echo "Triggers the monitoring state machine manually for testing."
    echo "If s3-key is not provided, lists recent .jsonl files in the bucket."
    exit 1
fi

STATE_MACHINE_ARN="$1"
S3_BUCKET="$2"
S3_KEY="${3:-}"

if [ -z "$S3_KEY" ]; then
    echo "No s3-key provided. Listing recent .jsonl files in bucket..."
    aws s3api list-objects-v2 \
      --bucket "$S3_BUCKET" \
      --query "reverse(sort_by(Contents, &LastModified))[:5].Key" \
      --output text | tr '\t' '\n' | grep '\.jsonl$' | head -5
    echo ""
    echo "Re-run with one of the above keys as the third argument."
    exit 0
fi

echo "Starting state machine execution..."
EXECUTION_ARN=$(aws stepfunctions start-execution \
  --state-machine-arn "$STATE_MACHINE_ARN" \
  --input "{\"detail\":{\"bucket\":{\"name\":\"$S3_BUCKET\"},\"object\":{\"key\":\"$S3_KEY\"}}}" \
  --query 'executionArn' \
  --output text)

echo "Execution started: $EXECUTION_ARN"
echo ""
echo "Monitor progress:"
echo "  aws stepfunctions describe-execution --execution-arn $EXECUTION_ARN"
