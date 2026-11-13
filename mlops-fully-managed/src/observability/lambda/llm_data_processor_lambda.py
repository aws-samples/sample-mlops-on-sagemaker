"""
LLM monitoring Lambda - processes batches of DataCapture JSONL records
dispatched by Step Functions Distributed Map.

Receives input shaped like:
{
  "BatchInput": { "s3_bucket": "...", "s3_key": "..." },
  "Items": [ {captureData record}, ... ]
}
"""
import json
import logging
import os
import base64
from typing import Dict, Any
from datetime import datetime

import boto3
import mlflow

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SAGEMAKER_ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
DATA_CAPTURE_BUCKET = os.environ["DATA_CAPTURE_BUCKET"]


def decode_base64_data(encoded_data: str) -> Any:
    try:
        decoded_bytes = base64.b64decode(encoded_data)
        decoded_str = decoded_bytes.decode("utf-8")
        try:
            return json.loads(decoded_str)
        except json.JSONDecodeError:
            return decoded_str
    except Exception as e:
        logger.error(f"Error decoding base64 data: {e}")
        return f"Error decoding: {str(e)}"


def parse_data_capture_record(record: Dict) -> Dict:
    parsed_record = {
        "timestamp": record.get("eventMetadata", {}).get("inferenceTime"),
        "event_id": record.get("eventMetadata", {}).get("eventId"),
    }

    capture_data = record.get("captureData", {})
    endpoint_input = capture_data.get("endpointInput", {})
    endpoint_output = capture_data.get("endpointOutput", {})

    if endpoint_input:
        encoding = endpoint_input.get("encoding", "")
        data = endpoint_input.get("data", "")
        if encoding == "BASE64":
            parsed_record["request"] = decode_base64_data(data)
        else:
            try:
                parsed_record["request"] = json.loads(data) if data else {}
            except (json.JSONDecodeError, TypeError):
                parsed_record["request"] = data

    if endpoint_output:
        encoding = endpoint_output.get("encoding", "")
        data = endpoint_output.get("data", "")
        observed_content_type = endpoint_output.get("observedContentType")

        if encoding == "BASE64":
            parsed_record["response"] = decode_base64_data(data)
        else:
            try:
                parsed_record["response"] = json.loads(data) if data else {}
            except (json.JSONDecodeError, TypeError):
                parsed_record["response"] = data

        response_data = parsed_record.get("response", {})
        if isinstance(response_data, dict):
            if "code" in response_data:
                parsed_record["status_code"] = response_data["code"]
                parsed_record["error_message"] = response_data.get("message", "")
                parsed_record["is_error"] = True
            elif observed_content_type is None:
                parsed_record["is_error"] = True
                parsed_record["status_code"] = response_data.get("code", 500)
            else:
                parsed_record["status_code"] = 200
                parsed_record["is_error"] = False

    return parsed_record


def log_trace_to_mlflow(event_data: Dict, s3_file_key: str) -> None:
    event_id = event_data.get("event_id", "unknown")
    inference_time = event_data.get("timestamp", datetime.now().isoformat())
    input_data = event_data.get("request", {})
    output_data = event_data.get("response", {})
    is_error = event_data.get("is_error", False)

    additional_trace_attr = {
        "s3_bucket_name": DATA_CAPTURE_BUCKET,
        "s3_file_key": s3_file_key,
        "sagemaker_endpoint_name": SAGEMAKER_ENDPOINT_NAME,
        "event_id": event_id,
        "inference_time": inference_time,
        "status_code": str(event_data.get("status_code", 200)),
    }

    with mlflow.start_span(name=s3_file_key) as span:
        span.set_inputs({
            "prompt": input_data.get("inputs", ""),
            "parameters": input_data.get("parameters", {}),
        })
        span.set_attributes(additional_trace_attr)
        if is_error:
            span.set_status("ERROR")
        span.set_outputs(output_data)

    logger.info(f"Logged trace for event {event_id}")


def handler(event: Dict, context: Any) -> Dict:
    """Process a batch of JSONL records from Distributed Map."""
    logger.info(f"Received batch event with {len(event.get('Items', []))} items")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    batch_input = event.get("BatchInput", {})
    s3_key = batch_input.get("s3_key", "")
    items = event.get("Items", [])

    processed_count = 0
    error_count = 0

    for record in items:
        try:
            parsed_record = parse_data_capture_record(record)
            log_trace_to_mlflow(parsed_record, s3_key)
            processed_count += 1
            if parsed_record.get("is_error"):
                error_count += 1
        except Exception as e:
            logger.error(f"Error processing record: {e}", exc_info=True)
            continue

    logger.info(f"Processed {processed_count}/{len(items)} records")

    return {
        "records_processed": processed_count,
        "errors_found": error_count,
        "s3_key": s3_key,
    }
