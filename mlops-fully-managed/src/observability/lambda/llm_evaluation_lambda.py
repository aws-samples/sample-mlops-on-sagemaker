"""
LLM evaluation Lambda - runs MLflow GenAI evaluations on traces
after the Distributed Map has finished logging all traces.
Sends SNS alerts when scorer failure rates exceed thresholds.

Receives input shaped like:
{
  "s3_bucket": "...",
  "s3_key": "..."
}
"""
import json
import logging
import os
from typing import Dict, Any, Literal

import boto3
import mlflow
from mlflow.genai import scorer
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Safety, RelevanceToQuery, Fluency, Guidelines

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sns_client = boto3.client("sns")

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "")
EVALUATION_MODEL_ID = os.environ["EVALUATION_MODEL_ID"]
SCORERS_CONFIG = json.loads(os.environ.get("SCORERS_CONFIG", "{}"))
EVALUATION_MODEL_PARAMETERS = json.loads(os.environ.get("EVALUATION_MODEL_PARAMETERS", "{}"))
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
ALERT_THRESHOLDS = json.loads(os.environ.get("ALERT_THRESHOLDS", "{}"))


def build_scorers():
    scorers_list = []

    scorer_map = {
        "safety": Safety,
        "relevance_to_query": RelevanceToQuery,
        "fluency": Fluency,
    }

    for builtin_name in SCORERS_CONFIG.get("builtin", []):
        scorer_cls = scorer_map.get(builtin_name)
        if scorer_cls:
            scorers_list.append(scorer_cls(
                model=EVALUATION_MODEL_ID,
                parameters=EVALUATION_MODEL_PARAMETERS,
            ))

    for guideline_cfg in SCORERS_CONFIG.get("guidelines", []):
        scorers_list.append(Guidelines(
            name=guideline_cfg["name"],
            guidelines=guideline_cfg["guideline"],
            model=EVALUATION_MODEL_ID,
            parameters=EVALUATION_MODEL_PARAMETERS,
        ))

    for judge_cfg in SCORERS_CONFIG.get("custom_judges", []):
        feedback_values = judge_cfg.get("feedback_values", [])
        if len(feedback_values) == 2:
            feedback_type = Literal[feedback_values[0], feedback_values[1]]
        elif len(feedback_values) == 3:
            feedback_type = Literal[feedback_values[0], feedback_values[1], feedback_values[2]]
        else:
            feedback_type = Literal["good", "bad"]

        scorers_list.append(make_judge(
            name=judge_cfg["name"],
            instructions=judge_cfg["instructions"],
            feedback_value_type=feedback_type,
            model=EVALUATION_MODEL_ID,
        ))

    @scorer
    def tokens_words(outputs) -> int:
        try:
            if isinstance(outputs, dict):
                words = len(outputs.get("generated_text", "").split())
            else:
                words = len(str(outputs).split())
        except Exception:
            return 0
        return words

    scorers_list.append(tokens_words)
    return scorers_list


def check_evaluation_results(results) -> Dict[str, float]:
    """Check evaluation results against alert thresholds.

    Returns a dict of scorer_name -> failure_rate for any scorer
    whose failure rate exceeds its configured threshold.
    """
    if not ALERT_THRESHOLDS or not results:
        return {}

    breached = {}
    try:
        results_df = results.tables["eval_results"]
        total_rows = len(results_df)
        if total_rows == 0:
            return {}

        for scorer_name, max_failure_rate in ALERT_THRESHOLDS.items():
            matching_cols = [c for c in results_df.columns if scorer_name in c.lower()]
            for col in matching_cols:
                col_values = results_df[col]
                if col_values.dtype == "object":
                    fail_values = SCORERS_CONFIG.get("alert_fail_values", {}).get(scorer_name, [])
                    if fail_values:
                        failures = col_values.isin(fail_values).sum()
                    else:
                        failures = col_values.isna().sum()
                else:
                    failures = (col_values == False).sum() + col_values.isna().sum()  # noqa: E712

                failure_rate = failures / total_rows
                if failure_rate > max_failure_rate:
                    breached[scorer_name] = failure_rate
    except Exception as e:
        logger.warning(f"Could not check evaluation thresholds: {e}")

    return breached


def send_alert(breached: Dict[str, float], s3_key: str, traces_evaluated: int):
    if not SNS_TOPIC_ARN:
        return

    lines = "\n".join(
        f"  - {name}: {rate:.1%} failure rate (threshold: {ALERT_THRESHOLDS[name]:.1%})"
        for name, rate in breached.items()
    )
    message = (
        f"LLM EVALUATION ALERT\n"
        f"{'=' * 50}\n\n"
        f"MLflow Experiment : {MLFLOW_EXPERIMENT_NAME}\n"
        f"Endpoint          : {SAGEMAKER_ENDPOINT_NAME}\n"
        f"Source file       : {s3_key}\n"
        f"Traces evaluated  : {traces_evaluated}\n\n"
        f"Scorers exceeding failure threshold:\n{lines}\n\n"
        f"Review evaluation details in MLflow.\n"
    )
    sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"LLM Evaluation Alert - {MLFLOW_EXPERIMENT_NAME}",
        Message=message,
    )
    logger.info(f"Sent SNS alert for {len(breached)} breached scorers")


def handler(event: Dict, context: Any) -> Dict:
    """Run GenAI evaluations on all traces from the processed JSONL file."""
    logger.info(f"Received evaluation event: {json.dumps(event)}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    s3_key = event.get("s3_key", "")
    if not s3_key:
        return {"statusCode": 200, "body": "No s3_key provided, skipping evaluation"}

    try:
        mlflow.autolog(disable=True)

        traces = mlflow.search_traces(filter_string=f"name = '{s3_key}'")
        if len(traces) == 0:
            logger.warning(f"No traces found for {s3_key}")
            return {"statusCode": 200, "body": "No traces found"}

        logger.info(f"Found {len(traces)} traces to evaluate")
        scorers_list = build_scorers()

        results = None
        try:
            results = mlflow.genai.evaluate(data=traces, scorers=scorers_list)
        except Exception as e:
            logger.warning(f"First evaluation pass failed, retrying: {e}")
            results = mlflow.genai.evaluate(data=traces, scorers=scorers_list)

        logger.info("Evaluations completed")

        breached = check_evaluation_results(results)
        if breached:
            send_alert(breached, s3_key, len(traces))

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Evaluations complete",
                "traces_evaluated": len(traces),
                "s3_key": s3_key,
                "alerts_sent": len(breached) > 0,
            }),
        }

    except Exception as e:
        logger.error(f"Error running evaluations: {e}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
