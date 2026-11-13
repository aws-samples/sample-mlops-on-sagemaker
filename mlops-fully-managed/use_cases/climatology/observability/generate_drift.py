"""
Send traffic to the climatology endpoint that is warmer than the baseline, so the
observability pipeline flags data drift

The defaults send one burst that fits in a single Lambda batch (Step Functions caps a
batch near 240 KB, about 50 of these requests); raise --rounds and --warmup-rounds to
ramp the shift up gradually instead

Run from use_cases/climatology:
    uv run python observability/generate_drift.py --env deploy.climatology-realtime.yaml
"""

import argparse
import io
import pathlib
import sys
import time

import boto3
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
from invoke import DESERIALIZER_MAP, SERIALIZER_MAP, resolve_endpoint_name  # noqa: E402
from sagemaker.core.resources import Endpoint  # noqa: E402

# Temperature features that shift together when the climate warms; deltas, anomalies
# and location stay as they were
WARMED_COLUMNS = [
    "tmax_lag_1",
    "tmax_lag_2",
    "tmax_lag_3",
    "tmax_lag_7",
    "tmin_lag_1",
    "tmax_roll_7",
    "tmax_roll_30",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Invoke the endpoint with drifting payloads")
    parser.add_argument("--env", default="deploy.climatology-realtime.yaml", help="Deploy config in deploy/")
    parser.add_argument("--obs-env", default="observability/observability.predictive.yaml", help="Observability config")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds, warming a little more each one")
    parser.add_argument("--requests-per-round", type=int, default=40)
    parser.add_argument("--rows-per-request", type=int, default=20)
    parser.add_argument("--max-shift", type=float, default=3.0, help="Degrees C added to temperatures by the last round")
    parser.add_argument("--warmup-rounds", type=int, default=0, help="Rounds sent unshifted before warming starts")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between requests")
    return parser.parse_args()


def load_reference(obs_config):
    predictive = obs_config["predictive_ml"]
    obj = boto3.client("s3").get_object(Bucket=predictive["reference_bucket"], Key=predictive["baseline_s3_key"])
    return pd.read_csv(obj["Body"])[predictive["feature_columns"]]


def shift_for_round(round_index, args):
    if round_index < args.warmup_rounds:
        return 0.0
    ramp_rounds = max(args.rounds - args.warmup_rounds, 1)
    return args.max_shift * min((round_index - args.warmup_rounds + 1) / ramp_rounds, 1.0)


def main():
    args = parse_args()
    with open(f"deploy/{args.env}") as f:
        deploy_config = yaml.safe_load(f)
    with open(args.obs_env) as f:
        obs_config = yaml.safe_load(f)

    serializer = SERIALIZER_MAP[deploy_config["serializer"]]()
    deserializer = DESERIALIZER_MAP[deploy_config["deserializer"]]()
    endpoint_name = resolve_endpoint_name(deploy_config)
    endpoint = Endpoint.get(endpoint_name=endpoint_name)
    if endpoint is None:
        raise ValueError(f"No endpoint named '{endpoint_name}'. Deploy it first with deploy.py --env {args.env}.")

    reference = load_reference(obs_config)
    rng = np.random.default_rng()

    for round_index in range(args.rounds):
        shift = shift_for_round(round_index, args)
        for _ in range(args.requests_per_round):
            batch = reference.sample(args.rows_per_request, replace=True, random_state=rng).copy()
            batch[WARMED_COLUMNS] += shift
            buffer = io.StringIO()
            batch.to_csv(buffer, header=False, index=False)

            result = endpoint.invoke(
                body=serializer.serialize(buffer.getvalue()),
                content_type=serializer.CONTENT_TYPE,
                accept=deserializer.ACCEPT[0],
            )
            if result is None:
                raise RuntimeError(f"Endpoint '{endpoint_name}' returned no response.")
            if args.sleep:
                time.sleep(args.sleep)

        sent = (round_index + 1) * args.requests_per_round
        print(f"round {round_index + 1}/{args.rounds}: +{shift:.2f} C, {sent} requests sent")


if __name__ == "__main__":
    main()
