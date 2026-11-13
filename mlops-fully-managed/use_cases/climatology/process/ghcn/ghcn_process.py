#!/usr/bin/env python

import argparse
import json
import logging
import os
import socket
import sys
from glob import glob

import boto3
import numpy as np
import pandas as pd
import mlflow
import sagemaker_mlflow  # noqa: F401

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Set by the engine when this runs as a pipeline step, so this run nests under the
# one named after the execution. Empty when the script is run on its own
MLFLOW_PARENT_RUN_ID = os.environ.get("MLFLOW_PARENT_RUN_ID") or None

# GHCN-Daily is one CSV per station, one row per station/day/measurement, so a
# single station going back to 1930 is ~435k rows across ~50 element types
RAW_COLUMNS = frozenset({"ID", "DATE", "ELEMENT", "DATA_VALUE"})
ELEMENTS = ["TMAX", "TMIN", "PRCP"]

# Temperatures are tenths of a degree C and precipitation tenths of a mm
TENTHS = 10.0

# Today's daily high, predicted from this station's own recent history
TARGET = "tmax"

# Fixed-width offsets from the GHCN-Daily readme for ghcnd-stations.txt
STATION_FIELDS = {
    "ID": (0, 11),
    "LATITUDE": (12, 20),
    "LONGITUDE": (21, 30),
    "ELEVATION": (31, 37),
}


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="/opt/ml/processing/raw")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing")
    # Station metadata is one small file, kept outside the sharded raw prefix
    # because anything under it goes to a single instance
    parser.add_argument("--stations_s3_uri", type=str, required=True)
    # Trimming history keeps the demo to minutes rather than hours
    parser.add_argument("--start_year", type=int, default=2015)
    # Chronological split; a random one leaks future weather through the lags
    parser.add_argument("--validation_start_year", type=int, default=2023)
    parser.add_argument("--test_start_year", type=int, default=2024)
    parser.add_argument("--mlflow_arn", type=str, default=None)
    parser.add_argument("--mlflow_experiment_name", type=str, default="Default")

    return parser.parse_known_args()


def current_host():
    """
    Name of the instance this copy of the script is running on

    Every instance runs the same code against its own shard, so the host name is
    what keeps their output files from colliding in the shared output prefix
    """
    resource_config = "/opt/ml/config/resourceconfig.json"
    if os.path.exists(resource_config):
        with open(resource_config) as f:
            return json.load(f)["current_host"]
    return os.environ.get("SM_CURRENT_HOST") or socket.gethostname()


def cluster_hosts():
    """
    Every instance in this job, in the order all of them see

    Only used to pick a leader: the instances have no way to talk to each other,
    but they do agree on this list, so the first name in it can own the step's
    MLflow run without any coordination
    """
    resource_config = "/opt/ml/config/resourceconfig.json"
    if os.path.exists(resource_config):
        with open(resource_config) as f:
            return json.load(f)["hosts"]
    return [current_host()]


def read_station_metadata(s3_uri):
    """
    Read latitude, longitude and elevation per station ID
    """
    bucket, key = s3_uri.replace("s3://", "", 1).split("/", 1)
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()

    rows = []
    for line in body.decode("utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(
            {
                name: line[start:end].strip()
                for name, (start, end) in STATION_FIELDS.items()
            }
        )

    stations = pd.DataFrame(rows)
    for column in ("LATITUDE", "LONGITUDE", "ELEVATION"):
        stations[column] = pd.to_numeric(stations[column], errors="coerce")

    # -999.9 is the archive's sentinel for unknown elevation
    stations.loc[stations["ELEVATION"] <= -999, "ELEVATION"] = np.nan

    logger.info(f"Read metadata for {len(stations)} stations")
    return stations.dropna(subset=["LATITUDE", "LONGITUDE"])


def read_shard(input_dir, start_year):
    """
    Read and reshape every station CSV handed to this instance

    ShardedByS3Key splits the input objects across the cluster, so this instance
    sees roughly 1/instance_count of the stations. Files are filtered one at a
    time before being accumulated because a raw file is ~20x the slice we keep
    """
    paths = sorted(glob(os.path.join(input_dir, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")

    logger.info(f"This instance was handed {len(paths)} station files")

    cutoff = start_year * 10000
    frames = []
    raw_rows = 0

    for index, path in enumerate(paths, start=1):
        df = pd.read_csv(path, usecols=lambda column: column in RAW_COLUMNS)
        raw_rows += len(df)

        df = df[df["ELEMENT"].isin(ELEMENTS) & (df["DATE"] >= cutoff)]
        if not df.empty:
            frames.append(df)

        if index % 100 == 0 or index == len(paths):
            logger.info(f"Read {index}/{len(paths)} files ({raw_rows:,} raw rows)")

    if not frames:
        raise ValueError(
            f"No {'/'.join(ELEMENTS)} observations on or after {start_year} in this shard"
        )

    long_df = pd.concat(frames, ignore_index=True)
    logger.info(f"Kept {len(long_df):,} of {raw_rows:,} raw rows")

    # One row per station-day, one column per element
    wide = long_df.pivot_table(
        index=["ID", "DATE"], columns="ELEMENT", values="DATA_VALUE", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    for element in ELEMENTS:
        if element not in wide.columns:
            wide[element] = np.nan
        wide[element] = wide[element] / TENTHS

    logger.info(f"Pivoted to {wide.shape[0]:,} station-days x {wide.shape[1]} columns")
    return wide, len(paths), raw_rows


def engineer_features(wide, stations):
    """
    Build backward-looking weather features per station

    Yesterday's high is already a strong predictor of today's, so the lags and
    rolling means carry the local trend, the day-of-year pair carries the season
    without a December/January discontinuity, and the station coordinates let a
    single model generalise across climates
    """
    wide = wide.sort_values(["ID", "DATE"]).reset_index(drop=True)
    by_station = wide.groupby("ID", sort=False)

    features = pd.DataFrame(index=wide.index)
    features[TARGET] = wide["TMAX"]

    for lag in (1, 2, 3, 7):
        features[f"tmax_lag_{lag}"] = by_station["TMAX"].shift(lag)
    features["tmin_lag_1"] = by_station["TMIN"].shift(1)
    features["prcp_lag_1"] = by_station["PRCP"].shift(1)

    # Roll the already-shifted series so today never enters its own window, and
    # group again so a window cannot span two stations
    shifted_tmax = by_station["TMAX"].shift(1)
    for window in (7, 30):
        features[f"tmax_roll_{window}"] = (
            shifted_tmax.groupby(wide["ID"])
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    features["tmax_delta_1"] = features["tmax_lag_1"] - features["tmax_lag_2"]
    features["tmax_anomaly_7"] = features["tmax_lag_1"] - features["tmax_roll_7"]

    dates = pd.to_datetime(wide["DATE"], format="%Y%m%d")
    day_of_year = dates.dt.dayofyear
    features["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    coordinates = wide[["ID"]].merge(stations, on="ID", how="left")
    features["latitude"] = coordinates["LATITUDE"].values
    features["longitude"] = coordinates["LONGITUDE"].values
    features["elevation"] = coordinates["ELEVATION"].values

    # Only used to cut the splits, never a feature
    features["year"] = dates.dt.year

    rows_in = len(features)
    # The lag and rolling warm-up leaves NaNs in each station's first 30 days
    features = features.dropna().reset_index(drop=True)
    logger.info(
        f"Feature warm-up and gaps dropped {rows_in - len(features):,} of {rows_in:,} rows"
    )

    return features


def split(df, args):
    """
    Split chronologically so no future observation informs a past prediction
    """
    splits = {
        "train": df[df["year"] < args.validation_start_year],
        "validation": df[
            (df["year"] >= args.validation_start_year)
            & (df["year"] < args.test_start_year)
        ],
        "test": df[df["year"] >= args.test_start_year],
    }
    return {name: part.drop(columns=["year"]) for name, part in splits.items()}


def write_splits(splits, output_dir, host):
    """
    Write each split as a headed CSV with the target in the first column

    Every instance uploads to the same S3 prefix, so the host name has to be in
    the file name or the instances would overwrite each other's shard
    """
    row_counts = {}
    for name, split_df in splits.items():
        split_dir = os.path.join(output_dir, name)
        os.makedirs(split_dir, exist_ok=True)

        path = os.path.join(split_dir, f"{name}-{host}.csv")
        split_df.to_csv(path, index=False)
        row_counts[name] = int(split_df.shape[0])

        logger.info(
            f"Wrote {split_df.shape[0]:,} rows x {split_df.shape[1]} cols to {path}"
        )

    return row_counts


def start(args, host):
    """
    Turn this instance's shard of GHCN station files into train/validation/test
    """
    stations = read_station_metadata(args.stations_s3_uri)
    wide, file_count, raw_rows = read_shard(args.input_dir, args.start_year)
    features = engineer_features(wide, stations)
    splits = split(features, args)
    row_counts = write_splits(splits, args.output_dir, host)

    # Only the leader has a run open. Logging unconditionally would make MLflow
    # start one implicitly on every other instance, which is the per-instance
    # fan-out this step deliberately keeps out of the run tree
    if not mlflow.active_run():
        return

    mlflow.log_params(
        {
            "host": host,
            "start_year": args.start_year,
            "validation_start_year": args.validation_start_year,
            "test_start_year": args.test_start_year,
            # Minus the target and the year column used for splitting
            "n_features": features.shape[1] - 2,
            "target": TARGET,
        }
    )
    mlflow.log_metrics(
        {
            # The leader's shard, not the whole job: the other instances handle an
            # equal share of the station files without logging, so these scale by
            # instance_count
            "station_files_in_shard": file_count,
            "stations_in_shard": int(wide["ID"].nunique()),
            "raw_rows_read": raw_rows,
            "feature_rows": int(features.shape[0]),
            **{f"{name}_rows": count for name, count in row_counts.items()},
        }
    )


if __name__ == "__main__":
    args, _ = parse_args()
    host = current_host()
    hosts = cluster_hosts()

    # Every instance processes its own shard, but only the leader opens a run, so
    # the step appears in the run tree once instead of once per instance
    if host == hosts[0]:
        mlflow.set_tracking_uri(args.mlflow_arn)
        mlflow.set_experiment(args.mlflow_experiment_name)

        with mlflow.start_run(
            run_name="climatology-ghcn-processing", parent_run_id=MLFLOW_PARENT_RUN_ID
        ):
            mlflow.set_tag("step", "processing")
            # The fan-out the run's shard metrics have to be read against
            mlflow.log_param("instance_count", len(hosts))
            start(args, host)
    else:
        logger.info(f"{host} is not the leader; processing its shard untracked")
        start(args, host)
