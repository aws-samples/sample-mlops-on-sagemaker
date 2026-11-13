"""
Inference handler for the managed SageMaker scikit-learn container
"""

import io
import logging
import os

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    """
    Load the model bundle the training job saved

    Training saves the feature names alongside the estimator so the column order
    lives with the model instead of being duplicated here
    """
    bundle = joblib.load(os.path.join(model_dir, "model.joblib"))
    if isinstance(bundle, dict):
        return bundle["model"], bundle.get("feature_names")
    return bundle, None


def input_fn(request_body, request_content_type):
    """
    Parse a headerless CSV batch into a frame of features

    The transform job's input_filter has already dropped the target column, so
    every field that arrives here is a feature
    """
    if not request_content_type or not request_content_type.startswith("text/csv"):
        raise ValueError(f"Unsupported content type: {request_content_type}")

    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    return pd.read_csv(io.StringIO(request_body), header=None)


def predict_fn(input_data, model):
    """
    Score one batch of records
    """
    estimator, feature_names = model

    if feature_names is not None:
        if input_data.shape[1] != len(feature_names):
            raise ValueError(
                f"Model expects {len(feature_names)} features, got {input_data.shape[1]}; "
                "check the transform job's input_filter"
            )
        # XGBoost validates feature names when it was fitted on a named frame
        input_data = input_data.set_axis(feature_names, axis=1)

    return estimator.predict(input_data)


def output_fn(prediction, accept):
    """
    Serialize predictions as newline-delimited CSV
    """
    body = "\n".join(f"{value:.4f}" for value in prediction.tolist())
    return body, "text/csv"
