import argparse
import json
import pathlib

import yaml
from deploy import endpoint_name as resolve_endpoint_name

from sagemaker.core.resources import Endpoint
from sagemaker.core.serializers import (
    DataSerializer,
    NumpySerializer,
    JSONSerializer,
    CSVSerializer,
    TorchTensorSerializer,
    StringSerializer,
)
from sagemaker.core.deserializers import (
    BytesDeserializer,
    NumpyDeserializer,
    JSONDeserializer,
    PandasDeserializer,
    TorchTensorDeserializer,
    StringDeserializer,
)

SERIALIZER_MAP = {
    "csv": CSVSerializer,
    "json": JSONSerializer,
    "string": StringSerializer,
    "array": NumpySerializer,
    "tensor": TorchTensorSerializer,
    "bytes": DataSerializer,
}

DESERIALIZER_MAP = {
    "dataframe": PandasDeserializer,
    "json": JSONDeserializer,
    "string": StringDeserializer,
    "array": NumpyDeserializer,
    "tensor": TorchTensorDeserializer,
    "bytes": BytesDeserializer,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Endpoint invocation launcher")
    parser.add_argument(
        "--env",
        default="deploy.yaml",
        help="YAML config file name in deploy/ (default: deploy.yaml)",
    )
    return parser.parse_args()


def load_payload(payload_path, serializer_name):
    """Read the request payload off disk in the shape its serializer expects.

    `bytes` is the exception: DataSerializer opens the file itself, so it takes the
    path rather than the contents. A `.json` payload is parsed into a Python object
    because every other serializer builds its own wire format from one -- a list of
    feature vectors serializes just as well to csv or .npy as it does to JSON.
    Anything else is handed over as text.
    """
    if serializer_name == "bytes":
        return payload_path

    path = pathlib.Path(payload_path)
    if path.suffix.lower() == ".json":
        with open(path) as f:
            return json.load(f)
    return path.read_text()


def invoke_model(env=False):
    args = parse_args()
    if env:
        args.env = env

    with open(f"deploy/{args.env}") as f:
        invoke_env_config = yaml.safe_load(f)

    serializer_name = invoke_env_config["serializer"]
    serializer = SERIALIZER_MAP[serializer_name]()
    deserializer = DESERIALIZER_MAP[invoke_env_config["deserializer"]]()

    payload = load_payload(invoke_env_config["payload_path"], serializer_name)

    # Resolved the same way deploy.py resolves it, so one config file serves both.
    endpoint_name = resolve_endpoint_name(invoke_env_config)
    endpoint = Endpoint.get(endpoint_name=endpoint_name)
    if endpoint is None:
        raise ValueError(
            f"No endpoint named '{endpoint_name}'. Deploy it first with "
            f"deploy.py --env {args.env}."
        )

    result = endpoint.invoke(
        body=serializer.serialize(payload),
        content_type=serializer.CONTENT_TYPE,
        # A deserializer advertises every content type it handles; the endpoint wants
        # a single Accept header, so ask for the one it prefers.
        accept=deserializer.ACCEPT[0],
    )
    if result is None:
        raise RuntimeError(f"Endpoint '{endpoint_name}' returned no response.")

    print(deserializer.deserialize(result.body, result.content_type))


if __name__ == "__main__":
    invoke_model()
