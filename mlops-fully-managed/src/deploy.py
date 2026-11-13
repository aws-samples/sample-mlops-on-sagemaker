import argparse
import io
import os
import re
import tarfile
import time

import boto3
import yaml
from botocore.exceptions import ClientError
from model_sources.huggingface import (
    print_deploy_config,
    serving_environment,
    staged_artifact,
    staged_source,
)
from job_secrets import secrets_env
from model_sources.jumpstart import print_recipe, print_search

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.resources import Endpoint, EndpointConfig, Model, ModelPackage
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.shapes import (
    AdditionalModelDataSource,
    ContainerDefinition,
    InstancePool,
    ModelAccessConfig,
    ModelDataSource,
    ProductionVariant,
    ProductionVariantCapacityReservationConfig,
    S3ModelDataSource,
)
from sagemaker.core.transformer import Transformer
from sagemaker.core.utils.exceptions import FailedStatusError
from sagemaker.core.utils.utils import serialize
from sagemaker.serve.utils.types import ModelServer
from sagemaker.core.training.configs import SourceCode
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.workflow.pipeline_context import PipelineSession

# How the model is served: a persistent endpoint, or a Batch Transform job
DEPLOY_TYPES = ("real-time", "batch")

# Where the weights come from; a config declares exactly one
MODEL_SOURCES = ("model_package_arn", "model_data_s3_uri", "huggingface")

# One variant per endpoint; this engine does not do traffic splitting
VARIANT_NAME = "AllTraffic"

# Statuses an endpoint moves out of on its own, given time
TRANSIENT_STATUSES = ("Creating", "Updating", "SystemUpdating", "RollingBack")

# PREFIX_AWARE sends requests sharing a prompt prefix to the same instance, so an
# LLM endpoint reuses the KV cache instead of recomputing the prefix
ROUTING_STRATEGIES = ("RANDOM", "LEAST_OUTSTANDING_REQUESTS", "PREFIX_AWARE")

CAPTURE_MODES = ("Input", "Output", "InputAndOutput")

# InstancePools accepts at most five types
MAX_INSTANCE_POOLS = 5

# Endpoint configs are stamped with the deploy time, so --delete can recognize the
# ones this config made without matching a longer endpoint name that starts the same
CONFIG_STAMP = "%Y%m%d-%H%M%S"
CONFIG_STAMP_PATTERN = r"\d{8}-\d{6}"


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy launcher")
    parser.add_argument(
        "--env",
        default="deploy.yaml",
        help="YAML config file name in deploy/ (default: deploy.yaml)",
    )
    parser.add_argument(
        "--local",
        default="false",
        help="Local deployment; Docker required (default: false)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Tear the endpoint down instead of deploying it",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Stage a huggingface config's weights to S3 without deploying",
    )
    parser.add_argument(
        "--search-jumpstart",
        help="List JumpStart models whose id matches these terms, and exit",
    )
    parser.add_argument(
        "--jumpstart",
        help="Print a deploy config for this JumpStart model id, and exit",
    )
    parser.add_argument(
        "--vllm-recipe",
        help="Print what vLLM publishes for this HuggingFace model id, and exit",
    )
    parser.add_argument(
        "--jumpstart-config",
        help="Inference config to read, if not the one the hub ranks first",
    )
    parser.add_argument(
        "--instance-type",
        help="Instance type to write the JumpStart config for",
    )
    return parser.parse_args()


def load_deploy_config(env):
    args = parse_args()
    if env:
        args.env = env

    with open(f"deploy/{args.env}") as f:
        deploy_env_config = yaml.safe_load(f)

    deploy_type = deploy_env_config.get("deploy_type", "real-time")
    if deploy_type not in DEPLOY_TYPES:
        raise ValueError(
            f"deploy_type must be one of {DEPLOY_TYPES}, got '{deploy_type}'"
        )

    # Caught here rather than letting resolve_model_artifact pick by precedence, which
    # would serve one source and leave no sign the other was ignored
    sources = [key for key in MODEL_SOURCES if deploy_env_config.get(key)]
    if len(sources) > 1:
        raise ValueError(f"Set one model source, not {sources}")

    return args, deploy_env_config, deploy_type


def resolve_model_artifact(deploy_env_config):
    """
    S3 uri of the model.tar.gz to serve

    A registered model package is the normal path; model_data_s3_uri is the
    escape hatch for an artifact a training job wrote but nobody registered. A
    huggingface block is weights that are not in S3 yet, so asking for them stages
    them there first
    """
    if deploy_env_config.get("huggingface"):
        return staged_artifact(deploy_env_config)

    if deploy_env_config.get("model_package_arn"):
        model_package_arn = deploy_env_config["model_package_arn"]
        model_package = ModelPackage.get(model_package_arn)
        if model_package is None:
            raise ValueError(f"No model package found for {model_package_arn}")
        specification = model_package.inference_specification
        if specification is None or not specification.containers:
            raise ValueError(f"{model_package_arn} has no inference containers")
        data_source = specification.containers[0].model_data_source
        if data_source is None or data_source.s3_data_source is None:
            raise ValueError(f"{model_package_arn} has no model artifact")
        return data_source.s3_data_source.s3_uri

    if deploy_env_config.get("model_data_s3_uri"):
        return deploy_env_config["model_data_s3_uri"]

    raise ValueError(f"Set one of {MODEL_SOURCES} in the deploy config")


# ---------------------------------------------------------------------------
# Real-time inference
# ---------------------------------------------------------------------------


def endpoint_name(deploy_env_config):
    """
    Name of the endpoint a deploy config owns

    Defaults to model_name, which is what a config written before this key existed
    already got, and is how invoke.py resolves the endpoint from the same file
    """
    return deploy_env_config.get("endpoint_name") or deploy_env_config["model_name"]


def find(get, **kwargs):
    """
    Resource.get() raises when the resource is absent; this answers None instead

    Every step of a deploy has to be create-if-absent: a re-run must not fail on the
    endpoint it created last time, and a teardown must not fail on what is already
    gone. sagemaker.core has no ResourceNotFound type - a missing name surfaces as a
    botocore ValidationException, so that is the only code swallowed here
    """
    try:
        return get(**kwargs)
    except ClientError as error:
        if error.response["Error"]["Code"] == "ValidationException":
            return None
        raise


def instance_plan(deploy_env_config):
    """
    Instance types the endpoint may run on, best first

    The first entry is the shape to run on and anything after it is a capacity
    fallback, which is the usual reason a GPU endpoint never comes up

    One entry deploys as InstanceType rather than a pool of one, which is the shape of
    request these endpoints have actually come up on. train.py and process.py keep
    their own instance_type; this is the endpoint's key only
    """
    return (deploy_env_config["instance_preferences"] or [])[:MAX_INSTANCE_POOLS]


def resolve_routing(deploy_env_config):
    """
    The variant's RoutingConfig in API shape, or None to leave SageMaker's default

    Returned already serialized because sagemaker.core's typed
    ProductionVariantRoutingConfig carries only routing_strategy and forbids extra
    fields, so PREFIX_AWARE's two tuning knobs have nowhere to live on the shape
    """
    routing = deploy_env_config.get("routing") or {}
    strategy = routing.get("strategy")
    tuning = {
        "PrefixLength": routing.get("prefix_length"),
        "ConcurrencyThreshold": routing.get("concurrency_threshold"),
    }
    tuning = {key: int(value) for key, value in tuning.items() if value is not None}

    if not strategy:
        if tuning:
            raise ValueError(
                "routing.prefix_length and routing.concurrency_threshold need "
                "routing.strategy: PREFIX_AWARE"
            )
        return None
    if strategy not in ROUTING_STRATEGIES:
        raise ValueError(
            f"routing.strategy must be one of {ROUTING_STRATEGIES}, got '{strategy}'"
        )
    if tuning and strategy != "PREFIX_AWARE":
        raise ValueError(
            f"routing.prefix_length and routing.concurrency_threshold only apply to "
            f"PREFIX_AWARE routing, not {strategy}"
        )

    if not tuning:
        return {"RoutingStrategy": strategy}

    return {"RoutingStrategy": strategy, "PrefixAwareRoutingConfig": tuning}


def resolve_data_capture(deploy_env_config):
    """
    The endpoint config's DataCaptureConfig in API shape, or None to leave capture off

    A data_capture block turns capture on unless it sets enabled: false, and has to
    name where the captured JSONL goes
    """
    capture = deploy_env_config.get("data_capture") or {}
    if not capture or not capture.get("enabled", True):
        return None

    destination = capture.get("destination_s3_uri")
    if not destination or not str(destination).startswith("s3://"):
        raise ValueError("data_capture.destination_s3_uri must be an s3:// uri")

    sampling = int(capture.get("sampling_percentage", 100))
    if not 0 <= sampling <= 100:
        raise ValueError(
            f"data_capture.sampling_percentage must be 0-100, got {sampling}"
        )

    modes = capture.get("capture_modes") or ["Input", "Output"]
    unknown = [mode for mode in modes if mode not in CAPTURE_MODES]
    if unknown:
        raise ValueError(
            f"data_capture.capture_modes must be drawn from {CAPTURE_MODES}, got {unknown}"
        )

    config = {
        "EnableCapture": True,
        "InitialSamplingPercentage": sampling,
        "DestinationS3Uri": destination,
        "CaptureOptions": [{"CaptureMode": mode} for mode in modes],
    }
    if capture.get("kms_key_id"):
        config["KmsKeyId"] = capture["kms_key_id"]

    # Without these, SageMaker base64-encodes payloads whose content type it does
    # not recognize as text
    content_types = {
        "JsonContentTypes": capture.get("json_content_types"),
        "CsvContentTypes": capture.get("csv_content_types"),
    }
    content_types = {key: value for key, value in content_types.items() if value}
    if content_types:
        config["CaptureContentTypeHeader"] = content_types

    return config


def production_variant(deploy_env_config, model_name):
    """
    The endpoint's single production variant, serialized to its API shape

    Built as a typed shape first so the SDK validates the field names and types we
    can express, then serialized so the routing config - which the shape cannot
    carry - can be dropped in alongside them
    """
    plan = instance_plan(deploy_env_config)

    variant = ProductionVariant(
        variant_name=VARIANT_NAME,
        model_name=model_name,
        initial_instance_count=int(deploy_env_config.get("instance_count", 1)),
        # An LLM is tens of GB of weights to pull before the container can answer a
        # health check, and both clocks have to cover it
        model_data_download_timeout_in_seconds=int(
            deploy_env_config.get("model_data_download_timeout_seconds", 1800)
        ),
        container_startup_health_check_timeout_in_seconds=int(
            deploy_env_config.get("startup_health_check_timeout_seconds", 1800)
        ),
    )

    if len(plan) > 1:
        # InstancePools replaces instance_type rather than supplementing it: an
        # ordered list SageMaker falls back through on InsufficientInstanceCapacity
        # instead of failing the create
        variant.instance_pools = [
            InstancePool(instance_type=instance_type, priority=priority)
            for priority, instance_type in enumerate(plan, 1)
        ]
        if deploy_env_config.get("instance_provision_timeout_seconds"):
            variant.variant_instance_provision_timeout_in_seconds = int(
                deploy_env_config["instance_provision_timeout_seconds"]
            )
    else:
        variant.instance_type = plan[0]

    # The NVIDIA driver belongs to the host, not the image, so a container built
    # against a newer CUDA than the default AMI carries needs this to start at all
    if deploy_env_config.get("inference_ami_version"):
        variant.inference_ami_version = deploy_env_config["inference_ami_version"]

    if deploy_env_config.get("volume_size_gb"):
        variant.volume_size_in_gb = int(deploy_env_config["volume_size_gb"])

    # Reserved GPU capacity, which is how an endpoint gets an instance type on-demand
    # cannot supply. This is the training-plan arn, not the reserved-capacity arn the
    # plan contains, and the plan has to have been bought targeting endpoints
    if deploy_env_config.get("training_plan_arn"):
        variant.capacity_reservation_config = ProductionVariantCapacityReservationConfig(
            # The only preference the API accepts: the endpoint draws from the
            # reservation and stops serving once it expires
            capacity_reservation_preference="capacity-reservations-only",
            ml_reservation_arn=deploy_env_config["training_plan_arn"],
        )

    serialized = serialize(variant)

    routing = resolve_routing(deploy_env_config)
    if routing:
        serialized["RoutingConfig"] = routing

    return serialized


def endpoint_config_name(deploy_env_config):
    return f"{endpoint_name(deploy_env_config)}-{time.strftime(CONFIG_STAMP)}"


def create_endpoint_config(deploy_env_config, model_name, session):
    """
    A fresh endpoint config for this deploy, created through the boto3 client

    EndpointConfig.create cannot express prefix-aware routing, and
    ModelBuilder.deploy reaches neither that nor instance pools. Calling the API
    with a variant we serialized ourselves keeps one code path for every option the
    service accepts rather than a typed path plus an escape hatch

    The name is stamped with the deploy time because UpdateEndpoint needs a config
    the endpoint is not already pointing at

    No ExecutionRoleArn here: the API takes it only for inference-component
    endpoints, and a variant that names a model gets the role off that model
    """
    config_name = endpoint_config_name(deploy_env_config)
    request = {
        "EndpointConfigName": config_name,
        "ProductionVariants": [production_variant(deploy_env_config, model_name)],
    }

    data_capture = resolve_data_capture(deploy_env_config)
    if data_capture:
        request["DataCaptureConfig"] = data_capture

    session.sagemaker_client.create_endpoint_config(**request)
    print(f"Created endpoint config {config_name}")

    return config_name


def apply_endpoint_config(name, config_name):
    """
    Point the endpoint at a config, creating the endpoint if it is not there yet

    An existing endpoint is updated rather than replaced: UpdateEndpoint stands the
    new fleet up before retiring the old one, so a re-deploy picks up a changed
    image or instance type without dropping traffic. An endpoint in a failed state
    is deleted instead, because UpdateEndpoint is rejected from there and nothing
    else clears it
    """
    endpoint = find(Endpoint.get, endpoint_name=name)

    if endpoint and endpoint.endpoint_status in TRANSIENT_STATUSES:
        print(
            f"Endpoint {name} is {endpoint.endpoint_status}; waiting for it to settle"
        )
        try:
            endpoint.wait_for_status("InService")
        except FailedStatusError:
            # Landing mid-create means watching that create fail, which is the state
            # the next branch clears rather than something to raise from here
            pass

    # Matches Failed and UpdateRollbackFailed, the same test the SDK's waiter makes
    if endpoint and "failed" in str(endpoint.endpoint_status).lower():
        reason = getattr(endpoint, "failure_reason", None)
        print(f"Endpoint {name} is {endpoint.endpoint_status} ({reason}); replacing it")
        endpoint.delete()
        endpoint.wait_for_delete()
        endpoint = None

    if endpoint:
        print(f"Updating endpoint {name}")
        endpoint.endpoint_config_name = config_name
        endpoint.update()
    else:
        print(f"Creating endpoint {name}")
        endpoint = Endpoint.create(endpoint_name=name, endpoint_config_name=config_name)
        if endpoint is None:
            raise RuntimeError(f"CreateEndpoint returned no resource for {name}")

    endpoint.wait_for_status("InService")

    return endpoint


def s3_model_data_source(s3_uri, accept_eula=False):
    """
    An artifact's S3DataSource, with its layout read off the uri

    A uri ending in / is a prefix of uncompressed files, which is how JumpStart
    publishes LLM weights and the only form a serving container can load without
    unpacking tens of GB first. Anything else is the single model.tar.gz a training
    job writes

    accept_eula is how a gated model - a Llama, anything JumpStart marks Gated - says
    the licence has been agreed to. The download is refused without it
    """
    access = ModelAccessConfig(accept_eula=True) if accept_eula else None

    if s3_uri.endswith("/"):
        return S3ModelDataSource(
            s3_uri=s3_uri,
            s3_data_type="S3Prefix",
            compression_type="None",
            model_access_config=access,
        )

    return S3ModelDataSource(
        s3_uri=s3_uri,
        s3_data_type="S3Object",
        compression_type="Gzip",
        model_access_config=access,
    )


def container_definition(deploy_env_config, env):
    """
    The container the endpoint runs, pointed at the model artifact

    Built when the config names no handler code. A serving container like vLLM reads
    the weights out of /opt/ml/model and exposes its own inference API, so there is
    no script of ours to pack in and no reason to repack the artifact at all
    """
    container = ContainerDefinition(
        image=deploy_env_config["image_uri"],
        model_data_source=ModelDataSource(
            s3_data_source=s3_model_data_source(
                resolve_model_artifact(deploy_env_config),
                accept_eula=bool(deploy_env_config.get("accept_eula", False)),
            )
        ),
        environment=env,
    )

    # Extra weights the container wants beside the main artifact - a speculative
    # decoding draft model, a LoRA adapter - each mounted at
    # /opt/ml/additional-model-data-sources/<channel_name>
    sources = deploy_env_config.get("additional_model_data_sources") or []
    if sources:
        container.additional_model_data_sources = [
            AdditionalModelDataSource(
                channel_name=source["channel_name"],
                s3_data_source=s3_model_data_source(
                    resolve_extra_artifact(deploy_env_config, source)
                ),
            )
            for source in sources
        ]

    return container


def resolve_extra_artifact(deploy_env_config, source):
    """
    S3 uri of one extra artifact, staging it from the Hub when named that way

    A channel gives either an s3_uri it already lives at or a huggingface block to
    stage it from, and asking for both leaves no way to tell which one the endpoint
    actually mounted
    """
    named = [key for key in ("s3_uri", "huggingface") if source.get(key)]
    if len(named) != 1:
        raise ValueError(
            f"additional_model_data_sources channel '{source['channel_name']}' needs "
            f"exactly one of s3_uri or huggingface, got {named or 'neither'}"
        )

    if source.get("s3_uri"):
        return source["s3_uri"]

    return staged_source(deploy_env_config, source)


def build_handler_model(deploy_env_config, model_name, env, instance_type, session):
    """
    Create the model with our inference handler packed into its artifact

    ModelBuilder repacks source_code_dir into the model.tar.gz and points the model
    server at the entry script, which is how a framework container ends up running
    our code. It only understands a compressed artifact, so a container that serves
    itself takes the container-only path instead
    """
    model_server = deploy_env_config["model_server"]
    source_code_config = SourceCode(
        source_dir=deploy_env_config["source_code_dir"],
        entry_script=deploy_env_config["entry_script"],
    )

    model_builder = ModelBuilder(
        image_uri=deploy_env_config["image_uri"],
        model_server=getattr(ModelServer, model_server),
        source_code=source_code_config,
        s3_model_data_url=resolve_model_artifact(deploy_env_config),
        instance_type=instance_type,
        env_vars={
            **env,
            "OPTION_ENTRYPOINT": f"code/{deploy_env_config['entry_script']}",
        },
        sagemaker_session=session,
        role_arn=deploy_env_config["sagemaker_execution_role"],
    )

    model_builder.model_path = "/tmp/sagemaker/model-builder"
    os.makedirs(model_builder.model_path, exist_ok=True)
    # Workaround: SDK bug - repack_model expects dependencies to be a list of
    # file paths, but ModelBuilder sets it to {'auto': True}
    model_builder.dependencies = []  # type: ignore[assignment]

    return model_builder.build(model_name=model_name, mode=Mode.SAGEMAKER_ENDPOINT)


def build_model(deploy_env_config, model_name, env, instance_type, session):
    """
    Create the SageMaker Model the endpoint serves, replacing one of the same name

    A Model is immutable, so replacing it is the only way a re-deploy picks up a new
    image, artifact or environment. The endpoint keeps serving the old definition
    until apply_endpoint_config swaps in the config built from this one
    """
    existing_model = find(Model.get, model_name=model_name)
    if existing_model:
        print(f"Replacing model {model_name}")
        existing_model.delete()

    if deploy_env_config.get("source_code_dir"):
        if deploy_env_config.get("model_server"):
            return build_handler_model(
                deploy_env_config, model_name, env, instance_type, session
            )
        # A framework container such as scikit-learn runs the inference toolkit,
        # which no ModelServer covers, so hand it the code the way batch does
        code_s3_uri = upload_source_dir(
            deploy_env_config["source_code_dir"],
            deploy_env_config.get(
                "code_s3_uri",
                f"s3://{session.default_bucket()}/{model_name}/code/sourcedir.tar.gz",
            ),
        )
        env = {
            "SAGEMAKER_PROGRAM": deploy_env_config["entry_script"],
            "SAGEMAKER_SUBMIT_DIRECTORY": code_s3_uri,
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_REGION": session.boto_region_name,
            **env,
        }

    print(f"Creating model {model_name}")

    return Model.create(
        model_name=model_name,
        primary_container=container_definition(deploy_env_config, env),
        execution_role_arn=deploy_env_config["sagemaker_execution_role"],
        enable_network_isolation=bool(
            deploy_env_config.get("enable_network_isolation", False)
        ),
        session=session.boto_session,
    )


def published_environment(deploy_env_config):
    """
    Serving settings published for a HuggingFace model, or nothing

    Only for the huggingface path: a model_data_s3_uri config was either written by
    hand or generated from the hub, and in both cases its environment block is already
    the published answer. Set vllm_recipe: false to deploy on the config alone
    """
    if not deploy_env_config.get("huggingface"):
        return {}
    # Off unless asked for: a generated config already carries every serving setting,
    # and a deploy that looked settings up again would not be the config that was swept
    if not deploy_env_config.get("vllm_recipe"):
        return {}

    environment, source = serving_environment(deploy_env_config)
    print(f"Serving settings: {source}")

    return environment


def container_environment(deploy_env_config, secret_config):
    """
    The container's environment, with every value a string the API will accept

    YAML types a bare 1 as an int and true as a bool, neither of which CreateModel
    takes, and an env var is a string either way. A key with nothing after the colon
    becomes an empty value rather than the string "None", since that is how a serving
    container is told to pass a flag that takes no argument

    Published settings go underneath the config's own environment block, so a recipe
    fills in what was not written down and never overrides a deliberate choice
    """
    if secret_config and deploy_env_config.get("enable_network_isolation"):
        raise ValueError(
            "secrets_arn needs the container to call Secrets Manager, which "
            "enable_network_isolation blocks"
        )

    env = {
        **secret_config,
        **published_environment(deploy_env_config),
        **(deploy_env_config.get("environment") or {}),
    }

    return {str(key): "" if value is None else str(value) for key, value in env.items()}


def deploy_endpoint(deploy_env_config, secret_config):
    """
    Build the model and put it behind a persistent endpoint
    """
    name = endpoint_name(deploy_env_config)
    model_name = deploy_env_config["model_name"]
    plan = instance_plan(deploy_env_config)
    session = Session()

    build_model(
        deploy_env_config,
        model_name,
        container_environment(deploy_env_config, secret_config),
        plan[0],
        session,
    )

    config_name = create_endpoint_config(deploy_env_config, model_name, session)
    endpoint = apply_endpoint_config(name, config_name)

    print(f"\n{name} is InService on {' -> '.join(plan)}")

    return endpoint


def owned_endpoint_configs(deploy_env_config):
    """
    The endpoint configs this deploy config created, and nothing else

    ListEndpointConfigs only filters on a substring, which would also match the
    configs of an endpoint whose name merely starts with this one - so the stamped
    names are matched exactly before anything gets deleted
    """
    name = endpoint_name(deploy_env_config)
    owned = re.compile(rf"{re.escape(name)}(-{CONFIG_STAMP_PATTERN})?$")

    return [
        config
        for config in EndpointConfig.get_all(name_contains=name)
        if owned.fullmatch(str(config.endpoint_config_name))
    ]


def delete_endpoint(deploy_env_config):
    """
    Stop the endpoint billing, leaving the model behind

    These endpoints bill for their instances whether or not anything is calling
    them, which is why a teardown path matters more than it looks. Endpoint first and
    then its configs, because a config in use by an endpoint will not delete

    The model stays: it costs nothing to keep, build_model replaces one of the same
    name on the next deploy, and deleting it only means rebuilding it to redeploy
    """
    name = endpoint_name(deploy_env_config)

    endpoint = find(Endpoint.get, endpoint_name=name)
    if endpoint:
        print(f"Deleting endpoint {name}")
        endpoint.delete()
        endpoint.wait_for_delete()

    for config in owned_endpoint_configs(deploy_env_config):
        print(f"Deleting endpoint config {config.endpoint_config_name}")
        config.delete()

    print(f"\n{name} torn down; model kept")


# ---------------------------------------------------------------------------
# Batch Transform
# ---------------------------------------------------------------------------


def split_s3_uri(s3_uri):
    _, _, rest = s3_uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def upload_source_dir(source_code_dir, s3_uri):
    """
    Tar the handler directory and upload it for the container to pull at startup

    The inference toolkit accepts an S3 tarball for SAGEMAKER_SUBMIT_DIRECTORY,
    which keeps the handler out of the model artifact - the artifact can then be
    produced by a step that knows nothing about serving, and swapped without a
    repack
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in sorted(os.listdir(source_code_dir)):
            if name.startswith(".") or name == "__pycache__":
                continue
            tar.add(os.path.join(source_code_dir, name), arcname=name)

    bucket, key = split_s3_uri(s3_uri)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"Uploaded handler code: {s3_uri}")

    return s3_uri


def batch_model_name(deploy_env_config, batch_config):
    """
    Name of the SageMaker Model the transform job runs

    Suffixed by default so a batch run cannot overwrite the model definition
    sitting behind a real-time endpoint of the same name
    """
    return batch_config.get("model_name", f"{deploy_env_config['model_name']}-batch")


def build_batch_model(deploy_env_config, batch_config, secret_config, session):
    """
    Build the model the transform job scores with

    CreateModel validates that the artifact exists in S3, so when the artifact
    comes from an earlier pipeline step this cannot run at definition time. Under
    a PipelineSession build() returns step arguments instead of creating the
    model, so the caller can make it a step that depends on whatever stages the
    artifact
    """
    model_name = batch_model_name(deploy_env_config, batch_config)
    model_artifact = resolve_model_artifact(deploy_env_config)

    default_code_uri = (
        f"s3://{session.default_bucket()}/{model_name}/code/sourcedir.tar.gz"
    )
    code_s3_uri = upload_source_dir(
        deploy_env_config["source_code_dir"],
        batch_config.get("code_s3_uri", default_code_uri),
    )

    model_builder = ModelBuilder(
        image_uri=deploy_env_config["image_uri"],
        # Plain uri, not an S3DataSource dict: Batch Transform rejects a model
        # whose container definition carries ModelDataSource
        s3_model_data_url=model_artifact,
        env_vars={
            **secret_config,
            "SAGEMAKER_PROGRAM": deploy_env_config["entry_script"],
            "SAGEMAKER_SUBMIT_DIRECTORY": code_s3_uri,
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_REGION": session.boto_region_name,
        },
        sagemaker_session=session,
        role_arn=deploy_env_config["sagemaker_execution_role"],
    )

    model_builder.model_path = "/tmp/sagemaker/model-builder"
    os.makedirs(model_builder.model_path, exist_ok=True)
    # Workaround: SDK bug - repack_model expects dependencies to be a list of
    # file paths, but ModelBuilder sets it to {'auto': True}
    model_builder.dependencies = []  # type: ignore[assignment]

    return model_name, model_builder.build(model_name=model_name)


def make_transformer(deploy_env_config, batch_config, model_name, session):
    """
    A Transformer and the arguments to run it, built from the deploy config
    """
    transformer = Transformer(
        model_name=model_name,
        instance_count=deploy_env_config["instance_count"],
        instance_type=instance_plan(deploy_env_config)[0],
        output_path=batch_config["output_s3_path"],
        strategy=batch_config.get("strategy", "MultiRecord"),
        assemble_with=batch_config.get("assemble_with", "Line"),
        accept=batch_config.get("accept", "text/csv"),
        max_payload=batch_config.get("max_payload_mb"),
        max_concurrent_transforms=batch_config.get("max_concurrent_transforms"),
        base_transform_job_name=deploy_env_config["model_name"],
        sagemaker_session=session,
    )

    transform_args = {
        "data": batch_config["input_s3_path"],
        "data_type": batch_config.get("input_data_type", "S3Prefix"),
        "content_type": batch_config.get("content_type", "text/csv"),
        "split_type": batch_config.get("split_type", "Line"),
        # Filters slice each record before and after scoring, so a labelled
        # CSV can be scored without a separate feature-only copy
        "input_filter": batch_config.get("input_filter"),
        "output_filter": batch_config.get("output_filter"),
        "join_source": batch_config.get("join_source"),
        "wait": True,
        "logs": True,
    }

    return transformer, transform_args


def make_batch_transform(env=False, session=None):
    """
    Everything needed to run a batch deploy config: the arguments that create the
    model, and the transformer that scores with it

    Pass a PipelineSession to get step arguments back instead of a created model
    """
    _, deploy_env_config, deploy_type = load_deploy_config(env)
    if deploy_type != "batch":
        raise ValueError(
            f"make_batch_transform needs deploy_type 'batch', got '{deploy_type}'"
        )

    secret_config = secrets_env(deploy_env_config)
    batch_config = deploy_env_config.get("batch", {})
    session = session or Session()

    model_name, model_args = build_batch_model(
        deploy_env_config, batch_config, secret_config, session
    )
    transformer, transform_args = make_transformer(
        deploy_env_config, batch_config, model_name, session
    )

    return model_args, transformer, transform_args, deploy_env_config


def run_transformer(transformer, transform_args, deploy_env_config):
    transformer_transform = transformer.transform(**transform_args)

    if not isinstance(transformer.sagemaker_session, PipelineSession):
        print(f"\nBatch transform complete for {deploy_env_config['model_name']}!")
        print(f"Predictions: {transformer.output_path}")

    return transformer_transform


def deploy_model(env=False, delete=False):
    args = parse_args()

    # Discovery rather than deployment: these read the JumpStart hub and never open a
    # config file, so they answer before one has been written
    if args.search_jumpstart:
        return print_search(args.search_jumpstart)
    if args.jumpstart:
        return print_recipe(
            args.jumpstart,
            instance_type=args.instance_type,
            config_name=args.jumpstart_config,
        )
    if args.vllm_recipe:
        if not args.instance_type:
            raise ValueError(
                "--vllm-recipe needs --instance-type: which knobs exist depends on "
                "the GPU the recipe is being read for"
            )
        return print_deploy_config(args.vllm_recipe, args.instance_type)

    args, deploy_env_config, deploy_type = load_deploy_config(env)

    if delete or args.delete:
        if deploy_type != "real-time":
            raise ValueError(
                "--delete tears down an endpoint; a batch transform job leaves "
                "nothing running to tear down"
            )
        return delete_endpoint(deploy_env_config)

    # Pre-warm the prefix so the download is not sitting in front of an endpoint that
    # is waiting on reserved capacity
    if args.stage_only:
        if not deploy_env_config.get("huggingface"):
            raise ValueError("--stage-only needs a huggingface block to stage")
        print(resolve_model_artifact(deploy_env_config))
        # Extra channels are staged too, since pre-warming that leaves a draft model
        # to download in front of the endpoint is not pre-warming
        for source in deploy_env_config.get("additional_model_data_sources") or []:
            print(resolve_extra_artifact(deploy_env_config, source))
        return None

    if deploy_type == "batch":
        _, transformer, transform_args, batch_env_config = make_batch_transform(env=env)
        return run_transformer(transformer, transform_args, batch_env_config)

    secret_config = secrets_env(deploy_env_config)
    return deploy_endpoint(deploy_env_config, secret_config)


if __name__ == "__main__":
    deploy_model()
