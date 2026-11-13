"""
Read deployment recipes out of the SageMaker JumpStart public hub
and generate a deploy config
"""

import json
import re
import textwrap

import boto3

# The managed hub every account can read; a private hub would be named here instead

PUBLIC_HUB = "SageMakerPublicHub"

# Marks a config as generated from a hub recipe rather than hand-tuned, so the config
# file, the model and the endpoint all say where their settings came from
JUMPSTART_SUFFIX = "-jumpstart"

# What CreateModel and CreateEndpoint accept for a name, less the -YYYYMMDD-HHMMSS that
# deploy.py stamps onto the endpoint config it builds from that name
NAME_LIMIT = 63 - len("-20260101-000000")

# Deploy-config keys this module fills in from a recipe, in the order they are printed
RECIPE_KEYS = (
    "model_name",
    "endpoint_name",
    "model_data_s3_uri",
    "accept_eula",
    "additional_model_data_sources",
    "image_uri",
    "instance_preferences",
    "instance_count",
    "model_data_download_timeout_seconds",
    "startup_health_check_timeout_seconds",
    "environment",
    "enable_network_isolation",
    "serializer",
    "deserializer",
    "payload_path",
    "sagemaker_execution_role",
)


def client(region=None):
    return boto3.client("sagemaker", region_name=region)


def search(query, region=None, hub=PUBLIC_HUB):
    """
    Hub model ids

    ListHubContents has no search parameter, so the whole catalogue is paged and
    """
    terms = [term.lower() for term in query.split()]
    sagemaker = client(region)
    matches = []

    # Paged by hand: botocore models no paginator for this operation, and MaxResults
    # is capped at 100
    token = None
    while True:
        request = {"HubName": hub, "HubContentType": "Model", "MaxResults": 100}
        if token:
            request["NextToken"] = token

        response = sagemaker.list_hub_contents(**request)
        for summary in response["HubContentSummaries"]:
            name = summary["HubContentName"]
            if all(term in name.lower() for term in terms):
                matches.append(name)

        token = response.get("NextToken")
        if not token:
            return sorted(matches)


def describe(model_id, region=None, hub=PUBLIC_HUB):
    """
    A hub model's document
    """
    response = client(region).describe_hub_content(
        HubName=hub, HubContentType="Model", HubContentName=model_id
    )

    return response["HubContentVersion"], json.loads(response["HubContentDocument"])


def hub_model_for(huggingface_id, region=None, hub=PUBLIC_HUB):
    """
    The hub model serving a HuggingFace repo

    Returns None when no hub model claims the repo, which is the usual answer for
    anything recent enough to be deployed from HuggingFace in the first place
    """
    name = short_model_name(huggingface_id, {"HuggingfaceId": huggingface_id})
    wanted = huggingface_id.lower()

    # The full name first, then its leading word, so qwen3-6-27b-fp8 falls back to
    # qwen3 rather than giving up when the hub spells the version differently
    for query in (name, name.split("-")[0]):
        for candidate in search(query, region, hub):
            version, document = describe(candidate, region, hub)
            if (document.get("HuggingfaceId") or "").lower() == wanted:
                return candidate, version, document

    return None


def config_names(document):
    """
    The model's inference configs that are ranked by AWS;
    first is best
    """
    rankings = document.get("InferenceConfigRankings") or {}
    overall = rankings.get("overall") or {}

    return list(overall.get("Rankings") or [])


def container_repository(merged):
    """
    The ECR repository name out of a config's image uri
    """
    return (merged.get("HostingEcrUri") or "").split("/")[-1].split(":")[0]


def openai_api_container(merged):
    """
    Whether the config's container serves the OpenAI API rather than TGI's schema
    """
    return container_repository(merged) in ("vllm", "sglang")


def resolve_config(document, config_name=None):
    """
    One inference config merged over the model's top-level defaults

    A config component overrides the document's own hosting keys, so reading a key
    means asking the component first and the document second. Models that publish no
    configs at all answer entirely from the document
    """
    components = document.get("InferenceConfigComponents") or {}
    if config_name is None:
        # The hub's own first choice, which is what the console deploys. A ranking can
        # name a config the model does not actually publish, so it is filtered first
        ranked = (name for name in config_names(document) if name in components)
        config_name = next(ranked, None)

    if config_name is not None and config_name not in components:
        raise ValueError(
            f"No inference config '{config_name}'; this model publishes "
            f"{sorted(components) or 'none'}"
        )

    component = components.get(config_name, {})

    return config_name, {**document, **component}


def parallel_degree(env):
    """
    The environment's tensor-parallel setting
    """
    for name in (
        "SM_VLLM_TENSOR_PARALLEL_SIZE",
        "SM_SGLANG_TENSOR_PARALLEL_SIZE",
        "OPTION_TENSOR_PARALLEL_DEGREE",
        "TENSOR_PARALLEL_DEGREE",
    ):
        if env.get(name):
            return env[name]

    return None


def single_gpu_instance_types(merged):
    """
    Supported single-GPU instance types the recipe runs on
    """
    variants = (merged.get("HostingInstanceTypeVariants") or {}).get("Variants") or {}
    supported = merged.get("SupportedInferenceInstanceTypes") or []

    single = []
    for instance_type in supported:
        properties = (variants.get(instance_type) or {}).get("Properties") or {}
        requirements = properties.get("ResourceRequirements") or {}
        if requirements.get("NumAccelerators") == 1:
            single.append(instance_type)

    return single


def instance_environment(merged, instance_type):
    """
    The recipe's environment with the chosen instance type's overrides applied

    Tensor-parallel degree and context length are per instance type
    """
    env = {
        variable["Name"]: variable.get("Default")
        for variable in merged.get("InferenceEnvironmentVariables") or []
        if variable.get("Default") is not None
    }

    variants = (merged.get("HostingInstanceTypeVariants") or {}).get("Variants") or {}
    properties = (variants.get(instance_type) or {}).get("Properties") or {}
    env.update(properties.get("EnvironmentVariables") or {})

    return env


def additional_data_sources(merged):
    """
    Extra artifacts the container expects beside the weights, flattened
    """
    grouped = merged.get("HostingAdditionalDataSources") or {}

    return [
        {
            "channel_name": source["ChannelName"],
            "s3_uri": source["S3DataSource"]["S3Uri"],
        }
        for sources in grouped.values()
        for source in sources
    ]


def sanitize(name):
    """
    A name SageMaker takes: lower case, alphanumeric, single dashes

    Model versions carry dots like Qwen3.6-27B-FP8 which CreateModel rejects
    """
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def short_model_name(model_id, document):
    """
    What people call the model, without the vendor prefix the hub id carries
    """
    huggingface_id = document.get("HuggingfaceId")

    return sanitize(huggingface_id.split("/")[-1] if huggingface_id else model_id)


def short_instance_type(instance_type):
    """
    ml.g7e.4xlarge as g7e-4x, which is how the configs in use_cases name their shape
    """
    return sanitize(instance_type.removeprefix("ml.").replace("xlarge", "x"))


def config_stem(model_id, document, instance_type):
    """
    The name shared by the generated config file, the model, and the endpoint
    """
    tail = JUMPSTART_SUFFIX
    if instance_type:
        tail += f"-{short_instance_type(instance_type)}"
    model = short_model_name(model_id, document)[: NAME_LIMIT - len(tail)]

    return f"{model.strip('-')}{tail}"


def recipe(model_id, region=None, instance_type=None, config_name=None, hub=PUBLIC_HUB):
    """
    Everything needed to deploy a hub model

    instance_type defaults to a single-GPU option where the recipe has one, otherwise it falls back to the recipe's own default
    """
    version, document = describe(model_id, region, hub)
    config_name, merged = resolve_config(document, config_name)

    single = single_gpu_instance_types(merged)
    default_type = merged.get("DefaultInferenceInstanceType")
    instance_type = instance_type or (single[0] if single else default_type)
    stem = config_stem(model_id, document, instance_type)
    supported = merged.get("SupportedInferenceInstanceTypes") or []
    env = instance_environment(merged, instance_type)

    if instance_type not in supported and parallel_degree(env) is None:
        raise ValueError(
            f"Inference config '{config_name}' publishes no serving parameters for "
            f"{instance_type}, only for {supported}. Pick one of those, or a config "
            f"that covers {instance_type}"
        )

    return {
        "model_id": model_id,
        "stem": stem,
        "hub_version": version,
        "config_name": config_name,
        "config_choices": config_names(document),
        "huggingface_id": document.get("HuggingfaceId"),
        "license": document.get("License"),
        "model_access": document.get("ModelAccess"),
        "eula_uri": document.get("HostingEulaUri"),
        "context_window": document.get("ContextWindow"),
        "supported_instance_types": merged.get("SupportedInferenceInstanceTypes") or [],
        "single_gpu_instance_types": single,
        "default_instance_type": default_type,
        "default_payload": default_payload(merged),
        "config": {
            # Model, instance shape, and the -jumpstart suffix, so two generated configs
            # for the same model on different hardware never collide
            "model_name": stem,
            "endpoint_name": stem,
            "model_data_s3_uri": merged.get("HostingArtifactUri"),
            "accept_eula": True if document.get("GatedBucket") else None,
            "additional_model_data_sources": additional_data_sources(merged) or None,
            "image_uri": merged.get("HostingEcrUri"),
            "instance_preferences": [instance_type],
            "instance_count": 1,
            "model_data_download_timeout_seconds": merged.get(
                "ModelDataDownloadTimeout"
            ),
            "startup_health_check_timeout_seconds": merged.get(
                "ContainerStartupHealthCheckTimeout"
            ),
            "environment": env,
            "enable_network_isolation": merged.get("InferenceEnableNetworkIsolation"),
            "serializer": "json",
            "deserializer": "json",
            "payload_path": f"./deploy/payload.{stem}.json",
            "sagemaker_execution_role": "<your SageMaker execution role arn>",
        },
    }


def default_payload(merged):
    """
    A request body the model responds to, if the hub published one
    """
    payloads = merged.get("DefaultPayloads") or {}
    body = next(
        (payload["Body"] for payload in payloads.values() if payload.get("Body")), None
    )

    if not isinstance(body, dict) or not openai_api_container(merged):
        return body

    content = body.get("inputs") or body.get("prompt")
    if not isinstance(content, str):
        return body

    # Only the prompt and the length carry over; the rest of TGI's parameters have no
    # straight equivalent and the server defaults them anyway
    parameters = body.get("parameters") or {}

    return {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": body.get("max_tokens", parameters.get("max_new_tokens", 256)),
    }


def env_scalar(setting):
    setting = str(setting)
    needs_quoting = setting[:1] in ("'", '"', "{", "[", "&", "*", "!", "#", "%", "@")

    if setting == "" or needs_quoting or ": " in setting:
        return json.dumps(setting)

    return setting


def as_yaml(recipe_dict):
    """
    The recipe as a deploy config, ready to paste into deploy/deploy.x.yaml
    """
    config = recipe_dict["config"]
    lines = []

    for key in RECIPE_KEYS:
        value = config.get(key)
        if value is None:
            continue

        if key == "additional_model_data_sources":
            lines.append(f"{key}:")
            for source in value:
                lines.append(f"  - channel_name: {source['channel_name']}")
                lines.append(f"    s3_uri: \"{source['s3_uri']}\"")
        elif key == "environment":
            lines.append(f"{key}:")
            for name, setting in sorted(value.items()):
                lines.append(f"  {name}: {env_scalar(setting)}")
        elif isinstance(value, list):
            # instance_preferences and anything else that is a plain list, as a block
            # sequence rather than python's repr of one
            lines.append(f"{key}:")
            lines += [f"  - {entry}" for entry in value]
        elif isinstance(value, bool):
            lines.append(f"{key}: {str(value).lower()}")
        elif isinstance(value, str) and (value.startswith("s3://") or " " in value):
            lines.append(f'{key}: "{value}"')
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def print_search(query, region=None):
    matches = search(query, region)
    if not matches:
        print(f"No JumpStart models match '{query}'")
        return

    print(f"{len(matches)} JumpStart models match '{query}':\n")
    for model_id in matches:
        print(f"  {model_id}")
    print("\nRun deploy.py --jumpstart <model-id> for one of them")


def print_recipe(model_id, region=None, instance_type=None, config_name=None):
    found = recipe(model_id, region, instance_type, config_name)
    config = found["config"]

    print(f"{found['model_id']} (hub version {found['hub_version']})")
    print(f"  hugging face : {found['huggingface_id']}")
    print(f"  license      : {found['license']} ({found['model_access']})")
    if found["eula_uri"]:
        print(f"  eula         : {found['eula_uri']}")
    print(f"  inference config : {found['config_name']}")
    others = [n for n in found["config_choices"] if n != found["config_name"]]
    if others:
        # A model can publish twenty of these, so they wrap rather than run off the
        # edge of the terminal
        plural = "other" if len(others) == 1 else "others"
        print(
            textwrap.fill(
                f"{len(others)} {plural}, for --jumpstart-config: {', '.join(others)}",
                width=88,
                initial_indent="  ",
                subsequent_indent="    ",
            )
        )
    print(f"  single-gpu types : {found['single_gpu_instance_types'] or 'none'}")
    print(f"  all types        : {found['supported_instance_types']}")

    if config["accept_eula"]:
        print(
            "\n  Gated model: accept_eula below is you agreeing to the licence above, "
            "and the weights will not download without it"
        )
    if not found["single_gpu_instance_types"]:
        print(
            f"\n  No single-gpu option published, so these settings are written for "
            f"{config['instance_type']}. Moving to a different gpu count means "
            f"changing the tensor-parallel variables by hand"
        )

    print(f"\n# --- deploy/deploy.{found['stem']}.yaml ---")
    print(as_yaml(found))

    if found["default_payload"] is not None:
        print(f"\n# --- {config['payload_path']} ---")
        print(json.dumps(found["default_payload"], indent=2))
