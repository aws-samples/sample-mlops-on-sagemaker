import fnmatch
import json
import pathlib
import urllib.error
import urllib.request

from botocore.exceptions import ClientError
from job_secrets import SECRETS_ARN_ENV
from .jumpstart import (
    NAME_LIMIT,
    env_scalar,
    hub_model_for,
    instance_environment,
    resolve_config,
    sanitize,
    short_instance_type,
)
from .vllm_recipes import ENV_PREFIX as ENV_PREFIX_VLLM
from .vllm_recipes import architecture as vllm_architecture
from .vllm_recipes import recipe_document
from .vllm_recipes import translate as vllm_translate
from .vllm_recipes import print_recipe as vllm_print_recipe
from .vllm_recipes import recipe as vllm_recipe
from .vllm_recipes import serving_environment as vllm_serving_environment

from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.processing import FrameworkProcessor
from sagemaker.core.shapes import ProcessingOutput, ProcessingS3Output

HUB_API = "https://huggingface.co/api/models"

STAGING_FRAMEWORK = "pytorch"
STAGING_FRAMEWORK_VERSION = "2.8.0"
STAGING_INSTANCE_TYPE = "ml.m5.4xlarge"
# A 100 GB+ repo past a slow mirror + upload to S3
STAGING_MAX_RUNTIME_SECONDS = 14400

# Where the job writes the snapshot and the code it runs to get there
STAGING_OUTPUT_NAME = "weights"
STAGING_LOCAL_PATH = "/opt/ml/processing/weights"
STAGING_SOURCE_DIR = str(pathlib.Path(__file__).parent / "hf_staging")
STAGING_ENTRY_SCRIPT = "stage.py"
STAGING_REQUIREMENTS = "requirements.txt"

SECRET_KEY_ENV = "HF_SECRET_KEY"
DEFAULT_SECRET_KEY = "HF_TOKEN"

IGNORE_PATTERNS = ("*.msgpack", "*.h5", "*.ot", "original/**")

# The volume holds the snapshot and the copy SageMaker uploads from
VOLUME_HEADROOM = 2.5

# Smallest volume SageMaker will attach
MIN_VOLUME_GB = 30


def hf_config(deploy_env_config):
    """
    The huggingface block of a deploy config, checked for the keys staging needs
    """
    config = deploy_env_config.get("huggingface") or {}

    missing = [key for key in ("model_id", "stage_to_s3") if not config.get(key)]
    if missing:
        raise ValueError(f"huggingface config needs {missing}")

    return config


# Context length to write when nothing published one, bracketed by the GB left after
# the weights. Deliberately conservative: a config that loads and serves a short
# window beats one that OOMs, and the value is a line in the file to raise
CONTEXT_BRACKETS = ((48, 16384), (128, 32768), (256, 65536), (float("inf"), 131072))
CONTEXT_FLOOR = 16384

# Where SageMaker mounts each extra artifact channel
CHANNEL_MOUNT = "/opt/ml/additional-model-data-sources"

# Settings inside a recipe's speculative config that name a model to draft with
DRAFT_KEYS = ("model", "draft_model")


def unapplied_features(model_id, requested):
    """
    The recipe's opt-in features as commented-out settings, ready to uncomment

    Every knob the recipe knows about ends up in the file either way, which is the
    point - but these are opt-in upstream because they cost something. long_context
    asks for a million-token window that will not fit on a small card, text_only
    turns off multimodal. Listing them applied would reintroduce the OOMs that
    context_cap exists to prevent, so they are listed inert instead
    """
    document = recipe_document(model_id)
    if document is None:
        return []

    available = document.get("features") or {}
    opt_in = [
        name
        for name in document.get("opt_in_features") or []
        if name not in requested and (available.get(name) or {}).get("args")
    ]
    if not opt_in:
        return []

    lines = [
        "  #",
        "  # Opt-in features this model's recipe publishes, not applied above.",
        "  # Uncomment a block to turn one on; a key it repeats replaces the one",
        "  # above it",
    ]
    for name in opt_in:
        lines.append(f"  # {name}:")
        settings = vllm_translate(list(available[name]["args"]))
        for key, value in sorted(settings.items()):
            lines.append(f"  #   {key}: {env_scalar(value)}")

    return lines


def context_cap(size_gb, arch):
    """
    A context length the weights leave room to serve, as (tokens, how it was reached)

    Left unset, vLLM sizes its KV cache for whatever window the model advertises, and
    a model that nearly fills the card OOMs at load rather than starting small - three
    deploys died that way before this existed. Picking a correct length needs the KV
    bytes per token, which depends on layers, kv heads and head dim and does not
    generalise across MoE and Mamba hybrids, so this brackets by headroom instead and
    says so in the config. Raise it once an endpoint is up and the cache fits
    """
    if arch is None or not arch.get("memory_mib"):
        return CONTEXT_FLOOR, "no GPU memory figure for this instance"

    vram_gb = arch["memory_mib"] * arch["count"] / 1024
    headroom = vram_gb - size_gb

    for limit, tokens in CONTEXT_BRACKETS:
        if headroom < limit:
            return tokens, (
                f"{size_gb:.0f} GB of weights on {vram_gb:.0f} GB leaves "
                f"~{headroom:.0f} GB for KV cache"
            )

    return CONTEXT_BRACKETS[-1][1], (
        f"{headroom:.0f} GB spare after weights; raise if the model allows more"
    )


def config_stem(model_id, instance_type):
    """
    The name shared by the config file, the model, and the endpoint
    """
    tail = f"-hf-{short_instance_type(instance_type)}"
    model = sanitize(model_id.split("/")[-1])[: NAME_LIMIT - len(tail)]

    return f"{model.strip('-')}{tail}"


def deploy_config(model_id, instance_type, region=None, features=()):
    """
    A deploy config for a Hub model, ready to paste into deploy/deploy.x.yaml
    """
    metadata = describe(model_id)
    size_gb = download_size_gb(metadata, IGNORE_PATTERNS)
    # Off the instance, not off the recipe: the GPU count is knowable either way, and
    # defaulting it to 1 would under-shard a model on a multi-GPU shape
    arch = vllm_architecture(instance_type, region)
    count = arch["count"] if arch else 1

    stem = config_stem(model_id, instance_type)
    image = image_uris.retrieve(
        framework="vllm",
        region=region or Session().boto_region_name,
        image_scope="inference",
        instance_type=instance_type,
    )
    # Long enough for the staged prefix to land on a fresh instance, floored for small
    # repos and capped at what the API accepts
    timeout = max(1200, min(3600, int(size_gb * 45)))

    serving, _ = serving_environment(
        {
            "instance_preferences": [instance_type],
            "huggingface": {
                "model_id": model_id,
                "stage_to_s3": "unused-during-generation",
                "vllm_features": features,
            },
        },
        region,
    )

    lines = [
        f"# Generated by deploy.py --vllm-recipe {model_id} "
        f"--instance-type {instance_type}",
        f"model_name: {stem}",
        f"endpoint_name: {stem}",
        "huggingface:",
        f"  model_id: {model_id}",
        "  # The commit to stage",
        f"  # curl -s https://huggingface.co/api/models/{model_id} | jq -r .sha",
        f"  revision: {metadata['sha']}",
        '  stage_to_s3: "s3://<your staging bucket>/hf-weights/"',
    ]
    if metadata.get("gated"):
        lines.append("  secrets_arn: <arn of a secret holding a Hub token>")
    lines += [
        f"image_uri: {image}",
        "# First is the shape to run on; add more as capacity fallbacks",
        "instance_preferences:",
        f"  - {instance_type}",
        "instance_count: 1",
    ]
    # Merged rather than concatenated
    environment = {
        "ENDPOINT_SERVER_TIMEOUT": 3600,
        "HF_MODEL_ID": "/opt/ml/model",
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_CONTAINER_LOG_LEVEL": 20,
        "SAGEMAKER_MODEL_SERVER_TIMEOUT": 3600,
        "SAGEMAKER_MODEL_SERVER_WORKERS": 1,
        # vLLM DLC reads SM_VLLM_TENSOR_PARALLEL_SIZE and ignores
        # TENSOR_PARALLEL_DEGREE, which is used by DJL/LMI's. So
        # include both.
        "TENSOR_PARALLEL_DEGREE": count,
        f"{ENV_PREFIX_VLLM}TENSOR_PARALLEL_SIZE": count,
        **serving,
    }

    # Only when no source set one: a published figure was measured, this is a bracket
    capped = None
    if f"{ENV_PREFIX_VLLM}MAX_MODEL_LEN" not in environment:
        tokens, why = context_cap(size_gb, arch)
        environment[f"{ENV_PREFIX_VLLM}MAX_MODEL_LEN"] = tokens
        capped = why

    lines += [
        f"model_data_download_timeout_seconds: {timeout}",
        f"startup_health_check_timeout_seconds: {timeout}",
        "# Serving settings",
        "environment:",
    ]
    for key, value in sorted(environment.items()):
        if capped and key == f"{ENV_PREFIX_VLLM}MAX_MODEL_LEN":
            lines.append(f"  # Not published for this model; {capped}")
        lines.append(f"  {key}: {env_scalar(value)}")

    lines += unapplied_features(model_id, features)
    lines += [
        "enable_network_isolation: true",
        "serializer: json",
        "deserializer: json",
        "payload_path: ./deploy/payload.chat.json",
        'sagemaker_execution_role: "<your SageMaker execution role arn>"',
    ]

    return "\n".join(lines)


def print_deploy_config(model_id, instance_type, region=None, features=()):
    """
    The recipe, then a config built from it
    """
    vllm_print_recipe(model_id, instance_type, region)
    stem = config_stem(model_id, instance_type)
    print(f"\n# --- deploy/deploy.{stem}.yaml ---")
    print(deploy_config(model_id, instance_type, region, features))


def local_draft_paths(deploy_env_config, environment):
    """
    A speculative config's draft model as a path the container can actually read
    """
    key = f"{ENV_PREFIX_VLLM}SPECULATIVE_CONFIG"
    raw = environment.get(key)
    if not raw:
        return environment

    settings = json.loads(raw)
    channels = {
        (source.get("huggingface") or {}).get("model_id"): source["channel_name"]
        for source in deploy_env_config.get("additional_model_data_sources") or []
    }

    declared = {
        source["channel_name"]
        for source in deploy_env_config.get("additional_model_data_sources") or []
    }

    for draft_key in DRAFT_KEYS:
        draft = settings.get(draft_key)
        if not isinstance(draft, str):
            continue

        if draft.startswith(f"{CHANNEL_MOUNT}/"):
            if draft.rsplit("/", 1)[-1] not in declared:
                print(
                    f"Dropping speculative config: it drafts from {draft}, and no "
                    f"additional_model_data_sources channel supplies that"
                )
                return {k: v for k, v in environment.items() if k != key}
            continue
        if draft.startswith("/"):
            continue

        channel = channels.get(draft)
        if channel is None:
            raise ValueError(
                f"the recipe drafts with '{draft}', which the endpoint cannot fetch "
                f"with network isolation on. Stage it by adding a channel:\n"
                f"additional_model_data_sources:\n"
                f"  - channel_name: draft\n"
                f"    huggingface:\n"
                f"      model_id: {draft}"
            )
        settings[draft_key] = f"{CHANNEL_MOUNT}/{channel}"

    return {**environment, key: json.dumps(settings)}


def serving_environment(deploy_env_config, region=None):
    """
    Serving settings for a HuggingFace model

    Preferred source is the recipe vLLM publishes for this model on this instance's
    GPU. When nothing is published for the pairing, the hub's settings for the same model
    are the fallback.

    Returns an empty environment when neither has anything, which leaves the config's
    own environment block as the only source.
    """
    config = hf_config(deploy_env_config)
    model_id = config["model_id"]
    # First of instance_preferences is the shape to serve on; the rest are capacity
    # fallbacks. Resolved here rather than imported from deploy.py, which imports this
    instance_type = (deploy_env_config["instance_preferences"] or [])[0]
    features = config.get("vllm_features") or ()

    selected = vllm_recipe(model_id, instance_type, region, features)
    if selected and not selected["portable_only"]:
        return (
            local_draft_paths(deploy_env_config, vllm_serving_environment(selected)),
            f"vLLM recipe for {selected['hardware']} ({selected['status']})",
        )

    # A recipe for other hardware still knows the model's parsers and prompt format,
    # and the hub knows an instance type. Neither alone is the whole answer
    portable = vllm_serving_environment(selected) if selected else {}
    hub = {}
    sources = []

    found = hub_model_for(model_id, region)
    if found:
        hub_id, _, document = found
        _, merged = resolve_config(document)
        hub = instance_environment(merged, instance_type)
        sources.append(f"JumpStart hub tuning from {hub_id}")
    if portable and selected:
        sources.append(
            f"model settings from the vLLM recipe written for "
            f"{selected['recommended_hardware']}"
        )

    if not sources:
        return {}, "nothing published for this model"

    # The recipe is the better authority on anything model-shaped, so it lands on top.
    # The hub's own speculative config goes through the same draft check, since it
    # names a channel mount that only its own config declared
    combined = local_draft_paths(deploy_env_config, {**hub, **portable})

    return combined, " plus ".join(sources)


def describe(model_id, revision=None):
    """
    Hub metadata for a repo: its commit sha, whether it is gated, and its file sizes

    A repo the Hub will not describe answers 401 rather than 404, because a private one
    must not be distinguishable from one that never existed. Gated repos do answer
    unauthenticated, so sizing and pinning work before a token is in play
    """
    path = f"{model_id}/revision/{revision}" if revision else model_id

    try:
        with urllib.request.urlopen(f"{HUB_API}/{path}?blobs=true", timeout=30) as f:
            return json.load(f)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403, 404):
            at = f" at revision '{revision}'" if revision else ""
            raise ValueError(
                f"The Hub will not describe '{model_id}'{at}. Check the id for a "
                f"typo - a private repo answers the same way as a missing one"
            ) from error
        raise


def download_size_gb(metadata, ignore_patterns):
    """
    Size of the files staging will actually pull, in GB

    The ignored formats are counted out, since they are often most of a repo that
    publishes more than one and the number is here to size a volume
    """
    kept = [
        sibling
        for sibling in metadata.get("siblings") or []
        if not any(
            fnmatch.fnmatch(sibling.get("rfilename", ""), pattern)
            for pattern in ignore_patterns
        )
    ]

    return sum(sibling.get("size") or 0 for sibling in kept) / 1e9


def staged_layout(config, sha):
    """
    Where a snapshot of one commit lives: the weights prefix, and its marker

    The marker sits beside the prefix rather than inside it, so it is not mounted into
    /opt/ml/model with the weights
    """
    base = config["stage_to_s3"].rstrip("/")
    name = sanitize(config["model_id"])

    return f"{base}/{name}/{sha}/", f"{base}/{name}/{sha}.staged.json"


def split_s3_uri(s3_uri):
    """
    An s3:// uri as the bucket and key boto3 wants
    """
    bucket, _, key = s3_uri.removeprefix("s3://").partition("/")

    return bucket, key


def find_marker(marker_uri, s3_client):
    """
    The marker a finished staging job left, or None

    Written only after the job reports success, so finding it rules out the
    half-uploaded prefix a job that died partway leaves behind - which is what
    checking the prefix for objects instead would accept
    """
    bucket, key = split_s3_uri(marker_uri)

    try:
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as error:
        if error.response["Error"]["Code"] in ("NoSuchKey", "404", "NoSuchBucket"):
            return None
        raise

    return json.loads(body)


def staging_failure(processor, session):
    """
    Why the staging job failed, as a message worth reading

    The SDK raises AttributeError building its own UnexpectedStatusException, so a
    failed job otherwise arrives with the reason nowhere in the traceback
    """
    job = getattr(processor, "latest_job", None)
    name = getattr(job, "processing_job_name", None)
    if not name:
        return "The staging job failed before it was named"

    described = session.boto_session.client("sagemaker").describe_processing_job(
        ProcessingJobName=name
    )
    reason = described.get("FailureReason") or described.get("ProcessingJobStatus")

    return (
        f"Staging job {name} failed: {reason}. The script's own traceback is in its "
        f"CloudWatch log stream, under /aws/sagemaker/ProcessingJobs"
    )


def stage(deploy_env_config, config, metadata, sha, prefix, marker_uri, session):
    """
    Download a Hub repo into the staged prefix with a SageMaker processing job

    The job runs in the account rather than locally because the weights are tens to
    hundreds of GB: inside AWS that is a Hub-to-S3 copy on a 10 Gbps link, and the
    token stays in Secrets Manager for the job to read, which is the arrangement
    job_secrets.py exists to keep
    """
    ignore_patterns = config.get("ignore_patterns", list(IGNORE_PATTERNS))
    size_gb = download_size_gb(metadata, ignore_patterns)
    volume_gb = config.get("volume_size_gb") or max(
        MIN_VOLUME_GB, int(size_gb * VOLUME_HEADROOM) + 1
    )

    image_uri = config.get("image_uri") or image_uris.retrieve(
        framework=STAGING_FRAMEWORK,
        region=session.boto_region_name,
        version=STAGING_FRAMEWORK_VERSION,
        instance_type=config.get("instance_type", STAGING_INSTANCE_TYPE),
        image_scope="training",
    )

    # Only a gated repo needs a token, so an ungated one is staged without the ARN
    # rather than failing on a Secrets Manager permission it never needed
    env = {"AWS_DEFAULT_REGION": session.boto_region_name}
    if metadata.get("gated") and config.get("secrets_arn"):
        env[SECRETS_ARN_ENV] = str(config["secrets_arn"])
        env[SECRET_KEY_ENV] = str(config.get("secret_key", DEFAULT_SECRET_KEY))

    print(
        f"Staging {config['model_id']} at {sha[:12]} to {prefix}\n"
        f"  {size_gb:.1f} GB to download, on a {volume_gb} GB volume. One time per "
        f"commit; later deploys of this config find the marker and skip it"
    )

    processor = FrameworkProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=config.get("instance_type", STAGING_INSTANCE_TYPE),
        instance_count=1,
        volume_size_in_gb=volume_gb,
        max_runtime_in_seconds=int(
            config.get("max_runtime_seconds", STAGING_MAX_RUNTIME_SECONDS)
        ),
        role=deploy_env_config["sagemaker_execution_role"],
        env=env,
        # SageMaker appends a timestamp, and rejects a name ending in a dash, which is
        # what trimming a long repo id to fit leaves behind
        base_job_name=f"stage-{sanitize(config['model_id'])}"[:NAME_LIMIT].rstrip("-"),
        sagemaker_session=session,
    )

    arguments = ["--repo-id", config["model_id"], "--revision", sha]
    for pattern in ignore_patterns:
        arguments += ["--ignore-pattern", pattern]

    try:
        processor.run(
            source_dir=STAGING_SOURCE_DIR,
            code=STAGING_ENTRY_SCRIPT,
            requirements=STAGING_REQUIREMENTS,
            arguments=arguments,
            outputs=[
                ProcessingOutput(
                    output_name=STAGING_OUTPUT_NAME,
                    s3_output=ProcessingS3Output(
                        s3_uri=prefix,
                        local_path=STAGING_LOCAL_PATH,
                        s3_upload_mode="EndOfJob",
                    ),
                )
            ],
        )
    except Exception as error:
        raise RuntimeError(staging_failure(processor, session)) from error

    bucket, key = split_s3_uri(marker_uri)
    session.boto_session.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(
            {
                "model_id": config["model_id"],
                "revision": sha,
                "prefix": prefix,
                "download_size_gb": round(size_gb, 2),
                "ignore_patterns": ignore_patterns,
            },
            indent=2,
        ).encode(),
    )

    return prefix


def staged_artifact(deploy_env_config, session=None):
    """
    S3 prefix holding the config's main Hub repo, staging it the first time
    """
    return stage_repo(deploy_env_config, hf_config(deploy_env_config), session)


def staged_source(deploy_env_config, source, session=None):
    """
    S3 prefix for an extra artifact named as a Hub repo rather than an s3 uri

    A draft model for speculative decoding or a LoRA adapter is published on the Hub
    like any other repo, and staging it by hand to paste the prefix back in defeats
    having a staging path at all. Where to put it and which token to use default to
    the main repo's answers, since the usual case is one bucket and one account
    """
    config = {
        "stage_to_s3": (deploy_env_config.get("huggingface") or {}).get("stage_to_s3"),
        "secrets_arn": (deploy_env_config.get("huggingface") or {}).get("secrets_arn"),
        "secret_key": (deploy_env_config.get("huggingface") or {}).get("secret_key"),
        **{key: value for key, value in source["huggingface"].items() if value},
    }

    missing = [key for key in ("model_id", "stage_to_s3") if not config.get(key)]
    if missing:
        raise ValueError(
            f"additional_model_data_sources channel '{source['channel_name']}' needs "
            f"{missing}; stage_to_s3 is inherited from the huggingface block when the "
            f"channel leaves it out"
        )

    return stage_repo(deploy_env_config, config, session)


def stage_repo(deploy_env_config, config, session=None):
    """
    S3 prefix holding a Hub repo, staging it the first time it is asked for

    Keyed on the resolved commit rather than the branch name, so a config gets the same
    weights every deploy and pays for the download once. The endpoint then reads the
    prefix the way it reads a JumpStart one, with no token and no route to the Hub
    """
    metadata = describe(config["model_id"], config.get("revision"))

    sha = metadata.get("sha")
    if not sha:
        raise ValueError(f"The Hub published no commit sha for {config['model_id']}")

    session = session or Session()
    prefix, marker_uri = staged_layout(config, sha)

    marker = find_marker(marker_uri, session.boto_session.client("s3"))
    if marker:
        print(f"Using staged weights at {prefix}")
        return marker.get("prefix", prefix)

    if metadata.get("gated") and not config.get("secrets_arn"):
        raise ValueError(
            f"{config['model_id']} is gated, so staging it needs a token: set "
            f"huggingface.secrets_arn to a Secrets Manager secret holding one, from "
            f"an account that has accepted the terms on the model's Hub page"
        )

    return stage(deploy_env_config, config, metadata, sha, prefix, marker_uri, session)
