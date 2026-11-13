import json
import urllib.error
import urllib.request

import boto3
from botocore.exceptions import ClientError

RECIPES_HOST = "https://recipes.vllm.ai"

CATALOG_PATH = "/models.json"

# The GPU as EC2 names it, mapped to how recipes.vllm.ai keys a recipe (the hardware
# token) and to the architecture generation its hardware_overrides are keyed by
#
# Accelerators with no published recipe are still listed because knowing an L4 has no
# recipe is what sends the caller to the JumpStart fallback rather than guessing at one
GPU_HARDWARE = {
    "H100": ("h100", "hopper"),
    "H200": ("h200", "hopper"),
    "B200": ("b200", "blackwell"),
    "B300": ("b300", "blackwell"),
    "GB200": ("gb200", "blackwell"),
    "GB300": ("gb300", "blackwell"),
    "L40S": ("l40s", "ada"),
    "RTX PRO SERVER 6000": ("rtx_pro_6000", "blackwell"),
    # No recipes published for these as of the corpus scan; the family is still right
    "RTX PRO 4500": (None, "blackwell"),
    "A100": (None, "ampere"),
    "A10G": (None, "ampere"),
    "L4": (None, "ada"),
    "V100": (None, "volta"),
    "T4": (None, "turing"),
    "T4G": (None, "turing"),
}

# How a recipe says it will not run on hardware it nonetheless mentions
UNSUPPORTED = "unsupported"

# Tokens the argv carries that are not flags
ARGV_SKIP = ("vllm", "serve")

# vLLM takes nested settings as --speculative-config.method, which the SageMaker
# container instead reads as one JSON object in SM_VLLM_SPECULATIVE_CONFIG
NESTED_FLAG_SEPARATOR = "."

# Prefix the vLLM SageMaker container reads its serving arguments from
ENV_PREFIX = "SM_VLLM_"

# Flags whose value names a file; a recipe written against a vLLM checkout can point
# one at something only that checkout has
# Instance families that carry a GPU, so looking for a sibling shape does not page
# through the whole EC2 catalogue
GPU_FAMILIES = (
    "g4dn.*",
    "g5.*",
    "g6.*",
    "g6e.*",
    "gr6.*",
    "g7.*",
    "g7e.*",
    "p3.*",
    "p4d.*",
    "p4de.*",
    "p5.*",
    "p5e.*",
    "p5en.*",
    "p6-*",
    "p6e-*",
)

# Everything the endpoint can read lives under here: the staged weights and the extra
# channels SageMaker mounts beside them
STAGED_ROOT = "/opt/ml"

# A parser named by a plugin only exists because the plugin defines it, so dropping an
# unreachable plugin has to drop the name that depended on it
PLUGIN_DEPENDENTS = {
    "reasoning_parser_plugin": "reasoning_parser",
    "tool_parser_plugin": "tool_call_parser",
}

PATH_VALUED_FLAGS = frozenset(
    {
        "chat_template",
        "tool_parser_plugin",
        "reasoning_parser_plugin",
        "generation_config",
        "logits_processors",
        "speculative_config",
    }
)

# Flags describing how to talk to a model rather than how to fit it on a card
PORTABLE_FLAGS = frozenset(
    {
        "tool_call_parser",
        "tool_parser_plugin",
        "enable_auto_tool_choice",
        "reasoning_parser",
        "reasoning_parser_plugin",
        "reasoning_config",
        "trust_remote_code",
        "chat_template",
        "chat_template_content_format",
        "default_chat_template_kwargs",
        "tokenizer_mode",
        "config_format",
        "generation_config",
        "served_model_name",
    }
)


def fetch(path):
    """
    A JSON document from the recipe site, or None if it publishes none at that path
    """
    try:
        with urllib.request.urlopen(f"{RECIPES_HOST}{path}", timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def catalog():
    """
    HuggingFace id to recipe path, for every recipe the site publishes
    """
    return {
        entry["hf_id"].lower(): entry["json"]
        for entry in fetch(CATALOG_PATH) or []
        if entry.get("hf_id") and entry.get("json")
    }


def gpu(instance_type, region=None):
    """
    The GPU behind an ml.* instance type: its name, how many, and memory
    """
    try:
        described = boto3.client("ec2", region_name=region).describe_instance_types(
            InstanceTypes=[instance_type.removeprefix("ml.")]
        )
    except ClientError:
        # An unknown type, or no ec2:DescribeInstanceTypes. Either way there is no
        # architecture by which to look up a recipe
        return None

    types = described.get("InstanceTypes") or []
    gpus = (types[0].get("GpuInfo", {}).get("Gpus") or []) if types else []
    if not gpus:
        return None

    return {
        "name": gpus[0]["Name"],
        "count": int(gpus[0].get("Count") or 1),
        "memory_mib": (gpus[0].get("MemoryInfo") or {}).get("SizeInMiB"),
    }


def instance_siblings(instance_type, region=None):
    """
    Other ml.* shapes carrying the same GPU in the same number, cheapest first

    These are the only safe capacity fallbacks: tensor parallel is pinned to the GPU
    count and the weights have to fit the same memory, so a shape that differs in
    either is not a substitute. Same GPU and same count leaves vCPU and host memory as
    the only difference, which the serving container does not care about

    Ordered by vCPU because that is what the extra shapes charge for
    """
    profile = gpu(instance_type, region)
    if profile is None:
        return []

    # Paged: the GPU families run past one page of results, and a short read looks
    # like a shape simply having no alternative
    try:
        pages = boto3.client("ec2", region_name=region).get_paginator(
            "describe_instance_types"
        ).paginate(Filters=[{"Name": "instance-type", "Values": list(GPU_FAMILIES)}])
        candidates = [
            entry for page in pages for entry in page.get("InstanceTypes") or []
        ]
    except ClientError:
        return []

    siblings = []
    for candidate in candidates:
        gpus = (candidate.get("GpuInfo") or {}).get("Gpus") or []
        if not gpus:
            continue
        if gpus[0]["Name"] != profile["name"]:
            continue
        if int(gpus[0].get("Count") or 1) != profile["count"]:
            continue
        name = f"ml.{candidate['InstanceType']}"
        if name != instance_type:
            siblings.append(
                (int((candidate.get("VCpuInfo") or {}).get("DefaultVCpus") or 0), name)
            )

    return [name for _, name in sorted(siblings)]


def architecture(instance_type, region=None):
    """
    What an ml.* instance looks like to recipes.vllm.ai
    """
    profile = gpu(instance_type, region)
    if profile is None:
        return None

    token, family = GPU_HARDWARE.get(profile["name"].upper(), (None, None))
    if token is None:
        return {"tokens": [], "family": family, **profile}

    tokens = (
        [token]
        if profile["count"] == 1
        else [
            f"{token}_{profile['count']}x",
            token,
        ]
    )

    return {"tokens": tokens, "family": family, **profile}


def select_hardware(document, tokens):
    """
    The first hardware token this recipe will actually run on, or None
    """
    published = dict(document.get("meta", {}).get("hardware") or {})

    # Recipes sometimes leave their own recommended hardware out of the published
    # list. For example, GLM-5.3-Flash is written for h200 but does not list it. Having authored the command against that GPU is the stronger claim of the two, so it counts
    # unless the list contradicts it outright.
    recommended = (document.get("recommended_command") or {}).get("hardware")
    if recommended and recommended not in published:
        published[recommended] = "recommended"

    for token in tokens:
        status = published.get(token)
        if status and status != UNSUPPORTED:
            return token, status

    return None, None


def feature_requests(features):
    """
    Each requested feature as (name, overrides).

    A bare name takes the recipe's own settings.
    """
    for entry in features or ():
        if isinstance(entry, str):
            yield entry, {}
            continue

        if isinstance(entry, dict) and len(entry) == 1:
            ((name, overrides),) = entry.items()
            yield name, dict(overrides or {})
            continue

        raise ValueError(
            f"vllm_features takes a feature name or a single-key mapping of one "
            f"to its overrides, got {entry!r}"
        )


def apply_overrides(name, args, overrides):
    """
    A feature's arguments with the config's overrides folded in.

    An override naming one of the feature's own flags replaces that flag's value.
    Anything else goes inside the feature's JSON-valued flag, which is where vLLM
    keeps the settings worth tuning.
    """
    args = list(args)
    nested_at = None

    for key, value in overrides.items():
        flag_names = {flag_key(arg): index for index, arg in enumerate(args)}
        index = flag_names.get(flag_key(str(key)))

        if index is not None and index + 1 < len(args):
            args[index + 1] = str(value)
            continue

        if nested_at is None:
            nested_at = next(
                (
                    position
                    for position, arg in enumerate(args)
                    if position and arg.startswith("{")
                ),
                None,
            )
        if nested_at is None:
            raise ValueError(
                f"the '{name}' feature has no '{key}' flag and nothing to nest it in, "
                f"only {[a for a in args if a.startswith('--')]}"
            )

        settings = json.loads(args[nested_at])
        settings[str(key)] = value
        args[nested_at] = json.dumps(settings)

    return args


def recipe_document(model_id):
    """
    The raw recipe document for a model id, or None if the site publishes none
    """
    path = catalog().get(model_id.lower())

    return fetch(path) if path else None


def recipe(model_id, instance_type, region=None, features=()):
    """
    The vLLM recipe for a model on the GPU an instance type carries, or None

    None means the site publishes nothing for the model at all, or the instance has no
    GPU to serve it on, and the caller has only the hub to fall back to.
    """
    path = catalog().get(model_id.lower())
    if path is None:
        return None

    arch = architecture(instance_type, region)
    if arch is None:
        return None

    document = fetch(path)
    if document is None:
        return None

    token, status = select_hardware(document, arch["tokens"])

    command = document.get("recommended_command") or {}
    argv = list(command.get("argv") or [])
    # The recommended command's environment belongs to the hardware it was recommended
    # for, and an opaque VLLM_* toggle gives no way to tell a frontend choice from a
    # kernel selection, so none of it travels
    env = dict(command.get("env") or {}) if token else {}

    available = document.get("features") or {}
    for name, overrides in feature_requests(features):
        if name not in available:
            raise ValueError(
                f"{model_id} publishes no '{name}' feature, only {sorted(available)}"
            )

        # Some recipes announce a feature with nothing behind it, and asking for one
        # of those should say so rather than appear to turn something on
        published_args = available[name].get("args")
        if not published_args:
            raise ValueError(
                f"{model_id} lists '{name}' but publishes no arguments for it, so "
                f"there is nothing to turn on. Set the settings you want in the "
                f"config's environment block instead"
            )

        argv += apply_overrides(name, list(published_args), overrides)

    # Overrides hang off the architecture generation, not the specific card
    override = (document.get("hardware_overrides") or {}).get(arch["family"]) or {}
    argv += list(override.get("extra_args") or [])
    env.update(override.get("extra_env") or {})

    return {
        "model_id": model_id,
        "hardware": token,
        "status": status,
        "portable_only": token is None,
        "architecture": arch,
        "argv": argv,
        "env": env,
        "variants": document.get("variants") or {},
        "available_features": sorted(available),
        "opt_in_features": list(document.get("opt_in_features") or []),
        "recommended_hardware": command.get("hardware"),
        "node_count": command.get("node_count"),
        "guide": document.get("guide"),
    }


def json_scalar(value):
    """
    A flag's value as the JSON type it means, not the string argv carried it as

    A nested config is handed over as JSON rather than as command-line text, and
    vLLM validates the types inside it (e.g. num_speculative_tokens is an int and
    "1" is rejected)
    """
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass

    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    return value


def feature_knobs(args):
    """
    What a feature exposes to tune: its own flags, and the keys inside its JSON flag
    """
    flags = [arg for arg in args if arg.startswith("--")]
    nested = next((arg for arg in args if arg.startswith("{")), None)
    keys = sorted(json.loads(nested)) if nested else []

    return [flag.removeprefix("--") for flag in flags] + keys


def print_recipe(model_id, instance_type, region=None):
    """
    What vLLM publishes for a model on an instance, and what it turns into here

    The counterpart to --jumpstart for the HuggingFace path
    """
    arch = architecture(instance_type, region)
    if arch is None:
        return print(
            f"{instance_type} carries no GPU this engine can look a recipe up by"
        )

    print(f"{instance_type}: {arch['name']} x{arch['count']}, {arch['family']}")
    print(
        f"  hardware tokens tried : {arch['tokens'] or 'none published for this GPU'}"
    )

    path = catalog().get(model_id.lower())
    if path is None:
        return print(f"\n{model_id} has no vLLM recipe; a deploy falls back to the hub")

    document = fetch(path) or {}
    token, status = select_hardware(document, arch["tokens"])
    published = document.get("meta", {}).get("hardware") or {}
    command = document.get("recommended_command") or {}

    print(f"\n{model_id}")
    print(f"  published for   : {published}")
    print(
        f"  written for     : {command.get('hardware')}, {command.get('node_count')} node(s)"
    )
    for name, variant in (document.get("variants") or {}).items():
        floor = variant.get("vram_minimum_gb")
        print(f"  variant {name:8}: {variant.get('precision')}, needs {floor} GB VRAM")

    if token is None:
        print(
            f"\n  MATCH: none. {arch['tokens']} is not in the published set, so only "
            f"the model-shaped flags carry over and sizing comes from the hub"
        )
    else:
        print(f"\n  MATCH: {token} ({status}) - the whole recipe applies")

    opt_in = set(document.get("opt_in_features") or [])
    print("\n  features (vllm_features):")
    for name, feature in sorted((document.get("features") or {}).items()):
        mark = "opt-in " if name in opt_in else "default"
        knobs = feature_knobs(feature.get("args") or [])
        detail = f"tune: {knobs}" if knobs else "announced but carries no arguments"
        print(f"    {name:26} {mark}  {detail}")

    selected = recipe(model_id, instance_type, region)
    print("\n  environment without any feature requested:")
    for key, value in sorted(serving_environment(selected).items()):
        print(f"    {key} = {value!r}")


def unusable_path(key, value):
    """
    Whether a flag points at a file the serving container will not have

    The only files the endpoint has are the ones we staged, so a path is usable only
    under /opt/ml - the weights prefix or a channel mount. Everything else is written
    for a checkout or a volume mount the container does not get: gemma-4 asks for
    examples/tool_chat_template_gemma4.jinja, Nemotron-3-Nano for a bare
    nano_v3_reasoning_parser.py that vLLM resolves to the filesystem root. Each fails
    validation at startup rather than being ignored

    A JSON object is a value, not a path, so speculative_config passes through
    """
    if key not in PATH_VALUED_FLAGS or not isinstance(value, str):
        return False
    if not value or value.startswith("{"):
        return False

    return not value.startswith(f"{STAGED_ROOT}/")


def flag_key(flag):
    """
    A flag reduced to one spelling
    """
    return flag.removeprefix("--").replace("-", "_").lower()


def env_name(flag):
    return ENV_PREFIX + flag_key(flag).upper()


def translate(argv, portable_only=False):
    """
    A vLLM argv as SM_VLLM_* environment variables, and nothing else

    Separate from serving_environment so a feature's arguments can be rendered on
    their own - the opt-in sets a config lists but does not apply
    """
    nested = {}
    environment = {}
    dropped = set()

    index = 0
    while index < len(argv):
        token = argv[index]

        if not token.startswith("--"):
            # The launcher and the positional model id, which carries no flag
            index += 1
            continue

        following = argv[index + 1] if index + 1 < len(argv) else None
        has_value = following is not None and not following.startswith("--")
        value = following if has_value else ""

        if NESTED_FLAG_SEPARATOR in token:
            group, _, key = token.removeprefix("--").partition(NESTED_FLAG_SEPARATOR)
            # A nested group is sizing or sharding in every recipe that uses one, so
            # it never survives into a portable set
            if not portable_only:
                nested.setdefault(group, {})[key] = json_scalar(value)
        elif unusable_path(flag_key(token), value):
            dropped.add(flag_key(token))
            print(f"Dropping {token} {value}: not a file the container has")
        elif not portable_only or flag_key(token) in PORTABLE_FLAGS:
            environment[env_name(token)] = value

        index += 2 if has_value else 1

    for group, settings in nested.items():
        environment[env_name(group)] = json.dumps(settings)

    # A parser the dropped plugin defined is not in the container either, and naming
    # it is a startup KeyError rather than a fallback to a built-in one
    for plugin, dependent in PLUGIN_DEPENDENTS.items():
        if plugin in dropped and environment.pop(env_name(dependent), None):
            print(f"Dropping --{dependent.replace('_', '-')} too: {plugin} supplied it")

    return environment


def serving_environment(selected, model_path="/opt/ml/model"):
    """
    A recipe's argv as the SM_VLLM_* environment the SageMaker container serves from
    """
    portable_only = selected.get("portable_only")
    environment = translate(selected["argv"], portable_only)

    if portable_only:
        return environment

    environment[f"{ENV_PREFIX}MODEL"] = model_path
    environment[f"{ENV_PREFIX}TENSOR_PARALLEL_SIZE"] = str(
        selected["architecture"]["count"]
    )

    return {**selected["env"], **environment}
