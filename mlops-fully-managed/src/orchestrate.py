import argparse
import json
import re

import mlflow
import sagemaker_mlflow  # noqa: F401
import yaml
from mlflow.tracking import MlflowClient
from sagemaker.mlops.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TransformStep,
    # CacheConfig,
)
from sagemaker.mlops.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.fail_step import FailStep
from sagemaker.core.lambda_helper import Lambda
import boto3
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.core.workflow.conditions import (
    ConditionEquals,
    ConditionGreaterThan,
    ConditionGreaterThanOrEqualTo,
    ConditionLessThan,
    ConditionLessThanOrEqualTo,
)
from sagemaker.core.workflow.functions import JsonGet, Join
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.properties import PropertyFile

from process import (
    METADATA_FILE_NAME,
    METADATA_OUTPUT_NAME,
    create_processor,
    run_processor,
)
from train import make_model_trainer, run_model_trainer
from register import make_model_builder, register_model
from deploy import make_batch_transform, run_transformer

# The pipeline parameter and the container environment variable that carry the
# MLflow run a step should nest its own run under
MLFLOW_PARENT_RUN_ID_PARAM = "MlflowParentRunId"
MLFLOW_PARENT_RUN_ID_ENV = "MLFLOW_PARENT_RUN_ID"

# Studio and SageMaker Unified Studio both write the running space's metadata here;
# a local IDE has no such file
SPACE_METADATA_PATH = "/opt/ml/metadata/resource-metadata.json"

# How a gate config's `comparison` reads: the gate passes when <metric> <comparison> <threshold>
GATE_COMPARISONS = {
    "less_than": ConditionLessThan,
    "less_than_or_equal": ConditionLessThanOrEqualTo,
    "greater_than": ConditionGreaterThan,
    "greater_than_or_equal": ConditionGreaterThanOrEqualTo,
    "equal": ConditionEquals,
}

# Tags SMUS scopes its resource lists by, and the AdditionalMetadata fields that
# carry their values
SMUS_TAG_SOURCES = {
    "AmazonDataZoneDomain": "DataZoneDomainId",
    "AmazonDataZoneProject": "DataZoneProjectId",
    "AmazonDataZoneScopeName": "DataZoneScopeName",
    "AmazonDataZoneUser": "DataZoneUserId",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline launcher")
    parser.add_argument(
        "--env",
        default="orchestrate.yaml",
        help="YAML config file name in orchestrate/ (default: orchestrate.yaml)",
    )
    return parser.parse_args()


def model_card_lambda_name(model_name):
    return f"create-update-model-card-{re.sub(r'[^A-Za-z0-9_-]', '-', model_name)}"[:64]


def create_update_model_card_lambda(
    sagemaker_execution_role,
    lambda_execution_role,
    image_uri,
    mlflow_arn,
    mlflow_experiment_name,
    model_name,
    model_description,
    pipeline_session,
):
    lambda_client = boto3.client(
        "lambda", region_name=pipeline_session.boto_region_name
    )
    function_name = model_card_lambda_name(model_name)
    environment_vars = {
        "MLFLOW_ARN": mlflow_arn,
        "MLFLOW_EXPERIMENT_NAME": mlflow_experiment_name,
        "MODEL_NAME": model_name,
        "MODEL_DESCRIPTION": model_description,
    }
    environment = {"Variables": environment_vars}

    try:
        response = lambda_client.get_function(FunctionName=function_name)
        lambda_client.update_function_code(
            FunctionName=function_name,
            ImageUri=image_uri,
        )
        lambda_client.get_waiter("function_updated_v2").wait(FunctionName=function_name)
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Role=lambda_execution_role,
            MemorySize=10240,
            Timeout=900,
            Environment=environment,
        )
        function_arn = response["Configuration"]["FunctionArn"]
    except lambda_client.exceptions.ResourceNotFoundException:
        response = lambda_client.create_function(
            FunctionName=function_name,
            PackageType="Image",
            Code={"ImageUri": image_uri},
            Role=lambda_execution_role,
            MemorySize=10240,
            Timeout=900,
            Environment=environment,
        )
        function_arn = response["FunctionArn"]
    lambda_client.get_waiter("function_updated_v2").wait(FunctionName=function_name)
    return Lambda(
        function_arn=function_arn,
        script=None,  # We have a Docker image instead
        zipped_code_dir=None,  # We have a Docker image instead
        environment=environment_vars,
        execution_role_arn=lambda_execution_role,
        memory_size=10240,
        timeout=900,
    )


# ---------------------------------------------------------------------------
# SageMaker Unified Studio
# ---------------------------------------------------------------------------


def smus_project_tags(metadata_path=SPACE_METADATA_PATH):
    """
    The project tags a pipeline needs to belong to a SMUS project, or {} elsewhere
    """
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    additional_metadata = metadata.get("AdditionalMetadata") or {}
    if not additional_metadata.get("DataZoneDomainId"):
        return {}

    return {
        tag: additional_metadata[field]
        for tag, field in SMUS_TAG_SOURCES.items()
        if additional_metadata.get(field)
    }


# ---------------------------------------------------------------------------
# MLflow run nesting
# ---------------------------------------------------------------------------


def mlflow_tracking_target(node_configs):
    """
    The one MLflow server and experiment the pipeline's steps log to, or None
    """
    # Register nodes carry mlflow_arn only to read runs back for the Model Card
    targets = {
        (config["mlflow_arn"], config.get("mlflow_experiment_name", "Default"))
        for node, config in node_configs.items()
        if config.get("mlflow_arn") and not node.startswith("register")
    }
    if len(targets) > 1:
        raise ValueError(
            "the pipeline's steps log to more than one MLflow experiment, so their runs "
            f"cannot nest under a single parent: {sorted(targets)}"
        )
    return targets.pop() if targets else None


def attach_parent_run(job, env_attribute, node_config, parent_run_id):
    """
    Tell a job's container which MLflow run to nest its own run under
    """
    if parent_run_id is None or not node_config.get("mlflow_arn"):
        return
    setattr(
        job,
        env_attribute,
        {
            **(getattr(job, env_attribute) or {}),
            MLFLOW_PARENT_RUN_ID_ENV: parent_run_id,
        },
    )


def start_parent_run(tracking_target, pipeline_name):
    mlflow_arn, experiment_name = tracking_target
    mlflow.set_tracking_uri(mlflow_arn)
    mlflow.set_experiment(experiment_name)

    run = mlflow.start_run(run_name=f"{pipeline_name}-starting")
    mlflow.set_tags({"pipeline_name": pipeline_name, "step": "pipeline"})
    return run.info.run_id


def finish_parent_run(parent_run_id, execution_arn):
    """
    Name the parent run after the execution it groups, then close it

    The execution id does not exist until the pipeline has started, so the run is
    opened under a placeholder and renamed here
    """
    execution_id = execution_arn.rsplit("/", 1)[-1]
    mlflow.set_tag("pipeline_execution_arn", execution_arn)
    mlflow.end_run()
    MlflowClient().update_run(parent_run_id, name=execution_id)
    return execution_id


# ---------------------------------------------------------------------------
# DAG shape
# ---------------------------------------------------------------------------


def build_edges(pipeline_definition):
    """
    children and parents for every node named in the DAG, including the leaves
    """
    children = {node: list(deps or []) for node, deps in pipeline_definition.items()}
    parents = {node: [] for node in children}
    for node, kids in list(children.items()):
        for kid in kids:
            parents.setdefault(kid, []).append(node)
            children.setdefault(kid, [])
    return children, parents


def topo_order(children, parents):
    """
    Parents before children (Kahn, keeping yaml order among ready nodes)
    """
    pending = {node: len(node_parents) for node, node_parents in parents.items()}
    ready = [node for node, count in pending.items() if count == 0]
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for kid in children[node]:
            pending[kid] -= 1
            if pending[kid] == 0:
                ready.append(kid)
    if len(order) != len(pending):
        raise ValueError(
            "pipeline_definition is not a DAG; these nodes are in a cycle: "
            f"{sorted(set(pending) - set(order))}"
        )
    return order


def train_node_for(node, parents, orchestrate_env_config):
    """
    The training node from which a register node takes its model artifact, image and experiment

    `model_card_training_step` is only the fallback, for a register node whose parent is not a
    training step (like an eval step sitting between the two)
    """
    train_parents = [
        parent for parent in parents.get(node, []) if parent.startswith("train")
    ]
    if len(train_parents) > 1:
        raise ValueError(
            f"{node} has more than one training step as a parent ({train_parents}), so which "
            "model it registers is ambiguous; give each training step its own register node"
        )
    if train_parents:
        return train_parents[0]
    return orchestrate_env_config.get("model_card_training_step")


def node_stem(node, prefix):
    """`register.yaml` -> ``, `register.scikit-learn-model.yaml` -> `scikit-learn-model`."""
    return node.removesuffix(".yaml").removeprefix(prefix).strip(".")


def register_step_name(node):
    """`register.yaml` -> `register-model`; `register.xgboost.yaml` -> `register-model-xgboost`.

    Unique per register node; a DAG with two of them would otherwise define two steps under one
    name and fail at upsert
    """
    stem = node_stem(node, "register")
    if not stem:
        return "register-model"
    return f"register-model-{stem.replace('.', '-')}"


def model_card_step_name(node):
    """The card LambdaStep paired with a given register node, named the same way"""
    stem = node_stem(node, "register")
    if not stem:
        return "create_update_model_card"
    return f"create_update_model_card_{stem.replace('.', '_')}"


def batch_model_step_name(node):
    """The CreateModel ModelStep paired with a given deploy node"""
    stem = node_stem(node, "deploy")
    if not stem:
        return "create-batch-model"
    return f"create-batch-model-{stem.replace('.', '-')}"


def metadata_property_file(node, node_config):
    """
    How a later step reads the JSON a process node published on its metadata channel
    """
    if not node_config.get("metadata_output_s3_path"):
        return []
    stem = node_stem(node, "process").replace(".", "_")
    return [
        PropertyFile(
            name=f"{stem or 'process'}_metadata",
            output_name=METADATA_OUTPUT_NAME,
            path=METADATA_FILE_NAME,
        )
    ]


def training_job_source(node, train_node, parents, steps_dict, node_configs):
    """
    The training job a register node's model came from and the MLflow experiment holding its run
    """
    if train_node is not None:
        if train_node not in steps_dict:
            raise ValueError(
                f"the training step for {node} resolved to {train_node}, which is not a node in "
                f"pipeline_definition; either give {node} a train.* parent or point "
                f"model_card_training_step at one of {sorted(steps_dict)}"
            )
        return (
            steps_dict[train_node].properties.TrainingJobName,
            node_configs[train_node]["mlflow_experiment_name"],
        )
    for parent in parents.get(node, []):
        parent_step = steps_dict.get(parent)
        property_files = getattr(parent_step, "property_files", None)
        if property_files:
            training_job_name = JsonGet(
                step_name=parent_step.name,
                property_file=property_files[0],
                json_path="$.training_job_name",
            )
            return (
                training_job_name,
                node_configs[parent].get("mlflow_experiment_name", "Default"),
            )
    return None, None


def upstream_model_package_arn(node, parents, steps_dict):
    """
    The package a register parent creates for a process node that evaluates it
    """
    register_parents = [
        parent for parent in parents.get(node, []) if parent.startswith("register")
    ]
    if len(register_parents) > 1:
        raise ValueError(
            f"{node} has more than one register step as a parent ({register_parents}), "
            "so which model package it reads is ambiguous"
        )
    if register_parents:
        return steps_dict[register_parents[0]].properties.ModelPackageArn
    return None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def process_node(
    node,
    pipeline_session,
    steps_dict=None,
    train_node=None,
    training_job_name=None,
    model_package_arn=None,
    parent_run_id=None,
    depends_on=None,
):
    """One DAG node -> one pipeline step, plus that node's config for the caller that needs it"""
    steps_dict = {} if steps_dict is None else steps_dict
    depends_on = depends_on or None
    node_config = {}

    if node.startswith("process"):
        processor, processor_args, node_config = create_processor(env=node)
        processor.sagemaker_session = pipeline_session
        attach_parent_run(processor, "env", node_config, parent_run_id)
        if model_package_arn is not None:
            # Resolved at execution time so the script can pull the model it evaluates
            processor_args["arguments"] = [
                *(processor_args["arguments"] or []),
                "--model_package_arn",
                model_package_arn,
            ]
        step_args = run_processor(processor, processor_args)
        step = ProcessingStep(
            name=processor.base_job_name,
            display_name=processor.base_job_name,
            step_args=step_args,
            property_files=metadata_property_file(node, node_config),
            depends_on=depends_on,
            # cache_config=cache_config,
        )
    elif node.startswith("train"):
        model_trainer, input_config, node_config = make_model_trainer(env=node)
        model_trainer.sagemaker_session = pipeline_session
        attach_parent_run(model_trainer, "environment", node_config, parent_run_id)
        step_args = run_model_trainer(model_trainer, input_config, node_config)
        step = TrainingStep(
            name=model_trainer.base_job_name,
            display_name=model_trainer.base_job_name,
            step_args=step_args,
            depends_on=depends_on,
            # cache_config=cache_config,
        )
    elif node.startswith("deploy"):
        model_args, transformer, transform_args, node_config = make_batch_transform(
            env=node, session=pipeline_session
        )
        # CreateModel validates that the artifact exists in S3, so the model is
        # built by its own step after whatever stages it rather than now, and the
        # transform job reads the name off that step
        model_step = ModelStep(
            name=batch_model_step_name(node),
            display_name=batch_model_step_name(node),
            step_args=model_args,
            depends_on=depends_on,
        )
        steps_dict[batch_model_step_name(node)] = model_step
        transformer.model_name = model_step.properties.ModelName
        step_args = run_transformer(transformer, transform_args, node_config)
        step = TransformStep(
            name=node_config["model_name"],
            display_name=node_config["model_name"],
            step_args=step_args,
            depends_on=[model_step],
            # cache_config=cache_config,
        )
    elif node.startswith("register"):
        train_step = steps_dict.get(train_node) if train_node else None
        from_step = {}
        if train_step is not None:
            # Without this, register.py describes the `training_job_name` in the yaml - a job that
            # does not exist yet when the pipeline is defined. With it, the artifact and the image
            # are read off the training step instead.
            from_step = {
                "train_step": train_step,
                "train_step_image_uri": train_step.step_args.func_args[
                    0
                ].training_image,
            }
        model_builder, model_card, node_config, model_metrics = make_model_builder(
            env=node,
            skip_model_card=True,
            pipeline_session=pipeline_session,
            **from_step,
        )
        step_args = register_model(
            model_builder,
            model_card,
            node_config,
            model_metrics,
            customer_metadata=(
                {"training_job_name": training_job_name}
                if training_job_name is not None
                else None
            ),
        )
        step = ModelStep(
            name=register_step_name(node),
            display_name=register_step_name(node),
            step_args=step_args,
            depends_on=depends_on,
            # cache_config=cache_config,
        )
    else:
        raise ValueError(f"Unknown step type for env file: {node}")

    return step, node_config


def evaluation_node_for(node, children, node_configs):
    """
    The process child of a register node that scores its package and publishes the result

    Recognized by its metadata channel which is where the "Model Card Lambda" reads the evaluation from
    """
    evaluation_nodes = [
        child
        for child in children.get(node, [])
        if child.startswith("process")
        and node_configs[child].get("metadata_output_s3_path")
    ]
    if len(evaluation_nodes) > 1:
        raise ValueError(
            f"{node} has more than one evaluation step as a child ({evaluation_nodes}), so "
            "which one goes on the model card is ambiguous"
        )
    return evaluation_nodes[0] if evaluation_nodes else None


# ---------------------------------------------------------------------------
# Evaluation gates
# ---------------------------------------------------------------------------


def descendants_of(node, children):
    found = []
    pending = list(children.get(node, []))
    while pending:
        child = pending.pop(0)
        if child not in found:
            found.append(child)
            pending.extend(children.get(child, []))
    return found


def gate_node(node, parents, steps_dict):
    """
    The condition, and the step to run when it fails, for one gate.* node

    The metric is read from the evaluation data its parent published in metadata.json. A failure
    fails the pipeline
    """
    with open(f"gate/{node}") as f:
        gate_config = yaml.safe_load(f)

    evaluation_parents = [
        parent
        for parent in parents.get(node, [])
        if getattr(steps_dict.get(parent), "property_files", None)
    ]
    if len(evaluation_parents) != 1:
        raise ValueError(
            f"{node} needs exactly one parent that publishes metadata.json, like an evaluation "
            f"step; found {evaluation_parents}"
        )
    parent_step = steps_dict[evaluation_parents[0]]
    property_file = parent_step.property_files[0]

    metric = gate_config["metric"]
    comparison = gate_config["comparison"]
    if comparison not in GATE_COMPARISONS:
        raise ValueError(
            f"{node}: comparison must be one of {sorted(GATE_COMPARISONS)}, got '{comparison}'"
        )
    threshold = float(gate_config["threshold"])

    metric_value = JsonGet(
        step_name=parent_step.name,
        property_file=property_file,
        json_path=f"$.evaluation.metrics.{metric}",
    )
    condition = GATE_COMPARISONS[comparison](left=metric_value, right=threshold)
    reason = Join(
        on=" ",
        values=[
            f"{metric} of",
            metric_value,
            f"failed the gate, which requires {comparison} {threshold}",
        ],
    )

    fail_step = FailStep(
        name=f"{gate_config['step_name']}-failed",
        display_name=f"{gate_config['step_name']}-failed",
        error_message=reason,
    )

    return gate_config, condition, [fail_step]


def patch_pipeline_for_model_card(
    steps_dict,
    node,
    register_step,
    training_job_name,
    mlflow_experiment_name,
    orchestrate_env_config,
    register_env_config,
    pipeline_session,
    evaluation_node=None,
    evaluation_config=None,
):
    lambda_function = create_update_model_card_lambda(
        orchestrate_env_config["sagemaker_execution_role"],
        orchestrate_env_config["lambda_execution_role"],
        orchestrate_env_config["lambda_mlflow_image_uri"],
        register_env_config["mlflow_arn"],
        mlflow_experiment_name,
        register_env_config["model_name"],
        register_env_config["model_description"],
        pipeline_session,
    )

    inputs = {
        "training_job_name": training_job_name,
        "model_package_arn": register_step.properties.ModelPackageArn,
    }
    depends_on = [register_step]
    if evaluation_node is not None:
        # Wait for the evaluation so the card carries its results
        inputs["evaluation_metadata_s3_uri"] = (
            f"{evaluation_config['metadata_output_s3_path'].rstrip('/')}/{METADATA_FILE_NAME}"
        )
        depends_on.append(steps_dict[evaluation_node])

    # Create LambdaStep to create/update model card and associate it with the model package
    lambda_step = LambdaStep(
        name=model_card_step_name(node),
        display_name=model_card_step_name(node),
        lambda_func=lambda_function,
        inputs=inputs,
        outputs=[LambdaOutput("model_card_name", LambdaOutputTypeEnum.String)],
        depends_on=depends_on,
    )
    steps_dict[model_card_step_name(node)] = lambda_step
    return steps_dict


def run_pipeline(env=False):
    args = parse_args()
    if env:
        args.env = env

    with open(f"orchestrate/{args.env}") as f:
        orchestrate_env_config = yaml.safe_load(f)

    pipeline_definition = orchestrate_env_config["pipeline_definition"]
    children, parents = build_edges(pipeline_definition)

    # Cache Pipeline steps to reduce execution time on subsequent executions
    # cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    pipeline_session = PipelineSession()

    # A parameter rather than a value baked into the definition, so the pipeline can
    # be upserted once and every execution still gets its own parent run
    parent_run_id_param = ParameterString(
        name=MLFLOW_PARENT_RUN_ID_PARAM, default_value=""
    )

    # Build every node parents-first, so a step that reads an upstream step's properties can.
    steps_dict = {}
    node_configs = {}
    register_steps = []
    gates = {}
    # Which DAG node each step came from, so a gate can claim every step downstream of it
    step_owners = {}
    for node in topo_order(children, parents):
        if node.startswith("gate"):
            gate_config, condition, else_steps = gate_node(node, parents, steps_dict)
            gates[node] = (gate_config, condition, else_steps)
            node_configs[node] = gate_config
            continue

        train_node = (
            train_node_for(node, parents, orchestrate_env_config)
            if node.startswith("register")
            else None
        )
        training_job_name, mlflow_experiment_name = (
            training_job_source(node, train_node, parents, steps_dict, node_configs)
            if node.startswith("register")
            else (None, None)
        )
        step, node_config = process_node(
            node,
            pipeline_session,
            steps_dict=steps_dict,
            train_node=train_node,
            training_job_name=training_job_name,
            model_package_arn=(
                upstream_model_package_arn(node, parents, steps_dict)
                if node.startswith("process")
                else None
            ),
            parent_run_id=parent_run_id_param,
            # A gate parent is not a step yet; its children run inside its if branch instead
            depends_on=[
                steps_dict[parent]
                for parent in parents[node]
                if not parent.startswith("gate")
            ],
        )
        steps_dict[node] = step
        step_owners.update(
            {name: node for name in steps_dict if name not in step_owners}
        )
        node_configs[node] = node_config
        if node.startswith("register"):
            register_steps.append(
                (node, step, node_config, training_job_name, mlflow_experiment_name)
            )

    # Attach a LambdaStep after registration to create/update the Model Card and associate it with
    # the new model package version
    for (
        node,
        register_step,
        register_env_config,
        training_job_name,
        mlflow_experiment_name,
    ) in register_steps:
        if (
            register_env_config.get("mlflow_arn")
            and orchestrate_env_config.get("lambda_mlflow_image_uri")
            and training_job_name is not None
        ):
            evaluation_node = evaluation_node_for(node, children, node_configs)
            steps_dict = patch_pipeline_for_model_card(
                steps_dict,
                node,
                register_step,
                training_job_name,
                mlflow_experiment_name,
                orchestrate_env_config,
                register_env_config,
                pipeline_session,
                evaluation_node=evaluation_node,
                evaluation_config=node_configs.get(evaluation_node),
            )
            step_owners[model_card_step_name(node)] = node

    # A gate's downstream steps move into its if branch, which is the only place they may be
    # listed, so they run only when the condition passes
    for node, (gate_config, condition, else_steps) in gates.items():
        downstream = descendants_of(node, children)
        nested = [child for child in downstream if child.startswith("gate")]
        if nested:
            raise ValueError(
                f"{node} has gates downstream of it ({nested}); nest one gate at most"
            )
        if_step_names = [
            name for name, owner in step_owners.items() if owner in downstream
        ]
        evaluation_parent = next(
            parent for parent in parents[node] if parent in steps_dict
        )
        steps_dict[node] = ConditionStep(
            name=gate_config["step_name"],
            display_name=gate_config["step_name"],
            conditions=[condition],
            if_steps=[steps_dict.pop(name) for name in if_step_names],
            else_steps=else_steps,
            depends_on=[steps_dict[evaluation_parent]],
        )

    # Create pipeline/DAG
    pipeline = Pipeline(
        name=orchestrate_env_config["pipeline_name"],
        parameters=[parent_run_id_param],
        steps=list(steps_dict.values()),
        sagemaker_session=pipeline_session,
    )
    project_tags = smus_project_tags()
    if project_tags:
        print(f"SMUS project {project_tags['AmazonDataZoneProject']}; tagging pipeline")

    pipeline.upsert(
        role_arn=orchestrate_env_config["sagemaker_execution_role"],
        description=orchestrate_env_config["pipeline_description"],
        tags=project_tags or None,
    )

    tracking_target = mlflow_tracking_target(node_configs)
    if tracking_target is None:
        return pipeline.start()

    # Open the parent run first so its id can be passed in, then name it after the
    # pipeline execution it started
    parent_run_id = start_parent_run(
        tracking_target, orchestrate_env_config["pipeline_name"]
    )
    try:
        execution = pipeline.start(
            parameters={MLFLOW_PARENT_RUN_ID_PARAM: parent_run_id}
        )
    except Exception:
        mlflow.end_run(status="FAILED")
        raise

    execution_id = finish_parent_run(parent_run_id, execution.arn)
    print(f"\nExecution {execution_id}")
    print(f"Step runs nest under MLflow run {parent_run_id}")

    return execution


if __name__ == "__main__":
    run_pipeline()
