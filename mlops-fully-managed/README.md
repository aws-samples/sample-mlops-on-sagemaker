# MLOps on SageMaker AI with managed features

Here we'll focus on MLOps using SageMaker AI managed features such as SageMaker Processing, SageMaker Training, MLflow, SageMaker Model Registry, SageMaker Inference, and SageMaker Pipelines.

Note that observability will be a CloudFormation stack you deploy which will leverage serverless services that aren't built-in SageMaker AI features.

### Getting started

This repository uses uv to manage virtual environments. To get started, run:

```sh
uv venv .venv
uv sync
```

## Choose a use case

Navigate to the `./use_cases` folder to choose a use case, or create your own folder there. Then run the commands below from inside your use case folder (`use_cases/<use_case>`). Each one takes `--env` with the name of a YAML config file in the matching step folder, such as `process/`, `train/`, or `deploy/`.

### Data processing

Processing, training, and deployment jobs need a container image to run your code. An AWS managed framework image, such as the SageMaker scikit-learn or PyTorch containers, often works as is. If you need your own image, add a `docker/` folder to your use case with a Dockerfile and a `.env.docker.<name>` file (see `src/docker/` for examples), then build and push it to Amazon ECR:

```sh
bash ../../src/build.sh --env .env.docker.<name>
bash ../../src/push.sh --env .env.docker.<name>
```

This prints an image URI you can set as `image_uri` in the `process/process.x.yaml`, `train/train.x.yaml`, and `deploy/deploy.x.yaml` config files.

If your source code requires libraries not present in the Docker image, you can simply create a `requirements.txt` file and save it in the top level of your specified source code directory.

To start a processing job, create a `process/process.x.yaml` file, update its values, then run the following:

```sh
uv run python ../../src/process.py --env process.x.yaml
```

### Model training

Training jobs use the same kind of image as processing (see above).

When training models, it's often helpful to be able to do quick tests to debug your code. For models where your local training hardware suffices, you can do local training (you do need Docker locally installed):

```sh
uv run python ../../src/train.py --env train.x.yaml --local true
```

Once your training code is running without errors, you can scale up your training using SageMaker AI Training Jobs:

```sh
uv run python ../../src/train.py --env train.x.yaml
```

Notice that your model training code is separated from your docker image - a benefit to using the SageMaker Python SDK. This removes the headache of having to rebuild your Docker image any time you make a minor update to your training code. Additionally, if your source code requires libraries not present in the Docker image, you can simply create a `requirements.txt` file and save it in the top level of your specified source code directory.

If you specify a value for `MLFLOW_ARN` in your `train/train.x.yaml` file, you'll be provided with a pre-signed URL for the MLflow UI after your training job completes. Don't forget to actually update your training code to include MLflow API calls (see the training scripts in the use cases for examples).

To run on reserved capacity instead of on-demand, point `flexible_training_plan_arn` at a training plan:

```yaml
flexible_training_plan_arn: "arn:aws:sagemaker:us-west-2:<account>:training-plan/my-plan"
```

The job then draws from that reservation, which is how you get GPU capacity that would otherwise be hard to come by on demand. Leave the key out and the job runs on-demand as before.

Two things to watch. The `instance_type` and `instance_count` in your config have to match what the plan actually reserved, or `CreateTrainingJob` rejects the job. And the job only runs inside the plan's scheduled window, so submitting outside it leaves the job queued rather than starting it. `aws sagemaker list-training-plans` shows your plans, their status, and the windows they cover.

### Model registration

Once you're satisfied with your model training and have reviewed performance on validation and test datasets in the MLflow UI, you might consider registering the model in SageMaker AI Model Registry if you think it's a production candidate.

```sh
uv run python ../../src/register.py --env register.x.yaml
```

Model registration will pull in all your training job metadata and experiment data from MLflow tracking into a unified interface for your to compare different versions of models together and later promote them for deployment into QA and Production accounts.

### Model deployment

A `deploy/deploy.x.yaml` file describes one deployment. `deploy_type` picks the shape:

```yaml
deploy_type: real-time # real-time (default) or batch
```

`real-time` puts the model behind a persistent endpoint. `batch` runs a Batch Transform job over an S3 prefix and exits, which is the cheaper option when you don't need an endpoint sitting there between requests.

```sh
uv run python ../../src/deploy.py --env deploy.x.yaml
```

The model comes from `model_package_arn` if you registered one, or from `model_data_s3_uri` if you want to serve an artifact a training job.

Once it's `InService`:

```sh
uv run python ../../src/invoke.py --env deploy.x.yaml
```

Endpoints bill for their instances whether or not anything is calling them, so tear one down when you're done with it. This deletes the endpoint and the endpoint config it created, and leaves the model in place so a redeploy does not have to rebuild it:

```sh
uv run python ../../src/deploy.py --env deploy.x.yaml --delete
```

#### Finding and deploying a JumpStart model

Do a search:

```sh
uv run python ../../src/deploy.py --search-jumpstart "gpt oss"
```

Generate a deploy config:

```sh
uv run python ../../src/deploy.py --jumpstart openai-reasoning-gpt-oss-20b
```

#### Deploying a HuggingFace model

You can stage the HuggingFace model weights in S3 with a separate staging job (a SageMaker Processing job does this).

```yaml
huggingface:
  model_id: Qwen/Qwen3-8B
  # The commit to stage
  # curl -s https://huggingface.co/api/models/google/gemma-4-31b-it | jq -r .sha
  revision: b968826d9c46dd6066d109eabc6255188de91218
  stage_to_s3: "s3://your-bucket/hf-weights/"
```

```sh
uv run python ../../src/deploy.py --stage-only --env deploy.x.yaml
```

For gated repos, you need a HuggingFace token and only the staging job ever sees it:

```yaml
huggingface:
  model_id: meta-llama/Llama-3.1-8B-Instruct
  stage_to_s3: "s3://your-bucket/hf-weights/"
  secrets_arn: arn:aws:secretsmanager:us-west-2:111122223333:secret:hf-token-AbCdEf
  secret_key: HF_TOKEN
```

To generate a deploy config:

```sh
uv run python ../../src/deploy.py --vllm-recipe Qwen/Qwen3.6-27B-FP8 \
  --instance-type ml.g7e.2xlarge
```

This config defaults to a vLLM recipe if it exists, otherwise it defaults to a Jumpstart config.

Some containers want more than one artifact like a draft model for speculative decoding or a LoRA adapter:

```yaml
additional_model_data_sources:
  - channel_name: eagle
    s3_uri: "s3://.../lmi-eagle-heads/models/gpt-oss-20b-p-eagle/"
```

Each channel is mounted at `/opt/ml/additional-model-data-sources/<channel_name>`, which is the path you then reference from `environment`.

#### Reserved capacity

Fallbacks only help if something in the list has capacity. For instance types that on-demand effectively never supplies, you buy a training plan and point the endpoint at it:

```yaml
instance_preferences:
  - ml.p5.4xlarge
training_plan_arn: "arn:aws:sagemaker:us-west-2:<account>:training-plan/my-plan"
```

#### Request routing

By default SageMaker spreads requests at random. For an LLM endpoint that's usually the wrong choice:

```yaml
routing:
  strategy: PREFIX_AWARE
  prefix_length: 4096
  concurrency_threshold: 8
```

`PREFIX_AWARE` sends requests that share a prompt prefix to the same instance, so the model reuses its KV cache instead of recomputing the prefix.

#### Data capture

To record what an endpoint is asked and what it answers, add a `data_capture` block:

```yaml
data_capture:
  destination_s3_uri: "s3://your-bucket/data-capture/"
  sampling_percentage: 100
  capture_modes: [Input, Output]
```

SageMaker writes the sampled requests and responses as JSONL under that prefix, which is what `src/observability` reads. The execution role needs write access to the bucket. Set `enabled: false` to keep the block in your config with capture turned off.

### Workflow/pipeline orchestration

If you're using MLflow to tracking your model training, you'll need to first create a Docker image to run a Lambda function that will create/update a Model Card with MLflow data and associate it to your registered model (that way we see more metadata about the trained model in Model Registry).

```sh
bash ../../src/build.sh --env .env.docker.lambda_mlflow
bash ../../src/push.sh --env .env.docker.lambda_mlflow
```

To create the Lambda function, you'll need permissions. You can either update your SageMaker execution role:

```json
{
    "Effect": "Allow",
    "Principal": {
        "Service": "lambda.amazonaws.com"
    },
    "Action": "sts:AssumeRole"
}
```

Or create a separate role for the Lambda function and then update `orchestrate/orchestrate.yaml`:

```yaml
lambda_execution_role: "arn:aws:iam::<account>:role/service-role/LambdaModelCardRole"
```

Whatever role you choose, it'll also need permissions to read from MLflow Apps and/or Tracking Servers.

This orchestration module will pull step data from the `.yaml` environment files you've created for data processing, model training, and model registration in the DAG that you define in `orchestrate/orchestrate.yaml`:

```yaml
pipeline_definition:
  "process.x.yaml":
    - "train.x.yaml"
  "train.x.yaml":
    - "register.x.yaml"
```

Finally, to create the workflow/SageMaker AI Pipeline, run:

```sh
uv run python ../../src/orchestrate.py
```

or if you're appending a specific environment file name for orchestration:

```sh
uv run python ../../src/orchestrate.py --env orchestrate.x.yaml
```

### Observability

For observability, we'll be creating a few different images.

For deployed LLMs, we'll need two images: one will process request/response data captured in S3 in a distributed way and the other will do the evaluation on the aggregated data using MLflow.

```sh
bash ../../src/build.sh --env .env.docker.obs-evaluate
bash ../../src/push.sh --env .env.docker.obs-evaluate
bash ../../src/build.sh --env .env.docker.obs-llm
bash ../../src/push.sh --env .env.docker.obs-llm
```

For deployed predictive models, we'll just use one image that will both process request/response data captured in S3 and evaluate that data with evidently.

```sh
bash ../../src/build.sh --env .env.docker.obs-predictive
bash ../../src/push.sh --env .env.docker.obs-predictive
```

To set up observability for a deployed predictive model, run the following which will create a CloudFormation stack that'll use Step Functions for distributed processing of captured data files along with Evidently for drift detection:

`uv run python ../../src/observability/deploy.py --env observability.x.yaml`

### Secrets

Jobs that need a token, like a Hugging Face token to pull a gated model, should be read from AWS Secrets Manager at runtime. Create one secret holding a JSON object of every token your job needs:

```sh
aws secretsmanager create-secret \
  --name ml-e2e/tokens \
  --secret-string '{"HF_TOKEN": "hf_...", "OPENAI_API_KEY": "sk-..."}'
```

Then set its ARN in any `process/process.x.yaml`, `train/train.x.yaml`, or `deploy/deploy.x.yaml` config:

```yaml
secrets_arn: "arn:aws:secretsmanager:us-west-2:<account>:secret:ml-e2e/tokens-AbCdEf"
```

The engine passes only the ARN to the container, as the `SECRETS_ARN` environment variable. Your job code fetches the values itself:

```python
import boto3, json, os

blob = boto3.client("secretsmanager").get_secret_value(
    SecretId=os.environ["SECRETS_ARN"]
)
os.environ.update(json.loads(blob["SecretString"]))
```

Your SageMaker execution role will need read access to the secret:

```json
{
    "Effect": "Allow",
    "Action": "secretsmanager:GetSecretValue",
    "Resource": "arn:aws:secretsmanager:us-west-2:<account>:secret:ml-e2e/tokens-*"
}
```

If the secret is encrypted with a customer managed KMS key rather than the default `aws/secretsmanager` key, grant `kms:Decrypt` on that key as well.