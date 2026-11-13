# The container environment variable that carries the secret a job should read
SECRETS_ARN_ENV = "SECRETS_ARN"


def secrets_env(env_config):
    """
    The Secrets Manager ARN a job's container needs, as an environment map

    Only the ARN travels with the job. The container calls GetSecretValue itself at
    runtime, so the tokens never land in the job or pipeline definition - which is
    where they ended up when the engine resolved them before handing them over
    """
    secrets_arn = env_config.get("secrets_arn")
    if not secrets_arn:
        return {}
    return {SECRETS_ARN_ENV: str(secrets_arn)}
