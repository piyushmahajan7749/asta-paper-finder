import os

from ai2i.chain import ModelFamily
from ai2i.config import config_value
from pydantic import SecretStr

from mabool.data_model.config import cfg_schema


def get_api_key_for_model(model_family: ModelFamily) -> SecretStr | None:
    api_key: str | None = None

    match model_family:
        case "openai":
            # Prefer Azure OpenAI key if present. We allow either:
            # - `AZURE_OPENAI_API_KEY` env var (recommended for Azure deployments)
            # - `OPENAI_API_KEY` (legacy; can be either OpenAI or Azure key depending on setup)
            api_key = os.getenv("AZURE_OPENAI_API_KEY") or config_value(cfg_schema.openai_api_key, default=None)
        case "anthropic":
            api_key = config_value(cfg_schema.anthropic_api_key, default=None)
        case "google":
            api_key = config_value(cfg_schema.google_api_key, default=None)
        case _:
            pass

    return SecretStr(api_key) if api_key else None
