"""pydantic-settings; env-driven knobs.

Only the settings extraction/client.py needs right now. api.py adds its own config
surface (CORS, rate limiting, etc.) when Phase 3 builds it - not duplicated here.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCINTEL_")

    # "mock" is the only backend the local machine is ever allowed to run (D-001) - it's
    # the default so a missing .env can't accidentally point local dev at a live GPU endpoint.
    backend: Literal["mock", "http"] = "mock"
    vllm_base_url: str = "http://localhost:8000"
    vllm_api_key: str | None = None
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"


settings = Settings()
