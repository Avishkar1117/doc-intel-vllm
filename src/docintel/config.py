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

    # Phase 7: public endpoint protection + storage wiring (PROJECT_BRIEF.md §5/§7, D-003).
    # extract_api_key unset means the check is a no-op - local/mock/CI runs never set this
    # env var, and the live deployment sets it via a Key Vault secret reference, never a
    # literal value anywhere in config.
    extract_api_key: str | None = None
    rate_limit_per_minute: int = 20
    storage_account_name: str | None = None
    blob_container: str = "receipts"


settings = Settings()
