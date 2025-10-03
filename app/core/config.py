from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

from pydantic import Field, FieldValidationInfo, field_validator
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Central application configuration loaded from env vars or .env."""

    app_name: str = Field(default="Knowledge Graph SME Workbench")
    database_path: Path = Field(default_factory=lambda: Path.cwd() / "data" / "knowledge_graph.db")
    llm_endpoint: Optional[str] = Field(default=None, alias="LLM_ENDPOINT")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")
    embedding_endpoint: Optional[str] = Field(default=None, alias="EMBEDDING_ENDPOINT")
    embedding_api_key: Optional[str] = Field(default=None, alias="EMBEDDING_API_KEY")
    embedding_deployment: Optional[str] = Field(default=None, alias="EMBEDDING_DEPLOYMENT")
    model_source: str = Field(default="openai", alias="MODEL_SOURCE")
    log_level: str = Field(default="INFO")
    llm_temperature_extraction: float = Field(default=0.1, alias="LLM_TEMPERATURE_EXTRACTION")
    llm_temperature_retrieval: float = Field(default=0.0, alias="LLM_TEMPERATURE_RETRIEVAL")
    llm_temperature_plan: float = Field(default=0.0, alias="LLM_TEMPERATURE_PLAN")
    llm_temperature_answer: float = Field(default=0.3, alias="LLM_TEMPERATURE_ANSWER")
    llm_temperature_connections: float = Field(default=0.0, alias="LLM_TEMPERATURE_CONNECTIONS")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
    }

    @property
    def database_url(self) -> str:
        # SQLAlchemy expects POSIX-style paths; ensure parent directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.database_path.as_posix()}"


@functools.lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


settings = get_settings()
