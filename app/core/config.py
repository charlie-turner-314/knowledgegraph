from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Central application configuration loaded from env vars or .env."""

    app_name: str = Field(default="Knowledge Graph SME Workbench")
    database_path: Path = Field(default_factory=lambda: Path.cwd() / "data" / "knowledge_graph.db")
    gemma_endpoint: Optional[str] = Field(default=None, alias="GEMMA_ENDPOINT")
    gemma_api_key: Optional[str] = Field(default=None, alias="GEMMA_API_KEY")
    log_level: str = Field(default="INFO")

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
