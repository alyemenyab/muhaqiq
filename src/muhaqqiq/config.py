"""Runtime configuration.

Every knob has a working default so that `muhaqqiq research "..."` runs with an
empty environment. Keys only unlock *better* behaviour (real models, live web),
they are never required to see the agent work end to end.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["offline", "openrouter"]
SearchProvider = Literal["offline", "tavily"]


class Settings(BaseSettings):
    """Configuration resolved from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- model provider ----------------------------------------------------
    llm_provider: LLMProvider = Field("offline", alias="MUHAQQIQ_LLM_PROVIDER")
    openrouter_api_key: str | None = Field(None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        "https://openrouter.ai/api/v1", alias="MUHAQQIQ_OPENROUTER_BASE_URL"
    )
    model: str = Field("openai/gpt-4o", alias="MUHAQQIQ_MODEL")
    fast_model: str = Field("openai/gpt-4o-mini", alias="MUHAQQIQ_FAST_MODEL")
    temperature: float = Field(0.2, alias="MUHAQQIQ_TEMPERATURE")
    request_timeout: float = Field(90.0, alias="MUHAQQIQ_REQUEST_TIMEOUT")

    # --- search provider ---------------------------------------------------
    search_provider: SearchProvider = Field("offline", alias="MUHAQQIQ_SEARCH_PROVIDER")
    tavily_api_key: str | None = Field(None, alias="TAVILY_API_KEY")
    corpus_dir: Path = Field(Path("data/corpus"), alias="MUHAQQIQ_CORPUS_DIR")

    # --- agent behaviour ---------------------------------------------------
    max_subquestions: int = Field(5, ge=1, le=12, alias="MUHAQQIQ_MAX_SUBQUESTIONS")
    max_sources_per_subquestion: int = Field(
        4, ge=1, le=15, alias="MUHAQQIQ_MAX_SOURCES_PER_SUBQUESTION"
    )
    max_research_rounds: int = Field(2, ge=1, le=5, alias="MUHAQQIQ_MAX_RESEARCH_ROUNDS")
    min_citation_coverage: float = Field(
        0.8, ge=0.0, le=1.0, alias="MUHAQQIQ_MIN_CITATION_COVERAGE"
    )
    researcher_concurrency: int = Field(4, ge=1, le=16, alias="MUHAQQIQ_RESEARCHER_CONCURRENCY")

    # --- MCP ---------------------------------------------------------------
    mcp_host: str = Field("127.0.0.1", alias="MUHAQQIQ_MCP_HOST")
    mcp_port: int = Field(8765, alias="MUHAQQIQ_MCP_PORT")
    mcp_url: str | None = Field(None, alias="MUHAQQIQ_MCP_URL")
    use_mcp: bool = Field(False, alias="MUHAQQIQ_USE_MCP")

    # --- storage -----------------------------------------------------------
    db_path: Path = Field(Path(".muhaqqiq/runs.db"), alias="MUHAQQIQ_DB_PATH")
    output_dir: Path = Field(Path("out"), alias="MUHAQQIQ_OUTPUT_DIR")
    skills_dir: Path = Field(Path("skills"), alias="MUHAQQIQ_SKILLS_DIR")

    # --- observability -----------------------------------------------------
    langsmith_tracing: bool = Field(False, alias="LANGSMITH_TRACING")
    langsmith_project: str = Field("muhaqqiq", alias="LANGSMITH_PROJECT")

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_llm(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"", "none", "mock", "local"}:
                return "offline"
        return value

    @field_validator("search_provider", mode="before")
    @classmethod
    def _normalise_search(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"", "none", "mock", "local", "corpus"}:
                return "offline"
        return value

    @property
    def effective_llm_provider(self) -> LLMProvider:
        """Fall back to offline if the selected provider has no credentials."""
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            return "offline"
        return self.llm_provider

    @property
    def effective_search_provider(self) -> SearchProvider:
        if self.search_provider == "tavily" and not self.tavily_api_key:
            return "offline"
        return self.search_provider

    @property
    def degraded_reasons(self) -> list[str]:
        """Human-readable explanation of any silent downgrade."""
        reasons: list[str] = []
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            reasons.append("OPENROUTER_API_KEY missing — using the offline reasoning provider")
        if self.search_provider == "tavily" and not self.tavily_api_key:
            reasons.append("TAVILY_API_KEY missing — searching the bundled local corpus")
        return reasons

    def resolved_mcp_url(self) -> str:
        return self.mcp_url or f"http://{self.mcp_host}:{self.mcp_port}/mcp"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests that mutate the environment."""
    get_settings.cache_clear()
