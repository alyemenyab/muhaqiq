from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Every test gets a private database and an absolute path to the corpus.

    Without this, tests inherit whatever `.env` the developer happens to have and
    write into the real run store — which is exactly the class of hidden coupling
    the agent itself is built to avoid.
    """
    from muhaqqiq import config

    monkeypatch.setenv("MUHAQQIQ_LLM_PROVIDER", "offline")
    monkeypatch.setenv("MUHAQQIQ_SEARCH_PROVIDER", "offline")
    monkeypatch.setenv("MUHAQQIQ_CORPUS_DIR", str(REPO_ROOT / "data" / "corpus"))
    monkeypatch.setenv("MUHAQQIQ_SKILLS_DIR", str(REPO_ROOT / "skills"))
    monkeypatch.setenv("MUHAQQIQ_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("MUHAQQIQ_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("MUHAQQIQ_USE_MCP", "false")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    # `Settings` also reads a `.env` file; chdir keeps the repo's own out of scope.
    monkeypatch.chdir(tmp_path)
    config.reset_settings_cache()
    yield config.get_settings()
    config.reset_settings_cache()


@pytest.fixture
def corpus_dir() -> Path:
    return REPO_ROOT / "data" / "corpus"


@pytest.fixture
def skills_dir() -> Path:
    return REPO_ROOT / "skills"


@pytest.fixture(autouse=True)
def _quiet_env():
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
