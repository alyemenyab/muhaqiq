.DEFAULT_GOAL := help
.PHONY: help install demo test lint fmt api mcp docker clean

Q ?= How do multi-agent orchestration patterns compare, and what are the main risks of deploying them?

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Create the environment and install the project
	uv sync --extra dev

demo: ## Run the agent end to end with no API keys (Q="your question")
	uv run muhaqqiq research "$(Q)" --trace

test: ## Run the test suite
	uv run pytest -q

lint: ## Lint
	uv run ruff check src tests

fmt: ## Auto-fix lint issues
	uv run ruff check src tests --fix

api: ## Start the HTTP API on :8000
	uv run muhaqqiq serve --reload

mcp: ## Start the MCP tool server on :8765
	uv run python -m muhaqqiq.mcp_server --http

docker: ## Build and run both services
	docker compose up --build

clean: ## Remove generated artefacts
	rm -rf out .muhaqqiq .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
