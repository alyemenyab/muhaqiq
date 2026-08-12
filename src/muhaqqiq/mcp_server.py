"""FastMCP tool server.

Exposes the exact same `ToolRegistry` functions over the Model Context Protocol,
so the research tools can be consumed by Muhaqqiq, by another agent, or by any
MCP-capable client (an IDE, a desktop assistant) without importing this package.

Run it:

    uv run python -m muhaqqiq.mcp_server            # stdio (for IDE clients)
    uv run python -m muhaqqiq.mcp_server --http     # streamable HTTP on :8765

Point the agent at it:

    MUHAQQIQ_USE_MCP=true MUHAQQIQ_MCP_URL=http://127.0.0.1:8765/mcp \\
        uv run muhaqqiq research "..."
"""

from __future__ import annotations

import argparse
from typing import Any

from fastmcp import FastMCP

from .config import get_settings
from .store import RunStore
from .tools import ToolRegistry

mcp: FastMCP = FastMCP(
    name="muhaqqiq-research-tools",
    instructions=(
        "Retrieval tools for research agents. `web_search` returns ranked documents "
        "for a query; `fetch_document` returns one document in full; `corpus_stats` "
        "describes the offline corpus. In offline mode all results come from a "
        "bundled synthetic corpus and must not be treated as real-world facts."
    ),
)

_settings = get_settings()
_registry = ToolRegistry(settings=_settings, store=RunStore(_settings.db_path))


@mcp.tool
def web_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search for documents relevant to a query.

    Args:
        query: Keyword-style search query. Named entities work better than sentences.
        limit: Maximum number of documents to return (1-15).
    """
    return _registry.web_search(query, limit=max(1, min(limit, 15)))


@mcp.tool
def fetch_document(doc_id: str) -> dict[str, Any] | None:
    """Return the full text of a single document by its id or URL."""
    return _registry.fetch_document(doc_id)


@mcp.tool
def fetch_url(url: str, max_chars: int = 6000) -> dict[str, Any]:
    """Fetch a URL and return its extracted text content."""
    return _registry.fetch_url(url, max_chars=max_chars)


@mcp.tool
def corpus_stats() -> dict[str, Any]:
    """Describe the offline corpus: document count, publishers, and whether it is synthetic."""
    return _registry.corpus_stats()


def main() -> None:
    parser = argparse.ArgumentParser(description="Muhaqqiq MCP tool server")
    parser.add_argument("--http", action="store_true", help="serve over streamable HTTP")
    parser.add_argument("--host", default=_settings.mcp_host)
    parser.add_argument("--port", type=int, default=_settings.mcp_port)
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
