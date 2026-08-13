"""MCP-backed tool registry.

Drop-in replacement for `ToolRegistry` that calls the tools over the Model
Context Protocol instead of in-process. The graph does not know or care which
one it was given — that is the point of keeping the tool surface narrow.

Enabled with `MUHAQQIQ_USE_MCP=true`. If the server is unreachable the factory
falls back to the in-process registry and records the reason, because a research
run should not die because a sidecar is down.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, get_settings
from .store import RunStore
from .tools import ToolRegistry

log = logging.getLogger("muhaqqiq.mcp_client")


@dataclass
class MCPToolRegistry:
    """Same surface as `ToolRegistry`, backed by a remote MCP server."""

    url: str
    settings: Settings = field(default_factory=get_settings)
    calls: int = 0
    call_log: list[dict[str, Any]] = field(default_factory=list)
    transport: str = "mcp"

    def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        from fastmcp import Client  # imported lazily; only needed on this path

        async def run() -> Any:
            async with Client(self.url) as client:
                result = await client.call_tool(name, arguments)
                return _unwrap(result)

        self.calls += 1
        self.call_log.append({"tool": name, "transport": "mcp", **arguments})
        return asyncio.run(run())

    def web_search(self, query: str, limit: int = 5, context: str = "") -> list[dict[str, Any]]:
        result = self._call("web_search", {"query": query, "limit": limit, "context": context})
        return result if isinstance(result, list) else []

    def fetch_document(self, doc_id: str) -> dict[str, Any] | None:
        result = self._call("fetch_document", {"doc_id": doc_id})
        return result if isinstance(result, dict) else None

    def fetch_url(self, url: str, max_chars: int = 6000) -> dict[str, Any]:
        result = self._call("fetch_url", {"url": url, "max_chars": max_chars})
        return result if isinstance(result, dict) else {"url": url, "error": "bad response"}

    def corpus_stats(self) -> dict[str, Any]:
        result = self._call("corpus_stats", {})
        return result if isinstance(result, dict) else {}

    def specs(self) -> list[dict[str, str]]:
        return [
            {"name": "web_search", "description": "Search via MCP server."},
            {"name": "fetch_document", "description": "Fetch a document via MCP server."},
            {"name": "fetch_url", "description": "Fetch a URL via MCP server."},
            {"name": "corpus_stats", "description": "Corpus stats via MCP server."},
        ]


def _unwrap(result: Any) -> Any:
    """Normalise a FastMCP call result into plain Python."""
    data = getattr(result, "data", None)
    if data is not None:
        return data
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured.get("result", structured)
    content = getattr(result, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return result


def build_registry(
    settings: Settings | None = None, store: RunStore | None = None
) -> tuple[Any, list[str]]:
    """Return `(registry, warnings)` — MCP when configured and reachable, else in-process."""
    settings = settings or get_settings()
    warnings: list[str] = []
    if settings.use_mcp:
        url = settings.resolved_mcp_url()
        registry = MCPToolRegistry(url=url, settings=settings)
        try:
            registry.corpus_stats()
            log.info("using MCP tool server at %s", url)
            return registry, warnings
        except Exception as exc:  # noqa: BLE001 - any transport failure is a fallback
            warnings.append(f"MCP server at {url} unreachable ({exc}); using in-process tools")
    return ToolRegistry(settings=settings, store=store), warnings
