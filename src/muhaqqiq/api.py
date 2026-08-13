"""HTTP API.

Two surfaces over the same agent:

* A **native** API (`/v1/research`, `/v1/runs/...`) that returns the full typed
  `RunResult` — brief, plan, sources, audit, trace. This is what you want if you
  are building on top of Muhaqqiq.
* An **OpenResponses-compatible** endpoint (`POST /v1/responses`) that speaks the
  same request/response shape as a standard responses API, so existing clients
  can point at Muhaqqiq without knowing anything about it. The agent's entire
  multi-stage run collapses into one `output_text`, with the audit attached as
  metadata for anyone who looks.

Research runs take tens of seconds, not milliseconds. `POST /v1/research`
therefore accepts `?background=true`, returning a run id immediately and letting
the client poll — the pattern the deployment literature recommends for workloads
whose tail latency exceeds a typical gateway timeout.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import __version__
from .agent import run_research
from .config import get_settings
from .graph import build_deps, graph_mermaid
from .render import render_html
from .schemas import RunResult
from .skills import SkillLibrary
from .store import RunStore

log = logging.getLogger("muhaqqiq.api")

app = FastAPI(
    title="Muhaqqiq",
    version=__version__,
    summary="A multi-agent research agent that refuses to publish an uncited claim.",
    description=__doc__,
)

_background: dict[str, dict[str, Any]] = {}


def _store() -> RunStore:
    return RunStore(get_settings().db_path)


# --------------------------------------------------------------------------- #
# request / response models
# --------------------------------------------------------------------------- #
class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=8, examples=["How do multi-agent systems fail in production?"])
    depth: Literal["quick", "standard", "deep"] = "standard"
    language: str | None = None
    audience: str | None = None


class ResponsesRequest(BaseModel):
    """The OpenResponses-compatible request shape."""

    input: str | list[dict[str, Any]]
    model: str | None = None
    metadata: dict[str, Any] | None = None

    def as_question(self) -> str:
        if isinstance(self.input, str):
            return self.input
        parts: list[str] = []
        for item in self.input:
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    str(c.get("text", "")) for c in content if isinstance(c, dict)
                )
        return "\n".join(p for p in parts if p).strip()


# --------------------------------------------------------------------------- #
# meta endpoints
# --------------------------------------------------------------------------- #
@app.get("/healthz", tags=["meta"])
def healthz() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": settings.effective_llm_provider,
        "search_provider": settings.effective_search_provider,
        "degraded": settings.degraded_reasons,
    }


@app.get("/v1/config", tags=["meta"])
def config() -> dict[str, Any]:
    settings = get_settings()
    library = SkillLibrary.load(settings.skills_dir)
    return {
        "llm_provider": settings.effective_llm_provider,
        "model": settings.model,
        "search_provider": settings.effective_search_provider,
        "max_subquestions": settings.max_subquestions,
        "max_research_rounds": settings.max_research_rounds,
        "min_citation_coverage": settings.min_citation_coverage,
        "use_mcp": settings.use_mcp,
        "skills": [{"name": s.name, "stages": list(s.stages)} for s in library.skills],
    }


@app.get("/v1/tools", tags=["meta"])
def tools() -> dict[str, Any]:
    deps = build_deps()
    return {"tools": deps.tools.specs(), "corpus": deps.tools.corpus_stats()}


@app.get("/graph", response_class=PlainTextResponse, tags=["meta"])
def graph() -> str:
    return graph_mermaid()


# --------------------------------------------------------------------------- #
# native API
# --------------------------------------------------------------------------- #
@app.post("/v1/research", tags=["research"])
async def research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    background: bool = Query(False, description="Return immediately with a run id."),
) -> dict[str, Any]:
    if background:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        _background[run_id] = {"status": "running"}
        background_tasks.add_task(_run_in_background, run_id, request)
        return {"run_id": run_id, "status": "running", "poll": f"/v1/runs/{run_id}"}

    result = await run_in_threadpool(
        run_research,
        request.question,
        depth=request.depth,
        language=request.language,
        audience=request.audience,
    )
    return result.model_dump(mode="json")


def _run_in_background(run_id: str, request: ResearchRequest) -> None:
    try:
        run_research(
            request.question,
            depth=request.depth,
            language=request.language,
            audience=request.audience,
            run_id=run_id,
        )
        _background[run_id] = {"status": "completed"}
    except Exception as exc:  # noqa: BLE001 - surface the failure to the poller
        log.exception("background run %s failed", run_id)
        _background[run_id] = {"status": "failed", "error": str(exc)}


@app.get("/v1/runs", tags=["research"])
def list_runs(limit: int = Query(25, ge=1, le=200)) -> dict[str, Any]:
    return {"runs": _store().list_runs(limit)}


@app.get("/v1/runs/{run_id}", tags=["research"])
def get_run(run_id: str) -> dict[str, Any]:
    result = _store().get_run(run_id)
    if result is None:
        state = _background.get(run_id)
        if state:
            return {"run_id": run_id, **state}
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    return result.model_dump(mode="json")


@app.get("/v1/runs/{run_id}/report.md", response_class=PlainTextResponse, tags=["research"])
def get_run_markdown(run_id: str) -> str:
    return _require(run_id).markdown


@app.get("/v1/runs/{run_id}/report.html", response_class=HTMLResponse, tags=["research"])
def get_run_html(run_id: str) -> str:
    result = _require(run_id)
    return render_html(result.markdown, result.report.title)


def _require(run_id: str) -> RunResult:
    result = _store().get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    return result


# --------------------------------------------------------------------------- #
# OpenResponses-compatible surface
# --------------------------------------------------------------------------- #
@app.post("/v1/responses", tags=["compat"])
async def responses(request: ResponsesRequest) -> dict[str, Any]:
    """Run the agent through a standard responses-API shape."""
    question = request.as_question()
    if len(question) < 8:
        raise HTTPException(status_code=422, detail="input is too short to research")

    result: RunResult = await run_in_threadpool(run_research, question)
    return {
        "id": result.meta.run_id,
        "object": "response",
        "created_at": result.meta.created_at,
        "model": request.model or result.meta.model,
        "status": "completed",
        "output": [
            {
                "id": f"msg_{result.meta.run_id}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": result.markdown,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "start_index": 0,
                                "end_index": 0,
                                "url": source.url,
                                "title": f"[{source.id}] {source.title}",
                            }
                            for source in result.sources
                        ],
                    }
                ],
            }
        ],
        "output_text": result.markdown,
        "usage": {
            "tool_calls": result.meta.tool_calls,
            "research_rounds": result.meta.rounds_used,
            "duration_ms": result.meta.duration_ms,
        },
        "metadata": {
            **(request.metadata or {}),
            "verdict": result.verification.verdict.value,
            "citation_coverage": result.verification.citation_coverage,
            "sources_cited": result.verification.source_diversity,
            "degraded": result.meta.degraded,
        },
    }
