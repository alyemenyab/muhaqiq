"""The public entry point.

`run_research(...)` is the whole product surface: a question goes in, a validated
`RunResult` — brief, plan, sources, sub-answers, report, audit and trace — comes
out. The CLI, the HTTP API and the tests all go through this one function, so
there is exactly one code path to trust.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .graph import Deps, build_deps, build_graph
from .render import render_html
from .schemas import (
    GapReport,
    Report,
    ResearchBrief,
    ResearchPlan,
    RunMeta,
    RunResult,
    Source,
    SubAnswer,
    VerificationResult,
)
from .store import RunStore

log = logging.getLogger("muhaqqiq.agent")

RECURSION_HEADROOM = 12


def run_research(
    question: str,
    *,
    depth: str = "standard",
    language: str | None = None,
    audience: str | None = None,
    settings: Settings | None = None,
    deps: Deps | None = None,
    persist: bool = True,
    run_id: str | None = None,
) -> RunResult:
    """Execute one full research run and return the validated result."""
    settings = settings or get_settings()
    deps = deps or build_deps(settings)
    graph = build_graph(deps)

    run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
    deps.trace.event("start", f"run {run_id}", question=question, depth=depth)

    initial: dict[str, Any] = {
        "question": question,
        "depth": depth,
        "round": 0,
        "tool_calls": 0,
        "events": [],
        "warnings": list(deps.warnings or []),
    }
    if language:
        initial["language"] = language
    if audience:
        initial["audience"] = audience

    max_rounds = settings.max_research_rounds
    final_state = graph.invoke(
        initial,
        config={
            "recursion_limit": RECURSION_HEADROOM + max_rounds * 4,
            "max_concurrency": settings.researcher_concurrency,
        },
    )

    result = _assemble(run_id, final_state, deps, settings)
    if persist and deps.store is not None:
        try:
            deps.store.save_run(result)
        except Exception as exc:  # noqa: BLE001 - persistence must never fail a run
            log.warning("could not persist run %s: %s", run_id, exc)
    return result


def _assemble(run_id: str, state: dict[str, Any], deps: Deps, settings: Settings) -> RunResult:
    meta = RunMeta(
        run_id=run_id,
        llm_provider=getattr(deps.provider, "name", "unknown"),
        search_provider=settings.effective_search_provider,
        model=settings.model if settings.effective_llm_provider != "offline" else "offline-reasoner",
        duration_ms=deps.trace.elapsed_ms(),
        rounds_used=int(state.get("round", 1)),
        tool_calls=int(state.get("tool_calls", 0)),
        degraded=list(dict.fromkeys(state.get("warnings", []) or [])),
    )
    return RunResult(
        meta=meta,
        brief=ResearchBrief.model_validate(state["brief"]),
        plan=ResearchPlan.model_validate(state["plan"]),
        sources=[Source.model_validate(s) for s in state.get("final_sources", [])],
        subanswers=[SubAnswer.model_validate(a) for a in state.get("final_subanswers", [])],
        gap_reports=[GapReport.model_validate(g) for g in state.get("gap_reports", [])],
        report=Report.model_validate(state["report"]),
        verification=VerificationResult.model_validate(state["verification"]),
        markdown=state.get("markdown", ""),
        events=deps.trace.as_list(),
    )


def write_outputs(result: RunResult, output_dir: Path | str = "out") -> dict[str, Path]:
    """Write the report as markdown, HTML and JSON. Returns the paths written."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = result.meta.run_id

    md_path = out / f"{stem}.md"
    html_path = out / f"{stem}.html"
    json_path = out / f"{stem}.json"

    md_path.write_text(result.markdown, encoding="utf-8")
    html_path.write_text(render_html(result.markdown, result.report.title), encoding="utf-8")
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return {"markdown": md_path, "html": html_path, "json": json_path}


def load_run(run_id: str, settings: Settings | None = None) -> RunResult | None:
    settings = settings or get_settings()
    return RunStore(settings.db_path).get_run(run_id)
