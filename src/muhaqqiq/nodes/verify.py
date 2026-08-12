"""Stage 6 & 7 — audit the report, then render it.

Verification never rewrites the report. It attaches a verdict, and the renderer
prints that verdict at the top of the document. A reader who can see that
coverage was 62% treats the artefact differently from one who cannot, and hiding
that number would defeat the purpose of computing it.
"""

from __future__ import annotations

from typing import Any

from ..audit import audit
from ..observability import traceable
from ..render import render_markdown
from ..schemas import Report, Source
from ..state import ResearchState


@traceable("muhaqqiq.verify")
def verify_node(state: ResearchState, deps: Any) -> dict[str, Any]:
    report = Report.model_validate(state["report"])
    sources = [Source.model_validate(s) for s in state.get("final_sources", [])]
    result = audit(report, sources, min_coverage=deps.settings.min_citation_coverage)
    deps.trace.event(
        "verify",
        f"verdict={result.verdict.value} coverage={result.citation_coverage:.0%}",
        diversity=result.source_diversity,
        dangling=len(result.dangling_citations),
    )
    return {"verification": result.model_dump(mode="json"), "events": deps.trace.as_list()[-1:]}


@traceable("muhaqqiq.render")
def render_node(state: ResearchState, deps: Any) -> dict[str, Any]:
    markdown = render_markdown(state, settings=deps.settings)
    deps.trace.event("render", f"rendered {len(markdown)} characters of markdown")
    return {"markdown": markdown, "events": deps.trace.as_list()[-1:]}
