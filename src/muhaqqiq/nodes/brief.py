"""Stage 1 — turn a raw question into a bounded research brief."""

from __future__ import annotations

from typing import Any

from ..observability import traceable
from ..state import ResearchState


@traceable("muhaqqiq.brief")
def brief_node(state: ResearchState, deps: Any) -> dict[str, Any]:
    question = (state.get("question") or "").strip()
    if not question:
        raise ValueError("a research question is required")

    brief = deps.provider.brief(
        {
            "question": question,
            "depth": state.get("depth") or "standard",
            "language": state.get("language"),
            "audience": state.get("audience"),
        }
    )
    deps.trace.event("brief", "normalised the question", topic=brief.topic, depth=brief.depth.value)
    return {
        "brief": brief.model_dump(mode="json"),
        "round": 0,
        "events": deps.trace.as_list()[-1:],
    }
