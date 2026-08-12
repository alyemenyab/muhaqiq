"""Stage 4 — the critic.

Reads every sub-answer and decides whether the evidence is thin enough to
justify paying for another retrieval round. This is the graph's only cycle, and
it is bounded by `MUHAQQIQ_MAX_RESEARCH_ROUNDS`: an agent that can decide to keep
going is an agent that can decide to never stop.
"""

from __future__ import annotations

from typing import Any

from ..observability import traceable
from ..state import ResearchState


@traceable("muhaqqiq.critique")
def critique_node(state: ResearchState, deps: Any) -> dict[str, Any]:
    gaps = deps.provider.critique(
        {
            "subanswers": state.get("subanswers", []),
            "plan": state.get("plan"),
            "min_sources": 2,
        }
    )
    round_no = int(state.get("round", 1))
    rounds_left = round_no < deps.settings.max_research_rounds

    pending = gaps.followup_queries if (not gaps.sufficient and rounds_left) else {}
    deps.trace.event(
        "critique",
        (
            "coverage sufficient"
            if not pending
            else f"{len(pending)} facet(s) queued for another round"
        ),
        round=round_no,
        gaps=len(gaps.gaps),
        will_retry=bool(pending),
    )
    return {
        "gap_reports": [gaps.model_dump(mode="json")],
        "pending": pending,
        "events": deps.trace.as_list()[-1:],
    }


def should_retry(state: ResearchState) -> str:
    """Conditional edge: another research round, or move to synthesis."""
    return "dispatch" if state.get("pending") else "synthesize"
