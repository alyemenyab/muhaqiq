"""Stage 2 — decompose the brief, then fan the sub-questions out in parallel."""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from ..observability import traceable
from ..schemas import ResearchPlan
from ..state import ResearchState


@traceable("muhaqqiq.plan")
def plan_node(state: ResearchState, deps: Any) -> dict[str, Any]:
    brief = state["brief"]
    plan = deps.provider.plan(
        {"brief": brief, "max_subquestions": deps.settings.max_subquestions}
    )
    if not plan.subquestions:
        raise ValueError("planner produced no sub-questions")

    deps.trace.event(
        "plan",
        f"decomposed into {len(plan.subquestions)} sub-questions",
        subquestions=[s.id for s in plan.subquestions],
    )
    return {
        "plan": plan.model_dump(mode="json"),
        "pending": {s.id: s.search_queries for s in plan.subquestions},
        "events": deps.trace.as_list()[-1:],
    }


def dispatch_node(state: ResearchState, deps: Any) -> dict[str, Any]:
    """Bookkeeping only. The real work is the conditional edge below."""
    round_no = int(state.get("round", 0)) + 1
    pending = state.get("pending") or {}
    deps.trace.event(
        "dispatch",
        f"round {round_no}: dispatching {len(pending)} researcher(s)",
        targets=sorted(pending),
    )
    return {"round": round_no, "events": deps.trace.as_list()[-1:]}


def fan_out(state: ResearchState) -> list[Send]:
    """One `Send` per pending sub-question — this is the parallel step."""
    plan = ResearchPlan.model_validate(state["plan"])
    pending = state.get("pending") or {s.id: s.search_queries for s in plan.subquestions}
    round_no = int(state.get("round", 1))

    sends: list[Send] = []
    for index, sub in enumerate(plan.subquestions, start=1):
        if sub.id not in pending:
            continue
        queries = pending.get(sub.id) or sub.search_queries or [sub.question]
        sends.append(
            Send(
                "researcher",
                {
                    "subquestion": sub.model_dump(mode="json"),
                    "queries": queries,
                    "index": index,
                    "round": round_no,
                    "brief": state.get("brief", {}),
                },
            )
        )
    return sends
