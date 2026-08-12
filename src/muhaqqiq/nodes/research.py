"""Stage 3 — the researcher agent.

Many instances of this node run concurrently, one per sub-question. Each one
sees only its own sub-question and its own retrieved documents: workers cannot
read each other's findings, which is what stops a shared narrative from forming
before the evidence is in.

Source ids are provisional at this stage (`S101`, `S102`, … for worker 1) so
that parallel workers cannot collide on a number. They are renumbered to a clean
`S1…Sn` during synthesis.
"""

from __future__ import annotations

from typing import Any

from ..observability import traceable
from ..offline import score_credibility
from ..schemas import Source, SubAnswer
from ..state import ResearchState
from ..textkit import truncate


@traceable("muhaqqiq.researcher", run_type="tool")
def researcher_node(state: ResearchState, deps: Any) -> dict[str, Any]:
    payload: dict[str, Any] = dict(state)  # a Send payload arrives as the node's state
    sub = payload["subquestion"]
    index = int(payload.get("index", 1))
    round_no = int(payload.get("round", 1))
    queries = payload.get("queries") or sub.get("search_queries") or [sub["question"]]

    limit = deps.settings.max_sources_per_subquestion
    tool_calls_before = getattr(deps.tools, "calls", 0)

    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query in queries[:3]:
        for doc in deps.tools.web_search(query, limit=limit):
            key = doc.get("url") or doc.get("doc_id") or doc.get("title", "")
            if key and key in seen:
                continue
            seen.add(key)
            hits.append(doc)

    hits.sort(key=lambda d: -float(d.get("score", 0.0)))
    hits = _apply_relevance_floor(hits)[:limit]

    sources: list[Source] = []
    payloads: list[dict[str, Any]] = []
    for offset, doc in enumerate(hits, start=1):
        source_id = f"S{index * 100 + offset}"
        content = doc.get("content", "") or doc.get("snippet", "")
        credibility = doc.get("credibility") or score_credibility(
            doc.get("url", ""), doc.get("publisher", "")
        )
        source = Source(
            id=source_id,
            title=doc.get("title", "") or doc.get("doc_id", "untitled"),
            url=doc.get("url", ""),
            publisher=doc.get("publisher", ""),
            published=doc.get("published", ""),
            snippet=truncate(content, 400),
            relevance=min(1.0, round(float(doc.get("score", 0.0)), 4)),
            credibility=credibility,
            query=doc.get("query", ""),
        )
        sources.append(source)
        payloads.append({**source.model_dump(mode="json"), "content": content})

    answer: SubAnswer = deps.provider.answer_subquestion(
        {
            "subquestion": sub,
            "sources": payloads,
            # Ask for more snippets than sources: the synthesis stage de-duplicates
            # across sections, so each researcher needs spare material.
            "max_snippets": max(4, limit + 2),
        }
    )
    # Trust the graph, not the model, for identity and bookkeeping fields.
    answer = answer.model_copy(update={"subquestion_id": sub["id"], "rounds": round_no})
    answer = _drop_unknown_citations(answer, {s.id for s in sources})

    used = getattr(deps.tools, "calls", 0) - tool_calls_before
    deps.trace.event(
        "researcher",
        f"{sub['id']}: {len(sources)} source(s), {len(answer.evidence)} evidence item(s)",
        subquestion=sub["id"],
        round=round_no,
        sources=[s.id for s in sources],
    )
    return {
        "sources": [s.model_dump(mode="json") for s in sources],
        "subanswers": [answer.model_dump(mode="json")],
        "tool_calls": used,
        "events": deps.trace.as_list()[-1:],
    }


RELATIVE_FLOOR = 0.30
ABSOLUTE_FLOOR = 0.03


def _apply_relevance_floor(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Discard weak hits relative to the best one for this sub-question.

    Retrieval returns a long tail of documents that share a common word with the
    query and nothing else, and a researcher handed an off-topic document will
    dutifully quote it. The threshold is relative rather than absolute so it
    transfers between the local index and a live search provider, whose scores
    are on completely different scales.
    """
    if not hits:
        return hits
    top = float(hits[0].get("score", 0.0))
    if top <= 0:
        return hits[:1]
    threshold = max(ABSOLUTE_FLOOR, RELATIVE_FLOOR * top)
    kept = [h for h in hits if float(h.get("score", 0.0)) >= threshold]
    return kept or hits[:1]


def _drop_unknown_citations(answer: SubAnswer, known: set[str]) -> SubAnswer:
    """A worker may only cite what it actually retrieved."""
    import re

    def scrub(text: str) -> str:
        return re.sub(
            r"\[(S\d+)\]", lambda m: m.group(0) if m.group(1) in known else "", text or ""
        )

    evidence = [
        e.model_copy(update={"source_ids": [s for s in e.source_ids if s in known]})
        for e in answer.evidence
    ]
    return answer.model_copy(
        update={
            "answer": " ".join(scrub(answer.answer).split()),
            "evidence": [e for e in evidence if e.source_ids],
        }
    )
