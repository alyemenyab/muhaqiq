"""Stage 5 — renumber the citations, then write the report.

Parallel workers produce provisional source ids (`S101`, `S204`, …). Before the
report is written those are collapsed to `S1…Sn` in order of first appearance,
and every citation marker in every sub-answer is rewritten to match. Doing the
rewrite here — deterministically, in Python — means the writer never has to be
trusted with renumbering, which is exactly the kind of bookkeeping models get
subtly wrong.
"""

from __future__ import annotations

import re
from typing import Any

from ..observability import traceable
from ..schemas import CITATION_RE, Source, SubAnswer
from ..state import ResearchState


def renumber_sources(
    sources: list[dict[str, Any]], subanswers: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Collapse provisional ids to S1..Sn in order of first citation.

    Two things happen here. First, a document retrieved by more than one worker
    is aliased onto a single canonical id, so the same paper is not listed twice
    under two numbers. Second, the survivors are renumbered in the order a reader
    meets them, with retrieved-but-uncited sources kept at the end so the auditor
    can still report them.

    Returns `(sources, subanswers, mapping)`.
    """
    ordered_answers = sorted(subanswers, key=lambda a: _sub_key(a.get("subquestion_id", "")))

    # 1. alias duplicate documents onto one canonical provisional id
    canonical: dict[str, str] = {}
    alias: dict[str, str] = {}
    unique: list[dict[str, Any]] = []
    for source in sources:
        key = source.get("url") or source.get("doc_id") or source.get("title") or source["id"]
        if key in canonical:
            alias[source["id"]] = canonical[key]
        else:
            canonical[key] = source["id"]
            alias[source["id"]] = source["id"]
            unique.append(source)

    by_id = {s["id"]: s for s in unique}

    # 2. order by first appearance in the sub-answers
    appearance: list[str] = []
    for answer in ordered_answers:
        for num in CITATION_RE.findall(answer.get("answer", "")):
            sid = alias.get(f"S{num}", f"S{num}")
            if sid in by_id and sid not in appearance:
                appearance.append(sid)

    uncited = [s["id"] for s in unique if s["id"] not in appearance]
    order = appearance + uncited
    mapping = {old: f"S{i}" for i, old in enumerate(order, start=1)}

    new_sources = []
    for old in order:
        source = dict(by_id[old])
        source["id"] = mapping[old]
        new_sources.append(source)

    def resolve(old: str) -> str | None:
        return mapping.get(alias.get(old, old))

    def rewrite(text: str) -> str:
        return re.sub(
            r"\[(S\d+)\]",
            lambda m: (f"[{resolve(m.group(1))}]" if resolve(m.group(1)) else ""),
            text or "",
        )

    new_answers = []
    for answer in ordered_answers:
        updated = dict(answer)
        updated["answer"] = " ".join(rewrite(answer.get("answer", "")).split())
        updated["evidence"] = [
            {
                **ev,
                "source_ids": [
                    new
                    for new in (resolve(s) for s in ev.get("source_ids", []))
                    if new is not None
                ],
            }
            for ev in answer.get("evidence", [])
        ]
        new_answers.append(updated)

    return new_sources, new_answers, mapping


@traceable("muhaqqiq.synthesize")
def synthesize_node(state: ResearchState, deps: Any) -> dict[str, Any]:
    sources, subanswers, mapping = renumber_sources(
        state.get("sources", []), state.get("subanswers", [])
    )

    report = deps.provider.report(
        {
            "brief": state["brief"],
            "plan": state["plan"],
            "subanswers": subanswers,
            "sources": sources,
        }
    )
    report = _scrub_unknown_citations(report, {s["id"] for s in sources})

    deps.trace.event(
        "synthesize",
        f"wrote {len(report.sections)} section(s) from {len(sources)} source(s)",
        renumbered=len(mapping),
    )
    return {
        "final_sources": sources,
        "final_subanswers": subanswers,
        "report": report.model_dump(mode="json"),
        "events": deps.trace.as_list()[-1:],
    }


def _scrub_unknown_citations(report, known: set[str]):
    """Delete any marker the writer invented. Fabricated citations never ship."""

    def scrub(text: str) -> str:
        return " ".join(
            re.sub(
                r"\[(S\d+)\]", lambda m: m.group(0) if m.group(1) in known else "", text or ""
            ).split()
        )

    return report.model_copy(
        update={
            "executive_summary": scrub(report.executive_summary),
            "key_findings": [scrub(k) for k in report.key_findings],
            "sections": [s.model_copy(update={"body": scrub(s.body)}) for s in report.sections],
        }
    )


def _sub_key(sub_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in sub_id if ch.isdigit())
    return (int(digits) if digits else 999, sub_id)


__all__ = ["synthesize_node", "renumber_sources", "Source", "SubAnswer"]
