"""Graph state and its reducers.

The whole run is one typed dictionary. Reducers matter more than they look: the
researcher nodes execute in parallel and all write to `sources` and `subanswers`
at the same time, so a field that overwrites instead of merging would silently
discard every worker but one.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_sources(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append new sources, de-duplicating on provisional id only.

    Deliberately *not* de-duplicated by URL here. Two researchers that retrieve
    the same document each cite it under their own provisional id, and dropping
    one of those rows at merge time would orphan every citation pointing at it.
    Content-level de-duplication happens once, later, in `renumber_sources`,
    where the alias can be recorded and the markers rewritten to match.
    """
    merged = list(left or [])
    seen_ids = {s.get("id") for s in merged}
    for source in right or []:
        if source.get("id") in seen_ids:
            continue
        merged.append(source)
        seen_ids.add(source.get("id"))
    return sorted(merged, key=lambda s: _source_key(s.get("id", "")))


def _source_key(source_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in source_id if ch.isdigit())
    return (int(digits) if digits else 10**9, source_id)


def merge_subanswers(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Later rounds replace earlier answers for the same sub-question."""
    by_id: dict[str, dict[str, Any]] = {a["subquestion_id"]: a for a in (left or [])}
    for answer in right or []:
        by_id[answer["subquestion_id"]] = answer
    return sorted(by_id.values(), key=lambda a: _sort_key(a["subquestion_id"]))


def _sort_key(sub_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in sub_id if ch.isdigit())
    return (int(digits) if digits else 999, sub_id)


class ResearchState(TypedDict, total=False):
    # inputs
    question: str
    depth: str
    language: str
    audience: str

    # stage outputs
    brief: dict[str, Any]
    plan: dict[str, Any]
    sources: Annotated[list[dict[str, Any]], merge_sources]
    subanswers: Annotated[list[dict[str, Any]], merge_subanswers]
    gap_reports: Annotated[list[dict[str, Any]], operator.add]

    # Renumbered, reader-facing versions produced by the synthesis stage. These
    # deliberately have no reducer: synthesis overwrites them wholesale, because
    # renumbering changes the very ids the merge reducers de-duplicate on.
    final_sources: list[dict[str, Any]]
    final_subanswers: list[dict[str, Any]]

    report: dict[str, Any]
    verification: dict[str, Any]
    markdown: str

    # control
    round: int
    pending: dict[str, list[str]]
    tool_calls: Annotated[int, operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]
    warnings: Annotated[list[str], operator.add]
