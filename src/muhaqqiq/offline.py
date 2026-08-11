"""The offline reasoning provider.

A capstone that only runs when a grader happens to own an API key is a capstone
that does not run. `OfflineReasoner` implements the same structured-output
contract as the hosted provider, but does it with deterministic, extractive
logic over whatever sources the retrieval layer returned: it decomposes the
question into facets, pulls the highest-scoring sentences out of each source,
and attaches a real `[S#]` citation to every sentence it keeps.

It is a genuine (if unsophisticated) reasoner, not a canned response — swap
`MUHAQQIQ_LLM_PROVIDER=openrouter` and the exact same graph runs on a frontier
model. Everything downstream of this file is provider-agnostic.
"""

from __future__ import annotations

from typing import Any

from .schemas import (
    Credibility,
    Depth,
    Evidence,
    GapReport,
    Report,
    ReportSection,
    ResearchBrief,
    ResearchPlan,
    SubAnswer,
    SubQuestion,
)
from .textkit import (
    cited_segments,
    dedupe_preserving_order,
    keywords,
    overlap_score,
    sentences,
    strip_citations,
    titlecase,
    truncate,
)

FACETS: list[tuple[str, str, list[str]]] = [
    (
        "Definition and scope",
        "{topic} — what is it, and how is it currently defined and bounded?",
        ["definition", "overview", "what is"],
    ),
    (
        "Current state and evidence",
        "{topic} — what is the current state, and what evidence or data supports it?",
        ["current state", "statistics", "evidence"],
    ),
    (
        "Approaches and trade-offs",
        "{topic} — which approaches or methods dominate, and how do they compare?",
        ["approaches", "comparison", "trade-offs"],
    ),
    (
        "Risks and limitations",
        "{topic} — what are the main risks, limitations, or criticisms?",
        ["risks", "limitations", "criticism"],
    ),
    (
        "Outlook and implications",
        "{topic} — where is this heading, and what does that mean in practice?",
        ["outlook", "future", "implications"],
    ),
]

DEPTH_TO_COUNT = {Depth.QUICK: 3, Depth.STANDARD: 4, Depth.DEEP: 5}


class OfflineReasoner:
    """Deterministic structured-output engine. Same interface, no network."""

    name = "offline"

    # -- stage 1 ----------------------------------------------------------- #
    def brief(self, payload: dict[str, Any]) -> ResearchBrief:
        question = str(payload.get("question", "")).strip()
        kws = keywords(question, limit=4)
        topic = " ".join(kws) if kws else truncate(question, 40)
        depth = Depth(payload.get("depth") or Depth.STANDARD)
        audience = payload.get("audience") or "a technical reader who is new to the topic"
        language = payload.get("language") or ("ar" if _is_arabic(question) else "en")
        return ResearchBrief(
            question=question,
            topic=topic,
            audience=audience,
            depth=depth,
            language=language,
            scope_in=[f"Material that directly addresses: {truncate(question, 90)}"],
            scope_out=["Speculation not traceable to a retrieved source"],
            success_criteria=[
                "Every substantive claim carries a [S#] citation",
                "At least three distinct sources are used",
                "Limitations and disagreements between sources are stated explicitly",
            ],
        )

    # -- stage 2 ----------------------------------------------------------- #
    def plan(self, payload: dict[str, Any]) -> ResearchPlan:
        brief = _as_brief(payload["brief"])
        limit = int(payload.get("max_subquestions") or DEPTH_TO_COUNT[brief.depth])
        limit = max(1, min(limit, len(FACETS)))
        subs: list[SubQuestion] = []
        for idx, (label, template, query_hints) in enumerate(FACETS[:limit], start=1):
            question = template.format(topic=brief.topic, audience=brief.audience)
            queries = dedupe_preserving_order(
                [f"{brief.topic} {hint}" for hint in query_hints] + [brief.question]
            )
            subs.append(
                SubQuestion(
                    id=f"q{idx}",
                    question=question,
                    label=label,
                    rationale=f"Covers the '{label.lower()}' facet of the brief.",
                    search_queries=queries[:3],
                )
            )
        strategy = (
            f"Decompose '{brief.topic}' into {len(subs)} orthogonal facets so that each can be "
            "researched independently and in parallel, then reconcile the findings into a single "
            "report. Facets are chosen to cover definition, evidence, alternatives, risk and "
            "outlook, which together answer the brief without overlapping."
        )
        return ResearchPlan(strategy=strategy, subquestions=subs)

    # -- stage 3 ----------------------------------------------------------- #
    def answer_subquestion(self, payload: dict[str, Any]) -> SubAnswer:
        sub = payload["subquestion"]
        sub_id = sub["id"] if isinstance(sub, dict) else sub.id
        sub_q = sub["question"] if isinstance(sub, dict) else sub.question
        sources = payload.get("sources") or []
        max_snippets = int(payload.get("max_snippets", 4))

        scored: list[tuple[float, str, str]] = []  # (score, source_id, sentence)
        for src in sources:
            sid = src["id"] if isinstance(src, dict) else src.id
            # Prefer full content; the snippet is a truncated prefix of it, and
            # concatenating the two splices a half-sentence onto a whole one.
            if isinstance(src, dict):
                body = str(src.get("content") or src.get("snippet") or "")
            else:
                body = str(getattr(src, "content", "") or src.snippet or "")
            for sentence in sentences(body) or [body]:
                score = overlap_score(sub_q, sentence)
                if score > 0.02:
                    scored.append((score, sid, sentence))

        scored.sort(key=lambda t: (-t[0], t[1]))
        picked: list[tuple[float, str, str]] = []
        seen_sources: set[str] = set()
        # First pass: one strong sentence per source, so the answer is not
        # dominated by whichever document happens to be longest.
        for score, sid, sentence in scored:
            if sid not in seen_sources:
                picked.append((score, sid, sentence))
                seen_sources.add(sid)
        # Second pass: fill remaining budget with the next best sentences.
        for item in scored:
            if len(picked) >= max_snippets:
                break
            if item not in picked:
                picked.append(item)
        picked = picked[:max_snippets]

        if not picked:
            return SubAnswer(
                subquestion_id=sub_id,
                answer="No retrieved source addressed this sub-question.",
                evidence=[],
                unresolved=[sub_q],
            )

        evidence = [
            Evidence(
                claim=truncate(sentence, 260),
                source_ids=[sid],
                quote=truncate(sentence, 320),
                confidence=round(min(0.95, 0.45 + score), 2),
            )
            for score, sid, sentence in picked
        ]
        # The marker goes *inside* the sentence, before its final stop. A marker
        # stranded after the full stop belongs to the next sentence as far as any
        # downstream splitter is concerned, which quietly destroys coverage.
        body = " ".join(_cite(truncate(sentence, 260), sid) for _, sid, sentence in picked)
        unresolved = [] if len(picked) >= 2 else [f"Only one source covered: {sub_q}"]
        return SubAnswer(
            subquestion_id=sub_id, answer=body, evidence=evidence, unresolved=unresolved
        )

    # -- stage 4 ----------------------------------------------------------- #
    def critique(self, payload: dict[str, Any]) -> GapReport:
        subanswers = payload.get("subanswers") or []
        plan = payload.get("plan")
        min_sources = int(payload.get("min_sources", 2))
        gaps: list[str] = []
        followups: dict[str, list[str]] = {}
        for sa in subanswers:
            sa_obj = sa if isinstance(sa, SubAnswer) else SubAnswer.model_validate(sa)
            distinct = {s for e in sa_obj.evidence for s in e.source_ids}
            if len(distinct) < min_sources:
                gaps.append(
                    f"{sa_obj.subquestion_id}: only {len(distinct)} distinct source(s) supported this facet."
                )
                sub = _plan_lookup(plan, sa_obj.subquestion_id)
                if sub is not None:
                    followups[sa_obj.subquestion_id] = [
                        f"{sub.question} evidence",
                        f"{sub.question} study OR report",
                    ]
            if sa_obj.unresolved:
                gaps.extend(f"{sa_obj.subquestion_id}: {u}" for u in sa_obj.unresolved)
        return GapReport(
            sufficient=not followups,
            gaps=dedupe_preserving_order(gaps),
            followup_queries=followups,
            notes=(
                "Coverage is adequate across all facets."
                if not followups
                else f"{len(followups)} facet(s) need another retrieval round."
            ),
        )

    # -- stage 5 ----------------------------------------------------------- #
    def report(self, payload: dict[str, Any]) -> Report:
        brief = _as_brief(payload["brief"])
        plan = payload["plan"]
        plan_obj = plan if isinstance(plan, ResearchPlan) else ResearchPlan.model_validate(plan)
        subanswers = [
            sa if isinstance(sa, SubAnswer) else SubAnswer.model_validate(sa)
            for sa in payload.get("subanswers", [])
        ]

        sections: list[ReportSection] = []
        summary_bits: list[str] = []
        findings: list[tuple[str, str]] = []
        open_qs: list[str] = []
        seen_segments: set[str] = set()

        for sa in subanswers:
            sub = plan_obj.by_id(sa.subquestion_id)
            heading = (sub.label if sub and sub.label else None) or titlecase(
                truncate(sub.question if sub else sa.subquestion_id, 70)
            )

            # Parallel researchers frequently surface the same sentence for
            # different facets. Keep the first occurrence, drop the echoes.
            segments = cited_segments(sa.answer) or ([sa.answer] if sa.answer else [])
            kept: list[str] = []
            for segment in segments:
                fingerprint = strip_citations(segment).lower()[:120]
                if fingerprint and fingerprint in seen_segments:
                    continue
                seen_segments.add(fingerprint)
                kept.append(segment)
            # Never empty a section entirely: a facet that only found material
            # already used elsewhere still has to say something about itself.
            # If everything this facet found was already used elsewhere, say so.
            # Repeating a paragraph under a second heading would pad the report
            # and inflate its apparent coverage; admitting the overlap costs a
            # line and tells the reader something true.
            if not kept and segments:
                kept = [
                    "This facet returned no evidence beyond what is already cited above; "
                    "the retrieved sources overlap with the preceding sections."
                ]
                open_qs.append(
                    f"{heading}: retrieval found no material specific to this facet."
                )

            sections.append(ReportSection(heading=heading.rstrip("?"), body=" ".join(kept)))
            if kept and "[S" in kept[0]:
                summary_bits.append(kept[0])
            for ev in sa.evidence[:2]:
                cite = "".join(f"[{sid}]" for sid in ev.source_ids)
                if cite:
                    findings.append((strip_citations(truncate(ev.claim, 200)), cite))
            open_qs.extend(sa.unresolved)

        executive = " ".join(
            [f"This report answers: {brief.question}", *summary_bits[:4]]
        ).strip()

        seen_findings: set[str] = set()
        unique_findings: list[str] = []
        for claim, cite in findings:
            key = claim.lower()[:120]
            if key in seen_findings:
                continue
            seen_findings.add(key)
            unique_findings.append(_cite(claim, cite))

        title = brief.question.strip().rstrip("؟?")
        if len(title) > 110 or len(title) < 12:
            title = f"{titlecase(brief.topic)}: A Sourced Research Brief"

        return Report(
            title=title,
            executive_summary=executive,
            key_findings=unique_findings[:8],
            sections=sections,
            open_questions=dedupe_preserving_order(open_qs)[:6],
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cite(text: str, marker: str) -> str:
    """Attach a citation marker inside a sentence, before its final stop."""
    text = (text or "").strip()
    if not marker:
        return text
    if not marker.startswith("["):
        marker = f"[{marker}]"
    if text and text[-1] in ".!?،؟":
        return f"{text[:-1].rstrip()} {marker}{text[-1]}"
    return f"{text} {marker}."


def _as_brief(value: Any) -> ResearchBrief:
    return value if isinstance(value, ResearchBrief) else ResearchBrief.model_validate(value)


def _plan_lookup(plan: Any, sub_id: str) -> SubQuestion | None:
    if plan is None:
        return None
    plan_obj = plan if isinstance(plan, ResearchPlan) else ResearchPlan.model_validate(plan)
    return plan_obj.by_id(sub_id)


def _is_arabic(text: str) -> bool:
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    return arabic > max(3, len(text) * 0.2)


def score_credibility(url: str, publisher: str = "") -> Credibility:
    """Very small domain heuristic used to flag weak sourcing in the verifier."""
    blob = f"{url} {publisher}".lower()
    if any(k in blob for k in (".gov", ".edu", ".int", "who.int", "oecd", "nature.com", "arxiv.org", "ieee")):
        return Credibility.HIGH
    if any(k in blob for k in (".org", "reuters", "ft.com", "economist", "docs.", "documentation")):
        return Credibility.MEDIUM
    if any(k in blob for k in ("blogspot", "medium.com", "reddit", "quora", "forum", "substack")):
        return Credibility.LOW
    return Credibility.UNKNOWN if not blob.strip() else Credibility.MEDIUM
