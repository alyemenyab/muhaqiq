"""Structured outputs.

Every hop between agents in Muhaqqiq is a typed Pydantic object, never free text.
That is what makes the pipeline auditable: the planner cannot hand the researchers
a paragraph, it has to hand them a list of `SubQuestion`s, and the writer cannot
emit a claim without an `Evidence` object pointing at a `Source.id`.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

CITATION_RE = re.compile(r"\[S(\d+)\]")


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Strict(BaseModel):
    """Base model: reject unknown fields so a drifting LLM fails loudly."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# --------------------------------------------------------------------------- #
# Stage 1 — brief
# --------------------------------------------------------------------------- #
class Depth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class ResearchBrief(Strict):
    """Normalised statement of what the user actually wants to know."""

    question: str = Field(..., description="The user's question, restated precisely.")
    topic: str = Field(..., description="Short topic label, 2-6 words.")
    audience: str = Field("general technical reader", description="Who the report is for.")
    depth: Depth = Field(Depth.STANDARD)
    language: str = Field("en", description="ISO code for the output language.")
    scope_in: list[str] = Field(default_factory=list, description="Explicitly in scope.")
    scope_out: list[str] = Field(default_factory=list, description="Explicitly out of scope.")
    success_criteria: list[str] = Field(
        default_factory=list, description="What a good answer must contain."
    )

    @field_validator("question", "topic")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


# --------------------------------------------------------------------------- #
# Stage 2 — plan
# --------------------------------------------------------------------------- #
class SubQuestion(Strict):
    id: str = Field(..., description="Stable id, e.g. 'q1'.")
    question: str
    label: str = Field("", description="Short section heading, 2-5 words.")
    rationale: str = Field("", description="Why this sub-question matters to the brief.")
    search_queries: list[str] = Field(default_factory=list, min_length=0)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.fullmatch(r"q\d+", v):
            raise ValueError("sub-question id must look like 'q1'")
        return v


class ResearchPlan(Strict):
    strategy: str = Field("", description="One paragraph on how the topic is decomposed.")
    subquestions: list[SubQuestion] = Field(default_factory=list)

    def by_id(self, sub_id: str) -> SubQuestion | None:
        return next((s for s in self.subquestions if s.id == sub_id), None)


# --------------------------------------------------------------------------- #
# Stage 3 — evidence
# --------------------------------------------------------------------------- #
class Credibility(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Source(Strict):
    id: str = Field(..., description="Citation handle, e.g. 'S1'.")
    title: str
    url: str = ""
    publisher: str = ""
    published: str = ""
    snippet: str = ""
    relevance: float = Field(0.0, ge=0.0, le=1.0)
    credibility: Credibility = Credibility.UNKNOWN
    retrieved_at: str = Field(default_factory=utcnow)
    query: str = Field("", description="The query that surfaced this source.")

    @field_validator("id")
    @classmethod
    def _id_shape(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.fullmatch(r"S\d+", v):
            raise ValueError("source id must look like 'S1'")
        return v


class Evidence(Strict):
    claim: str = Field(..., description="A single factual statement.")
    source_ids: list[str] = Field(default_factory=list)
    quote: str = Field("", description="Supporting text lifted from the source.")
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class SubAnswer(Strict):
    subquestion_id: str
    answer: str = Field(..., description="Answer to the sub-question, with [S#] markers.")
    evidence: list[Evidence] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    rounds: int = Field(1, ge=1)

    @property
    def cited_source_ids(self) -> set[str]:
        return set(f"S{n}" for n in CITATION_RE.findall(self.answer))


# --------------------------------------------------------------------------- #
# Stage 4 — critique
# --------------------------------------------------------------------------- #
class GapReport(Strict):
    sufficient: bool = True
    gaps: list[str] = Field(default_factory=list)
    followup_queries: dict[str, list[str]] = Field(
        default_factory=dict, description="sub-question id -> extra queries to run"
    )
    notes: str = ""


# --------------------------------------------------------------------------- #
# Stage 5 — report
# --------------------------------------------------------------------------- #
class ReportSection(Strict):
    heading: str
    body: str = Field(..., description="Markdown body with [S#] citation markers.")


class Report(Strict):
    title: str
    executive_summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    def all_text(self) -> str:
        parts = [self.executive_summary, *self.key_findings]
        parts += [s.body for s in self.sections]
        return "\n\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Stage 6 — verification
# --------------------------------------------------------------------------- #
class Verdict(str, Enum):
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    FAIL = "fail"


class VerificationResult(Strict):
    verdict: Verdict = Verdict.PASS
    citation_coverage: float = Field(0.0, ge=0.0, le=1.0)
    total_claims: int = 0
    cited_claims: int = 0
    uncited_claims: list[str] = Field(default_factory=list)
    dangling_citations: list[str] = Field(default_factory=list)
    unused_sources: list[str] = Field(default_factory=list)
    source_diversity: int = 0
    low_credibility_sources: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Run envelope
# --------------------------------------------------------------------------- #
class RunMeta(Strict):
    run_id: str
    created_at: str = Field(default_factory=utcnow)
    llm_provider: str = "offline"
    search_provider: str = "offline"
    model: str = ""
    duration_ms: int = 0
    rounds_used: int = 1
    tool_calls: int = 0
    degraded: list[str] = Field(default_factory=list)


class RunResult(Strict):
    meta: RunMeta
    brief: ResearchBrief
    plan: ResearchPlan
    sources: list[Source] = Field(default_factory=list)
    subanswers: list[SubAnswer] = Field(default_factory=list)
    gap_reports: list[GapReport] = Field(default_factory=list)
    report: Report
    verification: VerificationResult
    markdown: str = ""
    events: list[dict[str, Any]] = Field(default_factory=list)
