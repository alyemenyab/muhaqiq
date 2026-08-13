"""Structured outputs are the contract between agents; a loose schema is a bug."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from muhaqqiq.schemas import (
    Credibility,
    Report,
    ReportSection,
    ResearchBrief,
    ResearchPlan,
    Source,
    SubAnswer,
    SubQuestion,
    Verdict,
)


def test_source_id_must_be_well_formed():
    assert Source(id="s3", title="t").id == "S3"
    with pytest.raises(ValidationError):
        Source(id="source-3", title="t")


def test_subquestion_id_must_be_well_formed():
    assert SubQuestion(id="Q2", question="why?").id == "q2"
    with pytest.raises(ValidationError):
        SubQuestion(id="first", question="why?")


def test_brief_rejects_an_empty_question():
    with pytest.raises(ValidationError):
        ResearchBrief(question="   ", topic="x")


def test_relevance_and_confidence_are_bounded():
    with pytest.raises(ValidationError):
        Source(id="S1", title="t", relevance=1.5)


def test_subanswer_extracts_its_own_citations():
    answer = SubAnswer(
        subquestion_id="q1", answer="Claim one [S1]. Claim two [S12] and [S1] again."
    )
    assert answer.cited_source_ids == {"S1", "S12"}


def test_plan_lookup_by_id():
    plan = ResearchPlan(subquestions=[SubQuestion(id="q1", question="a")])
    assert plan.by_id("q1") is not None
    assert plan.by_id("q9") is None


def test_report_all_text_covers_every_field_the_auditor_must_see():
    report = Report(
        title="t",
        executive_summary="summary [S1]",
        key_findings=["finding [S2]"],
        sections=[ReportSection(heading="h", body="body [S3]")],
    )
    text = report.all_text()
    assert all(marker in text for marker in ("[S1]", "[S2]", "[S3]"))


def test_enums_are_closed():
    assert Credibility("high") is Credibility.HIGH
    assert Verdict("fail") is Verdict.FAIL
    with pytest.raises(ValueError):
        Credibility("excellent")
