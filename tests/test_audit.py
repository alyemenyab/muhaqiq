"""The citation audit is the gate. These tests are the gate on the gate."""

from __future__ import annotations

from muhaqqiq.audit import audit, is_claim
from muhaqqiq.schemas import Credibility, Report, ReportSection, Source, Verdict


def _sources(*ids: str, credibility: Credibility = Credibility.HIGH) -> list[Source]:
    return [Source(id=i, title=f"Title {i}", url=f"https://x/{i}", credibility=credibility) for i in ids]


def _report(*bodies: str) -> Report:
    return Report(
        title="t",
        sections=[ReportSection(heading=f"h{i}", body=b) for i, b in enumerate(bodies)],
    )


def test_fully_cited_report_with_enough_diversity_passes():
    report = _report(
        "Agents fail on malformed tool arguments more often than on reasoning [S1].",
        "Retrieval grounding reduces but does not eliminate unsupported generation [S2].",
        "Supervisor patterns give the best reliability per unit of complexity [S3].",
    )
    result = audit(report, _sources("S1", "S2", "S3"))
    assert result.verdict is Verdict.PASS
    assert result.citation_coverage == 1.0
    assert result.source_diversity == 3


def test_uncited_claims_drag_coverage_below_threshold_and_fail():
    report = _report(
        "Agents fail on malformed tool arguments more often than on reasoning [S1].",
        "This sentence asserts something substantial without any attribution at all.",
        "Another entirely unattributed assertion about how these systems behave.",
    )
    result = audit(report, _sources("S1", "S2", "S3"))
    assert result.verdict is Verdict.FAIL
    assert result.citation_coverage < 0.8
    assert len(result.uncited_claims) == 2


def test_a_citation_pointing_at_nothing_is_a_hard_failure():
    report = _report(
        "Agents fail on malformed tool arguments more often than on reasoning [S9].",
        "Retrieval grounding reduces but does not eliminate unsupported generation [S1].",
        "Supervisor patterns give the best reliability per unit of complexity [S2].",
    )
    result = audit(report, _sources("S1", "S2", "S3"))
    assert result.verdict is Verdict.FAIL
    assert result.dangling_citations == ["S9"]


def test_single_source_report_passes_coverage_but_is_flagged():
    report = _report(
        "Agents fail on malformed tool arguments more often than on reasoning [S1].",
        "Retrieval grounding reduces but does not eliminate unsupported generation [S1].",
    )
    result = audit(report, _sources("S1"))
    assert result.verdict is Verdict.PASS_WITH_WARNINGS
    assert result.source_diversity == 1
    assert any("distinct source" in note for note in result.notes)


def test_low_credibility_sourcing_is_reported():
    report = _report(
        "Agents fail on malformed tool arguments more often than on reasoning [S1].",
        "Retrieval grounding reduces but does not eliminate unsupported generation [S2].",
        "Supervisor patterns give the best reliability per unit of complexity [S3].",
    )
    result = audit(report, _sources("S1", "S2", "S3", credibility=Credibility.LOW))
    assert result.verdict is Verdict.PASS_WITH_WARNINGS
    assert result.low_credibility_sources == ["S1", "S2", "S3"]


def test_retrieved_but_uncited_sources_are_reported_not_penalised():
    report = _report(
        "Agents fail on malformed tool arguments more often than on reasoning [S1].",
        "Retrieval grounding reduces but does not eliminate unsupported generation [S2].",
        "Supervisor patterns give the best reliability per unit of complexity [S3].",
    )
    result = audit(report, _sources("S1", "S2", "S3", "S4"))
    assert result.verdict is Verdict.PASS
    assert result.unused_sources == ["S4"]


def test_headings_and_questions_are_not_treated_as_claims():
    assert not is_claim("Sources:")
    assert not is_claim("short")
    assert not is_claim("Is this a claim that needs a citation attached to it?")
    assert is_claim("This is a substantive assertion of adequate length about agents.")


def test_empty_report_does_not_divide_by_zero():
    result = audit(Report(title="t"), _sources("S1"))
    assert result.total_claims == 0
    assert result.citation_coverage == 0.0
