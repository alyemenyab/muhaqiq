"""End-to-end behaviour of the agent graph."""

from __future__ import annotations

import pytest

from muhaqqiq.agent import run_research, write_outputs
from muhaqqiq.config import get_settings
from muhaqqiq.graph import build_deps, build_graph
from muhaqqiq.nodes.critique import should_retry
from muhaqqiq.nodes.plan import fan_out
from muhaqqiq.nodes.synthesize import renumber_sources
from muhaqqiq.schemas import Verdict

QUESTION = "How do multi-agent orchestration patterns compare, and what are their risks?"


@pytest.fixture(scope="module")
def _shared_question() -> str:
    return QUESTION


def test_agent_produces_a_verified_cited_report():
    result = run_research(QUESTION)

    assert result.report.sections, "the report has no sections"
    assert result.sources, "no sources were retrieved"
    assert result.verification.verdict in {Verdict.PASS, Verdict.PASS_WITH_WARNINGS}
    assert result.verification.citation_coverage >= 0.8
    assert result.verification.dangling_citations == []
    assert result.markdown.startswith("# ")


def test_every_cited_marker_resolves_to_a_real_source():
    result = run_research(QUESTION)
    known = {s.id for s in result.sources}
    cited = {f"S{n}" for n in __import__("re").findall(r"\[S(\d+)\]", result.markdown)}
    assert cited <= known, f"report cites sources that do not exist: {cited - known}"


def test_sources_are_renumbered_from_one_with_no_gaps():
    result = run_research(QUESTION)
    assert [s.id for s in result.sources] == [f"S{i}" for i in range(1, len(result.sources) + 1)]


def test_plan_fans_out_one_researcher_per_subquestion():
    result = run_research(QUESTION)
    assert len(result.subanswers) == len(result.plan.subquestions)
    assert {a.subquestion_id for a in result.subanswers} == {
        s.id for s in result.plan.subquestions
    }


def test_research_rounds_are_bounded(monkeypatch):
    """The critic can ask for another round; it must not be able to ask forever."""
    monkeypatch.setenv("MUHAQQIQ_MAX_RESEARCH_ROUNDS", "2")
    from muhaqqiq import config

    config.reset_settings_cache()
    result = run_research(QUESTION)
    assert result.meta.rounds_used <= 2
    assert len(result.gap_reports) <= 2


def test_run_is_persisted_and_retrievable():
    result = run_research(QUESTION)
    stored = build_deps().store.get_run(result.meta.run_id)
    assert stored is not None
    assert stored.report.title == result.report.title


def test_outputs_are_written_in_three_formats(tmp_path):
    result = run_research(QUESTION)
    paths = write_outputs(result, tmp_path / "out")
    assert set(paths) == {"markdown", "html", "json"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths.values())
    assert "<html" in paths["html"].read_text(encoding="utf-8")


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        run_research("   ")


def test_graph_compiles_and_exposes_its_nodes():
    graph = build_graph(build_deps(get_settings()))
    node_names = set(graph.get_graph().nodes)
    for expected in ("brief", "plan", "dispatch", "researcher", "critique", "synthesize", "verify", "render"):
        assert expected in node_names


# --------------------------------------------------------------------------- #
# unit-level checks on the tricky pure functions
# --------------------------------------------------------------------------- #
def test_fan_out_only_dispatches_pending_subquestions():
    state = {
        "plan": {
            "strategy": "",
            "subquestions": [
                {"id": "q1", "question": "a", "search_queries": ["x"]},
                {"id": "q2", "question": "b", "search_queries": ["y"]},
            ],
        },
        "pending": {"q2": ["y2"]},
        "round": 2,
        "brief": {"question": "q", "topic": "t"},
    }
    sends = fan_out(state)
    assert len(sends) == 1
    assert sends[0].arg["subquestion"]["id"] == "q2"
    assert sends[0].arg["queries"] == ["y2"]


def test_should_retry_routes_on_pending_work():
    assert should_retry({"pending": {"q1": ["more"]}}) == "dispatch"
    assert should_retry({"pending": {}}) == "synthesize"


def test_renumber_aliases_the_same_document_found_by_two_workers():
    """Two researchers retrieve one document under two provisional ids."""
    sources = [
        {"id": "S101", "title": "Same paper", "url": "https://x/1"},
        {"id": "S201", "title": "Same paper", "url": "https://x/1"},
        {"id": "S202", "title": "Other paper", "url": "https://x/2"},
    ]
    subanswers = [
        {"subquestion_id": "q1", "answer": "Alpha [S101].", "evidence": [{"claim": "a", "source_ids": ["S101"]}]},
        {"subquestion_id": "q2", "answer": "Beta [S201]. Gamma [S202].", "evidence": [{"claim": "b", "source_ids": ["S201"]}]},
    ]
    new_sources, new_answers, mapping = renumber_sources(sources, subanswers)

    assert [s["id"] for s in new_sources] == ["S1", "S2"]
    assert new_answers[0]["answer"] == "Alpha [S1]."
    # the duplicate was aliased onto S1, not renumbered into a second entry
    assert new_answers[1]["answer"] == "Beta [S1]. Gamma [S2]."
    assert new_answers[1]["evidence"][0]["source_ids"] == ["S1"]
    assert mapping["S101"] == "S1"


def test_renumber_keeps_uncited_sources_at_the_end():
    sources = [
        {"id": "S101", "title": "Cited", "url": "https://x/1"},
        {"id": "S102", "title": "Never cited", "url": "https://x/2"},
    ]
    subanswers = [{"subquestion_id": "q1", "answer": "Alpha [S101].", "evidence": []}]
    new_sources, _, _ = renumber_sources(sources, subanswers)
    assert [s["title"] for s in new_sources] == ["Cited", "Never cited"]


def test_renumber_drops_markers_for_sources_that_do_not_exist():
    subanswers = [{"subquestion_id": "q1", "answer": "Alpha [S999].", "evidence": []}]
    _, new_answers, _ = renumber_sources([], subanswers)
    assert "[S999]" not in new_answers[0]["answer"]
