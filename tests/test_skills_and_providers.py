"""Skills load from disk, and the hosted provider degrades instead of dying."""

from __future__ import annotations

import httpx
import pytest

from muhaqqiq.config import get_settings
from muhaqqiq.llm import LLMError, OpenRouterReasoner, build_provider, parse_json_object
from muhaqqiq.offline import OfflineReasoner
from muhaqqiq.schemas import ResearchBrief
from muhaqqiq.skills import SkillLibrary


# --------------------------------------------------------------------------- #
# skills
# --------------------------------------------------------------------------- #
def test_all_bundled_skills_load(skills_dir):
    library = SkillLibrary.load(skills_dir)
    assert {"research-planning", "source-triage", "report-writing", "citation-audit"} <= set(
        library.names()
    )


def test_skills_are_bound_to_stages(skills_dir):
    library = SkillLibrary.load(skills_dir)
    assert [s.name for s in library.for_stage("verify")] == ["citation-audit"]
    assert "dangling citations" in library.instructions_for("verify").lower()


def test_unknown_stage_yields_no_instructions(skills_dir):
    assert SkillLibrary.load(skills_dir).instructions_for("nonexistent") == ""


def test_missing_skills_directory_is_survivable(tmp_path):
    assert len(SkillLibrary.load(tmp_path / "absent")) == 0


def test_skill_without_frontmatter_still_loads(tmp_path):
    folder = tmp_path / "bare"
    folder.mkdir()
    (folder / "SKILL.md").write_text("Just instructions.", encoding="utf-8")
    library = SkillLibrary.load(tmp_path)
    assert library.get("bare") is not None
    assert library.get("bare").stages == ()


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #
def test_offline_is_selected_when_no_key_is_present():
    assert build_provider(get_settings()).name == "offline"


def test_openrouter_without_a_key_degrades_to_offline(monkeypatch):
    monkeypatch.setenv("MUHAQQIQ_LLM_PROVIDER", "openrouter")
    from muhaqqiq import config

    config.reset_settings_cache()
    settings = config.get_settings()
    assert settings.effective_llm_provider == "offline"
    assert settings.degraded_reasons


def test_openrouter_stage_falls_back_when_the_api_fails(monkeypatch, skills_dir):
    """A provider outage must degrade the report, not destroy the run."""
    monkeypatch.setenv("MUHAQQIQ_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    from muhaqqiq import config

    config.reset_settings_cache()

    def explode(*args, **kwargs):
        raise httpx.ConnectError("no network")

    transport = httpx.MockTransport(explode)
    reasoner = OpenRouterReasoner(
        settings=config.get_settings(),
        skills=SkillLibrary.load(skills_dir),
        client=httpx.Client(transport=transport),
    )
    brief = reasoner.brief({"question": "What is an agentic system?", "depth": "quick"})
    assert isinstance(brief, ResearchBrief)
    assert brief.question == "What is an agentic system?"


def test_openrouter_parses_a_structured_response(monkeypatch, skills_dir):
    monkeypatch.setenv("MUHAQQIQ_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    from muhaqqiq import config

    config.reset_settings_cache()

    payload = {
        "choices": [
            {
                "message": {
                    "content": '{"question":"Q?","topic":"T","audience":"a","depth":"deep","language":"en"}'
                }
            }
        ]
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    reasoner = OpenRouterReasoner(
        settings=config.get_settings(),
        skills=SkillLibrary.load(skills_dir),
        client=httpx.Client(transport=transport),
    )
    brief = reasoner.brief({"question": "ignored", "depth": "quick"})
    assert brief.topic == "T"
    assert brief.depth.value == "deep"


@pytest.mark.parametrize(
    "raw",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here it is: {"a": 1} Hope that helps.'],
)
def test_json_is_recovered_from_chatty_model_output(raw):
    assert parse_json_object(raw) == {"a": 1}


def test_unparseable_output_raises():
    with pytest.raises(LLMError):
        parse_json_object("no json here")


# --------------------------------------------------------------------------- #
# offline reasoner
# --------------------------------------------------------------------------- #
def test_offline_reasoner_is_deterministic():
    reasoner = OfflineReasoner()
    payload = {"question": "How do agents fail in production?", "depth": "standard"}
    assert reasoner.brief(payload) == reasoner.brief(payload)


def test_offline_reasoner_detects_arabic_and_sets_the_language():
    brief = OfflineReasoner().brief({"question": "ما هي أنظمة الذكاء الاصطناعي الوكيلية؟"})
    assert brief.language == "ar"


def test_offline_reasoner_reports_honestly_when_it_finds_nothing():
    answer = OfflineReasoner().answer_subquestion(
        {"subquestion": {"id": "q1", "question": "anything at all"}, "sources": []}
    )
    assert answer.evidence == []
    assert answer.unresolved
    assert "[S" not in answer.answer


def test_plan_respects_the_subquestion_ceiling():
    reasoner = OfflineReasoner()
    brief = reasoner.brief({"question": "How do agents fail?", "depth": "deep"})
    plan = reasoner.plan({"brief": brief, "max_subquestions": 2})
    assert len(plan.subquestions) == 2
