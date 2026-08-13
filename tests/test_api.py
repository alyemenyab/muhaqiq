"""The HTTP surface, including the OpenResponses-compatible endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from muhaqqiq.api import app

QUESTION = "How do multi-agent orchestration patterns compare, and what are their risks?"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_healthz_reports_the_effective_providers(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "offline"


def test_config_lists_the_loaded_skills(client):
    body = client.get("/v1/config").json()
    assert {s["name"] for s in body["skills"]} >= {"citation-audit", "report-writing"}


def test_tools_endpoint_describes_the_action_space(client):
    body = client.get("/v1/tools").json()
    assert {t["name"] for t in body["tools"]} >= {"web_search", "fetch_document"}
    assert body["corpus"]["synthetic"] is True


def test_graph_endpoint_returns_mermaid(client):
    text = client.get("/graph").text
    assert text.startswith("flowchart")
    assert "researcher" in text


def test_native_research_returns_the_full_typed_result(client):
    response = client.post("/v1/research", json={"question": QUESTION, "depth": "quick"})
    assert response.status_code == 200
    body = response.json()
    assert body["report"]["sections"]
    assert body["verification"]["verdict"] in {"pass", "pass_with_warnings"}
    assert body["meta"]["search_provider"] == "offline"


def test_research_rejects_a_question_that_is_too_short(client):
    assert client.post("/v1/research", json={"question": "hi"}).status_code == 422


def test_responses_endpoint_matches_the_openresponses_shape(client):
    response = client.post("/v1/responses", json={"input": QUESTION})
    assert response.status_code == 200
    body = response.json()

    assert body["object"] == "response"
    assert body["status"] == "completed"
    message = body["output"][0]
    assert message["type"] == "message" and message["role"] == "assistant"
    assert message["content"][0]["type"] == "output_text"
    assert body["output_text"].startswith("# ")
    assert body["metadata"]["verdict"] in {"pass", "pass_with_warnings"}
    assert message["content"][0]["annotations"], "sources should surface as annotations"


def test_responses_accepts_the_message_list_form(client):
    response = client.post(
        "/v1/responses",
        json={"input": [{"role": "user", "content": [{"type": "input_text", "text": QUESTION}]}]},
    )
    assert response.status_code == 200
    assert response.json()["output_text"]


def test_responses_rejects_empty_input(client):
    assert client.post("/v1/responses", json={"input": "hi"}).status_code == 422


def test_a_completed_run_is_retrievable_in_every_format(client):
    run_id = client.post("/v1/research", json={"question": QUESTION}).json()["meta"]["run_id"]

    assert client.get(f"/v1/runs/{run_id}").status_code == 200
    assert client.get(f"/v1/runs/{run_id}/report.md").text.startswith("# ")
    assert "<html" in client.get(f"/v1/runs/{run_id}/report.html").text
    assert any(r["run_id"] == run_id for r in client.get("/v1/runs").json()["runs"])


def test_unknown_run_is_a_404(client):
    assert client.get("/v1/runs/run_missing").status_code == 404


def test_background_mode_returns_a_run_id_immediately(client):
    body = client.post("/v1/research?background=true", json={"question": QUESTION}).json()
    assert body["status"] == "running"
    # TestClient drains background tasks before returning, so the run is stored by now.
    assert client.get(f"/v1/runs/{body['run_id']}").status_code == 200
