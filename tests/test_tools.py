"""Retrieval quality decides report quality; everything downstream just quotes it."""

from __future__ import annotations

from muhaqqiq.config import get_settings
from muhaqqiq.store import RunStore
from muhaqqiq.tools import CorpusIndex, Document, ToolRegistry


def test_corpus_loads_every_document(corpus_dir):
    index = CorpusIndex(corpus_dir)
    assert len(index.documents) >= 15
    assert all(doc.synthetic for doc in index.documents)


def test_missing_corpus_directory_is_not_fatal(tmp_path):
    index = CorpusIndex(tmp_path / "nope")
    assert index.documents == []
    assert index.search("anything") == []


def test_ranking_puts_the_on_topic_document_first(corpus_dir):
    index = CorpusIndex(corpus_dir)
    hits = index.search("prompt injection risk surface of agent deployments", limit=3)
    assert hits[0].doc_id == "agent-risks"


def test_topic_context_excludes_documents_from_another_domain(corpus_dir):
    """The regression this whole mechanism exists for.

    'outlook' is a rare word, so IDF gives it huge weight, and there is an
    unrelated energy paper with 'Outlook' in its title. Without the topic
    context that paper wins a search about agent engineering.
    """
    index = CorpusIndex(corpus_dir)

    unfiltered = index.search("agentic AI outlook future", limit=3)
    filtered = index.search("agentic AI outlook future", limit=3, context="agentic AI systems")

    assert any(h.doc_id.startswith("solar") for h in unfiltered)
    assert not any(h.doc_id.startswith("solar") for h in filtered)


def test_topic_context_survives_morphology(corpus_dir):
    """The user says 'agentic'; the corpus mostly says 'agent'."""
    index = CorpusIndex(corpus_dir)
    eligible = index._eligible("AI system agentic")
    assert len(eligible) > 5
    assert not any(doc.doc_id.startswith("solar") for doc in eligible)


def test_ubiquitous_topic_words_do_not_make_everything_eligible(corpus_dir):
    """'system' appears in most documents, so it must not confer eligibility."""
    index = CorpusIndex(corpus_dir)
    assert len(index._eligible("system")) == len(index.documents)


def test_registry_counts_and_logs_tool_calls():
    registry = ToolRegistry(settings=get_settings(), store=RunStore(":memory:"))
    registry.web_search("agent evaluation metrics", limit=2)
    registry.corpus_stats()
    assert registry.calls == 2
    assert registry.call_log[0]["tool"] == "web_search"


def test_repeated_queries_are_served_from_cache():
    registry = ToolRegistry(settings=get_settings(), store=RunStore(":memory:"))
    first = registry.web_search("agent evaluation metrics", limit=3)
    second = registry.web_search("agent evaluation metrics", limit=3)
    assert [d["doc_id"] for d in first] == [d["doc_id"] for d in second]
    assert registry.call_log[-1].get("cached") is True


def test_fetch_document_round_trips():
    registry = ToolRegistry(settings=get_settings(), store=RunStore(":memory:"))
    doc = registry.fetch_document("agent-risks")
    assert doc is not None and "prompt injection" in doc["content"].lower()
    assert registry.fetch_document("does-not-exist") is None


def test_fetch_url_refuses_the_network_in_offline_mode():
    registry = ToolRegistry(settings=get_settings(), store=RunStore(":memory:"))
    result = registry.fetch_url("https://example.com/not-in-corpus")
    assert "error" in result


def test_document_dict_round_trip():
    doc = Document(doc_id="d", title="t", url="u", content="c", score=0.5)
    assert Document.from_dict(doc.to_dict()) == doc
