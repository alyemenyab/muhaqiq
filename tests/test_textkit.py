"""The text layer is load-bearing: the auditor's verdict depends on it."""

from __future__ import annotations

import pytest

from muhaqqiq.textkit import (
    cited_segments,
    dedupe_preserving_order,
    keywords,
    root,
    sentences,
    stem,
    strip_citations,
    tokenize,
    truncate,
)


def test_tokenize_splits_hyphenated_words():
    assert "multi" in tokenize("multi-agent systems")
    assert "agent" in tokenize("multi-agent systems")


def test_stem_merges_plurals_but_not_unrelated_words():
    assert stem("systems") == stem("system")
    assert stem("policies") == "policy"
    assert stem("access") == "access"  # -ss must survive


def test_root_connects_morphological_variants():
    assert root("agentic") == root("agents") == "agent"
    assert root("evaluation").startswith("evalu")


def test_keywords_preserve_acronyms_and_reading_order():
    result = keywords("What makes an AI system agentic and safe?", limit=4)
    assert "AI" in result  # casing preserved, not lowercased
    assert result.index("AI") < result.index("agentic")  # order of appearance


def test_keywords_ignores_stopwords():
    assert keywords("the and of with") == []


def test_sentences_keeps_citation_with_its_sentence():
    text = "Agents fail on tool arguments [S1]. Retrieval reduces this [S2]."
    parts = sentences(text)
    assert len(parts) == 2
    assert all("[S" in p for p in parts)


def test_sentences_drops_markdown_scaffolding():
    assert sentences("## Heading\n\n- short") == []


def test_cited_segments_splits_on_markers():
    text = "First claim [S1]. Second claim [S2]. Third claim [S3]."
    segments = cited_segments(text)
    assert len(segments) == 3
    assert segments[1].endswith("[S2].")


def test_cited_segments_on_uncited_text_is_empty():
    assert cited_segments("no markers at all here") == []


def test_strip_citations_removes_markers_only():
    assert strip_citations("A claim [S12] stands.") == "A claim stands."


@pytest.mark.parametrize(
    "text,limit",
    [("a" * 100, 20), ("short", 20)],
)
def test_truncate_never_exceeds_limit(text, limit):
    assert len(truncate(text, limit)) <= limit


def test_dedupe_is_case_and_whitespace_insensitive():
    assert dedupe_preserving_order(["One  claim", "one claim", "Two"]) == ["One  claim", "Two"]
