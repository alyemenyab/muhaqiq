"""The tool layer.

Tools are defined exactly once, here, as plain Python functions on a
`ToolRegistry`. They are then exposed through two different transports:

* in-process, which is what the graph uses by default (zero operational
  overhead, trivial to test);
* over the Model Context Protocol, via `mcp_server.py`, which imports this same
  registry and wraps each function as an MCP tool.

Keeping one definition and two transports is deliberate — it is the compromise
the MCP literature recommends for small systems: local development stays simple
while deployment keeps the tool layer decoupled from the agent.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import Settings, get_settings
from .offline import score_credibility
from .store import RunStore
from .textkit import STOPWORDS, root, stem, tokenize, truncate

log = logging.getLogger("muhaqqiq.tools")

TAVILY_URL = "https://api.tavily.com/search"


@dataclass
class Document:
    """A retrieved document, before it becomes a citable `Source`."""

    doc_id: str
    title: str
    url: str = ""
    publisher: str = ""
    published: str = ""
    content: str = ""
    credibility: str = "unknown"
    score: float = 0.0
    synthetic: bool = False
    query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "url": self.url,
            "publisher": self.publisher,
            "published": self.published,
            "content": self.content,
            "credibility": self.credibility,
            "score": round(self.score, 4),
            "synthetic": self.synthetic,
            "query": self.query,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        return cls(
            doc_id=data.get("doc_id", ""),
            title=data.get("title", ""),
            url=data.get("url", ""),
            publisher=data.get("publisher", ""),
            published=data.get("published", ""),
            content=data.get("content", ""),
            credibility=data.get("credibility", "unknown"),
            score=float(data.get("score", 0.0)),
            synthetic=bool(data.get("synthetic", False)),
            query=data.get("query", ""),
        )


# --------------------------------------------------------------------------- #
# Corpus (offline) backend
# --------------------------------------------------------------------------- #
class CorpusIndex:
    """A small BM25 index over the bundled JSON corpus.

    Inverse document frequency matters more here than the ranking function does.
    Without it, a query like "AI system agentic risks" is dominated by "system",
    a word that appears in every document in the corpus and therefore
    discriminates between none of them — which is how a report on agent safety
    ends up citing a paper about solar inverters. Weighting each term by how rare
    it is fixes the ranking at its source, rather than patching it downstream.
    """

    K1 = 1.5
    B = 0.75
    DOC_FREQ_CEILING = 0.7  # a topic word in >70% of the corpus identifies nothing

    def __init__(self, corpus_dir: Path | str) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.documents: list[Document] = []
        self._df: Counter[str] = Counter()
        self._root_df: Counter[str] = Counter()
        self._doc_terms: dict[str, Counter[str]] = {}
        self._doc_roots: dict[str, set[str]] = {}
        self._doc_len: dict[str, int] = {}
        self._avg_len: float = 1.0
        self._load()
        self._index()

    def _load(self) -> None:
        if not self.corpus_dir.is_dir():
            log.warning("corpus directory %s does not exist", self.corpus_dir)
            return
        for path in sorted(self.corpus_dir.glob("*.json")):
            try:
                records = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("skipping corpus file %s: %s", path, exc)
                continue
            for record in records:
                self.documents.append(
                    Document(
                        doc_id=record.get("doc_id") or record.get("title", "")[:40],
                        title=record.get("title", ""),
                        url=record.get("url", ""),
                        publisher=record.get("publisher", ""),
                        published=record.get("published", ""),
                        content=record.get("content", ""),
                        credibility=record.get("credibility", "unknown"),
                        synthetic=bool(record.get("synthetic", False)),
                    )
                )
        log.info("loaded %d corpus documents from %s", len(self.documents), self.corpus_dir)

    def _index(self) -> None:
        for doc in self.documents:
            tokens = tokenize(f"{doc.title} {doc.publisher} {doc.content}")
            terms = Counter(stem(t) for t in tokens)
            roots = {root(t) for t in tokens}
            self._doc_terms[doc.doc_id] = terms
            self._doc_roots[doc.doc_id] = roots
            self._doc_len[doc.doc_id] = sum(terms.values())
            for term in terms:
                self._df[term] += 1
            for r in roots:
                self._root_df[r] += 1
        if self._doc_len:
            self._avg_len = sum(self._doc_len.values()) / len(self._doc_len)

    def _idf(self, term: str, counts: Counter[str] | None = None) -> float:
        n = len(self.documents)
        df = (counts if counts is not None else self._df).get(term, 0)
        if n == 0 or df == 0:
            return 0.0
        # Robertson/Sparck-Jones IDF, floored at zero so a term present in every
        # document contributes nothing instead of contributing negatively.
        return max(0.0, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))

    def score(self, query: str, doc: Document) -> float:
        terms = [stem(t) for t in tokenize(query) if t not in STOPWORDS]
        if not terms:
            return 0.0
        tf = self._doc_terms.get(doc.doc_id, Counter())
        dl = self._doc_len.get(doc.doc_id, 1)
        title_terms = {stem(t) for t in tokenize(doc.title)}
        total = 0.0
        for term in set(terms):
            f = tf.get(term, 0)
            if not f:
                continue
            idf = self._idf(term)
            if idf <= 0:
                continue
            norm = f * (self.K1 + 1) / (f + self.K1 * (1 - self.B + self.B * dl / self._avg_len))
            weight = 1.4 if term in title_terms else 1.0  # a title hit is a strong signal
            total += idf * norm * weight
        return total / len(set(terms))

    def _eligible(self, context: str) -> list[Document]:
        """Documents that mention at least one *discriminative* topic term.

        A topic phrase mixes words that identify it ("agentic", "AI") with words
        that do not ("system", in a corpus of systems papers). A word appearing
        in most of the corpus identifies nothing and is dropped; a document
        matching none of the words that remain is not about this topic, however
        well it scores on a rare facet word like "outlook".

        Matching is on `root`, not `stem`, because the corpus says "agent" where
        the user says "agentic", and a filter that misses that connection
        excludes the entire corpus.
        """
        n = len(self.documents)
        if not n:
            return self.documents
        ceiling = self.DOC_FREQ_CEILING * n
        terms = {
            root(t)
            for t in tokenize(context)
            if t not in STOPWORDS and 0 < self._root_df.get(root(t), 0) <= ceiling
        }
        if not terms:
            return self.documents
        eligible = [
            doc for doc in self.documents if terms & self._doc_roots.get(doc.doc_id, set())
        ]
        return eligible or self.documents

    def search(self, query: str, limit: int = 5, context: str = "") -> list[Document]:
        """Rank documents for `query`, optionally biased toward a topic `context`.

        The sub-questions all search the same subject with different facet words
        appended ("… risks", "… outlook"). Those facet words are rare, so IDF
        gives them enormous weight, and a document from an unrelated domain whose
        title happens to contain one wins the ranking outright.

        The fix is to let the topic decide *eligibility* and the query decide
        *order*, rather than blending the two into one number. A document must be
        recognisably about the topic to be considered at all; among those that
        are, the facet word is what sorts them.
        """
        candidates = self._eligible(context) if context else self.documents

        scored: list[Document] = []
        for doc in candidates:
            value = self.score(query, doc)
            if value <= 0:
                continue
            hit = Document(**{**doc.__dict__})
            hit.score = value
            hit.query = query
            scored.append(hit)
        scored.sort(key=lambda d: (-d.score, d.doc_id))
        return scored[:limit]


# --------------------------------------------------------------------------- #
# Tavily (live) backend
# --------------------------------------------------------------------------- #
def tavily_search(
    query: str, api_key: str, limit: int = 5, timeout: float = 30.0
) -> list[Document]:
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": limit,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(TAVILY_URL, json=payload)
        response.raise_for_status()
        data = response.json()
    docs: list[Document] = []
    for item in data.get("results", []):
        url = item.get("url", "")
        docs.append(
            Document(
                doc_id=url or item.get("title", ""),
                title=item.get("title", ""),
                url=url,
                publisher=_domain(url),
                published=item.get("published_date", "") or "",
                content=item.get("content", "") or "",
                credibility=score_credibility(url).value,
                score=float(item.get("score", 0.0)),
                synthetic=False,
                query=query,
            )
        )
    return docs


def _domain(url: str) -> str:
    if "//" not in url:
        return url
    return url.split("//", 1)[1].split("/", 1)[0].removeprefix("www.")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class ToolRegistry:
    """The agent's entire action space. If it is not here, the agent cannot do it."""

    settings: Settings = field(default_factory=get_settings)
    store: RunStore | None = None
    corpus: CorpusIndex | None = None
    calls: int = 0
    call_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.corpus is None:
            self.corpus = CorpusIndex(self.settings.corpus_dir)

    # -- tool 1 ------------------------------------------------------------ #
    def web_search(self, query: str, limit: int = 5, context: str = "") -> list[dict[str, Any]]:
        """Search for documents relevant to `query`, within the topic `context`.

        Uses Tavily when a key is configured, otherwise the bundled local corpus.
        Results are cached so repeated queries inside a run are free.
        """
        provider = self.settings.effective_search_provider
        self.calls += 1
        cache_key = f"{context}||{query}" if context else query
        cached = self.store.cache_get(cache_key, provider) if self.store else None
        if cached is not None:
            self.call_log.append({"tool": "web_search", "query": query, "cached": True})
            return cached[:limit]

        if provider == "tavily" and self.settings.tavily_api_key:
            try:
                # A live engine has no `context` parameter; folding the topic into
                # the query text is the closest equivalent.
                docs = tavily_search(
                    f"{context} {query}".strip(), self.settings.tavily_api_key, limit=limit
                )
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("tavily search failed (%s); falling back to corpus", exc)
                docs = self.corpus.search(query, limit=limit, context=context) if self.corpus else []
                provider = "offline"
        else:
            docs = self.corpus.search(query, limit=limit, context=context) if self.corpus else []

        payload = [d.to_dict() for d in docs]
        if self.store:
            self.store.cache_put(cache_key, provider, payload)
        self.call_log.append(
            {"tool": "web_search", "query": query, "provider": provider, "hits": len(payload)}
        )
        return payload

    # -- tool 2 ------------------------------------------------------------ #
    def fetch_document(self, doc_id: str) -> dict[str, Any] | None:
        """Return the full text of a corpus document by id."""
        self.calls += 1
        self.call_log.append({"tool": "fetch_document", "doc_id": doc_id})
        if not self.corpus:
            return None
        for doc in self.corpus.documents:
            if doc.doc_id == doc_id or doc.url == doc_id:
                return doc.to_dict()
        return None

    # -- tool 3 ------------------------------------------------------------ #
    def fetch_url(self, url: str, max_chars: int = 6000) -> dict[str, Any]:
        """Fetch a URL and return its text.

        Only reachable when a live search provider is configured; in offline mode
        it resolves against the corpus so the tool contract stays the same.
        """
        self.calls += 1
        self.call_log.append({"tool": "fetch_url", "url": url})
        local = self.fetch_document(url)
        if local:
            return local
        if self.settings.effective_search_provider == "offline":
            return {"url": url, "error": "offline mode: only corpus documents are fetchable"}
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url, headers={"User-Agent": "muhaqqiq/0.1"})
                response.raise_for_status()
                text = _strip_html(response.text)
        except httpx.HTTPError as exc:
            return {"url": url, "error": str(exc)}
        return {
            "doc_id": url,
            "title": _domain(url),
            "url": url,
            "publisher": _domain(url),
            "content": truncate(text, max_chars),
            "credibility": score_credibility(url).value,
        }

    # -- tool 4 ------------------------------------------------------------ #
    def corpus_stats(self) -> dict[str, Any]:
        """Describe what the offline corpus actually contains."""
        self.calls += 1
        docs = self.corpus.documents if self.corpus else []
        return {
            "documents": len(docs),
            "publishers": sorted({d.publisher for d in docs if d.publisher}),
            "synthetic": all(d.synthetic for d in docs) if docs else False,
            "corpus_dir": str(self.settings.corpus_dir),
        }

    # -- introspection ------------------------------------------------------ #
    def specs(self) -> list[dict[str, str]]:
        return [
            {
                "name": "web_search",
                "description": "Search the configured provider (Tavily or local corpus) for documents.",
            },
            {"name": "fetch_document", "description": "Return a corpus document by id."},
            {"name": "fetch_url", "description": "Fetch and extract the text of a URL."},
            {"name": "corpus_stats", "description": "Describe the offline corpus."},
        ]


_TAG_RE = None


def _strip_html(html: str) -> str:
    import re

    global _TAG_RE
    if _TAG_RE is None:
        _TAG_RE = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.DOTALL | re.I)
    return " ".join(_TAG_RE.sub(" ", html).split())
