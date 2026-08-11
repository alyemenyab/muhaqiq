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
from .textkit import STOPWORDS, stem, tokenize, truncate

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
    discriminates between none of them.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, corpus_dir: Path | str) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.documents: list[Document] = []
        self._df: Counter[str] = Counter()
        self._doc_terms: dict[str, Counter[str]] = {}
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
            self._doc_terms[doc.doc_id] = terms
            self._doc_len[doc.doc_id] = sum(terms.values())
            for term in terms:
                self._df[term] += 1
        if self._doc_len:
            self._avg_len = sum(self._doc_len.values()) / len(self._doc_len)

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self._df.get(term, 0)
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

    def search(self, query: str, limit: int = 5) -> list[Document]:
        scored: list[Document] = []
        for doc in self.documents:
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
    def web_search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for documents relevant to `query`.

        Uses Tavily when a key is configured, otherwise the bundled local corpus.
        Results are cached so repeated queries inside a run are free.
        """
        provider = self.settings.effective_search_provider
        self.calls += 1
        cached = self.store.cache_get(query, provider) if self.store else None
        if cached is not None:
            self.call_log.append({"tool": "web_search", "query": query, "cached": True})
            return cached[:limit]

        if provider == "tavily" and self.settings.tavily_api_key:
            try:
                docs = tavily_search(query, self.settings.tavily_api_key, limit=limit)
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("tavily search failed (%s); falling back to corpus", exc)
                docs = self.corpus.search(query, limit=limit) if self.corpus else []
                provider = "offline"
        else:
            docs = self.corpus.search(query, limit=limit) if self.corpus else []

        payload = [d.to_dict() for d in docs]
        if self.store:
            self.store.cache_put(query, provider, payload)
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
