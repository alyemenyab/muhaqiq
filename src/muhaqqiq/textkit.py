"""Small, dependency-free text utilities.

These power two things: the offline reasoning provider (so the agent can run with
no API keys at all) and the citation verifier (which has to split a report into
individual claims before it can check whether each one is attributed).
"""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into",
    "is", "it", "its", "may", "might", "more", "most", "must", "no", "not", "of", "on", "or",
    "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "to", "up", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "why", "will", "with", "would", "you", "your", "about", "also",
    "between", "during", "over", "under", "via", "vs", "using", "used", "use", "make", "made",
    "makes", "making", "really", "actually", "given", "doing", "done", "many", "much",
    "well", "get", "got", "let", "one", "two", "new", "old", "own", "any", "all", "each",
    "main", "usually", "typically", "generally", "currently", "today", "need", "want",
}

# Hyphens split: a corpus that writes "multi-agent" should still match a
# query for "agent", and "solar-dominant" should still match "solar".
_WORD_RE = re.compile(r"[A-Za-z؀-ۿ][A-Za-z0-9؀-ۿ\+\.]*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?؟])\s+(?=[A-Z؀-ۿ\"'\(])|\n{2,}")


def tokenize(text: str) -> list[str]:
    return [w.lower().strip(".-+") for w in _WORD_RE.findall(text or "") if w.strip(".-+")]


def stem(word: str) -> str:
    """Crude singulariser so 'system' and 'systems' are counted as one term."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and not word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


_ROOT_SUFFIXES = ("ically", "ical", "ation", "ition", "ness", "ity", "ing", "ive", "ed", "al", "ic")


def root(word: str) -> str:
    """A blunter stemmer than `stem`, used for topic matching only.

    `stem` exists to make counting sane ("system"/"systems"). `root` exists to
    make *recall* sane: a corpus about agents mostly writes "agent" and "agents",
    while the user writes "agentic", and a topic filter that misses that
    connection filters out the entire corpus. It over-stems by design, which is
    why ranking still uses `stem`.
    """
    w = stem(word.lower())
    for suffix in _ROOT_SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[: -len(suffix)]
    return w


def keywords(text: str, limit: int = 12) -> list[str]:
    """Frequency-ranked content words, returned in the order they first appear.

    Surface casing is preserved so acronyms survive ("AI", not "ai"), and the
    result reads as a phrase rather than as a bag of words.
    """
    surface: dict[str, str] = {}
    first_seen: dict[str, int] = {}
    counts: Counter[str] = Counter()
    for position, raw in enumerate(_WORD_RE.findall(text or "")):
        token = raw.lower().strip(".-+")
        # Two-letter words are noise unless they are acronyms ("AI", "ML", "IT").
        if not token or token in STOPWORDS or (len(token) <= 2 and not raw.isupper()):
            continue
        key = stem(token)
        counts[key] += 1
        if key not in first_seen:
            first_seen[key] = position
            surface[key] = raw
        elif raw.isupper() and not surface[key].isupper():
            surface[key] = raw
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    chosen = [k for k, _ in ranked[:limit]]
    chosen.sort(key=lambda k: first_seen[k])
    return [surface[k] for k in chosen]


def sentences(text: str, min_len: int = 25) -> list[str]:
    """Split prose into sentence-ish chunks, dropping markdown scaffolding."""
    cleaned = re.sub(r"^\s*(#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s?)", "", text or "", flags=re.MULTILINE)
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
    out: list[str] = []
    for block in _SENT_SPLIT_RE.split(cleaned):
        s = " ".join(block.split())
        if len(s) >= min_len:
            out.append(s)
    return out


def overlap_score(query: str, text: str) -> float:
    """Cosine-ish similarity over content-word sets, in [0, 1]."""
    q = {t for t in tokenize(query) if t not in STOPWORDS}
    d = {t for t in tokenize(text) if t not in STOPWORDS}
    if not q or not d:
        return 0.0
    inter = len(q & d)
    if not inter:
        return 0.0
    return inter / math.sqrt(len(q) * len(d))


def bm25ish(query: str, text: str, avg_len: float = 120.0) -> float:
    """Cheap length-normalised term-frequency score used to rank corpus hits."""
    q_terms = [t for t in tokenize(query) if t not in STOPWORDS]
    if not q_terms:
        return 0.0
    d_terms = tokenize(text)
    if not d_terms:
        return 0.0
    tf = Counter(d_terms)
    dl = len(d_terms)
    k1, b = 1.5, 0.75
    score = 0.0
    for term in set(q_terms):
        f = tf.get(term, 0)
        if not f:
            continue
        score += (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avg_len, 1.0)))
    return score / len(set(q_terms))


def truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rsplit(" ", 1)[0] + "…"


def titlecase(text: str) -> str:
    small = {"a", "an", "and", "the", "of", "for", "in", "on", "to", "with", "vs"}
    words = (text or "").split()
    out = []
    for i, w in enumerate(words):
        out.append(w if w.isupper() else (w.capitalize() if i == 0 or w.lower() not in small else w.lower()))
    return " ".join(out)


_CITED_SEGMENT_RE = re.compile(r".*?\[S\d+\][^\[]*?(?=(?:\s|$))", re.DOTALL)


def cited_segments(text: str) -> list[str]:
    """Split text into chunks that each end with (or contain) a citation marker.

    Used to de-duplicate evidence across report sections without stripping the
    citation off the sentence it belongs to.
    """
    segments = [" ".join(m.group(0).split()) for m in _CITED_SEGMENT_RE.finditer(text or "")]
    return [s for s in segments if s]


def strip_citations(text: str) -> str:
    return " ".join(re.sub(r"\[S\d+\]", " ", text or "").split())


def dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = " ".join(item.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
