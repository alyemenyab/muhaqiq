"""The citation auditor.

This is the acceptance gate described in `skills/citation-audit/SKILL.md`, and it
is deliberately implemented in Python rather than delegated to a model: asking a
language model whether its own output is grounded reintroduces the failure the
check exists to catch. Regex and set arithmetic do not hallucinate.
"""

from __future__ import annotations

import re

from .schemas import (
    CITATION_RE,
    Credibility,
    Report,
    Source,
    Verdict,
    VerificationResult,
)
from .textkit import sentences, truncate

# Sentences that are structural rather than factual do not need a citation.
_NON_CLAIM_PATTERNS = re.compile(
    r"^(this report|this facet|the following|see also|in summary|table \d|figure \d|source[s]?:)",
    re.IGNORECASE,
)
MIN_DIVERSITY = 3


def is_claim(sentence: str) -> bool:
    """A claim is a substantive assertion, not a heading or a piece of scaffolding."""
    s = sentence.strip()
    if len(s) < 40:
        return False
    if _NON_CLAIM_PATTERNS.match(s):
        return False
    if s.endswith("?"):
        return False
    return True


def audit(
    report: Report, sources: list[Source], min_coverage: float = 0.8
) -> VerificationResult:
    known_ids = {s.id for s in sources}
    credibility = {s.id: s.credibility for s in sources}

    claims = [s for s in sentences(report.all_text()) if is_claim(s)]
    cited = [c for c in claims if CITATION_RE.search(c)]
    uncited = [c for c in claims if not CITATION_RE.search(c)]
    coverage = (len(cited) / len(claims)) if claims else 0.0

    used_ids = {f"S{n}" for n in CITATION_RE.findall(report.all_text())}
    dangling = sorted(used_ids - known_ids, key=_num)
    unused = sorted(known_ids - used_ids, key=_num)
    low_cred = sorted(
        (sid for sid in used_ids & known_ids if credibility.get(sid) == Credibility.LOW),
        key=_num,
    )

    notes: list[str] = []
    if dangling:
        notes.append(
            f"{len(dangling)} citation marker(s) point at sources that do not exist: "
            + ", ".join(dangling)
        )
    if coverage < min_coverage:
        notes.append(
            f"Citation coverage {coverage:.0%} is below the required {min_coverage:.0%}."
        )
    if len(used_ids & known_ids) < MIN_DIVERSITY:
        notes.append(
            f"Only {len(used_ids & known_ids)} distinct source(s) were cited; "
            f"{MIN_DIVERSITY} is the minimum for a report rather than a summary."
        )
    if unused:
        notes.append(f"{len(unused)} retrieved source(s) were never cited: " + ", ".join(unused))
    if low_cred:
        notes.append("Low-credibility sources were cited: " + ", ".join(low_cred))
    if not notes:
        notes.append("All checks passed.")

    if dangling or coverage < min_coverage:
        verdict = Verdict.FAIL
    elif len(used_ids & known_ids) < MIN_DIVERSITY or low_cred:
        verdict = Verdict.PASS_WITH_WARNINGS
    else:
        verdict = Verdict.PASS

    return VerificationResult(
        verdict=verdict,
        citation_coverage=round(coverage, 4),
        total_claims=len(claims),
        cited_claims=len(cited),
        uncited_claims=[truncate(c, 160) for c in uncited[:10]],
        dangling_citations=dangling,
        unused_sources=unused,
        source_diversity=len(used_ids & known_ids),
        low_credibility_sources=low_cred,
        notes=notes,
    )


def _num(source_id: str) -> int:
    digits = "".join(ch for ch in source_id if ch.isdigit())
    return int(digits) if digits else 0
