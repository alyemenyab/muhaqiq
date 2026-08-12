"""Report rendering.

Markdown is the source of truth; HTML is generated from it. Both carry the same
three things a reader needs in order to calibrate trust: the verdict banner, the
numbered source list, and the provenance note when the run used the synthetic
offline corpus.
"""

from __future__ import annotations

import html as html_lib
from typing import Any

from .config import Settings, get_settings
from .schemas import (
    GapReport,
    Report,
    ResearchBrief,
    ResearchPlan,
    Source,
    Verdict,
    VerificationResult,
)

VERDICT_BANNER = {
    Verdict.PASS: ("✅", "Passed the citation audit"),
    Verdict.PASS_WITH_WARNINGS: ("⚠️", "Passed with warnings"),
    Verdict.FAIL: ("❌", "Failed the citation audit — read with caution"),
}


def render_markdown(state: dict[str, Any], settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    brief = ResearchBrief.model_validate(state["brief"])
    plan = ResearchPlan.model_validate(state.get("plan", {"subquestions": []}))
    report = Report.model_validate(state["report"])
    verification = VerificationResult.model_validate(state.get("verification", {}))
    sources = [Source.model_validate(s) for s in state.get("final_sources", [])]
    gap_reports = [GapReport.model_validate(g) for g in state.get("gap_reports", [])]

    icon, label = VERDICT_BANNER[verification.verdict]
    lines: list[str] = []

    lines.append(f"# {report.title}")
    lines.append("")
    lines.append(f"> **{icon} {label}.** Citation coverage "
                 f"**{verification.citation_coverage:.0%}** "
                 f"({verification.cited_claims}/{verification.total_claims} claims cited) · "
                 f"**{verification.source_diversity}** distinct sources cited · "
                 f"{len(state.get('gap_reports', []))} critique round(s).")
    if any(s.title and _is_synthetic(s) for s in sources):
        lines.append(">")
        lines.append("> ⚠️ **Provenance:** this run used the bundled *synthetic* demo corpus. "
                     "The citations below are internally consistent but the documents are "
                     "fictional. Configure a live search provider for real-world research.")
    lines.append("")

    lines.append("## Research question")
    lines.append("")
    lines.append(brief.question)
    lines.append("")

    if report.executive_summary:
        lines.append("## Executive summary")
        lines.append("")
        lines.append(report.executive_summary)
        lines.append("")

    if report.key_findings:
        lines.append("## Key findings")
        lines.append("")
        lines.extend(f"- {finding}" for finding in report.key_findings)
        lines.append("")

    for section in report.sections:
        lines.append(f"## {section.heading}")
        lines.append("")
        lines.append(section.body)
        lines.append("")

    if report.open_questions:
        lines.append("## Open questions")
        lines.append("")
        lines.extend(f"- {q}" for q in report.open_questions)
        lines.append("")

    lines.append("## Sources")
    lines.append("")
    if sources:
        for source in sources:
            bits = [f"**[{source.id}]** {source.title}"]
            meta = " · ".join(
                x for x in (source.publisher, source.published, f"credibility: {source.credibility.value}") if x
            )
            if meta:
                bits.append(f"  \n  {meta}")
            if source.url:
                bits.append(f"  \n  <{source.url}>")
            lines.append("- " + "".join(bits))
    else:
        lines.append("_No sources were retrieved._")
    lines.append("")

    lines.append("## Audit")
    lines.append("")
    lines.append(f"- **Verdict:** `{verification.verdict.value}`")
    lines.append(f"- **Citation coverage:** {verification.citation_coverage:.0%} "
                 f"(threshold {settings.min_citation_coverage:.0%})")
    lines.append(f"- **Distinct sources cited:** {verification.source_diversity}")
    for note in verification.notes:
        lines.append(f"- {note}")
    if verification.uncited_claims:
        lines.append("")
        lines.append("<details><summary>Claims without a citation</summary>")
        lines.append("")
        lines.extend(f"- {claim}" for claim in verification.uncited_claims)
        lines.append("")
        lines.append("</details>")
    lines.append("")

    lines.append("## How this report was produced")
    lines.append("")
    lines.append(f"**Plan.** {plan.strategy}")
    lines.append("")
    for sub in plan.subquestions:
        lines.append(f"- `{sub.id}` — {sub.question}")
    lines.append("")
    for index, gaps in enumerate(gap_reports, start=1):
        status = "sufficient" if gaps.sufficient else "gaps found"
        lines.append(f"**Critique round {index}.** {status}. {gaps.notes}")
        lines.extend(f"  - {gap}" for gap in gaps.gaps)
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_Generated by **Muhaqqiq** — AAASE capstone, SDAIA Academy._")
    return "\n".join(lines).rstrip() + "\n"


def _is_synthetic(source: Source) -> bool:
    return "example.org" in (source.url or "")


def render_html(markdown_text: str, title: str = "Muhaqqiq report") -> str:
    """Self-contained HTML. No CDN, no build step, prints cleanly."""
    try:
        from markdown_it import MarkdownIt

        body = MarkdownIt("commonmark", {"html": True}).enable("table").render(markdown_text)
    except Exception:  # pragma: no cover - markdown-it is a declared dependency
        body = "<pre>" + html_lib.escape(markdown_text) + "</pre>"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; --fg:#16181d; --bg:#fbfbfa; --muted:#5f6672;
           --rule:#e3e4e8; --accent:#2f6f4f; --code:#f2f2f0; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e6e7ea; --bg:#15171b; --muted:#9aa1ad; --rule:#2a2d34;
             --accent:#7fc9a0; --code:#1e2127; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
          font:16px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif; }}
  main {{ max-width: 46rem; margin: 0 auto; padding: 3rem 1.25rem 5rem; }}
  h1 {{ font-size: 1.9rem; line-height:1.25; margin: 0 0 1rem; letter-spacing:-.01em; }}
  h2 {{ font-size: 1.15rem; margin: 2.4rem 0 .6rem; padding-bottom:.35rem;
        border-bottom:1px solid var(--rule); }}
  blockquote {{ margin:1.5rem 0; padding:.9rem 1.1rem; border-left:3px solid var(--accent);
                background:var(--code); border-radius:0 6px 6px 0; color:var(--muted); }}
  blockquote strong {{ color:var(--fg); }}
  code {{ background:var(--code); padding:.12em .38em; border-radius:4px; font-size:.88em; }}
  a {{ color:var(--accent); }}
  li {{ margin:.35rem 0; }}
  details {{ margin:1rem 0; }}
  summary {{ cursor:pointer; color:var(--muted); }}
  hr {{ border:0; border-top:1px solid var(--rule); margin:2.5rem 0 1.5rem; }}
  em {{ color:var(--muted); }}
</style>
</head>
<body><main>
{body}
</main></body>
</html>
"""
