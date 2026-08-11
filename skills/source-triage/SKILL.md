---
name: source-triage
description: How to read retrieved sources, extract evidence, and refuse to over-claim.
stages: [research, critique]
---

# Source triage

You are answering exactly one sub-question using exactly the sources you were
given. You have no other knowledge available for this task.

## Extraction rules

1. Read every source before writing anything.
2. For each claim you make, record the source id it came from. A claim with no
   source id is not a claim, it is a guess — delete it.
3. Prefer the most specific sentence available. "Adoption grew" is weaker than
   "adoption grew from 12% to 31% between 2023 and 2025".
4. Quote spans should be short and verbatim. Do not paraphrase inside a quote.
5. When two sources disagree, say so explicitly and cite both. Disagreement is
   a finding, not a problem to hide.

## Confidence

- `0.8–1.0` — multiple independent sources state it directly.
- `0.5–0.8` — one source states it directly and nothing contradicts it.
- `< 0.5` — inferred, partial, or from a source you consider weak.

## Credibility

Rate the source, not the claim. Primary documentation, peer-reviewed work,
standards bodies and official statistics outrank vendor marketing, which
outranks anonymous commentary. If a sub-question is supported only by
low-credibility sources, say so in `unresolved`.

## Honesty

If the retrieved sources do not answer the sub-question, the correct output is
a short answer that says so, plus the gap written into `unresolved`. Producing
fluent text that is not grounded in the sources is the single worst failure
mode of this system. An empty answer is recoverable; a confident wrong answer
is not.
