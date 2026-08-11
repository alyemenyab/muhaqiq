---
name: citation-audit
description: The acceptance test a report must pass before it is allowed to reach the user.
stages: [verify]
---

# Citation audit

This is the gate. A report that fails this audit is not shipped as-is; it is
shipped with its failure visible at the top.

## Checks

1. **Coverage** — split the report body into claim-sized sentences. What share
   carries at least one `[S#]` marker? Below the configured threshold
   (default 0.8) the verdict cannot be `pass`.
2. **Dangling citations** — every `[S#]` marker must resolve to a source that
   actually exists in the run's source list. A marker pointing at nothing is a
   fabrication and forces the verdict to `fail`.
3. **Unused sources** — sources retrieved but never cited. Not an error, but
   worth reporting: it usually means retrieval was noisy or the writer was lazy.
4. **Diversity** — count distinct cited sources. A report resting on one source
   is a summary of that source, not research. Warn below three.
5. **Credibility mix** — list every cited source rated `low`. If low-credibility
   sources are the only support for a key finding, warn.

## Verdicts

- `pass` — coverage at or above threshold, no dangling citations, three or more
  distinct sources.
- `pass_with_warnings` — coverage met, but diversity or credibility is thin.
- `fail` — coverage below threshold, or any dangling citation.

## Reporting

Warnings belong in the delivered document, not only in the logs. The reader
must be able to see how much they should trust what they are reading without
running the tool themselves.
