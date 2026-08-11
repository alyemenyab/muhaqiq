---
name: research-planning
description: How to turn a vague user question into a bounded brief and a set of non-overlapping sub-questions.
stages: [brief, plan]
---

# Research planning

Your job is to convert an open question into work that can be executed in
parallel by researchers who cannot see each other's results.

## Writing the brief

1. Restate the question so that it is answerable. Remove pronouns, resolve
   implicit time references to explicit ones, and name the domain.
2. Decide the audience. If the user gave no signal, assume a technical reader
   who is competent but new to this specific topic.
3. Write scope boundaries. `scope_in` says what a good answer must cover;
   `scope_out` names the adjacent rabbit holes you are deliberately skipping.
4. Write 3–5 success criteria as testable statements. "Explains the trade-off
   between X and Y" is testable. "Is comprehensive" is not.

## Decomposing into sub-questions

- Produce between 3 and 5 sub-questions. Fewer than 3 and the report has no
  structure; more than 5 and the researchers duplicate each other's work.
- Sub-questions must be **orthogonal**. If two sub-questions would be answered
  by the same source paragraph, merge them.
- Cover, at minimum: what the thing *is*, what the *evidence* says, what the
  *alternatives* are, and what could go *wrong*.
- Every sub-question gets 2–3 concrete search queries. Write queries the way a
  librarian would, not the way a chatbot would: keywords and named entities,
  not full sentences.
- Never plan a sub-question you already know the answer to. The point of the
  round is retrieval, not recall.

## Anti-patterns

- Sub-questions that are really instructions ("summarise the above").
- Search queries that repeat the brief verbatim for every sub-question.
- Planning more facets than the requested depth allows.
