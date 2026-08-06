# Part 4 — LLM Narrative Evaluation

Status: **not started**, blocked on Part 3 (`narrative/`).

Goal: quantify whether the generated narratives are accurate (claims match
the surrogate), useful (a physicist can act on them), and calibrated
(abstains when the surrogate residual is high or agreement with ParT is
low).

Candidate axes, to be refined once Part 3 exists:

- Faithfulness: do numeric/qualitative claims in the narrative match the
  surrogate's local explanation $\varphi_k(j)$ and residual $r(j)$?
- Coverage vs. abstention: rate of correct abstention on jets where the
  surrogate poorly tracks ParT.
- Human evaluation: do domain experts find the narrative informative
  compared to a raw shape-function plot?

See also Part 5 in `docs/paper.md` for the planned extension of this
evaluation to JetClass and to a RAG-augmented narrative pipeline.
