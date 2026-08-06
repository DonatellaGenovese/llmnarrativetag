# Part 3 — LLM Narrative

Status: **not started**.

Goal: turn a surrogate explanation (global shape functions, a local
per-jet breakdown $\varphi_k(j)$, and the residual $r(j)$ — see
`docs/paper.md`, Part 2) into a natural-language narrative describing why
the tagger classified a given jet the way it did.

Open questions to resolve here (currently `[TODO]` in `docs/paper.md`):

- Prompt structure: what is handed to the model — raw $\varphi_k(j)$ values,
  a ranked list of top-contributing observables, rendered shape-function
  plots, or some combination?
- Model choice: which LLM, and why (cost, context length, instruction
  following on numeric claims)?
- Grounding/verification: how each quantitative claim in the narrative is
  checked against the surrogate output before being kept, and what triggers
  an abstention (e.g. large residual $r(j)$, low surrogate–teacher decision
  agreement).

No code yet — this is a placeholder until Part 3 is designed.
