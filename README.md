# llmnarrativetag

**A Verified Natural Language Explanation of Jet Tagger Behavior**

Deep jet taggers (ParticleNet, Particle Transformer) read an unordered set of
particle four-vectors, so attributions on individual constituents are not
physically meaningful. This project re-expresses the tagger's decision in a
space of physical substructure observables via an interpretable surrogate
model, then turns that surrogate explanation into natural language with an
LLM — with narrative claims checked against the surrogate itself, and the
system abstaining where the surrogate does not fit.

The full write-up, observable definitions, references, and open questions
live in [`docs/paper.md`](docs/paper.md).

## Structure

```
docs/                 Project write-up (docs/paper.md) and future notes
data/
  observables/        Observable definitions and feature-building code (Part 1)
  configs/            Observable-basis configs + weaver dataloader configs
surrogate/            GAM/EBM surrogate training, evaluation, audit (Part 2)
  scripts/            Shell entry points for each surrogate run
  metrics/            Lightweight JSON/TXT metrics from past runs (no model binaries)
narrative/            LLM narrative generation (Part 3) — scaffold, work in progress
evaluation/           Narrative evaluation against the surrogate (Part 4) — scaffold
scripts/              Teacher (ParT) inference / logit-dumping utilities
particle_transformer/ Vendored ParT training toolkit + teacher logits (see below)
```

## Relationship to the ParT codebase

The ParticleNet / Particle Transformer teacher models are trained with the
upstream [jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer)
toolkit (network definitions, weaver dataloaders, training loop). This repo
vendors the parts of that toolkit needed to reproduce training/inference —
`particle_transformer/` (network defs, dataloader, weaver configs, train
scripts) — copied from the upstream repo at commit `2925bdb`. See
[`particle_transformer/README.md`](particle_transformer/README.md) for what
is/isn't included and why.

Not part of this repository, in any form: the raw datasets themselves
(TopLandscape, QuarkGluon, JetClass), the trained model checkpoints
(`.pt`/`.pth`), and the ParT teacher logits already computed on
TopLandscape/QuarkGluon (`t = logit_diff`, the values the surrogate is fit
to reproduce) — these are large binary artifacts, kept local for now and
destined for external hosting (TBD) rather than the git repo.

## Status

- **Part 1 — Dataset & Observables**: done, see `data/observables/`.
- **Part 2 — XAI / surrogate (GAM/EBM)**: done, see `surrogate/`.
- **Part 3 — LLM narrative**: not started, see `narrative/README.md`.
- **Part 4 — Narrative evaluation**: not started, see `evaluation/README.md`.
- **Part 5 — Additional challenges** (JetClass, RAG): future work.
