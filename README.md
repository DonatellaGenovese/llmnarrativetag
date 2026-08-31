# llmnarrativetag

**A verified natural-language explanation of jet tagger behaviour.**

Deep jet taggers (ParticleNet, Particle Transformer) read an unordered set of
particle four-vectors, so an attribution assigned to an individual constituent
carries no standalone physical meaning. This project re-expresses the tagger's
decision in a space of physical substructure observables via an interpretable
surrogate, then turns that surrogate explanation into natural language with an
LLM — with every quantitative claim checked against the surrogate before it is
kept, and no narrative produced at all where the surrogate does not track the
tagger.

Dataset: **TopLandscape** (top vs QCD), anti-kT R = 0.8, pT ∈ [550, 650] GeV.
Teacher: **ParT-FineTune**, target `t = z_top − z_qcd`.

```
data/observables/     Observable definitions and feature building     (Part 1)
data/configs/         Observable basis + weaver dataloader configs
surrogate/            Ridge / EBM / GBDT surrogate distillation       (Part 2)
  scripts/            Shell entry points
  metrics/top/        JSON/TXT numbers from the official run (tracked)
narrative/            LLM narrative generation and verification       (Part 3)
evaluation/           LLM-as-a-judge + human annotation set           (Part 4)
scripts/              ParT teacher logit dumping (train / val / test)
particle_transformer/ Vendored ParT training toolkit (see the last section)
```

**Where to run what.** Parts 3 and 4 run from the repo root. Parts 1 and 2 run
from **`particle_transformer/`**, because that is where the datasets, the
teacher checkpoints and the weaver configs live: the symlinks under
`particle_transformer/surrogate/` assemble this repo's real code into the
package layout the ParT scripts expect, and `surrogate.build_features_top` and
`surrogate.fjcontrib_loader` resolve there and nowhere else. Everything assumes
`conda activate part-surrogate`, except the ParT inference steps, which use the
torch venv (`source .venv/bin/activate`).

---

## Part 1 — Observables

Ten observables, pre-registered in
[`data/observables/basis_definitions.md`](data/observables/basis_definitions.md)
before any fitting: `m`, `m_SD`, `tau21`, `tau32`, `C2_double_b1`,
`D2_double_b1`, `C3_double_b1`, `N3_b1`, `pT`, `n_const`. SoftDrop at
z_cut = 0.1, β = 0; N-subjettiness at β = 1 with OnePass_KT axes; energy
correlators at β = 1, with `C3` truncated to the leading 50 constituents and
`N3` to 40. All computed with fastjet-contrib through cppyy.

The names `C2_double_b1` / `C3_double_b1` are deliberate: they avoid the C₂
naming collision with the quark/gluon tagging literature.

Observables are computed from the **original parquet four-vectors**, not from
the weaver-padded `pf_*` arrays.

## Part 2 — Surrogate

ParT is the teacher; an interpretable student is fit to reproduce its logit
difference `t` rather than the truth label. Three rungs on the same observable
basis:

| model | role | test R² | Spearman | agreement with ParT |
|---|---|---|---|---|
| Ridge (log + z) | additive-linear lower bound | 0.886 | 0.938 | 0.932 |
| **EBM, 0 interactions** | **the interpretable surrogate** | **0.962** | **0.972** | **0.951** |
| GBDT (unconstrained) | upper bound, not interpretable | 0.980 | 0.984 | 0.967 |

Fit on train (1.21 M jets), selected on val (403 k), reported on test (404 k).
The Ridge↔EBM gap is what the linear form misses; the EBM↔GBDT gap is how much
of ParT's behaviour needs non-additivity. The EBM is the one the narrative is
built on: with zero interactions it is exactly decomposable, so each shape
function `f_k` is a function of one variable and can be plotted in full.

The explanation of a single jet is the vector of contributions
`φ_k(j) = f_k(x_k(j))` with the intercept `β₀`, satisfying
`β₀ + Σ φ_k(j) = ĝ(x(j))`, plus the residual `r(j) = z(j) − ĝ(x(j))` — the part
of the decision the observable basis does not express.

**Sanity check against the label.** The same three models fit on truth labels
instead of on `t` gain only +0.4 points of accuracy for the EBM
(0.922 vs 0.918, against ParT's 0.935), so the surrogate is imitating ParT
rather than relearning the physics from scratch.

```bash
cd particle_transformer            # see "Where to run what" above

# 1) ParT teacher logits (torch venv)
source .venv/bin/activate
./dump_toplandscape_train_val_logits.sh     # val then train; test already exists

# 2) features + Ridge/EBM/GBDT
conda activate part-surrogate
./surrogate/run_official_top.sh             # full;  add 5000 for a smoke run

# 3) classifier metrics
python -m surrogate.eval_classifier_metrics \
  --predictions surrogate/outputs/top/models/test_predictions.parquet

# 4) label-trained comparison
./surrogate/run_label_classifiers.sh
```

Feature tables and fitted models land in `surrogate/outputs/top/` and stay
local. The JSON/TXT numbers they produce are copied into
[`surrogate/metrics/top/`](surrogate/metrics/top/) and are tracked.

## Part 3 — Narrative

Turns the local decomposition into a natural-language account of why ParT
classified a jet as it did.

```
narrative/
  configs/narrative_top.yaml   paths, artefact shape, abstention thresholds, LLM settings
  configs/glossary_top.yaml    observable → physical reading (all entries reviewed)
  prompts/prompt_top.yaml      versioned system + task template, currently `top-v16`
  stats.py                     reference percentiles, frozen once from the test set
  artefact.py                  per-jet artefact: orientation, ranking, rounding, regime
  llm_client.py                one client for Gemini and the OpenAI-compatible providers
  generator.py                 glossary rendering, prompt assembly, prompt hashing
  verifier.py                  tag extraction, equality, completeness, bare-number sieve
  orchestrator.py              abstention gate → generate → verify → retry
  blocks_report.py             the pass-rate table, re-verifying every narrative
  outputs/top/                 reference stats and narratives (local, not tracked)
```

Data flows one way — `stats.py` → `artefact.py` → `generator.py` →
`llm_client.py` → `verifier.py` — with `orchestrator.py` driving and
`blocks_report.py` reading what it wrote. `artefact.py` and `verifier.py` can be
exercised with no API key.

### Selecting the jets

**Every model must narrate the same jets**, or half of any difference between
them is sampling noise. The sample is drawn once into a file and reused:

```bash
python -m narrative.orchestrator \
    --jets-file narrative/outputs/top/narratives/run100_jets.json \
    --sample 100 --sample-seed 2 --model gemini-3.5-flash-lite
```

If the file exists it is loaded verbatim and `--sample` / `--sample-seed` are
ignored, so every later run sees an identical set. `--jets 11133 14079` takes
explicit row indices; `--n 5` takes the *first* five rows and is for smoke
tests only — the TopLandscape test file is ordered in blocks of up to 100
identical labels, so its opening rows are not a sample of anything (the first
eight are all QCD, and a run on them scored four times higher than the same
model on a random sample of the same size).

Of any 100 jets, roughly 10 fall in the abstention regime and are never sent to
a model, so a run of 100 yields about 90 narratives.

### Running

```bash
python -m narrative.stats                   # once, and again after any surrogate refit
python -m narrative.artefact --jet 146      # inspect one artefact, offline
python -m narrative.artefact --validate 3000

python -m narrative.orchestrator \
    --backend vertex --model gemini-3.5-flash-lite \
    --jets-file narrative/outputs/top/narratives/run100_jets.json \
    --out narrative/outputs/top/narratives/scratch_gemini-3.5-flash-lite.jsonl

# Read the result. --blocks takes file prefixes, so this scores scratch_*.jsonl.
python -m narrative.blocks_report --blocks scratch
```

`blocks_report.py` **re-verifies every narrative** rather than trusting the
`violations` field each run recorded. That field was written by whatever
verifier was on disk at the time and it has drifted, so a rate must be computed
under one contract. Running it with no arguments scores the v16 block and
reproduces the paper's pass-rate table.

It prints per-family rates with Wilson intervals, the same rates inside each
block, and exact McNemar tests over the shared jets.

### What is on disk

**The paper's run is the `v16` block alone**: 100 jets drawn at seed 2 into
`run100_jets.json`, of which the gate withholds 10, leaving 90 narrated by each
of the four generators. That sample file is the one thing under `outputs/` that
is tracked — a result is not reproducible without knowing which jets it covered.

Five further blocks `b1`–`b5` (seeds 3–7) exist and are **deliberately not in
git**. They live in `extra_blocks/`, together with the `run_blocks.sh` that
produced them, because nothing in the paper rests on them: pooled with `v16`
they give 600 distinct jets, 553 narrated by all four models, and a paired frame
whose rates differ from the reported ones. Keeping them out of the repo keeps
the published material unambiguous. To rebuild that frame:

```bash
python -m narrative.blocks_report \
    --runs narrative/outputs/top/narratives extra_blocks \
    --blocks v16 b1 b2 b3 b4 b5
```

The prefix names the *block*, not the prompt version — `v16_*` is named after
the prompt only because it predates the block scheme. **Read `prompt_id` out of
the file rather than inferring it from the name.**

Superseded runs are gzipped under `narratives/archive/`: `top-v7`, prompt
versions `v8`–`v15`, and the rejected ablation arms `v17`–`v19`. Two traps
there. A higher version number is not a later contract — `v17`–`v19` postdate
`v16` and were all rejected, so `v16` is what ships. And the prompt *text* for
`v17`–`v19` was never committed, so only their outputs survive.

### Backends

| `--backend` | protocol | credential |
|---|---|---|
| `aistudio` | google-genai | `GOOGLE_API_KEY` |
| `vertex` | google-genai | application default credentials |
| `deepseek` | OpenAI-compatible | `DEEPSEEK_API_KEY` |
| `groq` | OpenAI-compatible | `GROQ_API_KEY` |
| `ollama` | OpenAI-compatible | `OLLAMA_API_KEY` |

`llm.thinking_budget: 0` disables reasoning on every backend; the client
translates it per provider and records what was actually applied, since not
every model honours it.

### What verification establishes

The narrator wraps every artefact-derived number in a tag naming its field, and
every reading of one in a tag carrying a word from a closed vocabulary. Checking
a claim is then a regular expression and an equality test, with no second model
in the loop. A tag closes with the name it opened with — `[V01]175.27[/V01]` —
so a pair whose halves disagree does not parse and is caught rather than
silently accepted.

Five constraint families, each pass/fail per narrative: `Faith` (every tagged
number matches), `Comp` (all 46 required tags present), `Form` (no malformed or
unknown tag), `NoBare` (no digit outside a tag), `Read` (every tagged word is
the one its number implies). `TOTAL` is their conjunction and is the delivery
criterion.

**Established:** every number comes from the artefact, is attributed to the
right field, carries the right sign and magnitude, none was dropped or smuggled
in untagged, and every tagged reading agrees with the number it reads.

**Not established:** that the physical interpretation respects the glossary,
that the guardrails held, or that the observables came in the required order.
Those are interpretive, they belong to Part 4, and no regex settles them.

Each run writes one JSONL, one line per jet, carrying the artefact, the regime,
the prompt id and hash, and **every attempt** — text, model, seed, temperature,
token usage, violations. Failed attempts are kept deliberately: the rate at
which narratives survive verification is itself a measurement.

### Gotchas, all of them measured

- **Sample the jets.** See above. This one cost forty points of apparent pass rate.
- **Do not describe fields the artefact does not carry.** With `n_other` /
  `phi_other` dropped from the artefact but still documented in the prompt, the
  model invented them and the pass rate went to zero. Those instructions now
  live in a `remainder` block appended only when the fields are present.
- **Seed rotation does not drive retries.** Four seeds at temperature 0 returned
  byte-identical text on 2 jets out of 5. Retries raise the temperature instead.
- **Calibrate a band on the population that is judged**, not on the whole set.
  Word-tag boundaries chosen as quartiles of all ten observables land elsewhere
  entirely on the five that must carry a judgement, which are extreme by
  selection. A band nobody inhabits is one the model stops believing in.
- **Read the raw text before blaming a design choice.** One model's formatting
  score was written off to the closing-tag convention; the actual cause was a
  non-ASCII hyphen inside the tag, `[B]‑0.95[/B]`, which renders identically to
  a minus sign and does not parse. That model only, 113 tags.
- **Change one thing per run.** Two prompt edits shipped together produced a run
  that could not be attributed and had to be redone as two single-change arms.
- **A prompt edit moves more than the rule it edits.** Changing one word-tag rule
  flipped the verdict on 50 of 90 jets for one model. `TOTAL` is a conjunction
  over 46 tag decisions, so it goes as pᴺ; at n = 90 it cannot resolve a prompt
  edit. Pooling to n = 553 is what buys the resolution back — at that size every
  pairwise model contrast separates, where at n = 90 none did.
- **Do not infer the prompt version from a file name.** Pooling runs from two
  prompt versions makes the version a confounder in every number that comes out.
- **Check whether an error is one-sided before blaming the model.** An error that
  never occurs in the opposite direction is a design defect, not noise.
- **Reasoning shares the output budget.** Thinking tokens are drawn from
  `max_output_tokens`; at 2048 the reasoning models returned truncated narratives
  that failed for lack of room rather than of ability. 8192 covers both.
- **Omitting a reasoning flag is not the same as disabling it.** DeepSeek's docs
  read as though non-thinking were the default; a request without the field comes
  back having spent reasoning tokens. Always send it explicitly.
- **Gemini 3.x is reachable only from `location: global`** on Vertex, and returns
  404 from every regional endpoint.

## Part 4 — Evaluation

What verification cannot settle is judged on three dimensions —
`linguistic_realization`, `internal_consistency`, `overall_satisfaction`, each
scored 0–3 — by **one judge, `claude-sonnet-5`**, and in parallel by human
annotators on the same items.

Eight candidate judges across four families were piloted; the rest were dropped
for reading the rubric too coarsely, and their verdicts are not kept. Reasoning
is pinned **off**: measured, it cost thinking tokens and latency without buying
sensitivity, and on the one probe with an unambiguous answer it made the verdict
worse. Sonnet 5 rejects sampling parameters outright, so its verdicts carry
whatever variation the model has — `--repeat` is how that is measured.

The judge is not one of the four generators: a model grading its own output is
the one configuration a judge study cannot use. It sees **the narrative alone** —
no artefact, no glossary — because that is what the human annotators see, and
their agreement only measures the judge if both were shown the same thing.

```bash
./run_judges.sh                     # scores block v16, resumable, holds a lock
python -m evaluation.judge_report   # → annotation_set/judge_{scores,reasons}.csv
python -m evaluation.agreement      # human ↔ judge tables (--latex for the paper)
```

**The judge is an aggregate baseline, not a substitute for reading a
narrative.** On the 20-item set it agrees with the annotators on 95% of the
positive/negative collapse, but its quadratic-weighted κ against them is 0.25
where the two annotators score 0.86 against each other, and of the six human
ratings at or below 1 it caught **none**. The scale is saturated — 95% of human
ratings are 2 or 3, and Internal Consistency is 3 on all 40 human judgements, so
its κ is 0/0 and is reported as n/a rather than as a number. Both facts are
true and only reporting both is honest; `agreement.py` prints them together for
that reason.

```
evaluation/
  judge.py                     the rubric runner
  judge_prompt.yaml            versioned judge prompt
  judge_report.py              verdicts → spreadsheet, joined on KEY.csv
  agreement.py                 human ↔ judge agreement, κ_w, negative recall
  judgments_v16_claude-sonnet-5.jsonl
  annotation_set/
    items/  items_tagged/      the 20 items as handed to annotators, ± tags
    probes/                    5 hand-written probes with known answers
    KEY.csv                    provenance the annotators do not see
    PROMPT.txt FIELDS.txt GLOSSARY.txt
    human_*.csv                one sheet per annotator
    judge_scores.csv           one row per item, one column per dimension
    judge_reasons.csv          the same verdicts with justification text
    judge_probes_sonnet.jsonl  probes
    judge_variance_sonnet.jsonl, judge_v1_rerun_sonnet.jsonl
```

Nothing in the report aggregates over generators: five narratives per generator
is far too few to compare generators, and a column that invited that comparison
would be read as one.

## Part 5 — Future work

JetClass as a second dataset, and a RAG-augmented narrative pipeline.

---

## Relationship to the ParT codebase

The teacher is trained with the upstream
[jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer)
toolkit. `particle_transformer/` vendors the parts needed to reproduce
training and inference — network definitions, dataloader, weaver configs, train
scripts — copied verbatim from upstream at commit `2925bdb`, including its own
README. It is kept verbatim on purpose, so it is the one directory in this repo
not to tidy.

The symlinks inside `particle_transformer/surrogate/` point back at this repo's
real code, so the ParT scripts resolve the paths they expect without a second
copy of anything. They are load-bearing, not a convenience: `surrogate/` at the
repo root holds only the model fitting, while the feature building and the
fastjet loader live under `data/observables/`, and it is the symlink tree that
puts both in one importable package. Run Part 2 from the repo root and the
imports fail.

**Quark/gluon tagging is out of scope.** The QuarkGluon basis and runs were
removed; the vendored upstream files that mention QuarkGluon are left as they
came.

## Not in this repository

The raw datasets (TopLandscape, JetClass), the trained checkpoints
(`.pt`/`.pth`), the ParT teacher logits, the surrogate feature tables and
fitted models, and the narrative run outputs. They are large, regenerable from
the scripts here, and kept local pending external hosting. The one exception is
`narrative/outputs/top/narratives/run100_jets.json`: it defines which jets a run
covered, so a result is not reproducible without it, and it is tracked.
