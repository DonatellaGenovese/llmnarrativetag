# Part 3 — LLM Narrative

Turns a surrogate explanation — the local decomposition φ_k(j), the intercept β₀
and the residual r(j) from Part 2 — into a natural-language account of why ParT
classified a jet the way it did, with every quantitative claim checked against
the surrogate before it is kept, and no narrative produced at all where the
surrogate does not track the tagger.

Status: **running end to end.** Five models narrated the same 100 jets; results
and their reading live in [`docs/results_narrative_eval.md`](../docs/results_narrative_eval.md).

## Layout

```
narrative/
  configs/narrative_top.yaml   paths, artefact shape, abstention thresholds, LLM settings
  configs/glossary_top.yaml    observable → physical reading (all entries reviewed)
  prompts/prompt_top.yaml      versioned system + task template, currently `top-v7`
  stats.py                     reference percentiles, frozen once from the test set
  artefact.py                  per-jet artefact: orientation, ranking, rounding, regime
  llm_client.py                one client for Gemini and the OpenAI-compatible providers
  generator.py                 glossary rendering, prompt assembly, prompt hashing
  verifier.py                  tag extraction, equality, completeness, bare-number sieve
  orchestrator.py              abstention gate → generate → verify → retry
  metrics.py                   aggregates a run's JSONL into the pass-rate table
  outputs/top/
    reference_stats/           grids.npz + meta.json, written by stats.py
    narratives/                one JSONL per run, plus the jet sample files
```

The data flows in one direction: `stats.py` → `artefact.py` → `generator.py` →
`llm_client.py` → `verifier.py`, with `orchestrator.py` driving the sequence and
`metrics.py` reading what it wrote. Each module depends only on the ones above
it, so `artefact.py` and `verifier.py` can be exercised without any API key.

## Selecting the jets

**Every model must narrate the same jets**, or half of any difference between
them is sampling noise rather than behaviour. The sample is therefore drawn once
into a file and reused.

```bash
# Draw 100 jets at random (seed 2) and record them. Created on first use.
python -m narrative.orchestrator --jets-file narrative/outputs/top/narratives/run100_jets.json \
                                 --sample 100 --sample-seed 2 --model gemini-2.5-flash-lite
```

If the file already exists it is loaded verbatim and `--sample` / `--sample-seed`
are ignored, so every later run against that file sees an identical set. The
file records `n`, `seed` and the row indices, and is small enough to commit.

Two other selectors exist, both narrower:

```bash
--jets 11133 14079 34638      # explicit row indices, for inspecting specific jets
--n 5                         # the FIRST five rows
```

**Do not use `--n` for measurement.** The TopLandscape test file is ordered in
blocks of up to 100 identical labels, so its opening rows are not a sample of
anything — the first eight are all QCD. An early 8-jet run on the first rows
scored 0.88 for a model that scores 0.21 on a random sample of the same size
distribution. `--n` is for smoke tests only.

Of any 100 jets sampled, roughly 10 fall in the abstention regime and are never
sent to a model, so a run of 100 yields about 90 narratives.

## Running

```bash
conda activate part-surrogate
export PYTHONPATH="$PWD"

# Once, and again after any surrogate refit — the percentiles describe a
# specific fitted model.
python -m narrative.stats

# Offline checks, no API key needed.
python -m narrative.artefact --jet 146          # inspect one artefact
python -m narrative.artefact --validate 3000    # additive identity + regime split

# Narrate. --model and --backend override the config, so one config serves a
# whole sweep without being edited between runs.
python -m narrative.orchestrator \
    --backend vertex --model gemini-3.5-flash-lite \
    --jets-file narrative/outputs/top/narratives/run100_jets.json \
    --out narrative/outputs/top/narratives/run100_gemini-3.5-flash-lite.jsonl

# Read the result.
python -m narrative.metrics narrative/outputs/top/narratives/run100_gemini-3.5-flash-lite.jsonl
```

A sweep is that last pair in a loop over models. Runs are long enough to be
worth detaching:

```bash
setsid nohup ./sweep.sh </dev/null >/dev/null 2>&1 &
```

### Backends and credentials

| `--backend` | protocol | credential |
|---|---|---|
| `aistudio` | google-genai | `GOOGLE_API_KEY` |
| `vertex` | google-genai | application default credentials |
| `deepseek` | OpenAI-compatible | `DEEPSEEK_API_KEY` |
| `groq` | OpenAI-compatible | `GROQ_API_KEY` |
| `ollama` | OpenAI-compatible | `OLLAMA_API_KEY` |

`llm.thinking_budget: 0` disables reasoning on every backend; the client
translates it per provider and records in each attempt what was actually
applied, since not every model honours it.

## What the run writes

One JSONL per run, one line per jet, carrying the artefact, the regime, the
prompt id and hash, and **every attempt** — text, model, seed, temperature,
token usage, and the list of violations. Failed attempts are kept deliberately:
the rate at which narratives survive verification is itself a measurement, and
it cannot be recovered from the accepted ones.

## What verification establishes

The narrator wraps every artefact-derived number in a tag naming its field, so
checking a quantitative claim is a regular expression and an equality test, with
no second model in the loop.

**Established:** every number in the narrative comes from the artefact, is
attributed to the right field, carries the right sign and magnitude, and none
was dropped or smuggled in untagged.

**Not established:** that the prose around a number reads it correctly, that the
physical interpretation respects the glossary, or that the guardrails held.
Those are interpretive, they belong to Part 4, and no regex settles them.

## Gotchas, all of them measured

- **Sample the jets.** See above. This one cost forty points of apparent pass
  rate.
- **Do not describe fields the artefact does not carry.** With `n_other` /
  `phi_other` dropped from the artefact but still documented in the prompt, the
  model invented them and the pass rate went to zero. Those instructions now
  live in a `remainder` block appended only when the fields are present.
- **Seed rotation does not drive retries.** Four seeds at temperature 0 returned
  byte-identical text on 2 jets out of 5. Retries raise the temperature instead
  (`retry_temperature_step`).
- **Reasoning shares the output budget.** Thinking tokens are drawn from
  `max_output_tokens`; at 2048 the reasoning models returned truncated
  narratives that failed verification for lack of room rather than of ability.
  8192 covers both.
- **Omitting a reasoning flag is not the same as disabling it.** DeepSeek's docs
  read as though non-thinking were the default; a request without the field
  comes back having spent reasoning tokens. Always send it explicitly.
- **Gemini 3.x is reachable only from `location: global`** on Vertex, and
  returns 404 from every regional endpoint.
