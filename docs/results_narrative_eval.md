# Part 4 (LLM Narrative evaluation)

## Setup

All narrators are Gemini models served through Vertex AI on the global endpoint, the only one from which the 3.x family is reachable. Reasoning is disabled on every model (`thinking_budget = 0`, verified as applied: zero reasoning tokens on the real prompt), temperature 0, seed 42, prompt `top-v7`. No tools, no multi-turn, no agentic loop: one call in, one narrative out.

Four models, spanning two generations across two price tiers:

| Model | USD / 1M in | USD / 1M out |
| --- | --- | --- |
| gemini-2.5-flash-lite | 0.10 | 0.40 |
| gemini-3.1-flash-lite | 0.25 | 1.50 |
| gemini-2.5-flash | 0.30 | 2.50 |
| gemini-3.5-flash-lite | 0.30 | 2.50 |
| gemini-3.5-flash | 1.50 | 9.00 |

`gemini-2.5-pro` was dropped: it is the only model that refuses to disable reasoning, and with reasoning tokens billed as output at 10 USD/1M it costs 25x the cheapest model for no measured benefit.

100 jets sampled at random from the test set, **the same 100 for every model**, so the comparison is paired and any difference is not sampling noise. The abstention gate skipped 10 of them without a model call, leaving 90 narrated. The run is **single-pass**: one attempt per jet, no rejection sampling, which measures first-attempt behaviour rather than what a retry loop can recover.

## What is measured

Verification is a regular expression plus an equality test over the tagged numbers. It yields four independent constraint families, each a pass/fail per narrative:

| Family | Fails when | How it is computed |
| --- | --- | --- |
| `Faith` | a tagged number does not match the artefact | Extract every `[TAG]number[/TAG]`, look the tag up in the artefact, compare the two numbers. Fails if any one differs. |
| `Comp` | a number present in the artefact never appears | The artefact defines 46 required tags — six scalar, four per observable. Fails if any of them never appears. |
| `Form` | a tag is malformed, unpaired, or not a field of this artefact | Fails if a tag is not a field of this artefact, or if any square bracket survives once every well-formed pair has been removed from the text. |
| `NoBare` | a digit appears outside any tag | Strip the well-formed pairs, then the leftover brackets, then the observable and glossary names. Fails if any digit is still standing. |
| **TOTAL** | any of the above | Fails if any family fails. The criterion under which a narrative is accepted and delivered. |

Three details behind that table. Numbers are compared numerically rather than as strings, so `38` for `38.0` passes while `2.4` for `2.37` does not, and the sign counts. The closing tag is matched as a backreference to the opening one, so `[V4]0.37[/V5]` does not parse as a pair and falls through to `Form`. The names stripped before the `NoBare` check include glossary tokens that mix letters with digits, such as the correlator formulas `e4*e2/e3^2`, since those are names rather than measurements — purely numeric tokens are deliberately not stripped.

Each family is a per-narrative pass/fail, not a count of numbers: a narrative fails `Faith` if any one of its tagged numbers is wrong, not in proportion to how many are. The figures reported below are therefore the fraction of narratives that pass, and they are strict. `Faith` = 0.90 for gemini-3.5-flash means that 90% of its narratives contain no wrong number at all — not that 90% of its numbers are right. The underlying per-number rate is far more forgiving: nine wrong values out of roughly 4100 tagged numbers. The strict reading is the operational one, since a narrative is delivered or withheld as a whole.

`Faith` and `Comp` correspond to the correctness and completeness checks of PONTE. `Form` and `NoBare` have no counterpart there: `NoBare` in particular makes the guarantee bidirectional, since it establishes not only that every tag is right but that no quantitative claim escaped tagging.

Alongside these: abstention rate, attempts, tokens, and cost per narrated jet. Proportions are reported with Wilson 95% intervals, and models are compared with exact McNemar tests on the paired jets.

## Results

| Model | Faith | Comp | Form | NoBare | TOTAL | 95% CI | USD/jet |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-2.5-flash-lite | 0.99 | 0.36 | 0.60 | 0.89 | **0.21** | [0.14, 0.31] | 0.0008 |
| gemini-3.5-flash-lite | 1.00 | 0.89 | 0.88 | 0.84 | **0.83** | [0.74, 0.90] | 0.0044 |
| gemini-2.5-flash | 1.00 | 0.98 | 1.00 | 0.78 | **0.77** | [0.67, 0.84] | 0.0044 |
| gemini-3.5-flash | 0.90 | 0.88 | 0.93 | 0.96 | **0.80** | [0.71, 0.87] | 0.0169 |

Paired comparisons:

| Contrast | Discordant | p | |
| --- | --- | --- | --- |
| 2.5-flash-lite vs 3.5-flash-lite — generation | 0 / 56 | <0.0001 | significant |
| 3.5-flash-lite vs 2.5-flash — equal price | 17 / 11 | 0.34 | not significant |
| 3.5-flash-lite vs 3.5-flash — tier | 15 / 12 | 0.70 | not significant |

Generation dominates and tier does not. The step from 2.5 to 3.5 within the lite tier is worth 62 points and is total: of 56 discordant jets, none are cases where the older model passed and the newer failed. Above that threshold the three top models are statistically indistinguishable at n = 90, and gemini-3.5-flash costs four times gemini-3.5-flash-lite without doing better.

Each model fails in its own way, which is what makes the four families worth separating rather than collapsing into one number. gemini-2.5-flash-lite collapses structurally, with 553 malformed tags and `Comp` at 0.36. gemini-2.5-flash is perfect on form but leaves bare digits. gemini-3.5-flash is the only model that misreports values — and inspecting the nine cases, eight are not fabrications but **field swaps within the same observable**, the rarity written into the slot of the value: `172.24` reported as `5.8`, which is that observable's own rarity. A reader would not catch that; the equality test does, deterministically.

## TODO

**More generators.** All four models come from a single family, so what is currently established is that the method survives a capability range within Gemini, not that it survives a change of generator. `gpt-oss-120b-maas` is the natural second family, since PONTE evaluates on GPT-OSS and the comparison would be direct rather than arbitrary; `llama-3.3-70b-instruct-maas` a third. Both are visible in Model Garden but return 404 until enabled on the project, and both require an OpenAI-compatible client path in addition to the current one. Anthropic models need a third path again, through `rawPredict`.

**Larger n for the fine contrasts.** At n = 90 the three top models cannot be separated. Resolving a 10-point difference needs roughly 150–230 paired jets, and a 5-point difference 300–900. Whether that is worth buying depends on whether the difference between 0.77 and 0.83 is a question anyone needs answered.

**The retry loop.** The reported numbers are single-pass. Rejection sampling roughly doubles the acceptance rate at a cost of ~1.6 calls per jet, but our loop is blind: the model is never told what it got wrong, only resampled at a higher temperature. PONTE instead feeds structured corrective feedback into a revision prompt. Blind versus informed refinement is a clean ablation, directly against theirs, and the effect is likely far larger than any difference between the models above.

### What verification does not check

The regex layer establishes that every number in a narrative comes from the artefact, is attributed to the right field, carries the right sign and magnitude, and that none was dropped or smuggled in untagged. It establishes nothing about the prose. The following are open, and are the substance of the remaining evaluation:

1. **Whether a number is read correctly.** A narrative can transcribe `rarity` as `0.6` and describe it as "not unusual", which inverts its meaning while passing every check. Instances are documented; the frequency is not measured, and measuring it needs human annotation rather than keyword matching.
2. **Whether the physical reading respects the glossary.** The glossary is the narrator's only licence to interpret a value. Whether it stays inside that licence, or supplies physics of its own, is unverified.
3. **Whether the guardrails hold.** Attributing the decision to the decomposition, counterfactual statements, references to the truth label or to the constituents — all forbidden by the prompt, none checked by the verifier.
4. **Order.** The task requires the observables in the order given; the verifier accepts any order.
5. **Whether the caveat and abstention regimes read as intended**, rather than merely being triggered.
6. **Usefulness.** Whether a physicist can act on the narrative, and whether it beats reading the shape functions directly. This needs a human study.

Items 1–4 are cheap to attack with a labelled sample of a few dozen narratives, which also yields a reference against which any automated proxy can be validated before being trusted.
