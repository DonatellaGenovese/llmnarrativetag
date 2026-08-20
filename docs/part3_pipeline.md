# Part 3 (LLM Narrative) — pipeline as built

## What the narrator is asked to explain

1. **Decision.** Which class ParT assigned, and how strong the score is relative to the other jets it assigns to that class.
2. **Adequacy.** How much of that score the additive decomposition reproduces.
3. **Contributions.** Which observables carry the largest contributions, and towards which class.
4. **Value.** The measured value of each, and what it means physically.
5. **Rarity.** How unusual that value is for the opposite class.

Point 2 is only partly the narrator's job. Whether the decomposition is adequate at all is decided before the model is called: the residual r(j) from Part 2 gates the jet into one of three regimes, and above |r| = 2 no narrative is requested. The narrator reports adequacy; it does not judge it.

## What is passed to the model

One JSON artefact per jet, plus the glossary and the template. Nothing else — in particular not the residual, not the raw teacher logit, and not the truth label.

| Field | What it is | How it is obtained | Occurs |
| --- | --- | --- | --- |
| `decision` | the class ParT assigned | sign of t(j), mapped to a string | once |
| `other_class` | the class it did not assign | complement of `decision` | once |
| `score_tagger` | ParT's score, oriented towards `decision` | \|t(j)\| clipped to the fitting range, rounded | once |
| `score_median` | median score across jets assigned to the same class | test-set statistic | once |
| `score_pct` | percentage of same-class jets scored below this one | test-set statistic | once |
| `score_reconstructed` | the score the decomposition reproduces | g(j), same orientation | once |
| `recon_pct` | percentage of jets reconstructed less closely than this one | statistic over residuals | once |
| `intercept` | the baseline the contributions add to | beta_0 of the surrogate, oriented | once |
| `i` | the observable's index, used to build its tags | position in `features`, 1..K | per observable |
| `name` | observable name | — | per observable |
| `value` | measured value on this jet | x_k(j), rounded | per observable |
| `phi` | contribution to the score, same orientation | phi_k(j) = f_k(x_k(j)) | per observable |
| `pct` | percentage of jets **the tagger assigned to** `other_class` whose value is at or below this one | test-set statistic, one decimal | per observable |
| `rarity` | percentage of that class lying further out in the same direction | min(pct, 100 − pct) | per observable |

Design choices behind the table:

- **Observables are ranked by \|phi\|, not by signed phi.** Ranking by signed value pushes the strongest *opposing* contributions to the bottom, where they would be the least visible and are physically the most interesting.
- **All ten observables are passed.** With the full basis there is no remainder, so `n_other` and `phi_other` are omitted entirely rather than sent as zeros. Describing a field the artefact does not carry makes the model invent it: with those two documented in the prompt but absent from the payload, the pass rate fell to zero.
- **`rarity` is passed alongside `pct` rather than left to be derived.** `pct` is one-sided: values near 0 and near 100 are both extreme and the middle is ordinary, so `pct` alone does not answer "how unusual is this". The narrator is forbidden to compute, so any quantity it must state has to be supplied.
- **`score_tagger` uses the clipped logit**, the same target the surrogate was fitted against; comparing g against the raw score would show the outer 1% as reconstruction failures that are really a fitting choice.

## Tagging and verification

Following the tag-based approach, the narrator must wrap every artefact-derived number in a tag naming its field. Verification is then a regular expression and an equality test, with no second model in the loop, unlike [11] where claims are extracted by a further LLM.

    [Z]number[/Z]              score_tagger
    [ZM]number[/ZM]            score_median
    [ZP]number[/ZP]            score_pct
    [G]number[/G]              score_reconstructed
    [GP]number[/GP]            recon_pct
    [B]number[/B]              intercept
    [V1]number[/V1]            value    of the observable whose `i` is 1
    [I1]number[/I1]            phi      of that same observable
    [P1]number[/P1]            pct      of that same observable
    [R1]number[/R1]            rarity   of that same observable
    [Q1]word[/Q1]              how unusual that rarity is, as one word

Tags are indexed by the observable's `i` rather than named after it. Name-based tags — `[P_C2_double_b1]…[/C2_double_b1]` — made mis-closed tags the dominant failure mode, since the model had to reproduce a fourteen-character identifier twice.

**`[Q<i>]` carries a reading rather than a number**, drawn from a closed vocabulary by a rule the prompt states:

| rarity | word |
| --- | --- |
| below 1 | `extreme` |
| 1 up to 10 | `unusual` |
| 10 up to 25 | `uncommon` |
| 25 or above | `ordinary` |

This is the mechanism that makes an interpretive claim checkable. The thresholds are given to the model, so verification asks whether it applied a declared rule, not whether it matched a rule invented afterwards; and the vocabulary is closed, so a paraphrase is a violation rather than a gap in a keyword list. Crucially it does not remove the opportunity to fail: the model still decides, and can still decide wrongly.

With ten observables a narrative carries 56 mandatory tags.

## What verification establishes

Five independent constraint families, each a pass/fail per narrative rather than a count of numbers:

| Family | Fails when | How it is computed |
| --- | --- | --- |
| `Faith` | a tagged number does not match the artefact | extract every `[TAG]number[/TAG]`, look the tag up in the artefact, compare numerically |
| `Comp` | a number present in the artefact never appears | 56 required tags; fails if any never appears |
| `Form` | a tag is malformed, unpaired, or not a field of this artefact | fails if a tag is unknown, or if any bracket survives once well-formed pairs are removed |
| `NoBare` | a digit appears outside any tag | strip pairs, brackets, and the observable and glossary names; fails if a digit remains |
| `Read` | a tagged word contradicts the value it reads | compute the expected word from `rarity` by the declared rule and compare |
| **TOTAL** | any of the above | the criterion under which a narrative is delivered |

`Faith` and `Comp` correspond to the correctness and completeness checks of PONTE. `Form`, `NoBare` and `Read` have no counterpart there. `NoBare` makes the numeric guarantee bidirectional — not only is every tag right, but no quantitative claim escaped tagging — and `Read` is the first check on the prose rather than on the digits.

**Established:** every number comes from the artefact, is attributed to the right field, carries the right sign and magnitude, none was dropped or smuggled in untagged, and the one reading that is tagged says what its number says.

**Not established:** the readings that are not yet tagged — the direction of a contribution, which side of the other class a value sits on, and whether `recon_pct` is read as better or worse; and, structurally, whether a physical claim is licensed by the glossary. The first three are the same mechanism applied again. The last is not reachable this way: a statement like "consistent with a top quark decay" is not a function of any number in the artefact, so there is no expected label to compare it against.

## The narrator prompt

System instruction:

> You are an expert XAI (Explainable AI) Interpreter and Data Narrator. Your role is to create a translation layer between a jet substructure analysis and a physicist reader. You translate mathematical outputs into clear, actionable natural language narratives. You report what the artefact contains.

The task template is fixed across jets and carries eight sections — BACKGROUND, SIGN CONVENTION, INPUT, OBSERVABLE GLOSSARY, TASK, TAGGING, GUARDRAILS, DATA INPUT — with the glossary and the artefact substituted in. Of the roughly 11 000 characters that reach the model, 57% is the template and 29% the glossary, both invariant; only the 14% that is the artefact JSON changes from jet to jet.

The guardrails, in full:

- The DECISION belongs to the tagger; the CONTRIBUTIONS belong to the decomposition. Never write that the tagger "used", "looked at", or "focused on" an observable.
- Report, do not compute. No sums, ratios, or numeric comparisons.
- Do not correct the artefact. An unexpected direction is a finding, not an error.
- No counterfactuals. Do not state what value would change the outcome.
- Do not mention the true label, whether the classification is correct, the constituents, the architecture, or observables not in the input.
- The only distributional information you have concerns `other_class`, via `pct` and `rarity`. You are told nothing about how an observable is distributed among jets assigned to `decision`.
- `rarity` is what says how unusual a value is. `pct` says only in which direction. Never read `pct` as a measure of unusualness: `pct` near 0 and `pct` near 100 are both extreme, and `pct` near 50 is ordinary.
- `pct` counts the jets at or below the value. A high `pct` therefore means few jets of `other_class` lie above it, not many.
- The glossary is the only licence you have to interpret a value. Do not add physics knowledge of your own.

The last guardrail is what makes the glossary load-bearing: its `meaning` text is the entire licence the narrator has to say what a value signifies, and it reaches the reader verbatim.

## Generation settings

One call per jet: no tools, no multi-turn, no agentic loop. The model translates a verifiable artefact rather than acting as an explanatory agent, which is the distinction Mayne et al. draw and PONTE adopts.

Reasoning is disabled on every model that permits it, and the client records for each call whether the request was honoured. Temperature 0, fixed seed. Retries, when enabled, raise the temperature: seeds alone do not move the output, and four seeds at temperature 0 returned byte-identical text on two jets out of five.
