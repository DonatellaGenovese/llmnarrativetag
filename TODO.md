# TODO

Open work, by area. State as of 2026-11-08.

## Evaluation: human and LLM-judge (to be redesigned)

The current judge protocol does not discriminate, and it cannot see whether a
narrative is physically correct. Both have to change before the judge numbers
are worth reporting.

**What the v17 run showed** (first attempts, no new calls needed to reproduce:
`evaluation/results/judgments_v17_first_*.jsonl`)

- Saturation. Share of 3s: Sonnet 5 86.9%, GLM 5.3 95.0%, Mistral Large 4 100%
  (32 narratives). Humans on the 20-item set average 2.2-2.4.
- No sensitivity to defects. Mean score of first attempts that pass verification
  minus those that fail: Sonnet +0.22 / +0.04 / +0.17 (ling / inte / over),
  GLM +0.13 / +0.03 / +0.01, Mistral 0.
- GLM vs Sonnet on the same 183 narratives: kappa_w 0.34 / 0.20 / 0.05, and GLM
  is the higher one in 33 of 33 disagreements on Overall. GLM ranks gpt-oss
  (11% verified) above gemini-3.5-flash-lite (52%).
- Against the humans (20 items): kappa_w Sonnet 0.25, GLM 0.06, Mistral 0.06;
  all three recover 0 of 6 human negatives.
- Conclusion: neither GLM nor Mistral Large can replace Sonnet as is, and
  Sonnet itself is only usable as a coarse aggregate. The problem is the
  protocol, not the model.

**To do**

- [ ] **Add a physical-correctness criterion.** Today the judge sees the
      narrative alone, tags stripped, by design (the same material as the human
      annotators), so it can judge fluency and self-consistency but not whether
      the physics is right. Decide what "physically correct" means here, e.g.:
      - the physical reading of each observable matches the glossary and the
        direction of its contribution (e.g. low tau32 read as three-pronged,
        pushing towards top);
      - no physics claimed beyond the glossary (invented mechanisms, wrong
        particle content, wrong mass scales);
      - the overall story is consistent with what top and QCD jets are;
      - coherence between the stated decision and the drivers cited.
      And decide what the judge is given to check it: the artefact and the
      glossary (then it is no longer the same input as the humans'), or the
      narrative alone with a physics rubric.
      **Decided 2026-11-13: the judge stays as it was, narrative alone, tags
      stripped; no artefact or glossary.** Judges from now on: GLM 5.3,
      Mistral Large 4, Muse Spark 1.3 (Sonnet dropped), on the top-3
      generators only. Note: the v16 human items (annotation_set/items) DID
      carry the artefact above the narrative; decide whether the new human
      set gives the narrative alone, as the judge.
- [ ] **Add the same criterion to the human annotation**, with a written rubric
      and anchored examples, so human and judge measure the same thing. Needs
      annotators with jet-physics background.
- [ ] **Fix the saturation of the existing dimensions.** Options to pilot, on
      the 20 human-annotated items first (GLM is near free on the Mistral plan):
      1. pairwise comparison: two narratives of the same jet from different
         generators, "which is better and why";
      2. anchored rubric: real 1- and 2-rated examples from the annotations in
         the prompt;
      3. binary checklist instead of 0-3 (explains the decision? follows the
         contribution order? uses undefined terms? states the caveat when
         required? ...).
      Keep the one with the best human agreement and negative recall.
- [ ] **Bump the judge prompt id** (`evaluation/judge_prompt.yaml`, now
      `judge-v2`) for any of the above, and keep v2 runnable for comparison.
- [ ] **Rebuild the annotation set on the new generation** (v17 or the larger
      one): `narrative/make_annotation_set.py`, `evaluation/agreement.py` and
      `evaluation/judge_report.py` still carry the v16 generator names.
- [ ] **Decide the judge(s) and the budget** before rerunning. Sonnet 5 costs
      ~$0.0115 per narrative here (~3.45k in, ~0.46k out); the Anthropic credit
      ran out at 183 of 332 first attempts. Options: Sonnet on a subsample,
      prompt caching on the fixed rubric, or GLM/Mistral if a redesigned
      protocol makes them agree with the humans.
- [ ] Judge the delivered narratives too (`--attempt final`), for the paired
      before/after-refinement comparison; not run yet.

## Top tagging

- [x] ParT fine-tune, seeds 42/43/44, manifests (acc 0.9435 ± 0.0002, AUC 0.9876 ± 0.0001)
- [x] Surrogates per seed, seed variance on 42, teacher comparison (EBM R² 0.936 ± 0.0003; φ stable)
- [x] PONTE-style refinement loop (5 attempts, verifier feedback, history)
- [x] Generation v17, 100 jets, 4 generators; reports after 1 and after 5 attempts
- [x] **vLLM serving, code** (2026-11-10): Ollama dropped (no longer serves
      gpt-oss free), gpt-oss-120b moved to a `vllm` backend served by us;
      `scripts/vllm_models.sh` (pinned revisions, TP), `scripts/setup_vllm.sh`
      (env vllm 0.32.0+cu129, the newest CUDA-12 build, driver 535; + weights), `scripts/slurm/vllm_generate.sbatch`
      (server on localhost + orchestrator, TEST=1 for the debug test),
      orchestrator `--workers` and `--extra-body`, run_generators `ONLY`/`WORKERS`/`CFG`.
      Weights on $FAST (2026-11-11): load 626 s -> 98 s for gpt-oss.
- [x] gpt-oss-120b on vLLM (2 GPUs): debug test, then v17 regenerated
      (TOTAL 0.07 first attempt, 0.78 after refinement; Ollama was 0.11 / 0.83,
      within noise). Ollama file kept as v17_gpt-oss-120b_ollama.jsonl
- [x] Qwen3.8-27B on vLLM (FP8, 1 GPU), thinking off: debug test, then v17
      (TOTAL 0.45 first attempt, 0.84 after refinement; on a par with
      gemini-3.5-flash-lite). Added to `generators`.
- [x] Qwen3.8-Flash-Next dropped (2026-11-11): needs a whole node and its test
      sat in the queue for hours; the 27B is enough. Weights deleted, removed from the scripts.
- [x] **Rank gate for top (2026-11-13)**, `narrative/gate_top.py`, same scheme
      as the AD gate: abstain if sign(g_hat) != sign(t), caveat if the
      quantile-matched ranks differ by more than X = 7.5 pct points (calibrated
      on val, caveat <= 30% of kept). Test: 68% full / 26% caveat / 6% abstain;
      the old residual gate gave 50 / 29 / 21 on the seed-42 teacher (its
      69.5% / 7.1% came from an earlier teacher), and 48% of its abstentions
      had |t| >= 5. On the v17 jets: 72 full, 20 caveat, 8 abstain (v17 ran
      58 / 25 / 17). Rerun `python -m narrative.gate_top` after any teacher
      or surrogate change.
- [x] Pre-500 fixes (2026-11-13): top artefact without `-0.0`; prompt
      `top-v17` and `ad-v4` (band-edge wording; [AR] 0.01 = "or less"); `W`
      case-sensitive. v17 / ad100 reports unchanged.
- [x] Top jet sample for the large run (2026-11-13):
      `narrative/select_top_jets.py` -> `top500_jets.json`, 150 top correct /
      150 qcd correct / 100 top wrong (QCD tagged top) / 100 qcd wrong (missed
      top), seed 3, v17 jets excluded. Under the rank gate 419 narrated; missed
      tops abstain most (45 of 100), as the surrogate often takes the right
      decision where ParT is wrong.
- [x] README updated (2026-11-13): both use cases, seed-42 surrogate numbers,
      v17 and ad100 delivery tables with attempts, AD E1/E2, new gate, judge
      protocol marked as under redesign.
- [x] pilot100 (2026-11-13), both use cases, 5 generators: reports in
      narrative/outputs/{top,anomaly}/narratives/pilot100_report_*.txt. The one
      deepseek Ident failure (AD jet 332892) was "adding small amounts on top":
      `top` now also allowed after "on" (2026-11-16); only that jet changed.
- [ ] AD jet sample for the large run (ad500 with select_anomaly_jets, excluding
      ad100; QCD in the extreme band: 184 available).
- [ ] Note: in the caveat regime the top artefact can still carry
      [GQ]better (2 of 20 caveat jets on the v17 sample): GQ ranks the logit
      residual, the gate ranks the scores. Check whether narratives contradict.
- [ ] Commit the work (nothing committed yet; all changes local on branch `anomaly`)
- [ ] Larger generation (more blocks) once the evaluation protocol is settled
- [ ] Compare against the published numbers in a single table (surrogate, verification, judges)

## Anomaly detection

Decisions (2026-11-08):
- [x] Detector: IAD on the frozen ParT latent `x_cls` (the method as it stands)
- [x] Score to explain: IAD latent at **f = 10%, 5% and 1%** (a detector that
      works decently first, then how the surrogate degrades as the signal thins)
- [x] Surrogate fit on **val** (the IAD classifier never sees it), reported on test
- [x] Working point: **10% background efficiency** defines "anomalous" (1% and
      0.1% still reported as secondary)
- [x] Mass-decorrelated score as a second target: dropped (E2b below)

### Experiment plan

**E1. Detector, 5 seeds** (GPU, one job, ~1 h)
- `anomaly.transfer --fractions 0.10 0.05 0.01 --seed s`, s = 0..4, on the
  latent only. Each run saves every classifier, its per-jet logit on
  train/val/test, and a run manifest. The supervised ceiling only on seed 0.
  **Done 2026-11-08** (job 63861300): top efficiency at 10% background
  0.954 ± 0.005 (f=10%), 0.933 ± 0.013 (5%), 0.862 ± 0.061 (1%); supervised
  0.972; mass cut 0.669. IAD on the observables dropped from the plan
  (kept in seed*/ from this run, `--spaces latent obs` to redo it).
- References kept: the supervised ceiling on the latent and the soft-drop-mass
  cut. The latent-distance baselines are dropped (`summarize_seeds --all` shows them).
- `anomaly.baselines` once (deterministic: mass distance, latent euclid/cosine).
- Read: AUC and top efficiency at 10% / 1% / 0.1% background, raw and
  mass-decorrelated, mean ± sd over seeds. Check against the branch's numbers
  (latent f=10%: 91.9% at 10% background).
- Code: `scripts/slurm/anomaly_iad.sbatch`; `anomaly/summarize_seeds.py`.

**E2. Surrogate on the AD score** (CPU, one job, < 1 h)
- For each seed and f ∈ {0.10, 0.05, 0.01}, latent: EBM / Ridge / GBDT fit to the IAD
  logit with `train_surrogates_top --scores anomaly/outputs/seed<s>/scores/iad_f<f>_latent`.
- Splits: **fit on half of val, early-stop / select on the other half** (by jet
  parity), **report on test**. Train is excluded (in-sample for the IAD).
- Read: R², Spearman, and decision agreement with the detector at the 10%
  background working point (1% secondary); EBM importances (expected: mass first; the
  question is how much substructure carries); mean ± sd over seeds; φ
  stability across seeds (as for the ParT teachers); model-seed variance on
  seed 0, f = 10%.
- **Done 2026-11-08** (job 63862147 + `anomaly.eval_surrogates` on the login,
  `anomaly/outputs/summary_surrogate.json`). EBM at f=10%: R² 0.761 ± 0.014,
  ADO 0.948 (detector vs truth 0.974), AUC 0.949, top eff. at 10% bkg 0.878;
  agreement at 10% 0.797 at the common threshold, 0.898 quantile-matched.
  Gate: the top-tagging thresholds 1/2 are too loose on the narrower IAD logit
  (3.4% abstain), scaled by sd too strict (41%, 60% of flagged jets).
- [x] **AD abstention gate** (`anomaly/gate.py`, 2026-11-09): on rarity
      (background efficiency) instead of the logit. Abstain when detector and
      EBM (quantile-matched) disagree on the 10% decision, caveat when both flag
      the jet but the rarities differ by more than ×10 (calibrated on the
      selection half: caveat <= 30% of flagged), the rarity shown to the LLM
      only as a band. f=10%, flagged jets: 64% full, 23% caveat, 13% abstain;
      not flagged: 93% full, 7% abstain. Writes gate.json and
      gate_reference.npz per run; `anomaly/outputs/summary_gate.json`.
- Code: val split helper; AD fidelity metrics at a working point (the current
  agreement metric uses the sign, which is the wrong threshold here);
  `seed_variance` and `compare_teachers` taking a scores directory.

**E2b. Mass-decorrelated target**: dropped (2026-11-09). The EBM importances
already show the explanation is not mass-dominated (n_const, m, C2, N3, tau32).

**E3. Narratives** (after E2, if the surrogate fidelity is acceptable)
- Fix the working point; class-agnostic `stats.py` / `artefact.py`;
  `prompt_anomaly.yaml` with `refine`; `narrative_anomaly.yaml`; glossary
  review; pilot on a few jets, then a 100-jet block with the generators and
  the refine loop. Judges with the redesigned protocol.

**E3 code (2026-11-11)**, decisions A-I of the AD narrative design:
- [x] `narrative/artefact_anomaly.py`: decision, detector rarity [AR] + band [A],
      surrogate band [AS] (word only), phi not oriented (direction words in the
      artefact), observable pct/rarity against the QCD reference of the gate,
      no logits, no label; regime from anomaly/gate.py
- [x] `narrative/verifier.py`: DEC/AR/A/AS tags, AD direction words,
      `identity_claim` (banned terms from the config); `feedback.py` IDENTITY
      line; orchestrator `task: anomaly`; blocks_report `Ident` column.
      Top-tagging v17 reports byte-identical after the change
- [x] `narrative/prompts/prompt_anomaly.yaml` (ad-v1, refine ad-refine-v1, caveat)
- [x] `narrative/configs/glossary_anomaly.yaml` (m, tau32, D2, pT rewritten without
      particle names; all reviewed 2026-11-11), `narrative_anomaly.yaml`
- [x] `narrative/select_anomaly_jets.py`: ad100_jets.json, 5 bands x top/QCD
      (23/23/24/20/10); regimes on it: 68 full, 14 caveat, 18 abstain.
      adpilot_jets.json: 6 jets, one per band/label cell
- [x] Pilot (gpt-oss, qwen on Leonardo), read it, then the 100-jet block
- [x] Gemini / DeepSeek on the block
- [x] **ad100 evaluation (2026-11-11)**: `ad100_report_{first,final}.txt`,
      `ad100_breakdown.txt` (by band, true label, regime) in
      narrative/outputs/anomaly/narratives/. Verifier fixes, top-tagging v17
      reports byte-identical after them:
      - `[DEC]background-like[/DEC]` was a malformed tag (WORD_RE took no
        hyphen) while artefact and prompt both say "background-like": now an
        alias of "background". gemini-3.5-flash first attempt 0.74 -> 0.88.
      - `identity_claim`: all 27 hits over 5 models and all attempts were false
        positives ("top five", verb "signal"); `banned_terms` narrowed (top as
        rank/position allowed, signal only as noun). gpt-oss final 0.76 -> 0.82.
      - Generation ran with the old verifier, so the refine loop chased these
        false positives (wasted attempts, mostly gpt-oss).
- [x] Prompt ad-v3 (2026-11-12): `[DEC]` copies `decision` as is
      (`background-like`, canonical in the verifier; `background` still
      accepted as alias for ad-v2 narratives). ad100 and v17 reports unchanged.
- [x] Caveat only when the two bands differ (2026-11-12, option A), in
      `artefact_anomaly.build`; the gate and its calibration untouched, its
      verdict kept as `meta.gate_regime`. Only possible inside `extreme`. On
      ad100: 332892, 384135 caveat -> full; regimes 70 full, 12 caveat, 18
      abstain; artefacts unchanged.
- [x] Refine loop stop criterion (2026-11-12): no early stop, max_attempts 5
      stays. `blocks_report --attempt final` now ends with ATTEMPTS TO DELIVERY
      (mean ± sd of the first passing attempt over jets delivered within 5,
      histogram, fails), re-verified. Saved in v17_report_final.txt and
      ad100_report_final.txt.
- [ ] Review of every piece, top and AD: `REVIEW_CHECKLIST.md`.
- Physics read (heuristic scan of prong observables, ~650 sentences): two
  direction misreadings in delivered narratives, flash-lite 101963 (high C2
  "favouring two-pronged") and gpt-oss 264653 (tau32 at pct 93 "modest
  three-prong character"). No unlicensed physics (new physics, particle
  origin). Caveat stated in 66/69 caveat narratives (flash-lite misses 3).
