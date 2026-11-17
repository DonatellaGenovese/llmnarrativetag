# Handoff — llmnarrativetag (state at 2026-11-13, evening)

For a new session picking up this work. Repo `~/llmnarrativetag` on CINECA
Leonardo (user `dgenoves`), branch `anomaly`, **nothing committed** (commit only
when asked). User writes in Italian; code comments in English. See also
`TODO.md` and `REVIEW_CHECKLIST.md`.

## Constraints
- SLURM account **`IscrC_XJET-HEP`**; partition `boost_usr_prod`; QOS `normal`
  (24 h) or `boost_qos_dbg`. Nodes 4×A100 64 GB, driver 535. Login node: limit
  threads (`OMP_NUM_THREADS=2 LOKY_MAX_CPU_COUNT=2`).
- **Ask before any paid API run** (Gemini/Vertex, DeepSeek, Mistral, Meta).
  Keys in `~/.deepseek_key`, `~/.mistral_key` (~12 € credit left), Vertex via
  `~/.gcp/vertex_env.sh`; Meta key not created yet (`~/.meta_key`). Anthropic
  credit exhausted. Never print keys.
- Envs: `conda activate part-surrogate` + `export PYTHONPATH=$PWD`.
- vLLM jobs: `sbatch --export=ALL,MODEL=..,BLOCK=..,JETS=..[,CFG=..] scripts/slurm/vllm_generate.sbatch`;
  **qwen3.8-27b needs `--gres=gpu:1 --cpus-per-task=8 --mem=120G`** (it fails on 2 GPUs).

## Done since 2026-11-11
- ad100 evaluated (5 generators); verifier fixes: `[DEC]background-like` accepted
  (canonical), `identity_claim` false positives removed (`top` as rank, `signal`
  as verb, `W` case-sensitive). v17/ad100 reports unchanged by later edits.
- Caveat in AD only when detector and surrogate bands differ (artefact_anomaly).
- **New top gate** `narrative/gate_top.py` (sign disagreement → abstain; rank
  difference > 7.5 pct points → caveat; calibrated on val): test 68/26/6 %
  full/caveat/abstain (old residual gate 50/29/21).
- Prompts `top-v17`, `ad-v4`; top artefact without `-0.0`.
- `blocks_report --attempt final` adds ATTEMPTS TO DELIVERY.
- Samples: `top500_jets.json` (decision × correctness 150/150/100/100, 419
  narrated), `pilot100_jets.json` (stratified subset of it).
- README rewritten (both use cases, current numbers).
- **Pilot100 done, both use cases, 5 generators**:
  `narrative/outputs/{top,anomaly}/narratives/pilot100_report_{first,final}.txt`.
  TOTAL first → final: top gemini-flash .81→1.00, deepseek .85→.99, lite
  .62→.91, qwen .56→.91, gpt-oss .15→.80; AD .85→1.00, .87→.99, .57→.96,
  .72→.98, .22→.85. One deepseek AD narrative fails Ident (check it).

## Decisions (2026-11-13)
- Judges: **GLM 5.3, Mistral Large 4, Muse Spark 1.3** (Sonnet dropped; code in
  `evaluation/judge.py`, `run_judges.sh`; Meta endpoint `https://api.meta.ai/v1`
  unverified, override `META_BASE_URL`).
- **The judge sees the narrative only** (tags stripped), as before; no artefact
  or glossary. Open: give the new human set the narrative only too (v16 human
  items carried the artefact).
- Judges evaluate only the **top-3 generators** (gemini-3.5-flash,
  deepseek-flash, + qwen3.8-27b proposed over flash-lite); humans a subsample
  (~60 narratives proposed). N jets per use case still to fix (200 proposed).

## Judge tests (2026-11-16) — paused here to work on slides and paper
- Rubric drafts `evaluation/judge_prompt_v3.yaml` (top) and
  `judge_prompt_ad_v1.yaml` (AD): judge-v2 format (0–3 anchors, narrative
  only), Internal Consistency dropped, **Physical Soundness** added (correctness
  caps the score, 3 needs a physical picture), Overall Satisfaction trimmed.
  Summary for colleagues: `evaluation/judge_rubrics.md`.
- `judge.py`: dimensions read from the prompt, `--config` (AD), `--jets`.
- Test on 120 narratives (20 jets × top-3 generators × 2 tasks,
  `evaluation/results/judge_test/`) and stress test with injected defects
  (`evaluation/make_stress.py`, `evaluation/results/judge_stress/`). Muse
  detects all injected physics errors (where headroom), GLM most; Q1/Q3 near
  ceiling on real narratives but react to gross defects. GLM–Muse agree on Q2.
- **Mistral Large 4 dropped**; judges kept: GLM 5.3 and Muse Spark 1.3 (Meta
  key in `~/.meta_key`, endpoint verified). Main judge still to choose.
- Open: reformulate Q3 (coherence?); Q2's 3 unreachable without a physical
  synthesis in the narratives.

## Next (earlier plan, partly done)
1. **Rubric**: write `judge-v3` (top) and `judge-ad-v1` (AD), keep `judge-v2`
   runnable. Agreed proposal, 4 questions, 0–3: Clarity, Internal Consistency,
   **Physical Soundness** (new: reading applied in the direction of the stated
   value, no invented physics; AD: no identity claims), Overall Satisfaction.
   Anti-saturation: the judge lists problems (quoted spans) per question first,
   then a fixed mapping (3 = none after checking the list, 2 = one minor, 1 =
   one serious or ≥2 minor, 0 = unreliable). `judge.py --from-runs` reads only
   the top config: extend to AD.
2. **Sensitivity test** before humans: ~30 real pilot narratives + 20 good ones
   with one injected defect per question. Accept if, for GLM and at least one
   other judge, max score < 60 % on real narratives and the targeted question
   drops on ≥ 80 % of the injected copies. Cost ~$0.5–1 per judge per variant
   (ask before running).
3. Then human set (~60 narratives, 2 physicists), judge–human agreement, then
   the large run (Gemini/DeepSeek cost: ask first) and judges on it.
