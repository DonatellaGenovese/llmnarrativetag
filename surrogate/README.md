# Surrogate pipeline (TopLandscape first)

Env: `conda activate part-surrogate` (or `source surrogate_env.sh`).

## Layout

| File | Role |
|---|---|
| `basis_definitions.md` | Pre-registered SoftDrop / τ / ECF definitions |
| `configs/top_basis.yaml` | Knobs (β, truncation, fit settings) |
| `fjcontrib_loader.py` | Load SoftDrop / Nsubjettiness / ECF via cppyy |
| `observables_top.py` | Per-jet observable computation |
| `build_features_top.py` | Join observables ↔ ParT logits → feature parquet |
| `audit.py` | Correlation / VIF / condition number (+ mass residual helper) |
| `train_surrogates_top.py` | Ridge (log+z) / EBM (raw) / GBDT (raw) on `t` |
| `run_official_top.sh` | Official train/val/test end-to-end |
| `run_prototype_top.sh` | Small-N smoke test (legacy single-file split) |

## 1) ParT teacher logits (train + val + test)

From repo root, with the torch venv:

```bash
source .venv/bin/activate
./dump_toplandscape_train_val_logits.sh   # val then train (--skip-inputs)
# test already exists as TopLandscape_ParT-FineTune_logits.parquet
```

Outputs under the FineTune `predict_output/`:
- `TopLandscape_ParT-FineTune_train_logits.parquet`
- `TopLandscape_ParT-FineTune_val_logits.parquet`
- `TopLandscape_ParT-FineTune_logits.parquet` (test)

## 2) Official surrogate (fit train / select val / report test)

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate part-surrogate
./surrogate/run_official_top.sh           # full
./surrogate/run_official_top.sh 5000      # smoke cap per split
```

Outputs: `surrogate/outputs/top/{features,audit,models}/`.

## 3) Post-hoc classifier metrics (after training)

```bash
# test only, from saved predictions (fast)
python -m surrogate.eval_classifier_metrics \
  --predictions surrogate/outputs/top/models/test_predictions.parquet

# all splits, reload models
python -m surrogate.eval_classifier_metrics
```

Reports MAE/RMSE vs ParT scores, decision agreement with ParT, Acc/AUC vs truth (+ ParT baseline).

## 4) Label-trained classifiers (comparison)

Same observables / splits, but train Logistic / EBM / GBDT **on truth labels** (not on ParT `t`):

```bash
conda activate part-surrogate
./surrogate/run_label_classifiers.sh top           # or qg / both
./surrogate/run_label_classifiers.sh both 5000     # smoke
```

Outputs: `surrogate/outputs/{top,qg}/models_label/` and `compare_surrogate_vs_label.json`
(Acc/AUC side-by-side: imitate-ParT vs train-on-label vs ParT baseline).

## QuarkGluon (official)

```bash
# 1) dump train logits (test already done)
source .venv/bin/activate
./dump_quarkgluon_train_logits.sh

# 2) features + Ridge/EBM/GBDT + classifier metrics
conda activate part-surrogate
./surrogate/run_official_qg.sh           # full
./surrogate/run_official_qg.sh 5000      # smoke
```

QG has no val files → val is carved from train with `train_val_split: 0.8889`.
Observables are analytic (no FastJet): `w_pf`, `pTD`, `C_02`, angularities, `S_frag`, ellipticity, mass, multiplicity, `S_PID`, `E_Q`.

## Notes

- Observables use **original parquet constituents**, not weaver `pf_*`.
- Target is `t = logit_diff` (Top: \(z_{\mathrm{top}}-z_{\mathrm{QCD}}\); QG: \(z_Q-z_G\)).
- Top names `C2_double_b1` / `C3_double_b1` avoid the C₂ naming collision with QuarkGluon papers.
