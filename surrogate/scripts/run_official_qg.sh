#!/bin/bash
# Official QuarkGluon surrogate: train(+val carve) / test with ParT teacher logits.
#
# Usage:
#   ./surrogate/run_official_qg.sh              # full
#   ./surrogate/run_official_qg.sh 5000         # cap (smoke)

set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MAX_JETS="${1:-}"
OUT="$REPO/surrogate/outputs/qg"
PRED_DIR="$REPO/training/QuarkGluon/ParT-FineTune/20260726-124952_example_ParticleTransformer_finetune_ranger_lr0.0001_batch512/predict_output"
DATA="$REPO/datasets/QuarkGluon"
CFG="$REPO/surrogate/configs/qg_basis.yaml"

set +u
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate part-surrogate
set -u
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

extra=()
if [[ -n "$MAX_JETS" ]]; then
  extra=(--max-jets "$MAX_JETS")
fi

echo "=== Build features (train all) ==="
python -m surrogate.build_features_qg \
  --data-parquet "${DATA}/train_file_*.parquet" \
  --pred-parquet "${PRED_DIR}/QuarkGluon_ParT-FineTune_kinpid_train_logits.parquet" \
  --out "${OUT}/features/features_train_all.parquet" \
  --config "$CFG" \
  "${extra[@]}"

echo "=== Carve val from train (train_val_split from config) ==="
python - <<PY
import yaml
from pathlib import Path
import pandas as pd
cfg = yaml.safe_load(open("$CFG"))
frac = float(cfg["train_val_split"])
df = pd.read_parquet("${OUT}/features/features_train_all.parquet")
n = len(df)
n_tr = int(round(frac * n))
df.iloc[:n_tr].to_parquet("${OUT}/features/features_train.parquet", index=False)
df.iloc[n_tr:].to_parquet("${OUT}/features/features_val.parquet", index=False)
print(f"train={n_tr}  val={n-n_tr}  (frac={frac})")
PY

TEST_PRED="${PRED_DIR}/QuarkGluon_ParT-FineTune_kinpid_test_logits.parquet"
if [[ ! -f "$TEST_PRED" ]]; then
  TEST_PRED="${PRED_DIR}/QuarkGluon_ParT-FineTune_kinpid_logits.parquet"
fi

echo "=== Build features (test) ==="
python -m surrogate.build_features_qg \
  --data-parquet "${DATA}/test_file_*.parquet" \
  --pred-parquet "$TEST_PRED" \
  --out "${OUT}/features/features_test.parquet" \
  --config "$CFG" \
  "${extra[@]}"

echo "=== Audit (train) ==="
python -m surrogate.audit \
  --features "${OUT}/features/features_train.parquet" \
  --out-dir "${OUT}/audit" \
  --obs w_pf pTD C_02 lambda_LHA lambda_21 r_lambda S_frag ellipticity mass pT n_pf n_Q S_PID E_Q

echo "=== Fit Ridge / EBM / GBDT ==="
python -m surrogate.train_surrogates_top \
  --train "${OUT}/features/features_train.parquet" \
  --val "${OUT}/features/features_val.parquet" \
  --test "${OUT}/features/features_test.parquet" \
  --config "$CFG" \
  --out-dir "${OUT}/models" \
  "${extra[@]}"

echo "=== Classifier metrics ==="
python -m surrogate.eval_classifier_metrics \
  --train "${OUT}/features/features_train.parquet" \
  --val "${OUT}/features/features_val.parquet" \
  --test "${OUT}/features/features_test.parquet" \
  --model-dir "${OUT}/models" \
  --config "$CFG" \
  --out "${OUT}/models/classifier_metrics.json"

echo "DONE → $OUT"
