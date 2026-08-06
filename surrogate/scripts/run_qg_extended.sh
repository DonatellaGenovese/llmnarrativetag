#!/bin/bash
# Rebuild QG features with extended observables and fit (compare to baseline 14).
set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

OUT="$REPO/surrogate/outputs/qg_ext"
PRED_DIR="$REPO/training/QuarkGluon/ParT-FineTune/20260726-124952_example_ParticleTransformer_finetune_ranger_lr0.0001_batch512/predict_output"
DATA="$REPO/datasets/QuarkGluon"
CFG="$REPO/surrogate/configs/qg_basis_ext.yaml"

set +u
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate part-surrogate
set -u
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Build extended features (train) ==="
python -m surrogate.build_features_qg \
  --data-parquet "${DATA}/train_file_*.parquet" \
  --pred-parquet "${PRED_DIR}/QuarkGluon_ParT-FineTune_kinpid_train_logits.parquet" \
  --out "${OUT}/features/features_train_all.parquet" \
  --config "$CFG"

python - <<PY
import yaml, pandas as pd
cfg = yaml.safe_load(open("$CFG"))
frac = float(cfg["train_val_split"])
df = pd.read_parquet("${OUT}/features/features_train_all.parquet")
n_tr = int(round(frac * len(df)))
df.iloc[:n_tr].to_parquet("${OUT}/features/features_train.parquet", index=False)
df.iloc[n_tr:].to_parquet("${OUT}/features/features_val.parquet", index=False)
print(f"train={n_tr} val={len(df)-n_tr}")
PY

TEST_PRED="${PRED_DIR}/QuarkGluon_ParT-FineTune_kinpid_test_logits.parquet"
[[ -f "$TEST_PRED" ]] || TEST_PRED="${PRED_DIR}/QuarkGluon_ParT-FineTune_kinpid_logits.parquet"

echo "=== Build extended features (test) ==="
python -m surrogate.build_features_qg \
  --data-parquet "${DATA}/test_file_*.parquet" \
  --pred-parquet "$TEST_PRED" \
  --out "${OUT}/features/features_test.parquet" \
  --config "$CFG"

echo "=== Fit ==="
python -m surrogate.train_surrogates_top \
  --train "${OUT}/features/features_train.parquet" \
  --val "${OUT}/features/features_val.parquet" \
  --test "${OUT}/features/features_test.parquet" \
  --config "$CFG" \
  --out-dir "${OUT}/models"

echo "=== Classifier metrics ==="
python -m surrogate.eval_classifier_metrics \
  --train "${OUT}/features/features_train.parquet" \
  --val "${OUT}/features/features_val.parquet" \
  --test "${OUT}/features/features_test.parquet" \
  --model-dir "${OUT}/models" \
  --config "$CFG" \
  --out "${OUT}/models/classifier_metrics.json"

echo "DONE → $OUT"
