#!/bin/bash
# Official TopLandscape surrogate: train/val/test with ParT teacher logits.
#
# Prerequisites: ParT logits parquet for train/val/test under predict_output/.
# Usage:
#   ./surrogate/run_official_top.sh              # full splits
#   ./surrogate/run_official_top.sh 5000         # cap each split (smoke)

set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MAX_JETS="${1:-}"
OUT="$REPO/surrogate/outputs/top"
PRED_DIR="$REPO/training/TopLandscape/ParT-FineTune/20260725-162630_example_ParticleTransformer_finetune_ranger_lr0.0001_batch512/predict_output"
DATA="$REPO/datasets/TopLandscape"

set +u
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate part-surrogate
set -u

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

build_one () {
  local split="$1"
  local data="$2"
  local pred="$3"
  local out="$4"
  local extra=()
  if [[ -n "$MAX_JETS" ]]; then
    extra=(--max-jets "$MAX_JETS")
  fi
  echo "=== Build features ($split) ==="
  python -m surrogate.build_features_top \
    --data-parquet "$data" \
    --pred-parquet "$pred" \
    --out "$out" \
    --config "$REPO/surrogate/configs/top_basis.yaml" \
    "${extra[@]}"
}

# Prefer named train/val dumps; fall back to legacy test-only name for test
TEST_PRED="${PRED_DIR}/TopLandscape_ParT-FineTune_test_logits.parquet"
if [[ ! -f "$TEST_PRED" ]]; then
  TEST_PRED="${PRED_DIR}/TopLandscape_ParT-FineTune_logits.parquet"
fi

build_one train "${DATA}/train_file.parquet" \
  "${PRED_DIR}/TopLandscape_ParT-FineTune_train_logits.parquet" \
  "${OUT}/features/features_train.parquet"

build_one val "${DATA}/val_file.parquet" \
  "${PRED_DIR}/TopLandscape_ParT-FineTune_val_logits.parquet" \
  "${OUT}/features/features_val.parquet"

build_one test "${DATA}/test_file.parquet" \
  "$TEST_PRED" \
  "${OUT}/features/features_test.parquet"

echo "=== Audit (train features) ==="
python -m surrogate.audit \
  --features "${OUT}/features/features_train.parquet" \
  --out-dir "${OUT}/audit"

echo "=== Fit Ridge / EBM / GBDT (official splits) ==="
extra=()
if [[ -n "$MAX_JETS" ]]; then
  extra=(--max-jets "$MAX_JETS")
fi
python -m surrogate.train_surrogates_top \
  --train "${OUT}/features/features_train.parquet" \
  --val "${OUT}/features/features_val.parquet" \
  --test "${OUT}/features/features_test.parquet" \
  --out-dir "${OUT}/models" \
  "${extra[@]}"

echo "DONE → $OUT"
