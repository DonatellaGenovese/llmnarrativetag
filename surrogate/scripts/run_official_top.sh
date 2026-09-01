#!/bin/bash
# Official TopLandscape surrogate: train/val/test with ParT teacher logits.
#
# Prerequisites: ParT logits parquet for train/val/test under predict_output/.
# Run from the repository root:
#   ./surrogate/scripts/run_official_top.sh              # full splits
#   ./surrogate/scripts/run_official_top.sh 5000         # cap each split (smoke)

set -eo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

MAX_JETS="${1:-}"
OUT="$REPO/surrogate/outputs/top"
# The teacher's data and checkpoints live inside the vendored ParT tree.
PT="$REPO/particle_transformer"
DATA="$PT/datasets/TopLandscape"

# Weaver names each training run after its start time, so the directory differs
# on every machine. Discover it rather than hardcode one; PRED_DIR from the
# environment wins, which is how you pick between two runs.
if [[ -z "${PRED_DIR:-}" ]]; then
  mapfile -t _found < <(ls -d "$PT"/training/TopLandscape/ParT-FineTune/*/predict_output 2>/dev/null)
  case ${#_found[@]} in
    0) echo "No predict_output under $PT/training/TopLandscape/ParT-FineTune/."
       echo "Dump the teacher logits first, or set PRED_DIR."; exit 1 ;;
    1) PRED_DIR="${_found[0]}" ;;
    *) echo "Several ParT runs found; set PRED_DIR to the one you want:"
       printf '  %s\n' "${_found[@]}"; exit 1 ;;
  esac
fi
echo "teacher logits: $PRED_DIR"

set +u
# Resolve conda without assuming an install prefix. `conda info --base` alone
# is not enough: these scripts are meant to be run detached, where conda is not
# on PATH, which is exactly when a hardcoded prefix used to be needed.
_base="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"
if [ -z "$_base" ] && command -v conda >/dev/null 2>&1; then _base="$(conda info --base)"; fi
for _c in "$_base" /opt/miniforge3 "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
  if [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ]; then . "$_c/etc/profile.d/conda.sh"; break; fi
done
command -v conda >/dev/null 2>&1 || { echo "conda not found: set CONDA_EXE"; exit 1; }
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
