#!/bin/bash
# Prototype TopLandscape surrogate chain (small N for review).
# Usage:
#   ./surrogate/run_prototype_top.sh [MAX_JETS=2000]

set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MAX_JETS="${1:-2000}"
OUT="$REPO/surrogate/outputs/top_proto"

# conda activate scripts are not `set -u` safe
set +u
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate part-surrogate
set -u

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== 1) Build features (max_jets=$MAX_JETS) ==="
python -m surrogate.build_features_top \
  --max-jets "$MAX_JETS" \
  --out "$OUT/features.parquet"

echo "=== 2) Correlation audit ==="
python -m surrogate.audit \
  --features "$OUT/features.parquet" \
  --out-dir "$OUT/audit"

echo "=== 3) Fit Ridge / EBM / GBDT ==="
python -m surrogate.train_surrogates_top \
  --features "$OUT/features.parquet" \
  --out-dir "$OUT/models"

echo "DONE → $OUT"
