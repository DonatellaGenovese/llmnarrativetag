#!/bin/bash
# Train Logistic / EBM / GBDT on truth labels (same features as ParT surrogates).
#
# Run from the repository root:
#   ./surrogate/scripts/run_label_classifiers.sh         # full TopLandscape
#   ./surrogate/scripts/run_label_classifiers.sh 5000    # smoke

set -eo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

MAX_JETS="${1:-}"

set +u
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate part-surrogate
set -u
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

extra=()
if [[ -n "$MAX_JETS" ]]; then
  extra=(--max-jets "$MAX_JETS")
fi

run_one () {
  local name="$1"
  local cfg="$2"
  local feat="$REPO/surrogate/outputs/${name}/features"
  local out="$REPO/surrogate/outputs/${name}/models_label"
  local surr_metrics="$REPO/surrogate/outputs/${name}/models/classifier_metrics.json"

  echo "=== Label classifiers ($name) ==="
  python -m surrogate.train_classifiers_label \
    --train "${feat}/features_train.parquet" \
    --val "${feat}/features_val.parquet" \
    --test "${feat}/features_test.parquet" \
    --config "$cfg" \
    --out-dir "$out" \
    "${extra[@]}"

  if [[ -f "$surr_metrics" ]]; then
    echo "=== Compare imitate-ParT vs train-on-label ($name) ==="
    python -m surrogate.compare_surrogate_vs_label \
      --surrogate-metrics "$surr_metrics" \
      --label-metrics "${out}/metrics_label.json" \
      --out "$REPO/surrogate/outputs/${name}/compare_surrogate_vs_label.json"
  else
    echo "WARN: missing $surr_metrics — skip compare"
  fi
}

run_one top "$REPO/surrogate/configs/top_basis.yaml"

echo "DONE"
