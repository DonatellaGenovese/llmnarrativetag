#!/bin/bash
# Train Logistic / EBM / GBDT on truth labels (same features as ParT surrogates).
#
# Usage:
#   ./surrogate/run_label_classifiers.sh top           # full TopLandscape
#   ./surrogate/run_label_classifiers.sh qg            # full QuarkGluon
#   ./surrogate/run_label_classifiers.sh both          # both
#   ./surrogate/run_label_classifiers.sh top 5000      # smoke

set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

DATASET="${1:-both}"
MAX_JETS="${2:-}"

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

case "$DATASET" in
  top)
    run_one top "$REPO/surrogate/configs/top_basis.yaml"
    ;;
  qg)
    run_one qg "$REPO/surrogate/configs/qg_basis.yaml"
    ;;
  both)
    run_one top "$REPO/surrogate/configs/top_basis.yaml"
    run_one qg "$REPO/surrogate/configs/qg_basis.yaml"
    ;;
  *)
    echo "Usage: $0 {top|qg|both} [max_jets]"
    exit 1
    ;;
esac

echo "DONE"
