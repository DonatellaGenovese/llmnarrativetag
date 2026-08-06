#!/bin/bash
# Run ParT / ParticleNet inference on the TopLandscape test set.
#
# Prerequisites:
#   1. TopLandscape data:  ./get_datasets.py TopLandscape
#   2. Trained weights from train_TopLandscape.sh (JetClass .pt files in models/
#      are NOT drop-in compatible — TopLandscape is a 2-class task).
#
# Usage:
#   ./infer_TopLandscape.sh ParT /path/to/net_best_epoch_state.pt [extra weaver args...]
#   ./infer_TopLandscape.sh ParT-FineTune training/TopLandscape/ParT-FineTune/.../net_best_epoch_state.pt
#   ./infer_TopLandscape.sh PN training/TopLandscape/PN/.../net_best_epoch_state.pt
#
# Output: ROOT file with scores/labels (default: predict_output next to the checkpoint).

set -euo pipefail
set -x

source env.sh

DATADIR=${DATADIR_TopLandscape:-./datasets/TopLandscape}
model=${1:?Usage: $0 MODEL CHECKPOINT [extra weaver args...]}
ckpt=${2:?Usage: $0 MODEL CHECKPOINT [extra weaver args...]}
shift 2

if [[ ! -f "$ckpt" ]]; then
  echo "Checkpoint not found: $ckpt"
  exit 1
fi

case "$model" in
  ParT)
    modelopts="networks/example_ParticleTransformer.py"
    ;;
  ParT-FineTune)
    modelopts="networks/example_ParticleTransformer_finetune.py"
    ;;
  PN)
    modelopts="networks/example_ParticleNet.py"
    ;;
  PN-FineTune)
    modelopts="networks/example_ParticleNet_finetune.py"
    ;;
  PFN)
    modelopts="networks/example_PFN.py"
    ;;
  PCNN)
    modelopts="networks/example_PCNN.py"
    ;;
  *)
    echo "Invalid model $model (expected ParT|ParT-FineTune|PN|PN-FineTune|PFN|PCNN)"
    exit 1
    ;;
esac

FEATURE_TYPE=kin
outdir="$(dirname "$ckpt")/predict_output"
mkdir -p "$outdir"

weaver \
  --predict \
  --data-test "${DATADIR}/test_file.parquet" \
  --data-config "data/TopLandscape/top_${FEATURE_TYPE}.yaml" \
  --network-config $modelopts \
  --model-prefix "$ckpt" \
  --gpus 0 \
  --batch-size 512 \
  --num-workers 1 --fetch-step 1 --in-memory \
  --predict-output "${outdir}/TopLandscape_${model}_pred.root" \
  "$@"
