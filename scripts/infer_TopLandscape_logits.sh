#!/bin/bash
# Predict on TopLandscape and save raw logits (+ softmax + jet observables).
#
# Usage:
#   ./infer_TopLandscape_logits.sh ParT-FineTune /path/to/net_best_epoch_state.pt
#
# Output parquet next to the checkpoint under predict_output/.

set -euo pipefail
set -x

source env.sh

DATADIR=${DATADIR_TopLandscape:-./datasets/TopLandscape}
model=${1:?Usage: $0 MODEL CHECKPOINT}
ckpt=${2:?Usage: $0 MODEL CHECKPOINT}

if [[ ! -f "$ckpt" ]]; then
  echo "Checkpoint not found: $ckpt"
  exit 1
fi

case "$model" in
  ParT) modelopts="networks/example_ParticleTransformer.py" ;;
  ParT-FineTune) modelopts="networks/example_ParticleTransformer_finetune.py" ;;
  PN) modelopts="networks/example_ParticleNet.py" ;;
  PN-FineTune) modelopts="networks/example_ParticleNet_finetune.py" ;;
  PFN) modelopts="networks/example_PFN.py" ;;
  PCNN) modelopts="networks/example_PCNN.py" ;;
  *)
    echo "Invalid model $model"
    exit 1
    ;;
esac

outdir="$(dirname "$ckpt")/predict_output"
mkdir -p "$outdir"

python predict_TopLandscape_logits.py \
  --data-test "${DATADIR}/test_file.parquet" \
  --data-config data/TopLandscape/top_kin_predict.yaml \
  --network-config "$modelopts" \
  --model-prefix "$ckpt" \
  --gpus 0 \
  --batch-size 512 \
  --num-workers 1 \
  --predict-output "${outdir}/TopLandscape_${model}_logits.parquet"
