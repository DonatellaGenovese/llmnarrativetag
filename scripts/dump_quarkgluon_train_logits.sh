#!/bin/bash
# Dump ParT logits on QuarkGluon train (scalars only).
# Test logits already exist from infer_QuarkGluon_logits.sh.
#
# Usage:
#   source .venv/bin/activate
#   ./dump_quarkgluon_train_logits.sh

set -eo pipefail
cd "$(dirname "$0")"
source env.sh
source .venv/bin/activate

CKPT="${1:-training/QuarkGluon/ParT-FineTune/20260726-124952_example_ParticleTransformer_finetune_ranger_lr0.0001_batch512/net_best_epoch_state.pt}"
DATADIR="${DATADIR_QuarkGluon:-./datasets/QuarkGluon}"
OUTDIR="$(dirname "$CKPT")/predict_output"
mkdir -p "$OUTDIR" logs

NET=networks/example_ParticleTransformer_finetune.py
CFG=data/QuarkGluon/qg_kinpid_predict.yaml

echo "=== Predicting QuarkGluon train ==="
python predict_TopLandscape_logits.py \
  --data-test "${DATADIR}/train_file_*.parquet" \
  --data-config "$CFG" \
  --network-config "$NET" \
  --model-prefix "$CKPT" \
  --gpus 0 \
  --batch-size 512 \
  --num-workers 1 \
  --skip-inputs \
  --predict-output "${OUTDIR}/QuarkGluon_ParT-FineTune_kinpid_train_logits.parquet"

# alias test dump name for clarity
if [[ -f "${OUTDIR}/QuarkGluon_ParT-FineTune_kinpid_logits.parquet" ]]; then
  ln -sfn QuarkGluon_ParT-FineTune_kinpid_logits.parquet \
    "${OUTDIR}/QuarkGluon_ParT-FineTune_kinpid_test_logits.parquet"
fi

echo "DONE"
ls -lh "${OUTDIR}"/QuarkGluon_ParT-FineTune_kinpid_*logits.parquet
