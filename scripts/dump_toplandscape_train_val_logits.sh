#!/bin/bash
# Dump ParT logits on TopLandscape train + val (scalars only; no pf_* tensors).
# Uses the existing best FineTune checkpoint.
#
# Run from the repository root:
#   ./scripts/dump_toplandscape_train_val_logits.sh

set -eo pipefail
# Weaver resolves its network and data configs relative to the ParT tree, and
# the datasets and checkpoints live there too, so this steps into it. Invoke it
# from the repository root; nothing here depends on your working directory.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../particle_transformer"
source env.sh
source .venv/bin/activate

CKPT="${1:-training/TopLandscape/ParT-FineTune/20260725-162630_example_ParticleTransformer_finetune_ranger_lr0.0001_batch512/net_best_epoch_state.pt}"
DATADIR="${DATADIR_TopLandscape:-./datasets/TopLandscape}"
OUTDIR="$(dirname "$CKPT")/predict_output"
mkdir -p "$OUTDIR" logs

NET=networks/example_ParticleTransformer_finetune.py
CFG=data/TopLandscape/top_kin_predict.yaml

run_one () {
  local split="$1"
  local infile="$2"
  local outfile="$3"
  echo "=== Predicting $split ==="
  python "$HERE/predict_TopLandscape_logits.py" \
    --data-test "$infile" \
    --data-config "$CFG" \
    --network-config "$NET" \
    --model-prefix "$CKPT" \
    --gpus 0 \
    --batch-size 512 \
    --num-workers 1 \
    --skip-inputs \
    --predict-output "$outfile"
}

run_one val  "${DATADIR}/val_file.parquet"  "${OUTDIR}/TopLandscape_ParT-FineTune_val_logits.parquet"
run_one train "${DATADIR}/train_file.parquet" "${OUTDIR}/TopLandscape_ParT-FineTune_train_logits.parquet"

echo "DONE"
ls -lh "${OUTDIR}"/TopLandscape_ParT-FineTune_{train,val,}_logits.parquet 2>/dev/null || \
  ls -lh "${OUTDIR}"/*logits.parquet
