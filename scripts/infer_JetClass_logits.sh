#!/bin/bash
# Inference on JetClass with official ParT JetClass weights (GPU).
#
# Usage:
#   source .venv/bin/activate && source env.sh
#   ./infer_JetClass_logits.sh full test          # paper default
#   ./infer_JetClass_logits.sh kinpid val
#   ./infer_JetClass_logits.sh full test 0.05     # fraction of files (smoke)

set -eo pipefail
cd "$(dirname "$0")"
source env.sh
source .venv/bin/activate

FEATURE_TYPE=${1:-full}   # kin | kinpid | full
SPLIT=${2:-test}          # val | test
FRAC=${3:-}               # optional: not used at file level; kept for future

DATADIR=${DATADIR_JetClass:-./datasets/JetClass}
case "$FEATURE_TYPE" in
  kin|kinpid|full) ;;
  *) echo "FEATURE_TYPE must be kin|kinpid|full"; exit 1 ;;
esac

CKPT="models/ParT_${FEATURE_TYPE}.pt"
CFG="data/JetClass/JetClass_${FEATURE_TYPE}.yaml"
NET="networks/example_ParticleTransformer.py"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT"; exit 1
fi
if [[ -z "${DATADIR_JetClass}" && ! -d "$DATADIR" ]]; then
  echo "JetClass data missing. Run: python download_jetclass_val_test.py -d datasets"
  exit 1
fi

OUTDIR="models/predict_output"
mkdir -p "$OUTDIR" logs

if [[ "$SPLIT" == "val" ]]; then
  DATA_TEST="${DATADIR}/Pythia/val_5M/*.root"
elif [[ "$SPLIT" == "test" ]]; then
  # one glob covering all 10 classes (sorted by shell/glob in predict script)
  DATA_TEST="${DATADIR}/Pythia/test_20M/*.root"
else
  echo "SPLIT must be val|test"; exit 1
fi

OUT="${OUTDIR}/JetClass_ParT_${FEATURE_TYPE}_${SPLIT}_logits.parquet"

echo "=== JetClass ParT inference ==="
echo "ckpt=$CKPT  config=$CFG  split=$SPLIT"
echo "data=$DATA_TEST"
echo "out=$OUT"

python predict_TopLandscape_logits.py \
  --data-test "$DATA_TEST" \
  --data-config "$CFG" \
  --network-config "$NET" \
  --model-prefix "$CKPT" \
  --gpus 0 \
  --batch-size 512 \
  --num-workers 2 \
  --skip-inputs \
  --predict-output "$OUT"

echo "DONE → $OUT"
