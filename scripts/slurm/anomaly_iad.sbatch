#!/bin/bash
# AD experiments E1: the unsupervised baselines once, then the IAD detector on
# five seeds at three signal fractions, on the frozen ParT latent and on the ten
# observables. See TODO.md, "Anomaly detection".
#
#   cd ~/llmnarrativetag && sbatch scripts/slurm/anomaly_iad.sbatch
#   sbatch --export=ALL,SEEDS="0 1",FRACTIONS="0.10" scripts/slurm/anomaly_iad.sbatch   # a subset
#
# Writes anomaly/outputs/baselines/ and anomaly/outputs/seed<s>/ (scores per jet
# on train/val/test, the classifiers, transfer_summary.json, run_manifest.json).
# The supervised ceiling is trained on the first seed only.

#SBATCH --job-name=anomaly-iad
#SBATCH --account=IscrC_XJET-HEP
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$HOME/llmnarrativetag}"
cd "$REPO"
# torch lives in the ParT venv; it carries pandas, pyarrow and scikit-learn too.
source particle_transformer/.venv/bin/activate
export PYTHONPATH="$REPO"

SEEDS=${SEEDS:-"0 1 2 3 4"}
FRACTIONS=${FRACTIONS:-"0.10 0.05 0.01"}
EMB=surrogate/outputs/top/embeddings
# The observables and labels are the same for every teacher; top/features is
# the current one (seed 42). Only the IAD's own logits are used downstream.
FEAT=surrogate/outputs/top/features

echo "### $(date '+%F %T')  node $(hostname)  job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU visible to torch')"
for s in train val test; do
  [ -f "$EMB/xcls_$s.npz" ] || { echo "missing $EMB/xcls_$s.npz"; exit 1; }
  [ -f "$FEAT/features_$s.parquet" ] || { echo "missing $FEAT/features_$s.parquet"; exit 1; }
done

echo "### baselines  $(date '+%T')"
python -m anomaly.baselines --emb-dir "$EMB" --feat-dir "$FEAT" --out-dir anomaly/outputs/baselines

first=1
for s in $SEEDS; do
  extra=()
  [ $first -eq 1 ] || extra=(--skip-supervised)
  first=0
  echo "### IAD seed $s  fractions $FRACTIONS  $(date '+%T')"
  python -m anomaly.transfer --seed "$s" --fractions $FRACTIONS \
    --emb-dir "$EMB" --feat-dir "$FEAT" --out-dir "anomaly/outputs/seed$s" "${extra[@]}"
done

echo "### summary  $(date '+%T')"
python -m anomaly.summarize_seeds --out anomaly/outputs/summary_iad.json

echo "### done  $(date '+%F %T')"
