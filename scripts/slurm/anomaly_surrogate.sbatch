#!/bin/bash
# AD experiments E2: Ridge / EBM / GBDT fit to the IAD logit, for every seed and
# signal fraction of E1 (latent input), on val (half fits, half selects) and
# reported on test; then the EBM's own seed variance on one detector. Needs the
# outputs of scripts/slurm/anomaly_iad.sbatch. See TODO.md, "Anomaly detection".
#
#   cd ~/llmnarrativetag && sbatch scripts/slurm/anomaly_surrogate.sbatch
#
# Writes, per detector,
#   anomaly/outputs/seed<s>/surrogate_inputs/iad_f<f>_latent/   the three tables
#   anomaly/outputs/seed<s>/surrogate/iad_f<f>_latent/          the fitted models
# and anomaly/outputs/seed<VAR_SEED>/seed_variance/iad_f<VAR_F>_latent/.
# The surrogates' own seed is the config's (42); seed_variance uses 42..46.

#SBATCH --job-name=anomaly-surrogate
#SBATCH --account=IscrC_XJET-HEP
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Quarter-node shape, as the other surrogate job: it schedules where a GPU job
# would. Nothing here uses the GPU.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$HOME/llmnarrativetag}"
cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate part-surrogate
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEEDS=${SEEDS:-"0 1 2 3 4"}
FRACTIONS=${FRACTIONS:-"0.1 0.05 0.01"}   # as the run names write them (f:g)
VAR_SEED=${VAR_SEED:-0}
VAR_F=${VAR_F:-0.1}
O=anomaly/outputs

echo "### $(date '+%F %T')  node $(hostname)  job $SLURM_JOB_ID  cpus $SLURM_CPUS_PER_TASK"
for s in $SEEDS; do
  for f in $FRACTIONS; do
    run=iad_f${f}_latent
    sc=$O/seed$s/scores/$run
    [ -d "$sc" ] || { echo "missing $sc (run anomaly_iad.sbatch first)"; exit 1; }
    inp=$O/seed$s/surrogate_inputs/$run
    echo "### seed $s  $run  $(date '+%T')"
    python -m anomaly.prepare_surrogate_inputs --scores "$sc" --out "$inp"
    python -m surrogate.train_surrogates_top \
      --train "$inp/features_train.parquet" --val "$inp/features_val.parquet" \
      --test "$inp/features_test.parquet" --out-dir "$O/seed$s/surrogate/$run" \
      | grep -E "target:|R²"
  done
done

echo "### seed variance: detector seed $VAR_SEED, f=$VAR_F, EBM seeds 42-46  $(date '+%T')"
python -m surrogate.seed_variance \
  --feat-dir "$O/seed$VAR_SEED/surrogate_inputs/iad_f${VAR_F}_latent" \
  --out-dir "$O/seed$VAR_SEED/seed_variance/iad_f${VAR_F}_latent"

echo "### evaluation  $(date '+%T')"
python -m anomaly.eval_surrogates --out $O/summary_surrogate.json

echo "### abstention gate  $(date '+%T')"
python -m anomaly.gate --out $O/summary_gate.json

echo "### done  $(date '+%F %T')"
