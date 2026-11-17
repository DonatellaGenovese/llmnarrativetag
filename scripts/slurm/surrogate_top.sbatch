#!/bin/bash
# Part 2 on Leonardo, for one ParT teacher: the feature table with that
# teacher's logits, then the Ridge / EBM / GBDT surrogates fit to t and their
# metrics. One output directory per teacher seed:
#
#   sbatch --export=ALL,SEED=42 scripts/slurm/surrogate_top.sbatch
#   sbatch --export=ALL,SEED=42,SEED_VARIANCE=1 scripts/slurm/surrogate_top.sbatch
#
#   teacher   particle_transformer/training/TopLandscape/ParT-FineTune/*_seed<SEED>/
#   output    surrogate/outputs/top_seed<SEED>/{features,models,seed_variance}
#
# PRED_DIR and OUT override both, for a run without a seed in its name.
#
# Features: the ten observables do not depend on the teacher, so by default they
# are copied from an existing table (OBS_FROM, default surrogate/outputs/
# top_unseeded/features) and only the six teacher columns are replaced, with a
# row-by-row alignment check (surrogate/swap_teacher.py): minutes, not the hour
# FastJet takes. Set OBS_FROM=none to recompute them from the constituents.
#
# SEED_VARIANCE=1 also refits the surrogates over five model seeds on this
# teacher (surrogate/seed_variance.py), the spread at a fixed teacher.
#
# The tracked numbers in surrogate/metrics/top/ are NOT overwritten: they belong
# to the published run, and the comparison is made against them.

#SBATCH --job-name=surrogate-top
#SBATCH --account=IscrC_XJET-HEP
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
# A quarter of a booster node, the shape GPU jobs take: it fits where one is
# freed, where a whole node's 32 cores rarely are. Nothing here uses the GPU;
# three single-threaded feature builds and a 4-bag EBM need eight cores.
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

PT=particle_transformer
DATA=$PT/datasets/TopLandscape

# --- which teacher, which output ----------------------------------------------
if [ -n "${PRED_DIR:-}" ]; then
  PRED=$PRED_DIR
  OUT=${OUT:?with PRED_DIR, set OUT too}
else
  SEED=${SEED:?set SEED (sbatch --export=ALL,SEED=42 ...) or PRED_DIR and OUT}
  mapfile -t runs < <(ls -d $PT/training/TopLandscape/ParT-FineTune/*_seed"${SEED}"/predict_output 2>/dev/null)
  [ ${#runs[@]} -eq 1 ] || { echo "expected one run with seed $SEED, found ${#runs[@]}:"; printf '  %s\n' "${runs[@]}"; exit 1; }
  PRED=${runs[0]}
  OUT=${OUT:-surrogate/outputs/top_seed${SEED}}
fi
echo "### $(date '+%F %T')  node $(hostname)  job $SLURM_JOB_ID  cpus $SLURM_CPUS_PER_TASK"
echo "teacher logits: $PRED"
grep -sE '"seed"' "$(dirname "$PRED")/run_manifest.json" || echo "  (no run_manifest.json beside it)"
echo "output:         $OUT"
mkdir -p "$OUT/features" "$OUT/models"

declare -A PREDF=( [train]=$PRED/TopLandscape_ParT-FineTune_train_logits.parquet
                   [val]=$PRED/TopLandscape_ParT-FineTune_val_logits.parquet
                   [test]=$PRED/TopLandscape_ParT-FineTune_logits.parquet )
for s in train val test; do [ -f "${PREDF[$s]}" ] || { echo "missing ${PREDF[$s]}"; exit 1; }; done

# --- features --------------------------------------------------------------------
OBS_FROM=${OBS_FROM:-surrogate/outputs/top_unseeded/features}
if [ "$OBS_FROM" != "none" ]; then
  echo "### features: observables from $OBS_FROM, teacher columns from this run  $(date '+%T')"
  for s in train val test; do
    python -m surrogate.swap_teacher --features "$OBS_FROM/features_${s}.parquet" \
      --pred-parquet "${PREDF[$s]}" --out "$OUT/features/features_${s}.parquet"
  done
else
  # FastJet, three splits at once (one jet at a time, ~300 jets/s each).
  pids=()
  for s in train val test; do
    python -m surrogate.build_features_top \
      --data-parquet $DATA/${s}_file.parquet --pred-parquet "${PREDF[$s]}" \
      --out "$OUT/features/features_${s}.parquet" --config surrogate/configs/top_basis.yaml \
      > slurm_logs/features_${s}_${SLURM_JOB_ID}.log 2>&1 &
    pids+=($!)
    echo "### features $s started (log slurm_logs/features_${s}_${SLURM_JOB_ID}.log)"
  done
  for p in "${pids[@]}"; do
    wait "$p" || { echo "feature build failed, see slurm_logs/features_*_${SLURM_JOB_ID}.log"; exit 1; }
  done
fi
echo "### features done  $(date '+%T')"
OUT=$OUT python - <<'EOF'
import os, pandas as pd
for s in ("train", "val", "test"):
    d = pd.read_parquet(f"{os.environ['OUT']}/features/features_{s}.parquet")
    nan = {k: int(v) for k, v in d.isna().sum().items() if v}
    print(f"{s:5s} {len(d):>9,} jets, {d.shape[1]} cols, NaN: {nan or 'none'}")
EOF

# --- surrogates ----------------------------------------------------------------
echo "### surrogates  $(date '+%T')"
python -m surrogate.train_surrogates_top \
  --train "$OUT/features/features_train.parquet" \
  --val "$OUT/features/features_val.parquet" \
  --test "$OUT/features/features_test.parquet" \
  --out-dir "$OUT/models"

echo "### classifier metrics  $(date '+%T')"
python -m surrogate.eval_classifier_metrics \
  --predictions "$OUT/models/test_predictions.parquet" \
  --model-dir "$OUT/models"

if [ "${SEED_VARIANCE:-0}" = "1" ]; then
  echo "### seed variance (5 model seeds, this teacher)  $(date '+%T')"
  python -m surrogate.seed_variance --feat-dir "$OUT/features" --out-dir "$OUT/seed_variance"
fi

echo "### done  $(date '+%F %T')"
ls -lh "$OUT/features" "$OUT/models"
