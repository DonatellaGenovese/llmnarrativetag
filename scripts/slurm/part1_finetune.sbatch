#!/bin/bash
# Part 1 on Leonardo: fine-tune ParT on TopLandscape, then dump the raw logits
# of the best checkpoint on test, val and train. Those are the teacher scores
# the surrogate is fit to, and what surrogate/scripts/run_official_top.sh reads.
#
#   cd ~/llmnarrativetag && sbatch scripts/slurm/part1_finetune.sbatch
#   cd ~/llmnarrativetag && sbatch --export=ALL,SEED=7 scripts/slurm/part1_finetune.sbatch
#
# Seeded: SEED (default 42) fixes the initialisation, the data order and the
# kernels' choices, through scripts/weaver_seeded.py; the run directory carries
# it in its name (..._seed42) and a run_manifest.json with the seed, the
# command, the versions, the md5 of the checkpoint and logits, and the metrics.
#
# Checkpoints and logs land in $WORK (through the training/ and logs/ links in
# particle_transformer/), the job's own stdout in slurm_logs/.

#SBATCH --job-name=part-finetune
#SBATCH --account=IscrC_XJET-HEP
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$HOME/llmnarrativetag}"
cd "$REPO/particle_transformer"
source env.sh
source .venv/bin/activate

echo "### $(date '+%F %T')  node $(hostname)  job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU visible to torch')"

# --- fine-tune -------------------------------------------------------------
# 20 epochs, lr 1e-4 on the backbone and 50x on the new head, from ParT_kin.pt:
# the upstream recipe, unchanged. The vendored train script calls `weaver`;
# seeded_bin/ is put first on PATH so that call goes through weaver_seeded.py.
# COMMENT is the train script's own suffix for the run directory and log name.
export SEED=${SEED:-42}
export COMMENT="_seed${SEED}"
export PATH="$REPO/scripts/seeded_bin:$PATH"
[ "$(command -v weaver)" = "$REPO/scripts/seeded_bin/weaver" ] || { echo "seeded weaver not on PATH"; exit 1; }
echo "### fine-tune  seed $SEED  $(date '+%T')"
./train_TopLandscape.sh ParT-FineTune kin

# Weaver names the run after its start time; this job's run is the newest one
# carrying this seed.
RUN=$(ls -td training/TopLandscape/ParT-FineTune/*"${COMMENT}"/ | head -1)
CKPT=${RUN}net_best_epoch_state.pt
[ -f "$CKPT" ] || { echo "no best checkpoint at $CKPT"; exit 1; }
echo "### best checkpoint: $CKPT"

# --- raw logits ------------------------------------------------------------
echo "### logits test  $(date '+%T')"
"$REPO/scripts/infer_TopLandscape_logits.sh" ParT-FineTune "$PWD/$CKPT"
echo "### logits train + val  $(date '+%T')"
"$REPO/scripts/dump_toplandscape_train_val_logits.sh" "$CKPT"

# --- manifest --------------------------------------------------------------
echo "### manifest  $(date '+%T')"
LOG=$(ls -t logs/TopLandscape_ParT-FineTune_*"${COMMENT}".log | head -1)
python "$REPO/scripts/write_run_manifest.py" "$RUN" --seed "$SEED" --log "$LOG" \
  --command "SEED=$SEED COMMENT=$COMMENT ./train_TopLandscape.sh ParT-FineTune kin  (job $SLURM_JOB_ID)"

echo "### done  $(date '+%F %T')"
ls -lh "${RUN}predict_output/"
