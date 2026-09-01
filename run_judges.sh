#!/bin/bash
# Score every narrative of block v16 that passed verification, with the judge:
# claude-sonnet-5.
#
# Safe to re-run. Verdicts already written are skipped per narrative, so a
# shutdown, a Ctrl-C or a dropped connection costs only the call in flight.
# After an interruption, run the same command again.
#
#   ./run_judges.sh
#   setsid nohup ./run_judges.sh </dev/null >/dev/null 2>&1 &
#
# Detached survives closing the terminal and logging out, but not a shutdown or
# a suspend — which is what the resume is for.

cd "$(dirname "$0")"

# set -u after conda: the env's activate hooks read unset variables and abort
# the whole script under it.
# Resolve conda without assuming an install prefix. `conda info --base` alone
# is not enough: these scripts are meant to be run detached, where conda is not
# on PATH, which is exactly when a hardcoded prefix used to be needed.
_base="${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"
if [ -z "$_base" ] && command -v conda >/dev/null 2>&1; then _base="$(conda info --base)"; fi
for _c in "$_base" /opt/miniforge3 "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
  if [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ]; then . "$_c/etc/profile.d/conda.sh"; break; fi
done
command -v conda >/dev/null 2>&1 || { echo "conda not found: set CONDA_EXE"; exit 1; }
conda activate part-surrogate
set -u

export PYTHONPATH="$PWD"

LOG=evaluation/run_judges.log
mkdir -p evaluation
exec > >(tee -a "$LOG") 2>&1
echo "### started $(date '+%F %H:%M:%S')"

[ -f ~/.anthropic_key ] || [ -n "${ANTHROPIC_API_KEY:-}" ] || {
  echo "no credential: set ANTHROPIC_API_KEY or write it to ~/.anthropic_key"; exit 1; }

# Only one copy at a time: the boot job and a hand-launched run would otherwise
# both append to the same file.
exec 9>"evaluation/.judges.lock"
flock -n 9 || { echo "another run holds the lock; nothing to do"; exit 0; }

# Nothing left? Say so and stop, so the boot job costs a second a day once the
# sweep is finished rather than re-reading every file forever.
if [ -f evaluation/.judges.done ]; then
  echo "already complete (evaluation/.judges.done); remove that file to force a rerun"
  exit 0
fi

J=claude-sonnet-5
echo "===== $J $(date '+%H:%M') ====="
python -u evaluation/judge.py \
    --from-runs v16 --judges "$J" \
    --out "evaluation/results/judgments_v16_${J}.jsonl" \
  || { echo "FAILED $J — rerun this script to resume"; exit 1; }

touch evaluation/.judges.done
echo "ALL_DONE $(date '+%F %H:%M:%S')"
echo "report:  python -m evaluation.judge_report"
