#!/bin/bash
# Score every narrative of a block that passed verification, with each judge in
# JUDGES: zai-glm-5-3 and muse-spark-1.3 (2026-11-16; mistral-large-4 dropped after the judge tests; claude-sonnet-5, the judge
# of the earlier runs, is dropped but can still be named to resume them).
# The generators are read from the narrative config's `generators`.
#
# Safe to re-run. Verdicts already written are skipped per narrative, so a
# shutdown, a Ctrl-C or a dropped connection costs only the call in flight.
# After an interruption, run the same command again.
#
#   ./run_judges.sh [BLOCK]          # default v16
#   JUDGES="zai-glm-5-3" ./run_judges.sh v17         # a subset of judges
#   ATTEMPTS="final" ./run_judges.sh v17             # delivered narratives only
#
# Each block is judged twice by default: on every jet's first attempt (the model
# unaided, passed or not) and on the delivered narrative after the retry loop.
# The two share their narrative ids, so they pair up jet by jet.
#   setsid nohup ./run_judges.sh v17 </dev/null >/dev/null 2>&1 &
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

BLOCK=${1:-v16}
JUDGES=${JUDGES:-"zai-glm-5-3 muse-spark-1.3"}
ATTEMPTS=${ATTEMPTS:-"first final"}
echo "block $BLOCK, judges: $JUDGES, attempts: $ATTEMPTS"

# Fail before the first call, not after the first judge has spent an hour.
case " $JUDGES " in *" claude-"*)
  [ -f ~/.anthropic_key ] || [ -n "${ANTHROPIC_API_KEY:-}" ] || {
    echo "no credential: set ANTHROPIC_API_KEY or write it to ~/.anthropic_key"; exit 1; } ;;
esac
case " $JUDGES " in *" muse-"*)
  [ -f ~/.meta_key ] || [ -n "${META_API_KEY:-}" ] || {
    echo "no credential: set META_API_KEY or write it to ~/.meta_key"; exit 1; } ;;
esac
case " $JUDGES " in *" zai-glm-"*|*" mistral-"*)
  [ -f ~/.mistral_key ] || [ -n "${MISTRAL_API_KEY:-}" ] || {
    echo "no credential: set MISTRAL_API_KEY or write it to ~/.mistral_key"; exit 1; } ;;
esac

# Only one copy at a time: the boot job and a hand-launched run would otherwise
# both append to the same file.
exec 9>"evaluation/.judges.lock"
flock -n 9 || { echo "another run holds the lock; nothing to do"; exit 0; }

# Nothing left? Say so and stop, so the boot job costs a second a day once the
# sweep is finished rather than re-reading every file forever.
DONE="evaluation/.judges.${BLOCK}.done"
if [ -f "$DONE" ]; then
  echo "already complete ($DONE); remove that file to force a rerun"
  exit 0
fi

# One output file per attempt and judge, so each can be resumed, reported or
# dropped alone.
for A in $ATTEMPTS; do
  for J in $JUDGES; do
    echo "===== $BLOCK $A $J $(date '+%H:%M') ====="
    python -u evaluation/judge.py \
        --from-runs "$BLOCK" --attempt "$A" --judges "$J" \
        --out "evaluation/results/judgments_${BLOCK}_${A}_${J}.jsonl" \
      || { echo "FAILED $A $J — rerun this script to resume"; exit 1; }
  done
done

touch "$DONE"
echo "ALL_DONE $(date '+%F %H:%M:%S')"
echo "report:  python -m evaluation.judge_report"
