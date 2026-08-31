#!/bin/bash
# Five further blocks of 100 random test jets, narrated and verified by four
# models, on top of the original run100_jets.json.
#
# Safe to re-run: a (block, model) pair whose output already holds 100 lines is
# skipped, and a partial file — left by a reboot, a Ctrl-C, or a dropped
# connection — is deleted and redone. So after an interruption you just run the
# same command again.
#
#   ./run_blocks.sh                 # foreground, watch it go
#   setsid nohup ./run_blocks.sh </dev/null >/dev/null 2>&1 &   # detached
#
# Detached survives closing the terminal and logging out, but NOT a shutdown or
# a suspend. That is what the resume logic is for.

cd "$(dirname "$0")"

# `set -u` must come AFTER conda: the env's activate.d hooks read unset
# variables (ADDR2LINE among them) and abort the whole script under it.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate part-surrogate
set -u

export PYTHONPATH="$PWD"
export DEEPSEEK_API_KEY=$(cat ~/.deepseek_key)
export OLLAMA_API_KEY=$(cat ~/.ollama_key)

N=narrative/outputs/top/narratives
LOG=$N/run_blocks.log
mkdir -p "$N"

# Everything this script prints also lands in the log, so launching it with
# >/dev/null cannot hide a startup failure the way it did once already.
exec > >(tee -a "$LOG") 2>&1
echo "### started $(date '+%F %H:%M:%S')"

# Fail loudly and early rather than three hours later on the fourth model.
for v in DEEPSEEK_API_KEY OLLAMA_API_KEY; do
  [ -n "${!v}" ] || { echo "$v is empty — check ~/.${v%_API_KEY}_key"; exit 1; }
done
python -c "import narrative.orchestrator" \
  || { echo "cannot import narrative.orchestrator from $PWD"; exit 1; }
# The credential the SDK actually reads, not the gcloud CLI: gcloud is in its
# own env and is not on PATH here, while google-genai picks up the ADC file
# directly. Testing for the CLI reported a failure that did not exist.
[ -f "$HOME/.config/gcloud/application_default_credentials.json" ] \
  || [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] \
  || { echo "no Vertex ADC: run 'gcloud auth application-default login'"; exit 1; }
echo "startup ok — $(which python)"

# Seeds are fixed per block, so the same five samples come back on every run.
# The orchestrator writes the jet file on first use and loads it verbatim after,
# so a resumed run cannot silently draw a different sample.
declare -A SEED=( [b1]=3 [b2]=4 [b3]=5 [b4]=6 [b5]=7 )

run () {   # block backend model outname [extra args...]
  local b=$1 be=$2 m=$3 o=$4; shift 4
  local out="$N/${b}_${o}.jsonl"

  if [ -f "$out" ]; then
    local n e
    n=$(wc -l < "$out")
    # A rate-limited run still writes one record per jet, so line count alone
    # calls a file of 429s complete and skips it forever. Records carrying an
    # error are jets that were never narrated, and the file has to be redone.
    e=$(grep -c '"status": "error"' "$out" || true)
    if [ "$n" -eq 100 ] && [ "$e" -eq 0 ]; then
      echo "skip  $b $o (complete)"
      return
    fi
    if [ "$e" -gt 0 ]; then
      echo "redo  $b $o ($e/$n jets errored — quota or network)"
    else
      echo "redo  $b $o (partial: $n/100)"
    fi
    rm -f "$out"
  fi

  echo "===== $b $o $(date '+%F %H:%M') =====" >> "$LOG"
  echo "run   $b $o"
  python -u -m narrative.orchestrator \
      --backend "$be" --model "$m" "$@" \
      --jets-file "$N/run100_jets_${b}.json" \
      --sample 100 --sample-seed "${SEED[$b]}" \
      --out "$out" >> "$LOG" 2>&1 \
    || { echo "FAILED $b $o — rerun this script to retry" | tee -a "$LOG"; rm -f "$out"; }
}

for b in b1 b2 b3 b4 b5; do
  run "$b" vertex   gemini-3.5-flash-lite gemini-3.5-flash-lite
  run "$b" vertex   gemini-3.5-flash      gemini-3.5-flash
  run "$b" deepseek deepseek-v4-flash     deepseek-v4-flash
  run "$b" ollama   gpt-oss:120b          gpt-oss-120b --reasoning-effort low
done

echo
echo "done. $(ls $N/b[1-5]_*.jsonl 2>/dev/null | wc -l)/20 runs complete."
echo "report:  python -m narrative.blocks_report"
