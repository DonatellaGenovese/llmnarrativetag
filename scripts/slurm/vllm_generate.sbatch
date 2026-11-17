#!/bin/bash
# Serve one open-weights generator with vLLM on this node and narrate with it.
# The server listens on localhost only; the orchestrator calls it from the same
# job, so no tunnel and no internet are needed. Needs scripts/setup_vllm.sh
# (env + weights) run first on the login node. Models: scripts/vllm_models.sh.
#
# The GPU count must match the model's tensor parallelism; the header below is
# gpt-oss-120b's (2 GPUs). Override on the command line for the others:
#
#   # debug test: 3 jets, 30 min
#   sbatch --qos=boost_qos_dbg --time=00:30:00 \
#          --export=ALL,MODEL=gpt-oss-120b,TEST=1 scripts/slurm/vllm_generate.sbatch
#   # a block, e.g. v17 for gpt-oss
#   sbatch --export=ALL,MODEL=gpt-oss-120b,BLOCK=v17 scripts/slurm/vllm_generate.sbatch
#   # qwen3.8-27b: 1 GPU
#   sbatch --gres=gpu:1 --cpus-per-task=8 --mem=120G \
#          --export=ALL,MODEL=qwen3.8-27b,TEST=1 ... scripts/slurm/vllm_generate.sbatch
#
# Variables: MODEL (required), TEST=1 (3 jets, written to
# <outputs>/vllmtest_<MODEL>_<job>.jsonl), BLOCK and JETS (default v17 and its
# run100_jets.json), CFG (default the top-tagging config), WORKERS (default 48).
# Writes <outputs>/<BLOCK>_<MODEL>.jsonl and, beside it, <...>.vllm.json with
# the vLLM/torch versions, the weights' revision and the serve arguments.

#SBATCH --job-name=vllm-gen
#SBATCH --account=IscrC_XJET-HEP
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$HOME/llmnarrativetag}"
cd "$REPO"
source scripts/vllm_models.sh
source "$HOME/miniconda3/etc/profile.d/conda.sh"

MODEL=${MODEL:?set MODEL, e.g. --export=ALL,MODEL=gpt-oss-120b}
vllm_model "$MODEL"
CFG=${CFG:-narrative/configs/narrative_top.yaml}
BLOCK=${BLOCK:-v17}
JETS=${JETS:-narrative/outputs/top/narratives/run100_jets.json}
WORKERS=${WORKERS:-48}
OUT=$(conda run -n part-surrogate python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['outputs']['narratives'])" "$CFG")

n_gpu=$(nvidia-smi -L | wc -l)
echo "### $(date '+%F %T')  node $(hostname)  job $SLURM_JOB_ID  model $MODEL  gpus $n_gpu  tp $TP"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ "$n_gpu" -eq "$TP" ] || { echo "model $MODEL needs $TP GPUs, the job has $n_gpu (see the header)"; exit 1; }
[ -f "$MODEL_DIR/REVISION" ] || { echo "no weights at $MODEL_DIR (run scripts/setup_vllm.sh weights $MODEL)"; exit 1; }

# Offline: compute nodes have no internet, and nothing should be fetched anyway.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
PORT=$((20000 + SLURM_JOB_ID % 10000))
export VLLM_BASE_URL="http://127.0.0.1:$PORT/v1"
LOG="slurm_logs/vllm-server-$SLURM_JOB_ID.log"

conda activate "$VLLM_ENV"
VERSIONS=$(python -c "import json, torch, vllm; print(json.dumps({'vllm': vllm.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda}))")
echo "versions: $VERSIONS"
echo "### starting vLLM  $(date '+%T')  (log: $LOG)"
# shellcheck disable=SC2086
vllm serve "$MODEL_DIR" --served-model-name "$MODEL" --tensor-parallel-size "$TP" \
  --host 127.0.0.1 --port "$PORT" $SERVE_ARGS > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
conda deactivate

# Loading ~65-190 GB from /leonardo_work takes minutes; give it up to 40.
t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  if ! kill -0 $SERVER 2>/dev/null; then
    echo "vLLM exited during startup; last lines of $LOG:"; tail -40 "$LOG"; exit 1
  fi
  if [ $(( $(date +%s) - t0 )) -gt 2400 ]; then
    echo "vLLM not ready after 40 min"; tail -40 "$LOG"; exit 1
  fi
  sleep 15
done
echo "### vLLM ready after $(( $(date +%s) - t0 )) s"

if [ "${TEST:-0}" = "1" ]; then
  name="vllmtest_${MODEL}_${SLURM_JOB_ID}"
else
  name="${BLOCK}_${MODEL}"
fi
python3 - "$OUT/$name.vllm.json" <<EOF
import json, sys
json.dump({"model": "$MODEL", "repo": "$REPO", "revision": "$REV", "tensor_parallel": $TP,
           "serve_args": "$SERVE_ARGS", "versions": $VERSIONS, "node": "$(hostname)",
           "job": "$SLURM_JOB_ID", "workers": $WORKERS,
           "gpus": "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"},
          open(sys.argv[1], "w"), indent=2)
EOF

g0=$(date +%s)
if [ "${TEST:-0}" = "1" ]; then
  # Three jets of the shared sample, all at once: checks the contract end to end
  # and measures the speed the block estimate is made from.
  conda activate part-surrogate
  export PYTHONPATH="$REPO"
  rows=$(python -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['jets'][:3])))" "$JETS")
  gen=$(python - "$CFG" "$MODEL" <<'EOF'
import json, sys, yaml
g = next(g for g in yaml.safe_load(open(sys.argv[1]))["generators"] if g["name"] == sys.argv[2])
args = ["--backend", g["backend"], "--model", g["model"]]
if g.get("reasoning_effort"):
    args += ["--reasoning-effort", g["reasoning_effort"]]
if g.get("extra_body"):
    args += ["--extra-body", json.dumps(g["extra_body"], separators=(",", ":"))]
print(json.dumps(args))
EOF
)
  mapfile -t gen_args < <(python -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])))" "$gen")
  # shellcheck disable=SC2086
  python -m narrative.orchestrator --config "$CFG" "${gen_args[@]}" --jets $rows \
    --workers 3 --out "$OUT/$name.jsonl"
else
  CFG="$CFG" ONLY="$MODEL" WORKERS="$WORKERS" REPORT=0 \
    scripts/run_generators.sh "$BLOCK" "$JETS"
fi
secs=$(( $(date +%s) - g0 ))

# Speed, from the token counts every completion records.
# (Not `conda run`: it does not pass a heredoc through on stdin.)
conda activate part-surrogate
python - "$OUT/$name.jsonl" "$secs" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
u = [a["completion"]["usage"] for r in rows for a in r["attempts"] if "completion" in a]
secs = int(sys.argv[2])
inp = sum(x.get("prompt_token_count", 0) for x in u)
out = sum(x.get("candidates_token_count", 0) for x in u)
st = {}
for r in rows:
    st[r["status"]] = st.get(r["status"], 0) + 1
print(f"{len(rows)} jets {st}  {len(u)} calls in {secs} s:  {inp} prompt + {out} output tokens,"
      f"  {out / max(secs, 1):.0f} output tok/s overall")
EOF
echo "### done  $(date '+%F %T')"
