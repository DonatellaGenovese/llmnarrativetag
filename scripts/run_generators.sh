#!/bin/bash
# Narrate one block of jets with every generator in the config's `generators`.
#
#   ./scripts/run_generators.sh BLOCK JETS_FILE [extra orchestrator args...]
#   ./scripts/run_generators.sh v17 narrative/outputs/top/narratives/run100_jets.json
#
# Writes narrative/outputs/top/narratives/<BLOCK>_<name>.jsonl, the names that
# blocks_report and `judge.py --from-runs BLOCK` read back. A generator whose
# file already holds one record per jet and no errors is skipped, so after an
# interruption the same command resumes. API generators run on the login node
# (only HTTP calls); `vllm` generators run inside scripts/slurm/vllm_generate.sbatch,
# which starts the server and calls this script with ONLY set.
#
# Environment:
#   CFG=...        config (default narrative/configs/narrative_top.yaml)
#   ONLY="a b"     run only these generator names (default: all in the config)
#   WORKERS=N      jets narrated concurrently (default 1; >1 for a served model)
#   REPORT=0       skip the first/final reports at the end

set -eo pipefail
cd "$(dirname "$0")/.."
BLOCK=${1:?usage: $0 BLOCK JETS_FILE [orchestrator args]}
JETS=${2:?usage: $0 BLOCK JETS_FILE [orchestrator args]}
shift 2

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate part-surrogate
export PYTHONPATH="$PWD"
[ -f "$HOME/.gcp/vertex_env.sh" ] && source "$HOME/.gcp/vertex_env.sh"
for k in deepseek; do
  f="$HOME/.${k}_key"; v="$(echo "$k" | tr a-z A-Z)_API_KEY"
  [ -n "${!v:-}" ] || { [ -s "$f" ] && export "$v=$(cat "$f")"; }
done

CFG=${CFG:-narrative/configs/narrative_top.yaml}
OUT=$(python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['outputs']['narratives'])" "$CFG")
WORKERS=${WORKERS:-1}
mkdir -p "$OUT"
N_JETS=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['jets']))" "$JETS")

# name backend model effort extra_body(JSON), one generator per line
python - "$CFG" "${ONLY:-}" <<'EOF' |
import json, sys, yaml
only = set(sys.argv[2].split())
for g in yaml.safe_load(open(sys.argv[1]))["generators"]:
    if only and g["name"] not in only:
        continue
    if not only and g["backend"] == "vllm":
        # Served by us: needs a GPU job (scripts/slurm/vllm_generate.sbatch).
        print(f"skip  {g['name']} (vllm: run scripts/slurm/vllm_generate.sbatch)", file=sys.stderr)
        continue
    extra = json.dumps(g["extra_body"], separators=(",", ":")) if g.get("extra_body") else "-"
    print(g["name"], g["backend"], g["model"], g.get("reasoning_effort") or "-", extra)
EOF
while read -r name backend model effort extra_body; do
  out="$OUT/${BLOCK}_${name}.jsonl"
  if [ -f "$out" ]; then
    n=$(wc -l < "$out"); e=$(grep -c '"status": "error"' "$out" || true)
    if [ "$n" -eq "$N_JETS" ] && [ "$e" -eq 0 ]; then echo "skip  $name (complete)"; continue; fi
    echo "redo  $name ($n/$N_JETS records, $e errors)"; rm -f "$out"
  fi
  echo "===== $BLOCK $name ($backend $model) $(date '+%F %T')"
  extra=(); [ "$effort" != "-" ] && extra+=(--reasoning-effort "$effort")
  [ "$extra_body" != "-" ] && extra+=(--extra-body "$extra_body")
  python -m narrative.orchestrator --config "$CFG" --backend "$backend" --model "$model" "${extra[@]}" \
    --workers "$WORKERS" --jets-file "$JETS" --out "$out" "$@" < /dev/null
done

[ "${REPORT:-1}" = "0" ] && exit 0

# Both readings of the same run, kept beside it: after one attempt (the model
# unaided, the published contract) and after the refine loop (what is delivered).
for which in first final; do
  echo; echo "##### attempt: $which"
  python -m narrative.blocks_report --config "$CFG" --blocks "$BLOCK" --attempt "$which" \
    | tee "$OUT/${BLOCK}_report_${which}.txt"
done
