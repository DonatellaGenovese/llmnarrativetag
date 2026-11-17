#!/bin/bash
# Download vLLM model weights from a serial-partition node instead of the login
# node, so a long transfer is not at the mercy of the login node's process
# limits or of the ssh session. Writes to $WORK only (scripts/vllm_models.sh).
#
#   sbatch --export=ALL,MODELS="gpt-oss-120b" scripts/slurm/vllm_download.sbatch
#   sbatch --export=ALL,MODELS=qwen3.8-27b scripts/slurm/vllm_download.sbatch
#
# Needs the environment first (scripts/setup_vllm.sh env, on the login node).
# If this partition turns out to have no internet the job says so at once, and
# the download goes back to the login node with nohup (see setup_vllm.sh).

#SBATCH --job-name=vllm-download
#SBATCH --account=IscrC_XJET-HEP
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/llmnarrativetag}"
MODELS=${MODELS:?set MODELS, e.g. --export=ALL,MODELS=gpt-oss-120b}
echo "### $(date '+%F %T')  node $(hostname)  job $SLURM_JOB_ID  models $MODELS"
curl -sSfI --max-time 20 https://huggingface.co > /dev/null \
  || { echo "no internet on $(hostname): download on the login node instead"; exit 1; }
# shellcheck disable=SC2086
scripts/setup_vllm.sh weights $MODELS
scripts/setup_vllm.sh check
echo "### done  $(date '+%F %T')"
