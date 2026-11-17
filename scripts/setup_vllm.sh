#!/bin/bash
# vLLM on Leonardo: environment and weights. Run on the LOGIN node (compute
# nodes have no internet). See scripts/vllm_models.sh for the models.
#
#   scripts/setup_vllm.sh env                          # conda env + vllm==$VLLM_VERSION
#   scripts/setup_vllm.sh weights gpt-oss-120b [...]   # download, pinned revision
#   scripts/setup_vllm.sh check                        # what is installed / downloaded
#
# The downloads are large (gpt-oss ~65 GB, Qwen3.8-27B ~31 GB) and go to $WORK,
# never $HOME. Preferably from a serial-partition job,
#
#   sbatch --export=ALL,MODELS=gpt-oss-120b scripts/slurm/vllm_download.sbatch
#
# or, if that partition has no internet, on the login node in the background
# so a dropped ssh session does not kill them,
#
#   nohup scripts/setup_vllm.sh weights gpt-oss-120b > slurm_logs/setup_vllm.log 2>&1 &
#
# The environment is separate from part-surrogate and the ParT venv on purpose:
# vLLM pins its own torch/CUDA, which must not move the surrogate stack.

set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/vllm_models.sh
source "$HOME/miniconda3/etc/profile.d/conda.sh"
# Downloads and pip are network-bound; keep the login node's CPU limits happy.
export OMP_NUM_THREADS=2 MAX_JOBS=2

cmd=${1:?usage: $0 env | weights NAME... | check}
shift || true

case "$cmd" in
  env)
    if [ ! -d "$VLLM_ENV" ]; then
      conda create -y -p "$VLLM_ENV" python=3.12 pip
    fi
    conda activate "$VLLM_ENV"
    # The CUDA-12 build and a torch of the same CUDA, from vLLM's and PyTorch's
    # indexes (PyPI carries only the CUDA 13 build). Pinned with the local tag.
    pip install --no-cache-dir "vllm==$VLLM_VERSION+$VLLM_CUDA" "huggingface_hub[hf_transfer]" \
      --extra-index-url "https://wheels.vllm.ai/$VLLM_VERSION/$VLLM_CUDA" \
      --extra-index-url "https://download.pytorch.org/whl/$VLLM_CUDA"
    # torchcodec (audio decoding, unused here) comes from PyPI built for CUDA 13
    # and fails to load (libnvrtc.so.13) with an OSError vLLM does not catch, so
    # `vllm serve` dies at import (job 64086142). Without it vLLM falls back.
    pip uninstall -y torchcodec || true
    python - <<'EOF'
import torch, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "| built for CUDA", torch.version.cuda)
EOF
    # gpt-oss's tokenizer (openai-harmony) fetches its tiktoken vocabularies on
    # first use; the compute nodes cannot, so they are kept locally and pointed
    # at with TIKTOKEN_ENCODINGS_BASE (set in vllm_models.sh).
    mkdir -p "$TIKTOKEN_ENCODINGS_BASE"
    for enc in o200k_base cl100k_base; do
      [ -s "$TIKTOKEN_ENCODINGS_BASE/$enc.tiktoken" ] || curl -sSfL -o "$TIKTOKEN_ENCODINGS_BASE/$enc.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/$enc.tiktoken"
    done
    echo "Compute nodes run NVIDIA driver 535 (CUDA 12.2). A build for a newer CUDA 12.x"
    echo "usually runs through CUDA minor-version compatibility; the debug job checks it."
    ;;
  weights)
    [ $# -gt 0 ] || { echo "usage: $0 weights NAME..."; exit 1; }
    conda activate "$VLLM_ENV"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    for name in "$@"; do
      vllm_model "$name"
      echo "### $name  $REPO@$REV -> $MODEL_DIR  $(date '+%F %T')"
      args=()
      for pat in $EXCLUDE; do args+=(--exclude "$pat"); done
      hf download "$REPO" --revision "$REV" --local-dir "$MODEL_DIR" "${args[@]}"
      printf '%s\n%s\n' "$REPO" "$REV" > "$MODEL_DIR/REVISION"
      du -sh "$MODEL_DIR"
    done
    ;;
  check)
    if [ -d "$VLLM_ENV" ]; then
      conda activate "$VLLM_ENV"
      python -c "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__, '| CUDA', torch.version.cuda)"
    else
      echo "no environment at $VLLM_ENV (run: $0 env)"
    fi
    for name in gpt-oss-120b qwen3.8-27b; do
      vllm_model "$name"
      if [ -f "$MODEL_DIR/REVISION" ]; then
        echo "$name: $(tail -1 "$MODEL_DIR/REVISION")  $(du -sh "$MODEL_DIR" | cut -f1)"
      else
        echo "$name: not downloaded"
      fi
    done
    ;;
  *)
    echo "usage: $0 env | weights NAME... | check"; exit 1
    ;;
esac
