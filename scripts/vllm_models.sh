# Models we serve with vLLM on Leonardo, in one place. Sourced by
# scripts/setup_vllm.sh (download) and scripts/slurm/vllm_generate.sbatch (serve).
#
#   vllm_model NAME   sets REPO REV EXCLUDE TP SERVE_ARGS GPUS
#
# NAME is the generator name in the narrative configs and vLLM's
# --served-model-name. Revisions are pinned (Hugging Face commit, 2026-11-10):
# the weights are part of what a narrative run is reproduced from.
#
# Memory (A100 64 GB, 4 per node):
#   gpt-oss-120b         MXFP4, ~65 GB of weights   -> 2 GPUs, tensor parallel 2
#   qwen3.8-27b          FP8,   ~31 GB              -> 1 GPU
# The A100 has no native MXFP4/FP8 units; vLLM keeps the weights quantised and
# runs weight-only kernels, so memory is as above and decoding slower than on H100.

VLLM_DATA="${VLLM_DATA:-$WORK/dgenoves/llmnarrativetag_data}"
# The nodes run NVIDIA driver 535 (CUDA 12.2). vLLM >= 0.33 ships only CUDA 13
# builds, which need driver >= 580 and cannot run here; 0.32.0 is the newest
# with a CUDA 12.9 build, which runs on 535 through CUDA minor-version
# compatibility (as the ParT venv's cu126 torch does). It postdates the Qwen3.8
# releases (Aug 2026), so it should know their architectures.
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
VLLM_CUDA="${VLLM_CUDA:-cu129}"
VLLM_ENV="$VLLM_DATA/venv-vllm-$VLLM_VERSION-$VLLM_CUDA"
# Weights on $FAST, the project's all-flash area (1 TB, kept for the project's
# life + 6 months, no 40-day cleaning, no backup): ~1.5 GB/s sequential read
# against ~0.5 GB/s from $WORK (measured 2026-11-11). Copied there from
# $VLLM_DATA/hf_models, which still holds the originals.
VLLM_MODELS="${VLLM_MODELS:-${FAST:-/leonardo_scratch/fast/IscrC_XJET-HEP}/dgenoves/hf_models}"
# Everything lands in $WORK (project, 1 TB), nothing in $HOME (50 GB quota, mostly
# used already): weights, the HF cache, conda packages, and the caches vLLM,
# Triton and torch.compile write at runtime (by default under ~/.cache).
export HF_HOME="$VLLM_DATA/hf_cache"
export TIKTOKEN_ENCODINGS_BASE="$VLLM_DATA/tiktoken"
export XDG_CACHE_HOME="$VLLM_DATA/cache"
export VLLM_CACHE_ROOT="$VLLM_DATA/cache/vllm"
export TRITON_CACHE_DIR="$VLLM_DATA/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$VLLM_DATA/cache/inductor"
export CONDA_PKGS_DIRS="$VLLM_DATA/cache/conda_pkgs"
export PIP_CACHE_DIR="$VLLM_DATA/cache/pip"
export FLASHINFER_WORKSPACE_BASE="$VLLM_DATA/cache"
# FlashInfer's top-k/top-p sampler is compiled on first use with nvcc, which the
# nodes do not have (job 64114587 died there, after loading and compiling the
# model). vLLM's PyTorch sampler does the same sampling without a compile step.
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm_model() {
  # Common: prompts are ~6k tokens with the refine history, outputs ~1.5k.
  # Eager safetensors loading: whole files read sequentially. The default
  # (mmap + page-cache prefetch) loaded gpt-oss at ~100 MB/s from $WORK
  # (job 64118833, 626 s for 61 GiB) where a plain sequential read does ~500 MB/s.
  local common="--max-model-len 16384 --gpu-memory-utilization 0.92 --seed 42 --safetensors-load-strategy eager"
  case "$1" in
    gpt-oss-120b)
      REPO=openai/gpt-oss-120b
      REV=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
      # metal/ and original/ are other formats of the same weights (~130 GB)
      EXCLUDE="metal/* original/*"
      TP=2
      SERVE_ARGS="$common"
      ;;
    qwen3.8-27b)
      REPO=Qwen/Qwen3.8-27B-FP8
      REV=c3a462cc3d38cf351f3b26cbd763469eeb74ef81
      EXCLUDE=""
      TP=1
      # Thinking is switched off per request (extra_body in the config); the
      # parser keeps any reasoning out of the narrative text all the same.
      SERVE_ARGS="$common --reasoning-parser qwen3"
      ;;
    *)
      echo "unknown vLLM model '$1' (gpt-oss-120b, qwen3.8-27b)" >&2
      return 1
      ;;
  esac
  GPUS=$TP
  MODEL_DIR="$VLLM_MODELS/$1"
}
