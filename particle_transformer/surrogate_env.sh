#!/bin/bash
# Activate conda env part-surrogate (Miniforge at /opt/miniforge3)
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate part-surrogate
export ROOT_INCLUDE_PATH="${CONDA_PREFIX}/include${ROOT_INCLUDE_PATH:+:$ROOT_INCLUDE_PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "Activated: $CONDA_PREFIX"
python -c "import sys,interpret; print('python',sys.version.split()[0]); print('interpret',interpret.__version__)"
