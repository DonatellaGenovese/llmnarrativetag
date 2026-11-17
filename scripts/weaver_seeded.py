#!/usr/bin/env python3
"""Run weaver with every random generator fixed by one seed.

weaver has no seed option: the network's initialisation, the order the data are
read in and dropout all draw from unseeded generators, so two runs of the same
command differ and neither can be repeated. This fixes them before weaver
starts, from the environment variable SEED, and then hands the command line to
weaver unchanged:

    SEED=42 python scripts/weaver_seeded.py --data-train ... (weaver's own args)

What it covers:
  - python's `random`, numpy's global generator, torch on CPU and every GPU;
  - weaver's data workers, which seed themselves from torch's base seed
    (worker_info.seed), so fixing torch fixes the shuffling and the reading
    order too;
  - cuDNN autotuning off and deterministic kernels, which otherwise pick
    algorithms by timing and can differ from run to run.

What it cannot cover: a few CUDA kernels used under mixed precision have no
deterministic implementation. torch is asked for determinism with
`warn_only=True`, so those are reported in the log instead of stopping the run.
Two runs with the same seed therefore agree to within floating-point noise, not
bit for bit. The checkpoint, not the seed, is what makes downstream results
exactly reproducible; the seed makes the training repeatable.
"""

from __future__ import annotations

import os
import random
import sys


def main() -> int:
    seed_env = os.environ.get("SEED")
    if seed_env is None:
        sys.exit("SEED is not set: export SEED=<int> before calling weaver_seeded.py")
    seed = int(seed_env)

    # Has to be set before CUDA is initialised for cuBLAS to honour it.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[weaver_seeded] seed={seed}  torch={torch.__version__}  "
          f"cudnn.deterministic=True  deterministic_algorithms(warn_only=True)", flush=True)

    from weaver.train import main as weaver_main

    sys.argv[0] = "weaver"
    return weaver_main()


if __name__ == "__main__":
    sys.exit(main())
