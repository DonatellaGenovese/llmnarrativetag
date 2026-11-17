#!/usr/bin/env python3
"""Write run_manifest.json into a ParT fine-tune run directory.

Everything needed to say which teacher a result came from, and to check later
that the files on disk are still those files:

  - the seed (or null for a run trained before seeding existed);
  - the exact training command, from weaver's own log;
  - the repo commit and whether the tree had local edits;
  - torch / weaver / CUDA versions, the GPU, the node and the SLURM job;
  - md5 and size of the best checkpoint and of every logit file;
  - the metrics of the best checkpoint on each split, recomputed from the logits.

    python scripts/write_run_manifest.py RUN_DIR [--seed 42] [--command ...] [--log weaver.log]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def md5(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def split_metrics(path: Path) -> dict:
    """Accuracy, AUC and background rejection of the logits in one file."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve

    d = pd.read_parquet(path, columns=["label", "logit_diff"])
    y = (d["label"].to_numpy() == 0).astype(int)        # weaver index 0 = top
    t = d["logit_diff"].to_numpy()
    fpr, tpr, _ = roc_curve(y, t)
    return {
        "n": int(len(d)),
        "acc": round(float(((t > 0) == y).mean()), 6),
        "auc": round(float(roc_auc_score(y, t)), 6),
        "rej_at_eff_sig_0.5": round(float(1 / np.interp(0.5, tpr, fpr)), 1),
        "rej_at_eff_sig_0.3": round(float(1 / np.interp(0.3, tpr, fpr)), 1),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--seed", type=int, default=None, help="Omit for a run trained unseeded")
    p.add_argument("--command", default=None, help="The training command as it was run")
    p.add_argument("--log", type=Path, default=None,
                   help="weaver's own log; its first lines (the full weaver arguments) are kept")
    p.add_argument("--note", default="", help="Free text kept in the manifest")
    args = p.parse_args()
    run = args.run_dir.resolve()

    files = {}
    ckpt = run / "net_best_epoch_state.pt"
    if ckpt.exists():
        files[ckpt.name] = {"md5": md5(ckpt), "bytes": ckpt.stat().st_size}
    pred = run / "predict_output"
    for f in sorted(pred.glob("*.parquet")):
        files[f"predict_output/{f.name}"] = {"md5": md5(f), "bytes": f.stat().st_size}

    metrics = {}
    for split, name in (("train", "TopLandscape_ParT-FineTune_train_logits.parquet"),
                        ("val", "TopLandscape_ParT-FineTune_val_logits.parquet"),
                        ("test", "TopLandscape_ParT-FineTune_logits.parquet")):
        if (pred / name).exists():
            metrics[split] = split_metrics(pred / name)

    weaver_args = None
    if args.log and args.log.exists():
        # weaver opens its log with `INFO: args:` and then one ` - ('name', value)`
        # line per parsed argument. Kept as text: the values are reprs.
        lines = args.log.read_text(errors="replace").splitlines()
        start = next((i for i, l in enumerate(lines) if l.rstrip().endswith("INFO: args:")), None)
        if start is not None:
            weaver_args = {}
            for l in lines[start + 1:]:
                if not l.startswith(" - ('"):
                    break
                name, _, value = l[len(" - ('"):].partition("', ")
                weaver_args[name] = value.rstrip(")")

    # From the installed distributions: weaver-core exposes no __version__.
    from importlib.metadata import PackageNotFoundError, version
    versions = {"python": platform.python_version()}
    for dist in ("torch", "weaver-core", "numpy", "awkward", "uproot"):
        try:
            versions[dist] = version(dist)
        except PackageNotFoundError:
            pass
    gpu = None
    try:
        import torch
        versions["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass

    manifest = {
        "written": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run),
        "seed": args.seed,
        "seeded": args.seed is not None,
        "note": args.note,
        "repo": {"commit": git(repo, "rev-parse", "HEAD"),
                 "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
                 "dirty": bool(git(repo, "status", "--porcelain", "--untracked-files=no"))},
        "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME") if os.environ.get(k)},
        "host": socket.gethostname(),
        "gpu": gpu,
        "versions": versions,
        "training_command": args.command,
        "weaver_args": weaver_args,
        "files": files,
        "metrics": metrics,
    }
    out = run / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}")
    for s, m in metrics.items():
        print(f"  {s:5s} acc={m['acc']:.4f} auc={m['auc']:.4f} "
              f"rej50={m['rej_at_eff_sig_0.5']:.0f} rej30={m['rej_at_eff_sig_0.3']:.0f}")


if __name__ == "__main__":
    main()
