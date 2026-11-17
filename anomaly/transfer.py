"""Weakly supervised anomaly detection on ParT's frozen latent.

The encoder is never trained here. ParT's JetClass pretraining is run once, its
128-d class token is cached, and a small MLP is trained on top of it — the
transfer-learning mode Sophon uses, applied to the model we already have.

  IAD          idealised anomaly detection, the scheme resonance searches are
               benchmarked against. Two samples are built out of background
               alone plus an injected fraction f of signal:

                   reference = background half A            label 0
                   data      = background half B + f · top  label 1

               and a classifier learns to tell them apart. It never sees a
               per-jet label; it only knows that one sample is contaminated,
               which is what a sideband gives you in a real search. The trained
               classifier is then read on the independent test split.

  supervised   the same MLP trained on true top-versus-QCD labels. Not a
               detector, since it knows the signal: it is the ceiling, and it
               says how much of ParT survives in the frozen token.

Both run on two input spaces, the latent and the ten observables, so "is the
representation worth it" has an answer rather than an assumption.

Everything is read at fixed background efficiency and again with the soft-drop
mass dependence removed, because a mass distance alone already keeps 67% of top
at 10% background on this dataset.

Besides the summary, every trained classifier is kept with its logit for each
jet of every split, under anomaly/outputs/seed<S>/scores/<run>/ (see
`save_run`). That logit is what a surrogate is fit to, joined to the feature
table on `jet_id`.

One seed, --seed, fixes both sources of randomness: which top jets are drawn
into the contaminated sample, and the classifiers' initialisation, validation
split and batch order. Each seed writes its own directory, with a
run_manifest.json (seed, command, commit, versions, md5 of inputs and outputs).
Seed 0 reproduces the numbers this file gave before the option existed.

    python -m anomaly.transfer --seed 0
    for s in 0 1 2 3 4; do python -m anomaly.transfer --seed $s --fractions 0.01; done
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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .metrics import OBS, QCD_LABEL, TOP_LABEL, header, obs_matrix, report


class MLP(nn.Module):
    """Small on purpose: the representation is meant to be doing the work."""

    def __init__(self, dim: int, hidden=(512, 64)) -> None:
        super().__init__()
        layers, d = [], dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def score_mlp(model, x, device, batch=8192) -> np.ndarray:
    """The classifier's logit for every row of `x`, in row order."""
    xe = torch.tensor(x, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(xe[i:i + batch].to(device)).cpu().numpy()
                               for i in range(0, len(xe), batch)])


def fit_mlp(x, y, device, epochs=8, batch=4096, lr=1e-3, seed=0, n_val=50_000):
    """Train and early-stop on held-out loss; returns the best model.

    The validation loss is the classifier's own objective on part of the same
    mixture. Selecting on anomaly-detection performance instead would be
    supervision through the back door.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    va, tr = idx[:n_val], idx[n_val:]
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model = MLP(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        perm = tr[rng.permutation(len(tr))]
        for i in range(0, len(perm), batch):
            sl = perm[i:i + batch]
            loss = lossf(model(xt[sl].to(device)), yt[sl].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([lossf(model(xt[va[i:i + batch]].to(device)),
                                     yt[va[i:i + batch]].to(device)).item()
                               for i in range(0, len(va), batch)]))
        if v < best:
            best = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def standardise(fit, *others):
    mu, sd = fit.mean(0), fit.std(0) + 1e-6
    return [((a - mu) / sd).astype(np.float32) for a in (fit, *others)], (mu, sd)


def save_run(out_dir: Path, name: str, model, norm, scores: dict, jet_ids: dict,
             labels: dict, in_fit: np.ndarray | None, meta: dict) -> None:
    """One trained classifier and its per-jet logit on every split.

    Written as  <out_dir>/scores/<name>/{model.pt, scores_<split>.parquet}.
    The parquet files carry `jet_id`, so they join one-to-one onto the feature
    table: that join is how a surrogate is fit to this score. `label` keeps the
    feature table's convention, 0 = top and 1 = QCD.

    On the train split `in_fit` marks the jets the classifier was trained on.
    Their scores are in-sample and should not be used to fit a surrogate.
    """
    d = out_dir / "scores" / name
    d.mkdir(parents=True, exist_ok=True)
    mu, sd = norm
    torch.save({"state_dict": model.state_dict(), "dim": int(len(mu)),
                "mu": torch.tensor(mu), "sd": torch.tensor(sd), "meta": meta}, d / "model.pt")
    for split, s in scores.items():
        df = pd.DataFrame({"jet_id": jet_ids[split], "label": labels[split],
                           "score": s.astype(np.float64)})
        if split == "train" and in_fit is not None:
            df["in_fit"] = in_fit
        df.to_parquet(d / f"scores_{split}.parquet", index=False)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def _md5(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def write_manifest(out_dir: Path, repo: Path, args, inputs: list, results: dict) -> Path:
    """Everything needed to say which detector a score came from, and to check
    later that the files on disk are still those files."""
    versions = {"python": platform.python_version()}
    for dist in ("torch", "numpy", "pandas", "scikit-learn"):
        try:
            versions[dist] = version(dist)
        except PackageNotFoundError:
            pass
    versions["cuda"] = torch.version.cuda
    outputs = {str(f.relative_to(out_dir)): {"md5": _md5(f), "bytes": f.stat().st_size}
               for f in sorted(out_dir.glob("scores/*/*")) if f.is_file()}
    manifest = {
        "written": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "command": " ".join(["python -m anomaly.transfer", *sys.argv[1:]]),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "repo": {"commit": _git(repo, "rev-parse", "HEAD"),
                 "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
                 "dirty": bool(_git(repo, "status", "--porcelain", "--untracked-files=no"))},
        "slurm": {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME") if k in os.environ},
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": versions,
        "inputs": {str(f): {"md5": _md5(f), "bytes": f.stat().st_size} for f in inputs},
        "outputs": outputs,
        "test_metrics": results,
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, default=repo / "surrogate/outputs/top/embeddings")
    p.add_argument("--feat-dir", type=Path, default=repo / "surrogate/outputs/top/features")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.05, 0.02, 0.01],
                   help="Signal fractions injected into the contaminated sample")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--skip-supervised", action="store_true")
    # The observables were the comparison input in the branch's study; IAD on
    # them is no longer part of the plan (2026-11-08), so only the latent runs
    # by default. `--spaces latent obs` brings the comparison back.
    p.add_argument("--spaces", nargs="+", choices=["latent", "obs"], default=["latent"],
                   help="Input spaces to train on (default: latent only)")
    p.add_argument("--seed", type=int, default=0,
                   help="Fixes the injected top jets and the classifiers' training; "
                        "0 reproduces the numbers from before this option")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default anomaly/outputs/seed<SEED>")
    p.add_argument("--no-save-scores", action="store_true",
                   help="Only the summary; skip the per-jet scores and the models")
    args = p.parse_args()
    args.out_dir = args.out_dir or repo / f"anomaly/outputs/seed{args.seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The injection draw and every classifier fit derive from this one seed.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"seed {args.seed} → {args.out_dir}")

    # train and test are required; val is scored too when its embeddings exist,
    # since it is the split the classifier never touches.
    splits = ["train", "test"] + (["val"] if (args.emb_dir / "xcls_val.npz").exists() else [])
    emb, feat = {}, {}
    for s in splits:
        emb[s] = np.load(args.emb_dir / f"xcls_{s}.npz")
        # m_SD is already one of OBS; asking for it twice yields a duplicated
        # column, and every downstream shape goes wrong two functions later.
        feat[s] = pd.read_parquet(args.feat_dir / f"features_{s}.parquet",
                                  columns=OBS + ["label", "jet_id"])
        if not ((feat[s].label.to_numpy() == emb[s]["label"]).all()
                and (feat[s].jet_id.to_numpy() == emb[s]["jet_id"]).all()):
            raise SystemExit(f"{s}: embeddings and feature table are not row-aligned")
    sizes = ", ".join(f"{s} {len(emb[s]['label']):,}" for s in splits)
    print(f"splits: {sizes}  device: {device}\n")

    labels = {s: emb[s]["label"] for s in splits}
    jet_ids = {s: emb[s]["jet_id"] for s in splits}
    lab_tr, lab_te = labels["train"], labels["test"]
    y_te = (lab_te == TOP_LABEL).astype(int)
    m_te, is_bkg = feat["test"].m_SD.to_numpy(), lab_te == QCD_LABEL

    obs, med = {}, None
    for s in splits:                      # medians from train, applied everywhere
        o, med = obs_matrix(feat[s], med if s != "train" else None)
        obs[s] = o.astype(np.float32)
    spaces = {"latent": ("latente x_cls (128)", {s: emb[s]["x"].astype(np.float32) for s in splits}),
              "obs": ("10 osservabili", obs)}
    spaces = {k: v for k, v in spaces.items() if k in args.spaces}

    def fit_and_keep(key, label, X, fit_idx, y_fit, in_fit, meta):
        """Fit on X[train][fit_idx], report on test, save every split's logit."""
        Xn, norm = standardise(X["train"][fit_idx], *[X[s] for s in splits])
        model = fit_mlp(Xn[0], y_fit, device, args.epochs, seed=args.seed)
        scores = {s: score_mlp(model, Xn[1 + k], device) for k, s in enumerate(splits)}
        res = report(y_te, scores["test"], m_te, is_bkg, "  " + label)
        if not args.no_save_scores:
            save_run(args.out_dir, key, model, norm, scores, jet_ids, labels, in_fit,
                     {**meta, "space": key.rsplit("_", 1)[-1], "epochs": args.epochs,
                      "seed": args.seed,
                      "test_metrics": res})
        return res

    results = {}
    all_tr = np.arange(len(lab_tr))
    if not args.skip_supervised:
        print("SUPERVISIONATO — conosce le etichette, è il tetto\n")
        header()
        for sp, (label, X) in spaces.items():
            results[f"supervised | {label}"] = fit_and_keep(
                f"supervised_{sp}", label, X, all_tr, (lab_tr == TOP_LABEL).astype(np.float32),
                np.ones(len(lab_tr), bool), {"scheme": "supervised"})

    bkg = np.flatnonzero(lab_tr == QCD_LABEL)
    top = np.flatnonzero(lab_tr == TOP_LABEL)
    ref, dat = bkg[::2], bkg[1::2]        # two disjoint halves of the background
    for f in args.fractions:
        n_sig = int(round(f * len(dat)))
        sig = rng.choice(top, size=n_sig, replace=False)
        idx = np.concatenate([ref, dat, sig])
        y = np.concatenate([np.zeros(len(ref)), np.ones(len(dat) + n_sig)]).astype(np.float32)
        in_fit = np.zeros(len(lab_tr), bool)
        in_fit[idx] = True
        print(f"\nIAD  segnale iniettato {f:.0%}   riferimento {len(ref):,} · "
              f"dati {len(dat) + n_sig:,} (di cui {n_sig:,} top)\n")
        header()
        for sp, (label, X) in spaces.items():
            results[f"IAD f={f} | {label}"] = fit_and_keep(
                f"iad_f{f:g}_{sp}", label, X, idx, y, in_fit,
                {"scheme": "IAD", "signal_fraction": f, "n_reference": int(len(ref)),
                 "n_data": int(len(dat) + n_sig), "n_signal_injected": n_sig})

    (args.out_dir / "transfer_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out_dir / 'transfer_summary.json'}")
    if not args.no_save_scores:
        print(f"per-jet scores → {args.out_dir / 'scores'}/<run>/scores_<split>.parquet")
    inputs = ([args.emb_dir / f"xcls_{s}.npz" for s in splits]
              + [args.feat_dir / f"features_{s}.parquet" for s in splits])
    print(f"wrote {write_manifest(args.out_dir, repo, args, inputs, results)}")


if __name__ == "__main__":
    main()
