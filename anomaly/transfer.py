"""Transfer learning from ParT's frozen latent, the way Sophon uses it.

Sophon's second mode is to freeze the 128-d class token and train a small MLP on
top of it. The paper does that supervised, against a named signal. For anomaly
detection it uses the same latent inside a weakly supervised scheme, which needs
no per-jet label. Both are run here on our own JetClass-pretrained ParT.

  supervised   an MLP trained on top-versus-QCD labels. Not a detector: it knows
               the signal. It measures how much of ParT's power survives in the
               frozen token, and gives a ceiling for everything below.

  IAD          idealised anomaly detection, the scheme Sophon benchmarks. Two
               samples are built from background alone plus an injected signal
               fraction f: "data" = background half B plus f of top, "reference"
               = background half A. A classifier learns to tell them apart. It
               never sees a per-jet label, only that one sample is contaminated,
               which is what a sideband gives you in a real search.

Both are also run on the ten observables, the high-level input the paper
compares against, so the question "is the latent worth it" has an answer rather
than an assumption.

    python -m anomaly.transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .density import QCD_LABEL, WORKING_POINTS, working_points
from .sweep import OBS, obs_matrix

TOP_LABEL = 0


class MLP(nn.Module):
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


def fit_mlp(x, y, x_te, device, epochs=8, batch=4096, lr=1e-3, seed=0, n_val=50_000):
    """Train a binary classifier, early-stopping on held-out loss, score the test."""
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
            xb, yb = xt[sl].to(device), yt[sl].to(device)
            loss = lossf(model(xb), yb)
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
    xe = torch.tensor(x_te, dtype=torch.float32)
    with torch.no_grad():
        return np.concatenate([model(xe[i:i + 8192].to(device)).cpu().numpy()
                               for i in range(0, len(xe), 8192)])


def standardise(fit, *others):
    mu, sd = fit.mean(0), fit.std(0) + 1e-6
    return [((a - mu) / sd).astype(np.float32) for a in (fit, *others)]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, default=repo / "surrogate/outputs/top/embeddings")
    p.add_argument("--feat-dir", type=Path, default=repo / "surrogate/outputs/top/features")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.05, 0.02, 0.01])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=repo / "anomaly/outputs")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    tr, te = np.load(args.emb_dir / "xcls_train.npz"), np.load(args.emb_dir / "xcls_test.npz")
    f_tr = pd.read_parquet(args.feat_dir / "features_train.parquet", columns=OBS + ["label"])
    f_te = pd.read_parquet(args.feat_dir / "features_test.parquet", columns=OBS + ["label"])
    assert (f_tr.label.to_numpy() == tr["label"]).all() and (f_te.label.to_numpy() == te["label"]).all()
    lab_tr, lab_te = tr["label"], te["label"]
    y_te = (lab_te == TOP_LABEL).astype(int)

    o_tr, med = obs_matrix(f_tr)
    o_te, _ = obs_matrix(f_te, med)
    spaces = {"latent x_cls (128)": (tr["x"].astype(np.float32), te["x"].astype(np.float32)),
              "10 osservabili": (o_tr.astype(np.float32), o_te.astype(np.float32))}

    results, rows = {}, []
    # ------------------------------------------------------------- supervised
    print(f"{'':<34}{'AUC':>8}{'TPR@10%':>10}{'TPR@1%':>9}{'TPR@0.1%':>11}")
    for name, (X, Xe) in spaces.items():
        xs, xes = standardise(X, Xe)
        s = fit_mlp(xs, (lab_tr == TOP_LABEL).astype(np.float32), xes, device, args.epochs)
        r = working_points(y_te, s)
        results[f"supervised | {name}"] = r
        print(f"{'[supervisionato] ' + name:<34}{r['auc']:>8.4f}{r['tpr@0.1']:>10.3f}"
              f"{r['tpr@0.01']:>9.3f}{r['tpr@0.001']:>11.4f}")

    # -------------------------------------------------------------------- IAD
    bkg = np.flatnonzero(lab_tr == QCD_LABEL)
    top = np.flatnonzero(lab_tr == TOP_LABEL)
    ref, dat = bkg[::2], bkg[1::2]          # two disjoint halves of background
    for f in args.fractions:
        n_sig = int(round(f * len(dat)))
        sig = rng.choice(top, size=n_sig, replace=False)
        idx = np.concatenate([ref, dat, sig])
        y = np.concatenate([np.zeros(len(ref)), np.ones(len(dat) + n_sig)]).astype(np.float32)
        print(f"\nIAD  frazione di segnale {f:.0%}   "
              f"riferimento {len(ref):,}  dati {len(dat) + n_sig:,} (di cui {n_sig:,} top)")
        for name, (X, Xe) in spaces.items():
            xs, xes = standardise(X[idx], Xe)
            s = fit_mlp(xs, y, xes, device, args.epochs)
            r = working_points(y_te, s)
            results[f"IAD f={f} | {name}"] = r
            rows.append({"f": f, "space": name, **r})
            print(f"{'  ' + name:<34}{r['auc']:>8.4f}{r['tpr@0.1']:>10.3f}"
                  f"{r['tpr@0.01']:>9.3f}{r['tpr@0.001']:>11.4f}")

    (args.out_dir / "transfer_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out_dir / 'transfer_summary.json'}")


if __name__ == "__main__":
    main()
