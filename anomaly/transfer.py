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


def fit_mlp(x, y, x_te, device, epochs=8, batch=4096, lr=1e-3, seed=0, n_val=50_000):
    """Train, early-stop on held-out loss, and score the test split.

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
    p.add_argument("--fractions", type=float, nargs="+", default=[0.10, 0.05, 0.02, 0.01],
                   help="Signal fractions injected into the contaminated sample")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--skip-supervised", action="store_true")
    p.add_argument("--out-dir", type=Path, default=repo / "anomaly/outputs")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tr = np.load(args.emb_dir / "xcls_train.npz")
    te = np.load(args.emb_dir / "xcls_test.npz")
    f_tr = pd.read_parquet(args.feat_dir / "features_train.parquet", columns=OBS + ["label"])
    # m_SD is already one of OBS; asking for it twice yields a duplicated
    # column, and every downstream shape goes wrong two functions later.
    f_te = pd.read_parquet(args.feat_dir / "features_test.parquet", columns=OBS + ["label"])
    if not ((f_tr.label.to_numpy() == tr["label"]).all()
            and (f_te.label.to_numpy() == te["label"]).all()):
        raise SystemExit("embeddings and feature table are not row-aligned")

    lab_tr, lab_te = tr["label"], te["label"]
    y_te = (lab_te == TOP_LABEL).astype(int)
    m_te, is_bkg = f_te.m_SD.to_numpy(), lab_te == QCD_LABEL

    o_tr, med = obs_matrix(f_tr)
    o_te, _ = obs_matrix(f_te, med)
    spaces = {"latente x_cls (128)": (tr["x"].astype(np.float32), te["x"].astype(np.float32)),
              "10 osservabili": (o_tr.astype(np.float32), o_te.astype(np.float32))}

    results = {}
    if not args.skip_supervised:
        print("SUPERVISIONATO — conosce le etichette, è il tetto\n")
        header()
        for name, (X, Xe) in spaces.items():
            xs, xes = standardise(X, Xe)
            s = fit_mlp(xs, (lab_tr == TOP_LABEL).astype(np.float32), xes, device, args.epochs)
            results[f"supervised | {name}"] = report(y_te, s, m_te, is_bkg, "  " + name)

    bkg = np.flatnonzero(lab_tr == QCD_LABEL)
    top = np.flatnonzero(lab_tr == TOP_LABEL)
    ref, dat = bkg[::2], bkg[1::2]        # two disjoint halves of the background
    for f in args.fractions:
        n_sig = int(round(f * len(dat)))
        sig = rng.choice(top, size=n_sig, replace=False)
        idx = np.concatenate([ref, dat, sig])
        y = np.concatenate([np.zeros(len(ref)), np.ones(len(dat) + n_sig)]).astype(np.float32)
        print(f"\nIAD  segnale iniettato {f:.0%}   riferimento {len(ref):,} · "
              f"dati {len(dat) + n_sig:,} (di cui {n_sig:,} top)\n")
        header()
        for name, (X, Xe) in spaces.items():
            xs, xes = standardise(X[idx], Xe)
            s = fit_mlp(xs, y, xes, device, args.epochs)
            results[f"IAD f={f} | {name}"] = report(y_te, s, m_te, is_bkg, "  " + name)

    (args.out_dir / "transfer_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out_dir / 'transfer_summary.json'}")


if __name__ == "__main__":
    main()
