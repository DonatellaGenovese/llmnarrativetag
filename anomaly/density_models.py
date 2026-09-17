"""Richer density models on ParT's class token, and the pretraining head as a score.

anomaly.density showed that a diagonal Gaussian on x_cls already works (AUC
0.88, 16x at 0.1% background) while a full Gaussian does not. Two questions
follow, and this script answers both on the same fit and test sets.

  1. Does a better background density help?
       gmm_diag_k, gmm_full_k   Gaussian mixtures, score = -log p(x)
       flow                     a RealNVP normalising flow, score = -log p(x)

  2. How does a density score compare with simply reusing a classifier?
     The JetClass pretraining head (10 classes, one of them QCD) is still in
     the checkpoint, so its logits can be applied to the same x_cls:
       head_not_qcd   1 - p(QCD)                "departs from QCD, per a
                                                 classifier of 9 other kinds"
       head_top       p(Tbqq) + p(Tbl)          "looks like a top", which is
                                                 supervised by construction
       head_energy    -logsumexp(logits)        a standard out-of-distribution
                                                 score, no class singled out

Fit rows are the background half the decoder used (even rows of QCD train);
the flow early-stops on a 10% slice of those. Evaluation is the full test split
at fixed background efficiency, as before.

    python -m anomaly.density_models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from .density import QCD_LABEL, WORKING_POINTS, working_points

JETCLASS_LABELS = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]


class Coupling(nn.Module):
    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 256) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        xm = x * self.mask
        s, t = self.net(xm).chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        y = xm + (1 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(-1)


class RealNVP(nn.Module):
    def __init__(self, dim: int, n_layers: int = 8, hidden: int = 256) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        layers = []
        for _ in range(n_layers):
            m = torch.zeros(dim)
            m[torch.randperm(dim, generator=g)[: dim // 2]] = 1.0
            layers.append(Coupling(dim, m, hidden))
        self.layers = nn.ModuleList(layers)

    def log_prob(self, x):
        logdet = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld
        base = -0.5 * (x ** 2).sum(-1) - 0.5 * x.shape[1] * np.log(2 * np.pi)
        return base + logdet


def fit_flow(z_fit: np.ndarray, device, epochs: int, batch: int = 4096, lr: float = 1e-3):
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(z_fit))
    n_val = len(z_fit) // 10
    va, tr = torch.tensor(z_fit[idx[:n_val]], dtype=torch.float32), torch.tensor(z_fit[idx[n_val:]], dtype=torch.float32)
    flow = RealNVP(z_fit.shape[1]).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        flow.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), batch):
            xb = tr[perm[i:i + batch]].to(device)
            loss = -flow.log_prob(xb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
        flow.eval()
        with torch.no_grad():
            v = float(np.mean([-flow.log_prob(va[i:i + batch].to(device)).mean().item()
                               for i in range(0, len(va), batch)]))
        flag = ""
        if v < best - 1e-3:
            best, bad, flag = v, 0, "  <- best"
            best_state = {k: t.detach().clone() for k, t in flow.state_dict().items()}
        else:
            bad += 1
        print(f"  flow epoch {ep:>3}  val NLL/dim {v / z_fit.shape[1]:+.4f}{flag}")
        if bad >= 5:
            break
    flow.load_state_dict(best_state)
    return flow


@torch.no_grad()
def flow_score(flow, z: np.ndarray, device, batch: int = 8192) -> np.ndarray:
    t = torch.tensor(z, dtype=torch.float32)
    return np.concatenate([-flow.log_prob(t[i:i + batch].to(device)).cpu().numpy()
                           for i in range(0, len(t), batch)])


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, default=repo / "surrogate/outputs/top/embeddings")
    p.add_argument("--features", type=Path,
                   default=repo / "surrogate/outputs/top/features/features_test.parquet")
    p.add_argument("--checkpoint", type=Path, default=repo / "particle_transformer/models/ParT_kin.pt")
    p.add_argument("--gmm-k", type=int, nargs="+", default=[4, 16, 64])
    p.add_argument("--flow-epochs", type=int, default=60)
    p.add_argument("--out-dir", type=Path, default=repo / "anomaly/outputs")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, te = np.load(args.emb_dir / "xcls_train.npz"), np.load(args.emb_dir / "xcls_test.npz")
    fit_rows = (tr["label"] == QCD_LABEL) & (np.arange(len(tr["label"])) % 2 == 0)
    x_fit, x_te = tr["x"][fit_rows].astype(np.float64), te["x"].astype(np.float64)
    y_sig = (te["label"] != QCD_LABEL).astype(int)
    mu, sd = x_fit.mean(0), x_fit.std(0) + 1e-6
    z_fit, z_te = (x_fit - mu) / sd, (x_te - mu) / sd
    print(f"fit {len(z_fit):,} background tokens, test {len(z_te):,} jets")

    scores = {"euclid": (z_te ** 2).sum(1)}

    for k in args.gmm_k:
        for cov in ("diag", "full"):
            if cov == "full" and k > 16:
                continue
            gm = GaussianMixture(k, covariance_type=cov, random_state=0, max_iter=200,
                                 reg_covar=1e-4).fit(z_fit)
            scores[f"gmm_{cov}_{k}"] = -gm.score_samples(z_te)
            print(f"  gmm {cov} k={k} converged={gm.converged_}")

    flow = fit_flow(z_fit, device, args.flow_epochs)
    scores["flow"] = flow_score(flow, z_te, device)

    # The pretraining head, applied to the same tokens it was trained on top of.
    sdict = torch.load(args.checkpoint, map_location="cpu")
    W, b = sdict["mod.fc.0.weight"].double().numpy(), sdict["mod.fc.0.bias"].double().numpy()
    logits = x_te @ W.T + b
    lse = np.logaddexp.reduce(logits, axis=1)
    prob = np.exp(logits - lse[:, None])
    scores["head_not_qcd"] = 1.0 - prob[:, 0]
    scores["head_top"] = prob[:, JETCLASS_LABELS.index("Tbqq")] + prob[:, JETCLASS_LABELS.index("Tbl")]
    scores["head_energy"] = -lse

    feats = pd.read_parquet(args.features, columns=["n_const", "m_SD", "label"])
    assert (feats.label.to_numpy() == te["label"]).all(), "test rows not aligned"

    results = {}
    print(f"\n{'score':<15}{'AUC':>7}{'TPR@10%':>17}{'TPR@1%':>17}{'TPR@0.1%':>17}"
          f"{'rho(m_SD)':>11}{'rho(n_c)':>10}")
    for name, sc in scores.items():
        r = working_points(y_sig, sc)
        r["spearman_m_SD"] = float(pd.Series(sc).corr(feats.m_SD, method="spearman"))
        r["spearman_n_const"] = float(pd.Series(sc).corr(feats.n_const, method="spearman"))
        results[name] = r
        print(f"{name:<15}{r['auc']:>7.4f}"
              + "".join(f"{r[f'tpr@{t}']:>9.4f} ({r[f'gain@{t}']:>5.2f}x)" for t in WORKING_POINTS)
              + f"{r['spearman_m_SD']:>11.3f}{r['spearman_n_const']:>10.3f}")

    top_class = prob.argmax(1)
    print("\nJetClass head, predicted class on TopLandscape test:")
    for lab, name in ((0, "top"), (1, "QCD")):
        sel = te["label"] == lab
        counts = np.bincount(top_class[sel], minlength=10) / sel.sum()
        print(f"  {name:<4}" + "  ".join(f"{JETCLASS_LABELS[i]}={counts[i]:.2f}"
                                          for i in np.argsort(-counts)[:4]))

    pd.DataFrame({"jet_id": te["jet_id"], "label": te["label"],
                  **{f"d_{k}": v.astype(np.float32) for k, v in scores.items()}}
                 ).to_parquet(args.out_dir / "density_models_test.parquet", index=False)
    (args.out_dir / "density_models_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out_dir / 'density_models_test.parquet'}")


if __name__ == "__main__":
    main()
