"""A sweep of unsupervised anomaly scores, read at 10% background.

Target: 70-80% of top jets kept at 10% of QCD. The scores so far reach 54.5%
(Euclidean distance on x_cls) and 66.9% (mass alone). This tries the cheap,
legitimately unsupervised alternatives the density results point to, all fit on
the same background rows (even rows of QCD train) and read on the full test.

On x_cls:
  knn_k          mean distance to the k nearest background tokens. Non-parametric
                 distance in representation space, a standard out-of-distribution
                 score, and free of the Gaussian assumption.
  pcaproj_k      squared norm of the projection on the top k background principal
                 components, not whitened. Density results say the signal sits
                 along the large-variance directions; this keeps only those.
  cosine         1 - cosine similarity to the background centroid.

On the ten observables, the physics basis the surrogate will use:
  obs_euclid     standardised Euclidean distance from the background median, with
                 log applied to the mass-like and pT columns.
  obs_knn_k      kNN distance in that same standardised space.

Combined:
  concat_knn_k   kNN on x_cls and the observables side by side, each block scaled
                 to unit total variance so neither dominates by dimension count.

For reference only, a supervised ceiling: the fine-tuned ParT logit difference t,
which knows what a top is. Nothing unsupervised is expected to reach it.

    python -m anomaly.sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .density import QCD_LABEL, WORKING_POINTS, working_points

OBS = ["m", "m_SD", "tau21", "tau32", "C2_double_b1", "D2_double_b1",
       "C3_double_b1", "N3_b1", "pT", "n_const"]
LOG_COLS = {"m", "m_SD", "pT", "D2_double_b1", "N3_b1", "n_const"}


@torch.no_grad()
def knn_mean_dist(ref: np.ndarray, query: np.ndarray, ks, device, chunk: int = 2048):
    r = torch.tensor(ref, dtype=torch.float32, device=device)
    r2 = (r * r).sum(1)
    kmax = max(ks)
    out = {k: [] for k in ks}
    for i in range(0, len(query), chunk):
        q = torch.tensor(query[i:i + chunk], dtype=torch.float32, device=device)
        d2 = (q * q).sum(1, keepdim=True) + r2[None, :] - 2.0 * q @ r.T
        top = torch.topk(d2.clamp_min(0), kmax, dim=1, largest=False).values.sqrt()
        for k in ks:
            out[k].append(top[:, :k].mean(1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def obs_matrix(df: pd.DataFrame, med: dict | None = None):
    X = []
    med = dict(med or {})
    for c in OBS:
        v = df[c].to_numpy(dtype=np.float64)
        if c in LOG_COLS:
            v = np.log1p(np.clip(v, 0, None))
        if c not in med:
            med[c] = float(np.nanmedian(v))
        v = np.where(np.isfinite(v), v, med[c])
        X.append(v)
    return np.stack(X, 1), med


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, default=repo / "surrogate/outputs/top/embeddings")
    p.add_argument("--feat-dir", type=Path, default=repo / "surrogate/outputs/top/features")
    p.add_argument("--n-ref", type=int, default=100_000, help="Background reference size for kNN")
    p.add_argument("--out-dir", type=Path, default=repo / "anomaly/outputs")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    tr, te = np.load(args.emb_dir / "xcls_train.npz"), np.load(args.emb_dir / "xcls_test.npz")
    f_tr = pd.read_parquet(args.feat_dir / "features_train.parquet", columns=OBS + ["label"])
    f_te = pd.read_parquet(args.feat_dir / "features_test.parquet", columns=OBS + ["label", "t"])
    assert (f_tr.label.to_numpy() == tr["label"]).all() and (f_te.label.to_numpy() == te["label"]).all()

    fit = (tr["label"] == QCD_LABEL) & (np.arange(len(tr["label"])) % 2 == 0)
    y = (te["label"] != QCD_LABEL).astype(int)

    x_fit, x_te = tr["x"][fit].astype(np.float64), te["x"].astype(np.float64)
    mu, sd = x_fit.mean(0), x_fit.std(0) + 1e-6
    z_fit, z_te = (x_fit - mu) / sd, (x_te - mu) / sd

    o_fit, med = obs_matrix(f_tr[fit])
    o_te, _ = obs_matrix(f_te, med)
    oc = np.median(o_fit, 0)
    os_ = np.subtract(*np.percentile(o_fit, [75, 25], axis=0)) + 1e-9
    u_fit, u_te = (o_fit - oc) / os_, (o_te - oc) / os_

    ref_idx = rng.choice(len(z_fit), size=min(args.n_ref, len(z_fit)), replace=False)
    scores = {"euclid (x_cls)": (z_te ** 2).sum(1)}

    print(f"fit {len(z_fit):,} QCD, kNN reference {len(ref_idx):,}, test {len(z_te):,}")
    for k, v in knn_mean_dist(z_fit[ref_idx], z_te, (1, 10, 50), device).items():
        scores[f"knn_{k} (x_cls)"] = v

    evals, evecs = np.linalg.eigh(np.cov(z_fit, rowvar=False))
    evecs = evecs[:, ::-1]
    for k in (2, 4, 8, 16, 32):
        scores[f"pcaproj_{k} (x_cls)"] = ((z_te @ evecs[:, :k]) ** 2).sum(1)

    c = z_fit.mean(0)
    scores["cosine (x_cls)"] = 1 - (z_te @ c) / (np.linalg.norm(z_te, axis=1) * np.linalg.norm(c) + 1e-12)

    scores["obs_euclid"] = (u_te ** 2).sum(1)
    for k, v in knn_mean_dist(u_fit[ref_idx], u_te, (1, 10, 50), device).items():
        scores[f"obs_knn_{k}"] = v

    a_fit = np.hstack([z_fit / np.sqrt(z_fit.var(0).sum()), u_fit / np.sqrt(u_fit.var(0).sum())])
    a_te = np.hstack([z_te / np.sqrt(z_fit.var(0).sum()), u_te / np.sqrt(u_fit.var(0).sum())])
    for k, v in knn_mean_dist(a_fit[ref_idx], a_te, (10, 50), device).items():
        scores[f"concat_knn_{k}"] = v

    scores["[supervised] ParT fine-tune t"] = f_te.t.to_numpy()

    results = {}
    print(f"\n{'score':<32}{'AUC':>7}{'TPR@10%':>10}{'TPR@1%':>9}{'TPR@0.1%':>10}")
    for name, sc in sorted(scores.items(), key=lambda kv: -working_points(y, kv[1])["tpr@0.1"]):
        r = working_points(y, sc)
        results[name] = r
        print(f"{name:<32}{r['auc']:>7.4f}{r['tpr@0.1']:>10.3f}{r['tpr@0.01']:>9.3f}{r['tpr@0.001']:>10.4f}")

    (args.out_dir / "sweep_summary.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({"jet_id": te["jet_id"], "label": te["label"],
                  **{f"d_{k}": v.astype(np.float32) for k, v in scores.items()}}
                 ).to_parquet(args.out_dir / "sweep_scores_test.parquet", index=False)
    print(f"\nwrote {args.out_dir / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
