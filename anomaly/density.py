"""Anomaly scores from the density of background in ParT's latent space.

The constituent autoencoder failed in a specific way: its error tracks how hard a
jet is to reconstruct, which on this dataset means how sparse it is, and sparse
QCD fills the tail. A density score reconstructs nothing, so that failure mode
has nowhere to enter. It asks only how far a jet's class token sits from where
background tokens live.

Fit on the same background jets the decoder was fit on, the even rows of the
train split, and evaluated on the test split. Three scores, cheapest first:

  euclid      squared distance after per-dimension standardisation, a diagonal
              Gaussian. The baseline that ignores correlations.
  mahalanobis squared Mahalanobis distance under the background mean and a
              Ledoit-Wolf shrunk covariance, a full Gaussian.
  pca_k       reconstruction error from the top k principal components of the
              background, the linear autoencoder, for comparison at full scale.

Every score is reported at fixed background efficiency, because AUC averages
over thresholds nobody uses and hid the autoencoder's failure until the working
points were read.

    python -m anomaly.density
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve

QCD_LABEL = 1
WORKING_POINTS = (0.10, 0.01, 0.001)


def working_points(y_sig: np.ndarray, score: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y_sig, score)
    out = {"auc": float(roc_auc_score(y_sig, score))}
    for t in WORKING_POINTS:
        i = min(int(np.searchsorted(fpr, t)), len(tpr) - 1)
        out[f"tpr@{t}"] = float(tpr[i])
        out[f"gain@{t}"] = float(tpr[i] / max(fpr[i], 1e-12))
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, default=repo / "surrogate/outputs/top/embeddings")
    p.add_argument("--features", type=Path,
                   default=repo / "surrogate/outputs/top/features/features_test.parquet")
    p.add_argument("--pca-k", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--out-dir", type=Path, default=repo / "anomaly/outputs")
    args = p.parse_args()

    tr = np.load(args.emb_dir / "xcls_train.npz")
    te = np.load(args.emb_dir / "xcls_test.npz")
    lab_tr = tr["label"]
    fit_rows = (lab_tr == QCD_LABEL) & (np.arange(len(lab_tr)) % 2 == 0)
    x_fit = tr["x"][fit_rows].astype(np.float64)
    x_te = te["x"].astype(np.float64)
    y_sig = (te["label"] != QCD_LABEL).astype(int)
    print(f"fit on {len(x_fit):,} background tokens (train, even rows); "
          f"test {len(x_te):,} jets, {y_sig.sum():,} top")

    mu, sd = x_fit.mean(0), x_fit.std(0) + 1e-6
    z_fit, z_te = (x_fit - mu) / sd, (x_te - mu) / sd

    scores = {"euclid": (z_te ** 2).sum(1)}

    lw = LedoitWolf().fit(z_fit)
    prec = lw.precision_
    scores["mahalanobis"] = np.einsum("ij,jk,ik->i", z_te, prec, z_te)
    print(f"Ledoit-Wolf shrinkage = {lw.shrinkage_:.4f}")

    # PCA from the covariance already estimated, so it shares the fit.
    evals, evecs = np.linalg.eigh(np.cov(z_fit, rowvar=False))
    evecs = evecs[:, ::-1]
    for k in args.pca_k:
        v = evecs[:, :k]
        rec = z_te @ v @ v.T
        scores[f"pca_{k}"] = ((z_te - rec) ** 2).sum(1)

    feats = pd.read_parquet(args.features, columns=["n_const", "m", "label"])
    if len(feats) != len(x_te) or not (feats.label.to_numpy() == te["label"]).all():
        raise SystemExit("test embeddings and feature table are not row-aligned")

    results = {}
    print(f"\n{'score':<13}{'AUC':>7}{'TPR@10%':>17}{'TPR@1%':>17}{'TPR@0.1%':>17}"
          f"{'rho(n_const)':>14}{'%top in 0.1%':>14}")
    for name, sc in scores.items():
        r = working_points(y_sig, sc)
        rho = float(pd.Series(sc).corr(feats.n_const, method="spearman"))
        tail = sc >= np.quantile(sc, 0.999)
        r.update({"spearman_n_const": rho, "top_frac_tail_0.1pct": float(y_sig[tail].mean())})
        results[name] = r
        print(f"{name:<13}{r['auc']:>7.4f}"
              + "".join(f"{r[f'tpr@{t}']:>9.4f} ({r[f'gain@{t}']:.2f}x)" for t in WORKING_POINTS)
              + f"{rho:>14.3f}{r['top_frac_tail_0.1pct']:>14.1%}")

    best = max(results, key=lambda k: results[k]["gain@0.01"])
    pd.DataFrame({"jet_id": te["jet_id"], "label": te["label"],
                  **{f"d_{k}": v.astype(np.float32) for k, v in scores.items()}}
                 ).to_parquet(args.out_dir / "density_scores_test.parquet", index=False)
    (args.out_dir / "density_summary.json").write_text(json.dumps(
        {"n_fit": int(len(x_fit)), "lw_shrinkage": float(lw.shrinkage_),
         "results": results, "best_at_1pct": best}, indent=2))
    print(f"\nbest at 1% background: {best}")
    print(f"wrote {args.out_dir / 'density_scores_test.parquet'} and density_summary.json")


if __name__ == "__main__":
    main()
