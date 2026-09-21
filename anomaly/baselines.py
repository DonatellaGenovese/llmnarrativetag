"""What weak supervision is measured against: three unsupervised baselines.

These exist so the main result has a floor to stand on. Each is fit on
background alone — the even rows of the training split's QCD — and read on the
test split, at fixed background efficiency and again with the mass dependence
removed.

  mass distance    |m_SD - median_bkg| in background IQR units. One dimension,
                   no model, and on this dataset the strongest unsupervised
                   score there is. Anything that cannot beat it is a mass cut.
  latent euclid    squared distance from the background centroid in ParT's
                   frozen 128-d class token, per-dimension standardised.
  latent cosine    one minus the cosine similarity to the same centroid.

The two latent scores are the best of a long list that also included
Mahalanobis, PCA residuals, Gaussian mixtures, a normalising flow and kNN; all
of them did worse, and the better the density model the worse it got. The
distances survive; the likelihoods do not.

    python -m anomaly.baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import QCD_LABEL, TOP_LABEL, header, report


def fit_rows(label: np.ndarray, which: int) -> np.ndarray:
    """The half of a class the models are fit on: even rows, by convention.

    The odd rows are left alone so that a score is never normalised on jets the
    model has already seen.
    """
    return (label == which) & (np.arange(len(label)) % 2 == 0)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, default=repo / "surrogate/outputs/top/embeddings")
    p.add_argument("--feat-dir", type=Path, default=repo / "surrogate/outputs/top/features")
    p.add_argument("--out-dir", type=Path, default=repo / "anomaly/outputs")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tr = np.load(args.emb_dir / "xcls_train.npz")
    te = np.load(args.emb_dir / "xcls_test.npz")
    f_tr = pd.read_parquet(args.feat_dir / "features_train.parquet", columns=["m_SD", "label"])
    f_te = pd.read_parquet(args.feat_dir / "features_test.parquet", columns=["m_SD", "label"])
    if not ((f_tr.label.to_numpy() == tr["label"]).all()
            and (f_te.label.to_numpy() == te["label"]).all()):
        raise SystemExit("embeddings and feature table are not row-aligned")

    lab_tr, lab_te = tr["label"], te["label"]
    m_tr, m_te = f_tr.m_SD.to_numpy(), f_te.m_SD.to_numpy()
    y = (lab_te == TOP_LABEL).astype(int)
    is_bkg = lab_te == QCD_LABEL

    fit = fit_rows(lab_tr, QCD_LABEL)
    x = tr["x"][fit].astype(np.float64)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = (te["x"].astype(np.float64) - mu) / sd
    centroid = ((x - mu) / sd).mean(0)

    ref = m_tr[fit]
    med, iqr = np.median(ref), np.subtract(*np.percentile(ref, [75, 25]))

    scores = {
        "distanza in massa (1-d)": np.abs(m_te - med) / iqr,
        "latente, euclid": (z ** 2).sum(1),
        "latente, cosine": 1 - (z @ centroid) / (np.linalg.norm(z, axis=1)
                                                 * np.linalg.norm(centroid) + 1e-12),
    }

    print(f"fit su {fit.sum():,} jet di fondo, test su {len(z):,} jet "
          f"({y.sum():,} top)\n")
    header()
    out = {n: report(y, s, m_te, is_bkg, n) for n, s in scores.items()}
    (args.out_dir / "baselines_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_dir / 'baselines_summary.json'}")


if __name__ == "__main__":
    main()
