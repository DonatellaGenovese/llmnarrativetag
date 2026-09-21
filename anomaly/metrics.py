"""How an anomaly score is read, and the observables it is read against.

Two conventions are fixed here, because every number in this package depends on
them and both were learned the hard way.

Scores are judged at fixed background efficiency, never by AUC alone. AUC
averages over thresholds nobody runs a search at: on this dataset a score
reached AUC 0.588 while keeping *fewer* top jets than a coin toss at 10%, 1%
and 0.1% background.

And every score is also read after decorrelating it from soft-drop mass. A
one-dimensional mass distance already keeps 67% of top at 10% background here,
so without that column there is no way to tell a detector from a mass cut.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

TOP_LABEL, QCD_LABEL = 0, 1          # the feature table's convention
WORKING_POINTS = (0.10, 0.01, 0.001)


def working_points(y_sig: np.ndarray, score: np.ndarray) -> dict:
    """AUC, and signal efficiency and gain at each fixed background efficiency."""
    fpr, tpr, _ = roc_curve(y_sig, score)
    out = {"auc": float(roc_auc_score(y_sig, score))}
    for t in WORKING_POINTS:
        i = min(int(np.searchsorted(fpr, t)), len(tpr) - 1)
        out[f"tpr@{t}"] = float(tpr[i])
        out[f"gain@{t}"] = float(tpr[i] / max(fpr[i], 1e-12))
    return out


def decorrelate(score: np.ndarray, mass: np.ndarray, is_bkg: np.ndarray,
                n_bins: int = 40) -> np.ndarray:
    """Each jet's score as its percentile among background jets of similar mass.

    A score that is only a mass cut goes flat under this map, which is the
    point of it: that is the null everything else has to beat.
    """
    edges = np.unique(np.quantile(mass[is_bkg], np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, mass, side="right") - 1, 0, len(edges) - 2)
    out = np.empty_like(score, dtype=np.float64)
    for k in range(len(edges) - 1):
        sel = b == k
        ref = np.sort(score[sel & is_bkg])
        out[sel] = np.searchsorted(ref, score[sel], side="right") / max(len(ref), 1)
    return out


def header() -> None:
    print(f"{'':<32}{'AUC':>7}{'TPR@10%':>9}{'TPR@1%':>8}{'TPR@.1%':>9}"
          f"  | decorrelato da m_SD: {'AUC':>5}{'TPR@10%':>9}{'TPR@1%':>8}")


def report(y_sig: np.ndarray, score: np.ndarray, mass: np.ndarray,
           is_bkg: np.ndarray, name: str) -> dict:
    """One line: the score as it is, and with the mass dependence taken out."""
    raw = working_points(y_sig, score)
    dec = working_points(y_sig, decorrelate(score, mass, is_bkg))
    print(f"{name:<32}{raw['auc']:>7.3f}{raw['tpr@0.1']:>9.3f}{raw['tpr@0.01']:>8.3f}"
          f"{raw['tpr@0.001']:>9.4f}  |{'':>22}{dec['auc']:>5.3f}"
          f"{dec['tpr@0.1']:>9.3f}{dec['tpr@0.01']:>8.3f}")
    return {"raw": raw, "mass_decorrelated": dec}


# The ten pre-registered observables, and the ones whose tails need a log before
# a distance in this space means anything.
OBS = ["m", "m_SD", "tau21", "tau32", "C2_double_b1", "D2_double_b1",
       "C3_double_b1", "N3_b1", "pT", "n_const"]
LOG_COLS = {"m", "m_SD", "pT", "D2_double_b1", "N3_b1", "n_const"}


def obs_matrix(df, med: dict | None = None):
    """The observables as a dense matrix, with training medians filled in.

    A handful of jets carry NaN in tau32, C3 and N3 — ten rows in 600k — and a
    single one of them is enough to send a training run to NaN.
    """
    X, med = [], dict(med or {})
    for c in OBS:
        v = df[c].to_numpy(dtype=np.float64)
        if c in LOG_COLS:
            v = np.log1p(np.clip(v, 0, None))
        if c not in med:
            med[c] = float(np.nanmedian(v))
        X.append(np.where(np.isfinite(v), v, med[c]))
    return np.stack(X, 1), med
