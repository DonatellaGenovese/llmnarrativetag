"""Two checks a density score on x_cls has to pass before it is trusted.

The Euclidean distance from the background centroid beats every better density
model, and the better the model the worse it does, while its correlation with
soft-drop mass falls from +0.68 to +0.10. That pattern fits a score that is
mostly a mass cut. So:

  1. Mass. A pure mass baseline is scored the same way: the distance of m_SD
     from the background median, in background IQR units, which is as
     model-agnostic as the latent distance itself. Then every score is
     evaluated inside narrow m_SD windows, where mass cannot separate anything,
     and after decorrelation: each jet's score replaced by its percentile among
     background jets of similar mass.

  2. Direction. Fit on top instead and look for QCD as the anomaly. A detector
     of distance from the background passes in both directions; a detector of
     "heavy and three-pronged" fails the second one.

    python -m anomaly.checks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .density import QCD_LABEL, WORKING_POINTS, working_points

TOP_LABEL = 0


def fmt(r: dict) -> str:
    return f"{r['auc']:>7.4f}" + "".join(
        f"{r[f'tpr@{t}']:>8.4f} ({r[f'gain@{t}']:>5.2f}x)" for t in WORKING_POINTS)


def euclid_fit_on(x_tr, lab_tr, fit_label, x_te):
    rows = (lab_tr == fit_label) & (np.arange(len(lab_tr)) % 2 == 0)
    x = x_tr[rows].astype(np.float64)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    return (((x_te.astype(np.float64) - mu) / sd) ** 2).sum(1), int(rows.sum())


def mass_distance(m_tr, lab_tr, fit_label, m_te):
    ref = m_tr[(lab_tr == fit_label) & (np.arange(len(lab_tr)) % 2 == 0)]
    med = np.median(ref)
    iqr = np.subtract(*np.percentile(ref, [75, 25]))
    return np.abs(m_te - med) / iqr


def decorrelate(score, mass, is_bkg, n_bins=40):
    """Percentile of each jet's score among background jets of similar mass."""
    edges = np.unique(np.quantile(mass[is_bkg], np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, mass, side="right") - 1, 0, len(edges) - 2)
    out = np.empty_like(score, dtype=np.float64)
    for k in range(len(edges) - 1):
        sel = b == k
        ref = np.sort(score[sel & is_bkg])
        out[sel] = np.searchsorted(ref, score[sel], side="right") / max(len(ref), 1)
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", type=Path, default=repo / "surrogate/outputs/top/embeddings")
    p.add_argument("--feat-dir", type=Path, default=repo / "surrogate/outputs/top/features")
    p.add_argument("--out-dir", type=Path, default=repo / "anomaly/outputs")
    args = p.parse_args()

    tr, te = np.load(args.emb_dir / "xcls_train.npz"), np.load(args.emb_dir / "xcls_test.npz")
    f_tr = pd.read_parquet(args.feat_dir / "features_train.parquet", columns=["m_SD", "label"])
    f_te = pd.read_parquet(args.feat_dir / "features_test.parquet", columns=["m_SD", "tau32", "label"])
    assert (f_tr.label.to_numpy() == tr["label"]).all() and (f_te.label.to_numpy() == te["label"]).all()
    lab_tr, lab_te = tr["label"], te["label"]
    m_tr, m_te = f_tr.m_SD.to_numpy(), f_te.m_SD.to_numpy()
    head_path = args.out_dir / "density_models_test.parquet"
    head = (pd.read_parquet(head_path)["d_head_not_qcd"].to_numpy() if head_path.exists() else None)

    out = {}
    hdr = f"{'':<34}{'AUC':>7}{'TPR@10%':>16}{'TPR@1%':>16}{'TPR@0.1%':>16}"

    # ---------------------------------------------------------------- 1. mass
    is_sig = (lab_te == TOP_LABEL).astype(int)
    is_bkg = lab_te == QCD_LABEL
    eu, n_fit = euclid_fit_on(tr["x"], lab_tr, QCD_LABEL, te["x"])
    md = mass_distance(m_tr, lab_tr, QCD_LABEL, m_te)
    cos_c = tr["x"][(lab_tr == QCD_LABEL) & (np.arange(len(lab_tr)) % 2 == 0)].astype(np.float64)
    mu_c, sd_c = cos_c.mean(0), cos_c.std(0) + 1e-6
    zc = (te["x"].astype(np.float64) - mu_c) / sd_c
    cen = ((cos_c - mu_c) / sd_c).mean(0)
    cosine = 1 - (zc @ cen) / (np.linalg.norm(zc, axis=1) * np.linalg.norm(cen) + 1e-12)
    scores = {"euclid (x_cls)": eu, "cosine (x_cls)": cosine, "mass distance only": md}
    if head is not None:
        scores["head_not_qcd"] = head

    print(f"1) MASS   fit on {n_fit:,} QCD, anomaly = top\n{hdr}")
    out["standard"] = {}
    for name, sc in scores.items():
        r = working_points(is_sig, sc)
        out["standard"][name] = r
        print(f"{name:<34}{fmt(r)}")
    print("  after decorrelation from m_SD (percentile among QCD of similar mass):")
    out["decorrelated"] = {}
    for name, sc in scores.items():
        r = working_points(is_sig, decorrelate(sc, m_te, is_bkg))
        out["decorrelated"][name] = r
        print(f"  {name:<32}{fmt(r)}")

    print("\n  inside m_SD windows (AUC | TPR at 1% of the window's own QCD):")
    print(f"  {'window [GeV]':<16}{'n top':>8}{'n QCD':>8}" + "".join(f"{n[:16]:>18}" for n in scores))
    out["windows"] = []
    for lo, hi in ((0, 60), (60, 120), (120, 150), (150, 170), (170, 190), (190, 250)):
        sel = (m_te >= lo) & (m_te < hi)
        y = is_sig[sel]
        if y.sum() < 200 or (1 - y).sum() < 200:
            continue
        row = {"lo": lo, "hi": hi, "n_top": int(y.sum()), "n_qcd": int((1 - y).sum())}
        cells = ""
        for name in scores:
            r = working_points(y, scores[name][sel])
            row[name] = r
            cells += f"{r['auc']:>9.3f} {r['gain@0.01']:>6.2f}x "
        out["windows"].append(row)
        print(f"  {f'{lo}-{hi}':<16}{row['n_top']:>8}{row['n_qcd']:>8}{cells}")

    # ----------------------------------------------------------- 2. direction
    is_sig_inv = (lab_te == QCD_LABEL).astype(int)
    eu_inv, n_fit_inv = euclid_fit_on(tr["x"], lab_tr, TOP_LABEL, te["x"])
    md_inv = mass_distance(m_tr, lab_tr, TOP_LABEL, m_te)
    print(f"\n2) DIRECTION   fit on {n_fit_inv:,} top, anomaly = QCD\n{hdr}")
    out["inverse"] = {}
    for name, sc in (("euclid (x_cls)", eu_inv), ("mass distance only", md_inv)):
        r = working_points(is_sig_inv, sc)
        out["inverse"][name] = r
        print(f"{name:<34}{fmt(r)}")
    r = working_points(is_sig_inv, decorrelate(eu_inv, m_te, lab_te == TOP_LABEL))
    out["inverse"]["euclid decorrelated"] = r
    print(f"{'euclid, decorrelated from m_SD':<34}{fmt(r)}")

    (args.out_dir / "checks_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_dir / 'checks_summary.json'}")


if __name__ == "__main__":
    main()
