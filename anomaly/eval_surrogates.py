"""How faithfully the surrogates reproduce the IAD detector, over seeds.

For every signal fraction, over the detector seeds that have a fitted surrogate
(anomaly/outputs/seed<s>/surrogate/iad_f<f>_latent/). The same four headline
metrics as the top-tagging surrogate (R², ADO, decision agreement, AUC), with
the two adaptations the AD score needs (a working-point threshold instead of
the sign, and top efficiency at fixed background next to the AUC):

  fidelity    R² and Spearman of Ridge / EBM / GBDT with the detector's logit
  ado         average decision ordering: over 500k fixed (top, QCD) test pairs,
              the share the surrogate orders as the detector does. Reference:
              the detector's own ordering against the truth (its AUC on the
              same pairs), as in surrogate.seed_variance
  decisions   at the working point. A jet is "anomalous" when the detector's
              logit exceeds the threshold that keeps WP of the background
              (QCD) jets of the selection half of val.
                common    the surrogate's decision uses the same threshold on
                          its own prediction (as top tagging uses the same sign)
                quantile  the surrogate's own threshold, keeping the same WP of
                          the selection-half QCD jets by its own prediction:
                          agreement of the two rankings at the working point,
                          free of the surrogate's shrinkage toward the mean
              Reported: agreement, recall (share of the detector's anomalous
              jets the surrogate flags), precision, flagged fractions. 10%
              background is the narratives' working point; 1% alongside.
  as a        the surrogate's own AUC (top vs QCD, truth labels, test) and top
  detector    efficiency at 10% background, against the detector's
  regions     EBM error by region of the detector score: RMSE in the QCD-like
              bulk (below the 10% threshold), among the flagged jets (above
              it) and above the 1% threshold, and Spearman among the flagged
  gate        the narrative abstention gate on r = t_clip - g_hat (EBM):
              full |r| < a, caveat a <= |r| < b, abstain |r| >= b, on all test
              jets and on the jets the detector flags at 10%. Two scales:
                absolute  a, b = 1, 2, the top-tagging thresholds as they are
                scaled    a, b = (1, 2) * sd(t_AD) / sd(t_top): the same
                          thresholds in units of the target's spread, since the
                          IAD logit is several times narrower than ParT's
  importance  the EBM's mean |φ| per observable, mean ± sd over seeds
  stability   between every pair of detector seeds, on the same test jets: φ
              correlation, same leading driver, the three leading drivers
              shared and agreeing in sign (surrogate.compare_teachers)

    python -m anomaly.eval_surrogates
    python -m anomaly.eval_surrogates --fractions 0.1 --out summary.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics as st
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from surrogate.compare_teachers import pair_stability
from surrogate.train_surrogates_top import prepare_raw_X, prepare_ridge_X

MODELS = ("ridge", "ebm", "gbdt")
WPS = (0.10, 0.01)
ADO_PAIRS = 500_000
GATE_ABS = (1.0, 2.0)


def tpr_at(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[min(int(np.searchsorted(fpr, fpr_target)), len(tpr) - 1)])


def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {"mean": float(st.mean(v)), "sd": float(st.stdev(v)) if len(v) > 1 else 0.0, "n": len(v)}


def fmt(d, nd=3):
    return f"{d['mean']:.{nd}f} ± {d['sd']:.{nd}f}"


def predict_all(mod: Path, df: pd.DataFrame) -> dict:
    """The three saved surrogates applied to a feature table (the selection half of val)."""
    out = {}
    r = joblib.load(mod / "ridge.joblib")
    Xr, _ = prepare_ridge_X(df, list(r["obs"]), r["cfg"]["ridge_log_features"], None, r["ridge_stats"])
    out["ridge"] = r["ridge"].predict(Xr)
    for m in ("ebm", "gbdt"):
        b = joblib.load(mod / f"{m}.joblib")
        X, _ = prepare_raw_X(df, list(b["obs"]), b["raw_medians"])
        out[m] = b[m].predict(X)
    return out


def decision_stats(d, s):
    return {
        "agreement": float((d == s).mean()),
        "recall": float(s[d].mean()) if d.any() else float("nan"),
        "precision": float(d[s].mean()) if s.any() else float("nan"),
        "flagged_detector": float(d.mean()), "flagged_surrogate": float(s.mean()),
    }


def gate(r, a, b):
    r = np.abs(r)
    return {"full": float((r < a).mean()), "caveat": float(((r >= a) & (r < b)).mean()),
            "abstain": float((r >= b).mean())}


def load(seed_dir: Path, run: str, idx: np.ndarray, pairs, sd_top: float) -> dict:
    inp = seed_dir / "surrogate_inputs" / run
    mod = seed_dir / "surrogate" / run
    metrics = json.loads((mod / "metrics.json").read_text())
    sel = pd.read_parquet(inp / "features_val.parquet")
    test = pd.read_parquet(inp / "features_test.parquet")
    pred = pd.read_parquet(mod / "test_predictions.parquet",
                           columns=["jet_id", "t_clip", "t_hat_ridge", "t_hat_ebm", "t_hat_gbdt"])
    if not (pred["jet_id"].to_numpy() == test["jet_id"].to_numpy()).all():
        raise SystemExit(f"{mod}: predictions and test table are not row-aligned")

    t = test["t"].to_numpy(np.float64)                 # the detector's logit, unclipped
    t_clip = pred["t_clip"].to_numpy(np.float64)
    y_top = (test["label"].to_numpy() == 0).astype(int)
    qcd_sel_mask = sel["label"].to_numpy() == 1
    qcd_sel = sel["t"].to_numpy()[qcd_sel_mask]
    thr = {wp: float(np.quantile(qcd_sel, 1 - wp)) for wp in WPS}
    sel_pred = predict_all(mod, sel.loc[qcd_sel_mask])
    pi, pj = pairs

    out = {"fidelity": {}, "ado": {}, "decisions": {}, "as_detector": {}, "thresholds": thr}
    out["ado"]["detector_vs_truth"] = float(np.mean(t[pi] > t[pj]))
    out["as_detector"]["detector"] = {"auc": float(roc_auc_score(y_top, t)), "tpr@0.1": tpr_at(y_top, t, 0.10)}
    for m in MODELS:
        yh = pred[f"t_hat_{m}"].to_numpy()
        out["fidelity"][m] = {"r2": metrics[m]["test"]["r2_all"], "spearman": metrics[m]["test"]["spearman"]}
        out["ado"][m] = float(np.mean(np.sign(yh[pi] - yh[pj]) == np.sign(t[pi] - t[pj])))
        out["as_detector"][m] = {"auc": float(roc_auc_score(y_top, yh)), "tpr@0.1": tpr_at(y_top, yh, 0.10)}
        for wp in WPS:
            d = t > thr[wp]
            thr_own = float(np.quantile(sel_pred[m], 1 - wp))
            out["decisions"][f"{m}@{wp:g}"] = decision_stats(d, yh > thr[wp])
            out["decisions"][f"{m}@{wp:g}:quantile"] = decision_stats(d, yh > thr_own)

    # Where the EBM errs, by region of the detector score.
    g = pred["t_hat_ebm"].to_numpy()
    r = t_clip - g
    regions = {"bulk": t <= thr[0.10], "flagged@10%": t > thr[0.10], "flagged@1%": t > thr[0.01]}
    out["regions"] = {k: {"rmse": float(np.sqrt(np.mean(r[mk] ** 2))), "share": float(mk.mean())}
                      for k, mk in regions.items()}
    fl = regions["flagged@10%"]
    out["regions"]["flagged@10%"]["spearman"] = float(pd.Series(t_clip[fl]).corr(pd.Series(g[fl]), method="spearman"))

    sd_t = float(np.std(t_clip))
    scale = sd_t / sd_top
    out["gate"] = {"sd_t": sd_t, "scale": scale}
    for name, (a, b) in (("absolute", GATE_ABS), ("scaled", (GATE_ABS[0] * scale, GATE_ABS[1] * scale))):
        out["gate"][name] = {"thresholds": [a, b], "all": gate(r, a, b), "flagged@10%": gate(r[fl], a, b)}

    bundle = joblib.load(mod / "ebm.joblib")
    obs = list(bundle["obs"])
    x = test.iloc[idx][obs].copy()
    for c in obs:
        x[c] = x[c].fillna(bundle["raw_medians"][c])
    out["importance"] = dict(metrics.get("ebm_importances") or [])
    out["_phi"] = np.asarray(bundle["ebm"].eval_terms(x), dtype=np.float64)
    out["_t"] = t[idx] - thr[0.10]                     # sign = the detector's decision at 10%
    out["obs"] = obs
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=repo / "anomaly/outputs")
    p.add_argument("--fractions", nargs="+", default=["0.1", "0.05", "0.01"])
    p.add_argument("--n-jets", type=int, default=20000, help="Test jets for the φ comparison")
    p.add_argument("--top-predictions", type=Path,
                   default=repo / "surrogate/outputs/top/models/test_predictions.parquet",
                   help="Top-tagging surrogate predictions, for the spread of ParT's logit (gate scale)")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    sd_top = float(np.std(pd.read_parquet(a.top_predictions, columns=["t_clip"])["t_clip"].to_numpy()))
    print(f"top-tagging target spread sd(t_clip) = {sd_top:.3f}  (scale for the gate thresholds)")

    summary = {"sd_t_top": sd_top}
    for f in a.fractions:
        run = f"iad_f{f}_latent"
        seed_dirs = sorted((d for d in a.dir.glob("seed*") if (d / "surrogate" / run / "metrics.json").exists()),
                           key=lambda d: int(re.sub(r"\D", "", d.name)))
        if not seed_dirs:
            print(f"f={f}: no fitted surrogate yet")
            continue
        test0 = pd.read_parquet(seed_dirs[0] / "surrogate_inputs" / run / "features_test.parquet",
                                columns=["jet_id", "label"])
        n_test = len(test0)
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n_test, size=min(a.n_jets, n_test), replace=False))
        y0 = test0["label"].to_numpy()
        pairs = (rng.choice(np.flatnonzero(y0 == 0), ADO_PAIRS), rng.choice(np.flatnonzero(y0 == 1), ADO_PAIRS))
        runs = {}
        for d in seed_dirs:
            lab = pd.read_parquet(d / "surrogate_inputs" / run / "features_test.parquet", columns=["jet_id", "label"])
            if not (lab["jet_id"].to_numpy() == test0["jet_id"].to_numpy()).all():
                raise SystemExit(f"{d}: test jets differ from {seed_dirs[0]}; the fixed pairs would not match")
            runs[d.name] = load(d, run, idx, pairs, sd_top)
        R = list(runs.values())
        obs = R[0]["obs"]

        s = {"seeds": list(runs),
             "fidelity": {m: {k: ms([r["fidelity"][m][k] for r in R]) for k in ("r2", "spearman")} for m in MODELS},
             "ado": {k: ms([r["ado"][k] for r in R]) for k in R[0]["ado"]},
             "decisions": {key: {k: ms([r["decisions"][key][k] for r in R]) for k in R[0]["decisions"][key]}
                           for key in R[0]["decisions"]},
             "as_detector": {m: {k: ms([r["as_detector"][m][k] for r in R]) for k in ("auc", "tpr@0.1")}
                             for m in ("detector",) + MODELS},
             "regions": {reg: {k: ms([r["regions"][reg][k] for r in R]) for k in R[0]["regions"][reg]}
                         for reg in R[0]["regions"]},
             "gate": {"sd_t": ms([r["gate"]["sd_t"] for r in R]), "scale": ms([r["gate"]["scale"] for r in R]),
                      **{sc: {"thresholds": [ms([r["gate"][sc]["thresholds"][i] for r in R]) for i in (0, 1)],
                              **{pop: {k: ms([r["gate"][sc][pop][k] for r in R]) for k in ("full", "caveat", "abstain")}
                                 for pop in ("all", "flagged@10%")}}
                         for sc in ("absolute", "scaled")}},
             "importance": {o: ms([r["importance"].get(o, np.nan) for r in R]) for o in obs}}
        sp = {f"{x} vs {y}": pair_stability(runs[x], runs[y]) for x, y in itertools.combinations(runs, 2)}
        if sp:
            keys = [k for k, v in next(iter(sp.values())).items() if isinstance(v, float)]
            s["stability"] = {k: ms([v[k] for v in sp.values()]) for k in keys}
        summary[run] = s

        print(f"\n===== IAD latent, f = {f}   ({len(R)} detector seeds: {', '.join(runs)})")
        print(f"{'':8s}{'R²':>16s}{'Spearman':>16s}{'ADO':>16s}{'AUC':>16s}{'TPR@10%bkg':>16s}")
        for m in ("detector",) + MODELS:
            fid = s["fidelity"].get(m)
            ado = s["ado"]["detector_vs_truth" if m == "detector" else m]
            print(f"{m:8s}{fmt(fid['r2']) if fid else '—':>16s}{fmt(fid['spearman']) if fid else '—':>16s}"
                  f"{fmt(ado):>16s}{fmt(s['as_detector'][m]['auc']):>16s}{fmt(s['as_detector'][m]['tpr@0.1']):>16s}")
        print("  (detector row: ADO = its ordering against the truth on the same pairs)")
        print(f"\n{'decisions':14s}{'agree':>16s}{'recall':>16s}{'precision':>16s}{'flag det':>10s}{'flag sur':>10s}")
        for m in MODELS:
            for wp in WPS:
                for kind in ("", ":quantile"):
                    dd = s["decisions"][f"{m}@{wp:g}{kind}"]
                    lab = f"{m}@{wp:g}{' q' if kind else ' c'}"
                    print(f"{lab:14s}{fmt(dd['agreement']):>16s}{fmt(dd['recall']):>16s}{fmt(dd['precision']):>16s}"
                          f"{dd['flagged_detector']['mean']:>10.3f}{dd['flagged_surrogate']['mean']:>10.3f}")
        print("  (c = common threshold, q = quantile-matched)")
        rg = s["regions"]
        print("\nEBM error by region:  " + "   ".join(
            f"{k} (share {v['share']['mean']:.3f}) RMSE {fmt(v['rmse'])}" for k, v in rg.items())
            + f"   Spearman among flagged {fmt(rg['flagged@10%']['spearman'])}")
        gt = s["gate"]
        print(f"gate: sd(t_AD) {fmt(gt['sd_t'])}  scale {fmt(gt['scale'])}")
        for sc in ("absolute", "scaled"):
            th = gt[sc]["thresholds"]
            print(f"  {sc:8s} thresholds {th[0]['mean']:.2f} / {th[1]['mean']:.2f}:  " + "   ".join(
                f"{pop}: full {gt[sc][pop]['full']['mean']:.3f} caveat {gt[sc][pop]['caveat']['mean']:.3f} "
                f"abstain {gt[sc][pop]['abstain']['mean']:.3f}" for pop in ("all", "flagged@10%")))
        print("EBM importance (mean |φ|):  " + ", ".join(
            f"{o} {s['importance'][o]['mean']:.2f}" for o in sorted(obs, key=lambda o: -s['importance'][o]['mean'])))
        if sp:
            st_ = s["stability"]
            print(f"stability over {len(sp)} seed pairs:  φ corr {fmt(st_['phi_corr_mean'])}   "
                  f"same leading driver {fmt(st_['jet_top1_same'])}   "
                  f"top-3 all shared {fmt(st_['jet_top3_all_shared'])}   sign {fmt(st_['jet_top3_sign_agree'])}")

    if a.out:
        a.out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
