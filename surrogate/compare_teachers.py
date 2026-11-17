"""How much the surrogate, and its explanations, depend on which ParT it imitates.

Each run directory is one teacher (a ParT fine-tune seed) with its feature table
and fitted surrogates, as written by scripts/slurm/surrogate_top.sbatch:

    surrogate/outputs/top_seed42/{features,models}/

Reported, as mean ± sd across teachers:

  teacher      ParT accuracy and AUC on the test split
  fidelity     R², Spearman and decision agreement of Ridge / EBM / GBDT with
               their own teacher; the cost of additivity (GBDT − EBM) and the
               basis gap (1 − R² GBDT)
  importances  the EBM's mean |φ| per observable, and whether its ranking holds
               (Kendall τ between every pair of teachers)
  regimes      the share of test jets narrated in full, with a caveat, or
               abstained on, at the narrative config's thresholds

and, between every pair of teachers, on the same test jets:

  decisions    how often the two ParTs decide alike
  φ per jet    for each observable, the correlation of its contribution across
               jets; for each jet, the rank correlation of the two drivers
               orderings (|φ|), whether the leading driver is the same, how many
               of the three leading drivers are shared, and whether those agree
               in sign. This is the explanation-level question: retrain the
               model, and is this jet still explained by the same observables?

    python -m surrogate.compare_teachers surrogate/outputs/top_seed4{2,3,4}
    python -m surrogate.compare_teachers DIR... --n-jets 50000 --out comparison.json

Spread across EBM model seeds at a fixed teacher is a different quantity, and
comes from surrogate/seed_variance.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

MODELS = ("ridge", "ebm", "gbdt")


def load_run(d: Path, idx: np.ndarray, thresholds: tuple) -> dict:
    feats = pd.read_parquet(d / "features/features_test.parquet")
    metrics = json.loads((d / "models/metrics.json").read_text())
    clf = json.loads((d / "models/classifier_metrics.json").read_text())
    clf = clf.get("test", clf)
    bundle = joblib.load(d / "models/ebm.joblib")
    obs = list(bundle["obs"])

    y_top = (feats["label"].to_numpy() == 0).astype(int)      # weaver index 0 = top
    t = feats["t"].to_numpy(np.float64)

    # Contributions on the sampled jets, NaNs filled as at fit time.
    x = feats.iloc[idx][obs].copy()
    for c in obs:
        x[c] = x[c].fillna(bundle["raw_medians"][c])
    phi = np.asarray(bundle["ebm"].eval_terms(x), dtype=np.float64)

    pred = pd.read_parquet(d / "models/test_predictions.parquet", columns=["t_clip", "t_hat_ebm"])
    r = np.abs(pred["t_clip"].to_numpy() - pred["t_hat_ebm"].to_numpy())
    full_t, cav_t = thresholds

    run = {
        "dir": str(d),
        "obs": obs,
        "teacher": {"acc": float(((t > 0) == y_top).mean()), "auc": float(roc_auc_score(y_top, t))},
        "fidelity": {m: {"r2": metrics[m]["test"]["r2_all"],
                         "spearman": metrics[m]["test"]["spearman"],
                         "agreement": clf[m]["agreement_with_part"],
                         "auc_truth": clf[m]["auc_vs_truth"]} for m in MODELS},
        "gaps": {"cost_of_additivity": metrics["gaps_test"]["cost_of_additivity_ebm_to_gbdt"],
                 "basis_gap": metrics["gaps_test"]["basis_insufficiency_1_minus_gbdt"]},
        "importance": dict(metrics.get("ebm_importances") or []),
        "regimes": {"full": float((r < full_t).mean()),
                    "caveat": float(((r >= full_t) & (r < cav_t)).mean()),
                    "abstain": float((r >= cav_t).mean())},
        "_t": t[idx],
        "_phi": phi,
    }
    run["label"] = d.name
    return run


def pair_stability(a: dict, b: dict) -> dict:
    """Agreement of two teachers' surrogates on the same jets."""
    pa, pb = a["_phi"], b["_phi"]
    obs = a["obs"]
    per_obs = {o: float(np.corrcoef(pa[:, k], pb[:, k])[0, 1]) for k, o in enumerate(obs)}

    abs_a, abs_b = np.abs(pa), np.abs(pb)
    order_a = np.argsort(-abs_a, axis=1)
    order_b = np.argsort(-abs_b, axis=1)
    rank_corr = np.array([spearmanr(abs_a[i], abs_b[i])[0] for i in range(len(pa))])
    top1 = (order_a[:, 0] == order_b[:, 0]).mean()
    top3_a, top3_b = order_a[:, :3], order_b[:, :3]
    shared3 = np.array([len(set(top3_a[i]) & set(top3_b[i])) for i in range(len(pa))])
    # Sign agreement on a's three leading drivers.
    rows = np.arange(len(pa))[:, None]
    sign3 = (np.sign(pa[rows, top3_a]) == np.sign(pb[rows, top3_a])).mean()

    return {
        "decision_agreement": float(((a["_t"] > 0) == (b["_t"] > 0)).mean()),
        "phi_corr_per_observable": per_obs,
        "phi_corr_mean": float(np.mean(list(per_obs.values()))),
        "jet_rank_corr_median": float(np.nanmedian(rank_corr)),
        "jet_top1_same": float(top1),
        "jet_top3_shared_mean": float(shared3.mean()),
        "jet_top3_all_shared": float((shared3 == 3).mean()),
        "jet_top3_sign_agree": float(sign3),
    }


def mean_sd(values: List[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def fmt(ms: Dict[str, float], nd: int = 4) -> str:
    return f"{ms['mean']:.{nd}f} ± {ms['sd']:.{nd}f}"


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="surrogate/outputs/top_seed<N> directories")
    p.add_argument("--n-jets", type=int, default=20000,
                   help="Test jets the per-jet φ comparison is computed on (same jets for every run)")
    p.add_argument("--config", type=Path, default=repo / "narrative/configs/narrative_top.yaml",
                   help="For the regime thresholds")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    g = yaml.safe_load(args.config.read_text())["abstention"]
    thresholds = (float(g["full_threshold"]), float(g["caveat_threshold"]))
    n_test = len(pd.read_parquet(args.runs[0] / "features/features_test.parquet", columns=["jet_id"]))
    idx = np.sort(np.random.default_rng(0).choice(n_test, size=min(args.n_jets, n_test), replace=False))

    runs = [load_run(d, idx, thresholds) for d in args.runs]
    obs = runs[0]["obs"]
    if any(r["obs"] != obs for r in runs):
        raise SystemExit("the runs do not share the same observable basis")
    labels = [r["label"] for r in runs]
    print(f"{len(runs)} teachers: {', '.join(labels)}   (φ on {len(idx):,} test jets)\n")

    summary = {
        "teachers": labels,
        "n_jets_phi": int(len(idx)),
        "thresholds": thresholds,
        "teacher": {k: mean_sd([r["teacher"][k] for r in runs]) for k in ("acc", "auc")},
        "fidelity": {m: {k: mean_sd([r["fidelity"][m][k] for r in runs])
                         for k in ("r2", "spearman", "agreement", "auc_truth")} for m in MODELS},
        "gaps": {k: mean_sd([r["gaps"][k] for r in runs]) for k in ("cost_of_additivity", "basis_gap")},
        "importance": {o: mean_sd([r["importance"].get(o, np.nan) for r in runs]) for o in obs},
        "regimes": {k: mean_sd([r["regimes"][k] for r in runs]) for k in ("full", "caveat", "abstain")},
        "per_teacher": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs],
    }

    taus = []
    pairs = {}
    for i, j in itertools.combinations(range(len(runs)), 2):
        a, b = runs[i], runs[j]
        ia = [a["importance"].get(o, 0) for o in obs]
        ib = [b["importance"].get(o, 0) for o in obs]
        tau = float(kendalltau(ia, ib)[0])
        taus.append(tau)
        pairs[f"{labels[i]} vs {labels[j]}"] = {"importance_kendall_tau": tau, **pair_stability(a, b)}
    summary["pairs"] = pairs
    if pairs:
        keys = [k for k, v in next(iter(pairs.values())).items() if isinstance(v, float)]
        summary["pairs_mean"] = {k: mean_sd([pv[k] for pv in pairs.values()]) for k in keys}

    # --- print -------------------------------------------------------------
    t = summary["teacher"]
    print(f"ParT          acc {fmt(t['acc'])}   AUC {fmt(t['auc'])}")
    print(f"\n{'surrogate':10s}{'R²':>20s}{'Spearman':>20s}{'agreement':>20s}{'AUC (truth)':>20s}")
    for m in MODELS:
        f = summary["fidelity"][m]
        print(f"{m:10s}" + "".join(f"{fmt(f[k]):>20s}" for k in ("r2", "spearman", "agreement", "auc_truth")))
    gp = summary["gaps"]
    print(f"\ncost of additivity (GBDT − EBM)  {fmt(gp['cost_of_additivity'])}"
          f"    basis gap (1 − R² GBDT)  {fmt(gp['basis_gap'])}")
    rg = summary["regimes"]
    print(f"regimes (|r| < {thresholds[0]:g} / < {thresholds[1]:g} / ≥)   full {fmt(rg['full'], 3)}"
          f"   caveat {fmt(rg['caveat'], 3)}   abstain {fmt(rg['abstain'], 3)}")
    print("\nEBM importance (mean |φ|), by mean:")
    for o in sorted(obs, key=lambda o: -summary["importance"][o]["mean"]):
        print(f"  {o:14s} {fmt(summary['importance'][o], 3)}")
    if pairs:
        pm = summary["pairs_mean"]
        print(f"\nbetween teachers (mean ± sd over {len(pairs)} pairs):")
        for k, label in (("importance_kendall_tau", "importance ranking, Kendall τ"),
                         ("decision_agreement", "same ParT decision"),
                         ("phi_corr_mean", "φ correlation across jets, mean over observables"),
                         ("jet_rank_corr_median", "per-jet driver ranking, median Spearman"),
                         ("jet_top1_same", "same leading driver"),
                         ("jet_top3_shared_mean", "leading three drivers shared (of 3)"),
                         ("jet_top3_all_shared", "all three leading drivers shared"),
                         ("jet_top3_sign_agree", "leading three drivers, same sign")):
            print(f"  {label:52s} {fmt(pm[k], 3)}")
        worst = min(obs, key=lambda o: np.mean([pv["phi_corr_per_observable"][o] for pv in pairs.values()]))
        print(f"  least stable observable: {worst} "
              f"(φ corr {np.mean([pv['phi_corr_per_observable'][worst] for pv in pairs.values()]):.3f})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
