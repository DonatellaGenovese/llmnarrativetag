"""Training variance of the surrogate ladder across random seeds.

The official run fits each model once, at `random_state = 42`. Two of the three
are stochastic — the EBM bags four outer models, the GBDT subsamples rows and
columns — so a single fit says nothing about how much of a reported gap is the
seed. This refits them across seeds on identical data and reports mean ± sd.

The quantity that actually needs this is the cost of additivity, R²(GBDT) −
R²(EBM) = 0.018 in the official run: a claim that ParT's decision is nearly
additive is only as strong as that gap is stable.

Ridge is fitted once. It is a closed-form convex solution with no seed.

This is also, despite the name, the **only** place average decision ordering is
computed — for the three surrogates against the teacher, and for the teacher
against the physical truth ordering. Nothing else in the repo produces ADO, so
this file is not the accessory analysis it looks like.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

from .train_surrogates_top import clip_target, prepare_raw_X, prepare_ridge_X

ADO_PAIRS = 500_000


def _metrics(y_clip, y_raw, pred, truth_top, part_top, ado_idx) -> Dict[str, float]:
    """Test-set metrics for one fitted model, matching the official definitions."""
    s_hat = 1.0 / (1.0 + np.exp(-pred))
    pred_top = (pred > 0).astype(int)
    i, j = ado_idx
    out = {
        "r2": float(r2_score(y_clip, pred)),
        "spearman": float(pd.Series(y_clip).corr(pd.Series(pred), method="spearman")),
        "agreement": float(np.mean(pred_top == part_top)),
        "acc": float(accuracy_score(truth_top, pred_top)),
        "auc": float(roc_auc_score(truth_top, s_hat)),
        "ado": float(np.mean(np.sign(pred[i] - pred[j]) == np.sign(y_raw[i] - y_raw[j]))),
        # Probability space, against the unclipped teacher score, matching
        # eval_classifier_metrics.
        "mae": float(np.mean(np.abs(s_hat - 1.0 / (1.0 + np.exp(-y_raw))))),
    }
    return out


def run(config_path: Path, out_dir: Path, seeds: List[int], max_jets: int | None) -> dict:
    from interpret.glassbox import ExplainableBoostingRegressor
    from xgboost import XGBRegressor

    cfg = yaml.safe_load(config_path.read_text())
    obs = list(cfg["observables"])
    fit = cfg["fit"]
    repo = config_path.resolve().parents[2]
    base = repo / "surrogate/outputs/top/features"

    def load(name):
        df = pd.read_parquet(base / f"features_{name}.parquet")
        return df.iloc[:max_jets].copy() if max_jets else df

    df_tr, df_va, df_te = load("train"), load("val"), load("test")
    y_tr, y_va, y_te = (d["t"].to_numpy(dtype=np.float64) for d in (df_tr, df_va, df_te))
    _, clip_info = clip_target(y_tr, *fit["target_clip_percentiles"])
    lo, hi = clip_info["t_clip_lo"], clip_info["t_clip_hi"]
    y_tr_c, y_va_c, y_te_c = (np.clip(y, lo, hi) for y in (y_tr, y_va, y_te))

    X_tr, medians = prepare_raw_X(df_tr, obs)
    X_va, _ = prepare_raw_X(df_va, obs, medians)
    X_te, _ = prepare_raw_X(df_te, obs, medians)

    truth_top = (df_te["label"].to_numpy() == 0).astype(int)
    part_top = (y_te > 0).astype(int)

    # ADO pairs: fixed across seeds so the metric varies only with the model.
    rng = np.random.default_rng(0)
    sig = np.flatnonzero(truth_top == 1)
    bkg = np.flatnonzero(truth_top == 0)
    ado_idx = (rng.choice(sig, ADO_PAIRS), rng.choice(bkg, ADO_PAIRS))

    # The teacher's own ordering, against the physical truth rather than against
    # itself. Every pair here is (signal, background), so the truth ordering puts
    # the first above the second and this is just the fraction ParT ranks that
    # way. It is the reference the surrogate ADOs are read against — "the
    # surrogate tracks the tagger about as closely as the tagger tracks the
    # physics" is a comparison between this number and the EBM's — so it is
    # computed on the same pairs and not quoted from ParT's AUC, which the
    # Mann-Whitney identity makes it equal to in expectation but not by
    # construction. Ties count against, as they do in `_metrics`.
    part_ado_vs_truth = float(np.mean(y_te[ado_idx[0]] > y_te[ado_idx[1]]))
    print(f"ParT ADO vs truth ordering: {part_ado_vs_truth:.4f}")

    results: Dict[str, List[Dict[str, float]]] = {"ebm": [], "gbdt": []}

    # Ridge once: deterministic, no seed.
    Xr_tr, ridge_stats = prepare_ridge_X(
        df_tr, obs, fit["ridge_log_features"], tuple(fit["ratio_clip_percentiles"]), None
    )
    Xr_te, _ = prepare_ridge_X(
        df_te, obs, fit["ridge_log_features"], tuple(fit["ratio_clip_percentiles"]), ridge_stats
    )
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    ridge = Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
    ridge.fit(Xr_tr, y_tr_c)
    ridge_m = _metrics(y_te_c, y_te, ridge.predict(Xr_te), truth_top, part_top, ado_idx)
    print(f"ridge (deterministic)  r2={ridge_m['r2']:.4f}")

    for seed in seeds:
        ebm = ExplainableBoostingRegressor(
            feature_names=obs,
            interactions=int(fit["ebm_interactions"]),
            max_bins=int(fit["ebm_max_bins"]),
            learning_rate=float(fit["ebm_learning_rate"]),
            max_rounds=int(fit["ebm_max_rounds"]),
            outer_bags=4,
            inner_bags=0,
            random_state=seed,
            n_jobs=-1,
        )
        ebm.fit(X_tr, y_tr_c)
        m = _metrics(y_te_c, y_te, ebm.predict(X_te), truth_top, part_top, ado_idx)
        results["ebm"].append(m)
        print(f"seed {seed:>4}  ebm   r2={m['r2']:.4f}  agr={m['agreement']:.4f}  acc={m['acc']:.4f}")

        gbt = XGBRegressor(
            n_estimators=2000, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror",
            n_jobs=-1, random_state=seed, early_stopping_rounds=50,
        )
        gbt.fit(X_tr, y_tr_c, eval_set=[(X_va, y_va_c)], verbose=False)
        m = _metrics(y_te_c, y_te, gbt.predict(X_te), truth_top, part_top, ado_idx)
        results["gbdt"].append(m)
        print(f"seed {seed:>4}  gbdt  r2={m['r2']:.4f}  agr={m['agreement']:.4f}  acc={m['acc']:.4f}")

    gaps = [
        {
            "nonlinearity": e["r2"] - ridge_m["r2"],
            "cost_of_additivity": g["r2"] - e["r2"],
            "basis_insufficiency": 1.0 - g["r2"],
        }
        for e, g in zip(results["ebm"], results["gbdt"])
    ]

    def agg(rows, key):
        v = np.array([r[key] for r in rows], dtype=float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()),
                "max": float(v.max())}

    summary = {
        "seeds": seeds,
        "n_test": int(len(df_te)),
        "ado_pairs": ADO_PAIRS,
        "part": {"ado_vs_truth": part_ado_vs_truth},
        "ridge": ridge_m,
        "ebm": {k: agg(results["ebm"], k) for k in results["ebm"][0]},
        "gbdt": {k: agg(results["gbdt"], k) for k in results["gbdt"][0]},
        "gaps": {k: agg(gaps, k) for k in gaps[0]},
        "per_seed": {"ebm": results["ebm"], "gbdt": results["gbdt"], "gaps": gaps},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seed_variance.json").write_text(json.dumps(summary, indent=2))

    # sd is printed to six decimals on purpose: the EBM's spread across seeds
    # sits around 1e-5, and four decimals would show it as exactly zero.
    print("\n%-24s %12s %12s %12s" % ("", "mean", "sd", "range"))
    for model in ("ebm", "gbdt"):
        for k in ("r2", "spearman", "ado", "agreement", "mae", "acc", "auc"):
            a = summary[model][k]
            print("%-24s %12.6f %12.6f %12.6f"
                  % (f"{model} {k}", a["mean"], a["sd"], a["max"] - a["min"]))
    for k, a in summary["gaps"].items():
        print("%-24s %12.6f %12.6f %12.6f" % (f"gap {k}", a["mean"], a["sd"],
                                              a["max"] - a["min"]))
    print("%-24s %12.6f" % ("ParT ado_vs_truth", part_ado_vs_truth))
    print(f"\nWrote → {out_dir / 'seed_variance.json'}")
    return summary


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=repo / "surrogate/configs/top_basis.yaml")
    p.add_argument("--out-dir", type=Path, default=repo / "surrogate/outputs/top/seed_variance")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--max-jets", type=int, default=None)
    args = p.parse_args()
    run(args.config, args.out_dir, args.seeds, args.max_jets)


if __name__ == "__main__":
    main()
