"""Fit Ridge / EBM / GBDT surrogates on ParT logit difference t.

Official protocol: fit on train, select on val, report on test.
Clip thresholds and Ridge ratio percentiles are fit on train only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    from interpret.glassbox import ExplainableBoostingRegressor
except ImportError as e:  # pragma: no cover
    raise SystemExit("interpret is required in the part-surrogate env") from e


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def clip_target(y: np.ndarray, lo_p: float, hi_p: float) -> Tuple[np.ndarray, Dict]:
    lo, hi = np.nanpercentile(y, [lo_p, hi_p])
    y_c = np.clip(y, lo, hi)
    return y_c, {"t_clip_lo": float(lo), "t_clip_hi": float(hi)}


def prepare_ridge_X(
    df: pd.DataFrame,
    cols: Sequence[str],
    log_cols: Sequence[str],
    ratio_clip: Tuple[float, float] | None,
    train_stats: Dict | None = None,
) -> Tuple[np.ndarray, Dict]:
    """Build Ridge features. If train_stats is None, compute clip/medians from this df."""
    X = df[list(cols)].to_numpy(dtype=np.float64).copy()
    name_to_i = {c: i for i, c in enumerate(cols)}
    stats: Dict = {"ratio_clips": {}, "medians": {}, "log_cols": list(log_cols)}

    if train_stats is None:
        for c in cols:
            if c in log_cols:
                continue
            i = name_to_i[c]
            if ratio_clip is not None:
                lo, hi = np.nanpercentile(X[:, i], list(ratio_clip))
                stats["ratio_clips"][c] = [float(lo), float(hi)]
                X[:, i] = np.clip(X[:, i], lo, hi)
            else:
                stats["ratio_clips"][c] = [None, None]
        for c in log_cols:
            i = name_to_i[c]
            X[:, i] = np.log(np.clip(X[:, i], 1e-8, None))
        for i, c in enumerate(cols):
            col = X[:, i]
            med = float(np.nanmedian(col))
            stats["medians"][c] = med
            col[~np.isfinite(col)] = med
            X[:, i] = col
        return X, stats

    # Apply train-fitted transforms
    for c in cols:
        if c in log_cols:
            continue
        i = name_to_i[c]
        lo, hi = train_stats["ratio_clips"].get(c, [None, None])
        if lo is not None and hi is not None:
            X[:, i] = np.clip(X[:, i], lo, hi)
    for c in log_cols:
        i = name_to_i[c]
        X[:, i] = np.log(np.clip(X[:, i], 1e-8, None))
    for i, c in enumerate(cols):
        col = X[:, i]
        med = train_stats["medians"][c]
        col[~np.isfinite(col)] = med
        X[:, i] = col
    return X, train_stats


def prepare_raw_X(
    df: pd.DataFrame,
    cols: Sequence[str],
    medians: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    X = df[list(cols)].to_numpy(dtype=np.float64).copy()
    if medians is None:
        medians = {}
        for i, c in enumerate(cols):
            col = X[:, i]
            med = float(np.nanmedian(col))
            medians[c] = med
            col[~np.isfinite(col)] = med
            X[:, i] = col
        return X, medians
    for i, c in enumerate(cols):
        col = X[:, i]
        col[~np.isfinite(col)] = medians[c]
        X[:, i] = col
    return X, medians


def stratified_r2(y_true: np.ndarray, y_pred: np.ndarray, by: np.ndarray, n_bins: int = 5) -> Dict[str, float]:
    out = {"r2_all": float(r2_score(y_true, y_pred))}
    qs = np.linspace(0, 100, n_bins + 1)
    edges = np.nanpercentile(by, qs)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b < n_bins - 1:
            m = (by >= lo) & (by < hi)
        else:
            m = (by >= lo) & (by <= hi)
        if m.sum() < 10:
            continue
        out[f"r2_bin{b}"] = float(r2_score(y_true[m], y_pred[m]))
    return out


def score_space_resid(t_true: np.ndarray, t_pred: np.ndarray) -> np.ndarray:
    s_true = 1.0 / (1.0 + np.exp(-t_true))
    s_pred = 1.0 / (1.0 + np.exp(-t_pred))
    return np.abs(s_pred - s_true)


def _eval_split(y: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
    return {
        **stratified_r2(y, yp, by=np.abs(y)),
        "spearman": float(pd.Series(y).corr(pd.Series(yp), method="spearman")),
        "mae_score": float(np.mean(score_space_resid(y, yp))),
    }


def train_all(
    train_parquet: Path,
    val_parquet: Path,
    test_parquet: Path,
    config_path: Path,
    out_dir: Path,
    max_jets: int | None = None,
) -> None:
    cfg = _load_yaml(config_path)
    obs: List[str] = list(cfg["observables"])
    fit = cfg["fit"]
    seed = int(cfg.get("seed", 42))

    def _load(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if max_jets is not None:
            df = df.iloc[:max_jets].copy()
        return df

    df_tr = _load(train_parquet)
    df_va = _load(val_parquet)
    df_te = _load(test_parquet)

    y_tr = df_tr["t"].to_numpy(dtype=np.float64)
    y_va = df_va["t"].to_numpy(dtype=np.float64)
    y_te = df_te["t"].to_numpy(dtype=np.float64)

    _, clip_info = clip_target(y_tr, *fit["target_clip_percentiles"])
    lo, hi = clip_info["t_clip_lo"], clip_info["t_clip_hi"]
    y_tr_c = np.clip(y_tr, lo, hi)
    y_va_c = np.clip(y_va, lo, hi)
    y_te_c = np.clip(y_te, lo, hi)

    Xr_tr, ridge_stats = prepare_ridge_X(
        df_tr, obs, fit["ridge_log_features"], tuple(fit["ratio_clip_percentiles"]), None
    )
    Xr_va, _ = prepare_ridge_X(df_va, obs, fit["ridge_log_features"], None, ridge_stats)
    Xr_te, _ = prepare_ridge_X(df_te, obs, fit["ridge_log_features"], None, ridge_stats)

    X_tr, raw_medians = prepare_raw_X(df_tr, obs, None)
    X_va, _ = prepare_raw_X(df_va, obs, raw_medians)
    X_te, _ = prepare_raw_X(df_te, obs, raw_medians)

    # --- Ridge ---
    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )
    ridge.fit(Xr_tr, y_tr_c)
    pred_ridge = {
        "train": ridge.predict(Xr_tr),
        "val": ridge.predict(Xr_va),
        "test": ridge.predict(Xr_te),
    }

    # --- EBM (raw) ---
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
    pred_ebm = {
        "train": ebm.predict(X_tr),
        "val": ebm.predict(X_va),
        "test": ebm.predict(X_te),
    }

    # --- GBDT ceiling (raw), early-stop on val ---
    gbt = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=seed,
        early_stopping_rounds=50,
    )
    gbt.fit(X_tr, y_tr_c, eval_set=[(X_va, y_va_c)], verbose=False)
    pred_gbt = {
        "train": gbt.predict(X_tr),
        "val": gbt.predict(X_va),
        "test": gbt.predict(X_te),
    }

    y_by = {"train": y_tr_c, "val": y_va_c, "test": y_te_c}
    metrics = {
        "clip": clip_info,
        "n": {"train": int(len(df_tr)), "val": int(len(df_va)), "test": int(len(df_te))},
        "paths": {
            "train": str(train_parquet),
            "val": str(val_parquet),
            "test": str(test_parquet),
        },
    }
    for name, preds in [("ridge", pred_ridge), ("ebm", pred_ebm), ("gbdt", pred_gbt)]:
        metrics[name] = {split: _eval_split(y_by[split], preds[split]) for split in ("train", "val", "test")}

    r2_ebm = metrics["ebm"]["test"]["r2_all"]
    r2_ridge = metrics["ridge"]["test"]["r2_all"]
    r2_gbt = metrics["gbdt"]["test"]["r2_all"]
    metrics["gaps_test"] = {
        "nonlinearity_used_ridge_to_ebm": r2_ebm - r2_ridge,
        "cost_of_additivity_ebm_to_gbdt": r2_gbt - r2_ebm,
        "basis_insufficiency_1_minus_gbdt": 1.0 - r2_gbt,
    }
    metrics["gaps_val"] = {
        "nonlinearity_used_ridge_to_ebm": metrics["ebm"]["val"]["r2_all"] - metrics["ridge"]["val"]["r2_all"],
        "cost_of_additivity_ebm_to_gbdt": metrics["gbdt"]["val"]["r2_all"] - metrics["ebm"]["val"]["r2_all"],
        "basis_insufficiency_1_minus_gbdt": 1.0 - metrics["gbdt"]["val"]["r2_all"],
    }

    try:
        ebm_imp = list(zip(obs, [float(x) for x in ebm.term_importances()[: len(obs)]]))
        metrics["ebm_importances"] = sorted(ebm_imp, key=lambda z: -z[1])
    except Exception as e:
        metrics["ebm_importances_error"] = str(e)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "ridge_stats.json").write_text(json.dumps(ridge_stats, indent=2))

    import joblib

    joblib.dump(
        {"ridge": ridge, "obs": obs, "cfg": fit, "ridge_stats": ridge_stats, "clip": clip_info},
        out_dir / "ridge.joblib",
    )
    joblib.dump({"ebm": ebm, "obs": obs, "raw_medians": raw_medians, "clip": clip_info}, out_dir / "ebm.joblib")
    joblib.dump({"gbdt": gbt, "obs": obs, "raw_medians": raw_medians, "clip": clip_info}, out_dir / "gbdt.joblib")

    te = df_te.copy()
    te["t_clip"] = y_te_c
    te["t_hat_ridge"] = pred_ridge["test"]
    te["t_hat_ebm"] = pred_ebm["test"]
    te["t_hat_gbdt"] = pred_gbt["test"]
    te["abs_ds_ebm"] = score_space_resid(y_te_c, pred_ebm["test"])
    te.to_parquet(out_dir / "test_predictions.parquet", index=False)

    print(json.dumps({"val": metrics["gaps_val"], "test": metrics["gaps_test"]}, indent=2))
    print(
        "R² ridge/ebm/gbdt  val:",
        metrics["ridge"]["val"]["r2_all"],
        metrics["ebm"]["val"]["r2_all"],
        metrics["gbdt"]["val"]["r2_all"],
    )
    print("R² ridge/ebm/gbdt test:", r2_ridge, r2_ebm, r2_gbt)
    print(f"Wrote → {out_dir}")


def main():
    repo = Path(__file__).resolve().parents[1]
    feat_dir = repo / "surrogate/outputs/top/features"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=feat_dir / "features_train.parquet")
    p.add_argument("--val", type=Path, default=feat_dir / "features_val.parquet")
    p.add_argument("--test", type=Path, default=feat_dir / "features_test.parquet")
    p.add_argument("--config", type=Path, default=repo / "surrogate/configs/top_basis.yaml")
    p.add_argument("--out-dir", type=Path, default=repo / "surrogate/outputs/top/models")
    p.add_argument("--max-jets", type=int, default=None, help="Cap each split (debug)")
    args = p.parse_args()
    train_all(args.train, args.val, args.test, args.config, args.out_dir, args.max_jets)


if __name__ == "__main__":
    main()
