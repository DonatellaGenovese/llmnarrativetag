"""Fit Logistic / EBM / GBDT classifiers on truth labels (not ParT imitation).

Same observable basis and preprocessing as the ParT-surrogate regressors.
Official protocol: fit on train, early-stop / select on val, report on test.

y = 1 iff label == positive_class_index (Top: top=0; QG: quark=0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except ImportError as e:  # pragma: no cover
    raise SystemExit("interpret is required in the part-surrogate env") from e

from .train_surrogates_top import prepare_raw_X, prepare_ridge_X


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _labels(df: pd.DataFrame, positive_class_index: int) -> np.ndarray:
    return (df["label"].to_numpy() == positive_class_index).astype(np.int64)


def _eval_clf(y: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, proba)),
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
    pos = int(cfg.get("positive_class_index", 0))

    def _load(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if max_jets is not None:
            df = df.iloc[:max_jets].copy()
        return df

    df_tr = _load(train_parquet)
    df_va = _load(val_parquet)
    df_te = _load(test_parquet)

    y_tr = _labels(df_tr, pos)
    y_va = _labels(df_va, pos)
    y_te = _labels(df_te, pos)

    Xr_tr, ridge_stats = prepare_ridge_X(
        df_tr, obs, fit["ridge_log_features"], tuple(fit["ratio_clip_percentiles"]), None
    )
    Xr_va, _ = prepare_ridge_X(df_va, obs, fit["ridge_log_features"], None, ridge_stats)
    Xr_te, _ = prepare_ridge_X(df_te, obs, fit["ridge_log_features"], None, ridge_stats)

    X_tr, raw_medians = prepare_raw_X(df_tr, obs, None)
    X_va, _ = prepare_raw_X(df_va, obs, raw_medians)
    X_te, _ = prepare_raw_X(df_te, obs, raw_medians)

    # --- Logistic (linear, analogous to Ridge surrogate) ---
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegressionCV(
                    Cs=np.logspace(-3, 3, 25),
                    cv=3,
                    scoring="roc_auc",
                    max_iter=2000,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
    logistic.fit(Xr_tr, y_tr)

    # --- EBM classifier (raw) ---
    ebm = ExplainableBoostingClassifier(
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
    ebm.fit(X_tr, y_tr)

    # --- GBDT classifier (raw), early-stop on val ---
    gbt = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=seed,
        early_stopping_rounds=50,
    )
    gbt.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    def _pack(model, Xr, X, kind: str):
        if kind == "logistic":
            proba = model.predict_proba(Xr)[:, 1]
            pred = (proba >= 0.5).astype(np.int64)
        elif kind == "ebm":
            proba = model.predict_proba(X)[:, 1]
            pred = model.predict(X).astype(np.int64)
        else:
            proba = model.predict_proba(X)[:, 1]
            pred = model.predict(X).astype(np.int64)
        return proba, pred

    y_by = {"train": y_tr, "val": y_va, "test": y_te}
    Xr_by = {"train": Xr_tr, "val": Xr_va, "test": Xr_te}
    X_by = {"train": X_tr, "val": X_va, "test": X_te}

    metrics: Dict = {
        "target": "label",
        "positive_class_index": pos,
        "n": {"train": int(len(df_tr)), "val": int(len(df_va)), "test": int(len(df_te))},
        "paths": {
            "train": str(train_parquet),
            "val": str(val_parquet),
            "test": str(test_parquet),
        },
    }

    preds_test: Dict[str, np.ndarray] = {}
    for name, model, kind in [
        ("logistic", logistic, "logistic"),
        ("ebm", ebm, "ebm"),
        ("gbdt", gbt, "gbdt"),
    ]:
        metrics[name] = {}
        for split in ("train", "val", "test"):
            proba, pred = _pack(model, Xr_by[split], X_by[split], kind)
            metrics[name][split] = _eval_clf(y_by[split], proba, pred)
            if split == "test":
                preds_test[f"proba_{name}"] = proba
                preds_test[f"pred_{name}"] = pred

    # ParT baseline on same jets (for side-by-side)
    for split, df, y in [
        ("train", df_tr, y_tr),
        ("val", df_va, y_va),
        ("test", df_te, y_te),
    ]:
        s = df["s"].to_numpy(dtype=np.float64) if "s" in df.columns else 1.0 / (1.0 + np.exp(-df["t"].to_numpy(dtype=np.float64)))
        if "pred_label" in df.columns:
            pred_part = (df["pred_label"].to_numpy() == pos).astype(np.int64)
        else:
            pred_part = (df["t"].to_numpy() > 0).astype(np.int64)
        metrics.setdefault("part", {})[split] = _eval_clf(y, s, pred_part)

    try:
        ebm_imp = list(zip(obs, [float(x) for x in ebm.term_importances()[: len(obs)]]))
        metrics["ebm_importances"] = sorted(ebm_imp, key=lambda z: -z[1])
    except Exception as e:
        metrics["ebm_importances_error"] = str(e)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_label.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "ridge_stats.json").write_text(json.dumps(ridge_stats, indent=2))

    joblib.dump(
        {"logistic": logistic, "obs": obs, "cfg": fit, "ridge_stats": ridge_stats, "pos": pos},
        out_dir / "logistic.joblib",
    )
    joblib.dump({"ebm": ebm, "obs": obs, "raw_medians": raw_medians, "pos": pos}, out_dir / "ebm_clf.joblib")
    joblib.dump({"gbdt": gbt, "obs": obs, "raw_medians": raw_medians, "pos": pos}, out_dir / "gbdt_clf.joblib")

    te = df_te.copy()
    te["y_pos"] = y_te
    for k, v in preds_test.items():
        te[k] = v
    te.to_parquet(out_dir / "test_predictions_label.parquet", index=False)

    print(
        "Acc logistic/ebm/gbdt/part  test:",
        metrics["logistic"]["test"]["acc"],
        metrics["ebm"]["test"]["acc"],
        metrics["gbdt"]["test"]["acc"],
        metrics["part"]["test"]["acc"],
    )
    print(
        "AUC logistic/ebm/gbdt/part  test:",
        metrics["logistic"]["test"]["auc"],
        metrics["ebm"]["test"]["auc"],
        metrics["gbdt"]["test"]["auc"],
        metrics["part"]["test"]["auc"],
    )
    print(f"Wrote → {out_dir}")


def main():
    repo = Path(__file__).resolve().parents[1]
    feat_dir = repo / "surrogate/outputs/top/features"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=feat_dir / "features_train.parquet")
    p.add_argument("--val", type=Path, default=feat_dir / "features_val.parquet")
    p.add_argument("--test", type=Path, default=feat_dir / "features_test.parquet")
    p.add_argument("--config", type=Path, default=repo / "surrogate/configs/top_basis.yaml")
    p.add_argument("--out-dir", type=Path, default=repo / "surrogate/outputs/top/models_label")
    p.add_argument("--max-jets", type=int, default=None, help="Cap each split (debug)")
    args = p.parse_args()
    train_all(args.train, args.val, args.test, args.config, args.out_dir, args.max_jets)


if __name__ == "__main__":
    main()
