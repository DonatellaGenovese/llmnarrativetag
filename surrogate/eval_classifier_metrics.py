"""Post-hoc classifier metrics for surrogates (no retrain).

Computes, per model and split:
  - MAE / RMSE in score space vs ParT
  - decision agreement with ParT (sign of t)
  - Accuracy / AUC vs truth (top = weaver label 0)
  - ParT baseline Acc / AUC on the same jets

Can use:
  A) saved test_predictions.parquet (test only, no model reload), or
  B) feature tables + joblib models (train/val/test).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score

from .train_surrogates_top import prepare_raw_X, prepare_ridge_X


def _sigmoid(t: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-t))


def classifier_metrics(
    t_true: np.ndarray,
    t_hat: np.ndarray,
    y_top: np.ndarray,
    pred_part_top: np.ndarray,
) -> Dict[str, float]:
    """y_top / pred_part_top are boolean-like: 1 = top."""
    s_true = _sigmoid(t_true)
    s_hat = _sigmoid(t_hat)
    pred_surr_top = (t_hat > 0).astype(np.int64)
    pred_part = pred_part_top.astype(np.int64)
    y = y_top.astype(np.int64)

    out = {
        "mae_score_vs_part": float(np.mean(np.abs(s_hat - s_true))),
        "rmse_score_vs_part": float(np.sqrt(np.mean((s_hat - s_true) ** 2))),
        "agreement_with_part": float(np.mean(pred_surr_top == pred_part)),
        "acc_vs_truth": float(accuracy_score(y, pred_surr_top)),
        "auc_vs_truth": float(roc_auc_score(y, s_hat)),
        "part_acc_vs_truth": float(accuracy_score(y, pred_part)),
        "part_auc_vs_truth": float(roc_auc_score(y, s_true)),
        "acc_gap_vs_part": float(accuracy_score(y, pred_part) - accuracy_score(y, pred_surr_top)),
        "auc_gap_vs_part": float(roc_auc_score(y, s_true) - roc_auc_score(y, s_hat)),
    }
    return out


def _truth_and_part_pred(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # weaver: 0=top, 1=QCD; t>0 ⇔ ParT predicts top
    y_top = (df["label"].to_numpy() == 0).astype(np.int64)
    if "pred_label" in df.columns:
        pred_part_top = (df["pred_label"].to_numpy() == 0).astype(np.int64)
    else:
        pred_part_top = (df["t"].to_numpy() > 0).astype(np.int64)
    return y_top, pred_part_top


def eval_from_predictions(pred_parquet: Path) -> Dict:
    df = pd.read_parquet(pred_parquet)
    y_top, pred_part = _truth_and_part_pred(df)
    t_true = df["t"].to_numpy(dtype=np.float64)
    metrics = {"n": int(len(df)), "source": str(pred_parquet)}
    for name, col in [("ridge", "t_hat_ridge"), ("ebm", "t_hat_ebm"), ("gbdt", "t_hat_gbdt")]:
        if col not in df.columns:
            continue
        metrics[name] = classifier_metrics(t_true, df[col].to_numpy(dtype=np.float64), y_top, pred_part)
    return metrics


def eval_from_models(
    train_parquet: Path,
    val_parquet: Path,
    test_parquet: Path,
    model_dir: Path,
    config_path: Path,
) -> Dict:
    cfg = yaml.safe_load(config_path.read_text())
    obs = list(cfg["observables"])

    ridge_blob = joblib.load(model_dir / "ridge.joblib")
    ebm_blob = joblib.load(model_dir / "ebm.joblib")
    gbdt_blob = joblib.load(model_dir / "gbdt.joblib")
    ridge = ridge_blob["ridge"]
    ebm = ebm_blob["ebm"]
    gbt = gbdt_blob["gbdt"]
    ridge_stats = ridge_blob["ridge_stats"]
    raw_medians = ebm_blob["raw_medians"]
    clip = ridge_blob.get("clip") or ebm_blob.get("clip")
    lo, hi = clip["t_clip_lo"], clip["t_clip_hi"]

    splits = {
        "train": pd.read_parquet(train_parquet),
        "val": pd.read_parquet(val_parquet),
        "test": pd.read_parquet(test_parquet),
    }

    metrics: Dict = {
        "n": {k: int(len(v)) for k, v in splits.items()},
        "paths": {k: str(v) for k, v in [("train", train_parquet), ("val", val_parquet), ("test", test_parquet)]},
    }

    for split, df in splits.items():
        Xr, _ = prepare_ridge_X(df, obs, cfg["fit"]["ridge_log_features"], None, ridge_stats)
        X, _ = prepare_raw_X(df, obs, raw_medians)
        t_true = np.clip(df["t"].to_numpy(dtype=np.float64), lo, hi)
        y_top, pred_part = _truth_and_part_pred(df)
        preds = {
            "ridge": ridge.predict(Xr),
            "ebm": ebm.predict(X),
            "gbdt": gbt.predict(X),
        }
        metrics[split] = {
            name: classifier_metrics(t_true, yp, y_top, pred_part) for name, yp in preds.items()
        }
    return metrics


def main():
    repo = Path(__file__).resolve().parents[1]
    out_root = repo / "surrogate/outputs/top"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="If set, eval test-only from this predictions parquet (fast)",
    )
    p.add_argument("--train", type=Path, default=out_root / "features/features_train.parquet")
    p.add_argument("--val", type=Path, default=out_root / "features/features_val.parquet")
    p.add_argument("--test", type=Path, default=out_root / "features/features_test.parquet")
    p.add_argument("--model-dir", type=Path, default=out_root / "models")
    p.add_argument("--config", type=Path, default=repo / "surrogate/configs/top_basis.yaml")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: model-dir/classifier_metrics.json)",
    )
    args = p.parse_args()

    if args.predictions is not None:
        metrics = {"test": eval_from_predictions(args.predictions)}
    else:
        metrics = eval_from_models(args.train, args.val, args.test, args.model_dir, args.config)

    out = args.out or (args.model_dir / "classifier_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote → {out}")


if __name__ == "__main__":
    main()
