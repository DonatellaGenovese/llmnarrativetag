"""Correlation audit and mass residualization for the observable basis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def correlation_matrix(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return df[list(cols)].corr(method="pearson")


def vif_table(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Variance inflation factors via inverse correlation diagonal."""
    x = df[list(cols)].to_numpy(dtype=np.float64)
    # drop non-finite rows
    mask = np.isfinite(x).all(axis=1)
    x = x[mask]
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)
    corr = np.corrcoef(x, rowvar=False)
    try:
        inv = np.linalg.inv(corr)
        vif = np.diag(inv)
    except np.linalg.LinAlgError:
        vif = np.full(len(cols), np.nan)
    return pd.DataFrame({"feature": list(cols), "VIF": vif}).sort_values("VIF", ascending=False)


def condition_number(df: pd.DataFrame, cols: Sequence[str]) -> float:
    x = df[list(cols)].to_numpy(dtype=np.float64)
    mask = np.isfinite(x).all(axis=1)
    x = x[mask]
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)
    s = np.linalg.svd(x, compute_uv=False)
    return float(s.max() / max(s.min(), 1e-12))


def mass_residualize(
    df: pd.DataFrame,
    features: Sequence[str],
    mass_col: str = "m",
    degree: int = 3,
    train_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Residualize selected features against a degree-d polynomial in mass.
    Fit the polynomial on train_mask rows (or all rows if None).
    """
    out = df.copy()
    m = out[mass_col].to_numpy(dtype=np.float64).reshape(-1, 1)
    if train_mask is None:
        train_mask = np.isfinite(m[:, 0])
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    # fit poly on finite train mass
    fit_idx = train_mask & np.isfinite(m[:, 0])
    poly.fit(m[fit_idx])
    M_all = poly.transform(np.nan_to_num(m, nan=np.nanmedian(m[fit_idx])))

    for col in features:
        y = out[col].to_numpy(dtype=np.float64)
        y_fit_mask = fit_idx & np.isfinite(y)
        reg = LinearRegression()
        reg.fit(M_all[y_fit_mask], y[y_fit_mask])
        y_hat = reg.predict(M_all)
        resid = y - y_hat
        resid[~np.isfinite(y)] = np.nan
        out[f"{col}_mres"] = resid
    return out


def run_audit(features_parquet: Path, out_dir: Path, obs_cols: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(features_parquet)
    corr = correlation_matrix(df, obs_cols)
    vif = vif_table(df, obs_cols)
    cond = condition_number(df, obs_cols)

    corr.to_csv(out_dir / "correlation_matrix.csv")
    vif.to_csv(out_dir / "vif.csv", index=False)
    (out_dir / "condition_number.txt").write_text(f"{cond:.6g}\n")
    print("Condition number:", cond)
    print(vif.to_string(index=False))
    print(f"Wrote audit → {out_dir}")


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features",
        type=Path,
        default=repo / "surrogate/outputs/top/features_test.parquet",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "surrogate/outputs/top/audit",
    )
    p.add_argument(
        "--obs",
        nargs="+",
        default=[
            "m",
            "m_SD",
            "tau21",
            "tau32",
            "C2_double_b1",
            "D2_double_b1",
            "C3_double_b1",
            "N3_b1",
            "pT",
            "n_const",
        ],
    )
    args = p.parse_args()
    run_audit(args.features, args.out_dir, args.obs)


if __name__ == "__main__":
    main()
