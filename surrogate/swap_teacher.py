"""Reuse a feature table's observables with another teacher's logits.

The ten observables are computed from the jet's constituents and do not depend
on which ParT produced the logits. Retraining the teacher with another seed
therefore changes only the teacher columns of the feature table, and recomputing
FastJet for every seed would cost an hour per split for identical numbers. This
copies the observables from an existing table and replaces only the columns
build_features_top.py takes from the prediction file:

    z_top  z_qcd  t  s  label  pred_label

Rows are matched by position, as build_features_top does, and the match is
checked rather than assumed: same number of jets, and the same jet_eta, jet_phi
and label in both files. A mismatch stops with an error.

    python -m surrogate.swap_teacher --features REF/features_test.parquet \\
        --pred-parquet RUN/predict_output/TopLandscape_ParT-FineTune_logits.parquet \\
        --out NEW/features_test.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TEACHER_COLS = ("z_top", "z_qcd", "t", "s", "label", "pred_label")


def swap(features: Path, pred_parquet: Path, out: Path, config: Path) -> None:
    tgt = yaml.safe_load(config.read_text())["targets"]
    ref = pd.read_parquet(features)
    src = {"z_top": tgt["z_top"], "z_qcd": tgt["z_qcd"], "t": tgt["t"], "s": tgt["s"],
           "label": tgt["label"], "pred_label": "pred_label"}
    pred = pd.read_parquet(pred_parquet, columns=sorted(set(src.values()) | {"jet_eta", "jet_phi"}))

    if len(pred) != len(ref):
        raise SystemExit(f"{pred_parquet.name}: {len(pred):,} jets, the feature table has {len(ref):,}")
    for c in ("jet_eta", "jet_phi"):
        d = np.abs(pred[c].to_numpy(np.float64) - ref[c].to_numpy(np.float64))
        if not np.all(d < 1e-4):
            raise SystemExit(f"rows do not line up: {c} differs on {int((d >= 1e-4).sum()):,} jets")
    if not (pred[tgt["label"]].to_numpy() == ref["label"].to_numpy()).all():
        raise SystemExit("rows do not line up: labels differ")

    df = ref.copy()
    for col, name in src.items():
        dtype = np.int64 if col in ("label", "pred_label") else np.float64
        df[col] = pred[name].to_numpy(dtype)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    changed = float(np.mean(df["pred_label"].to_numpy() != ref["pred_label"].to_numpy()))
    print(f"wrote {len(df):,} jets → {out}  (observables from {features}; "
          f"decision differs from the reference teacher on {changed:.2%} of jets)")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, required=True, help="Existing feature table")
    p.add_argument("--pred-parquet", type=Path, required=True, help="The new teacher's logits")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=repo / "surrogate/configs/top_basis.yaml")
    a = p.parse_args()
    swap(a.features, a.pred_parquet, a.out, a.config)


if __name__ == "__main__":
    main()
