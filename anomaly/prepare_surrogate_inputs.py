"""Feature tables for fitting a surrogate to an IAD score.

The IAD classifier is trained on the train split, so its scores there are
in-sample. The surrogate is therefore fit on val, which the classifier never
sees, and reported on test. Val is halved by jet_id parity: one half fits the
surrogate, the other selects it (the GBDT's early stopping, and the "val"
metrics). The three tables are written under the names the surrogate scripts
already read,

    <out>/features_train.parquet   val, even jet_id   fit
    <out>/features_val.parquet     val, odd jet_id    selection
    <out>/features_test.parquet    test               report

with the target column `t` holding the IAD logit and `t_part` keeping ParT's,
so surrogate.train_surrogates_top and surrogate.seed_variance run on them
unchanged. `label` keeps the feature table's convention, 0 = top. Every row is
joined on jet_id and checked: same jets, same labels.

    python -m anomaly.prepare_surrogate_inputs \\
        --scores anomaly/outputs/seed0/scores/iad_f0.1_latent \\
        --out anomaly/outputs/seed0/surrogate_inputs/iad_f0.1_latent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def joined(features: Path, scores: Path) -> pd.DataFrame:
    f = pd.read_parquet(features)
    s = pd.read_parquet(scores, columns=["jet_id", "label", "score"])
    if s["jet_id"].duplicated().any():
        raise SystemExit(f"{scores}: duplicated jet_id")
    d = f.merge(s.rename(columns={"label": "label_sc"}), on="jet_id", how="inner", validate="one_to_one")
    if len(d) != len(f):
        raise SystemExit(f"{scores}: {len(f) - len(d):,} jets of {features.name} have no score")
    if not (d["label"] == d["label_sc"]).all():
        raise SystemExit(f"{scores}: labels disagree with {features.name}")
    d = d.drop(columns=["label_sc"]).rename(columns={"t": "t_part"})
    d["t"] = d.pop("score").astype("float64")
    return d


def prepare(feat_dir: Path, scores_dir: Path, out: Path) -> None:
    val = joined(feat_dir / "features_val.parquet", scores_dir / "scores_val.parquet")
    test = joined(feat_dir / "features_test.parquet", scores_dir / "scores_test.parquet")
    fit, sel = val[val["jet_id"] % 2 == 0], val[val["jet_id"] % 2 == 1]
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("train", fit), ("val", sel), ("test", test)):
        df.reset_index(drop=True).to_parquet(out / f"features_{name}.parquet", index=False)
    (out / "SOURCE.txt").write_text(
        f"target t = IAD logit from {scores_dir}\n"
        f"features_train = val, even jet_id ({len(fit):,} jets)  fit\n"
        f"features_val   = val, odd jet_id  ({len(sel):,} jets)  selection\n"
        f"features_test  = test             ({len(test):,} jets)  report\n"
        f"observables and labels from {feat_dir}\n")
    print(f"{scores_dir.name}: fit {len(fit):,} · select {len(sel):,} · test {len(test):,} → {out}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", type=Path, required=True, help="anomaly/outputs/seed<s>/scores/<run>")
    p.add_argument("--feat-dir", type=Path, default=repo / "surrogate/outputs/top/features")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    prepare(a.feat_dir, a.scores, a.out)


if __name__ == "__main__":
    main()
