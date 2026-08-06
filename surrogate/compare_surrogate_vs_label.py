"""Side-by-side Acc/AUC: ParT-imitation surrogates vs label-trained classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _get(d: Dict, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def compare(surrogate_clf_metrics: Path, label_metrics: Path, out: Path) -> Dict:
    surr = json.loads(surrogate_clf_metrics.read_text())
    lab = json.loads(label_metrics.read_text())

    # Map model families: ridge↔logistic, ebm↔ebm, gbdt↔gbdt
    pairs = [("ridge", "logistic"), ("ebm", "ebm"), ("gbdt", "gbdt")]
    split = "test"
    rows = {}
    for s_name, l_name in pairs:
        rows[s_name] = {
            "imitate_part": {
                "acc": _get(surr, split, s_name, "acc_vs_truth"),
                "auc": _get(surr, split, s_name, "auc_vs_truth"),
            },
            "train_on_label": {
                "acc": _get(lab, l_name, split, "acc"),
                "auc": _get(lab, l_name, split, "auc"),
            },
        }
        ia = rows[s_name]["imitate_part"]["acc"]
        la = rows[s_name]["train_on_label"]["acc"]
        iu = rows[s_name]["imitate_part"]["auc"]
        lu = rows[s_name]["train_on_label"]["auc"]
        if None not in (ia, la, iu, lu):
            rows[s_name]["delta_label_minus_imitate"] = {
                "acc": float(la) - float(ia),
                "auc": float(lu) - float(iu),
            }

    part = {
        "acc": _get(surr, split, "ridge", "part_acc_vs_truth")
        or _get(lab, "part", split, "acc"),
        "auc": _get(surr, split, "ridge", "part_auc_vs_truth")
        or _get(lab, "part", split, "auc"),
    }

    out_obj = {
        "split": split,
        "part_baseline": part,
        "models": rows,
        "sources": {
            "surrogate_classifier_metrics": str(surrogate_clf_metrics),
            "label_metrics": str(label_metrics),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_obj, indent=2))
    print(json.dumps(out_obj, indent=2))
    print(f"Wrote → {out}")
    return out_obj


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--surrogate-metrics",
        type=Path,
        default=repo / "surrogate/outputs/top/models/classifier_metrics.json",
    )
    p.add_argument(
        "--label-metrics",
        type=Path,
        default=repo / "surrogate/outputs/top/models_label/metrics_label.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=repo / "surrogate/outputs/top/compare_surrogate_vs_label.json",
    )
    args = p.parse_args()
    compare(args.surrogate_metrics, args.label_metrics, args.out)


if __name__ == "__main__":
    main()
