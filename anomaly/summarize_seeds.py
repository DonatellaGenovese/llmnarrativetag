"""The IAD detector over seeds: mean ± sd of every working-point metric.

Reads anomaly/outputs/seed<s>/transfer_summary.json for every seed present, and
anomaly/outputs/baselines/baselines_summary.json, and prints one row per
configuration (scheme, signal fraction, input space): AUC and top efficiency at
10% / 1% / 0.1% background, as the score is and with its soft-drop-mass
dependence removed. 10% background is the working point the narratives use.

    python -m anomaly.summarize_seeds
    python -m anomaly.summarize_seeds --out anomaly/outputs/summary_iad.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

COLS = [("raw", "auc", "AUC"), ("raw", "tpr@0.1", "TPR@10%"), ("raw", "tpr@0.01", "TPR@1%"),
        ("raw", "tpr@0.001", "TPR@.1%"), ("mass_decorrelated", "auc", "dAUC"),
        ("mass_decorrelated", "tpr@0.1", "dTPR@10%"), ("mass_decorrelated", "tpr@0.01", "dTPR@1%")]


def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=repo / "anomaly/outputs")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--all", action="store_true",
                   help="Also the configurations dropped from the plan: IAD and supervised on "
                        "the observables, and the latent-distance baselines")
    a = p.parse_args()
    # The plan (2026-11-08): IAD on the latent, against two references, the
    # supervised ceiling on the same latent and the soft-drop-mass cut.
    keep_cfg = (lambda k: True) if a.all else (lambda k: "osservabili" not in k)
    keep_base = (lambda k: True) if a.all else (lambda k: "massa" in k)

    runs = {}
    for d in sorted(a.dir.glob("seed*")):
        f = d / "transfer_summary.json"
        if f.exists():
            runs[int(re.sub(r"\D", "", d.name))] = json.loads(f.read_text())
    if not runs:
        raise SystemExit(f"no seed*/transfer_summary.json under {a.dir}")
    seeds = sorted(runs)
    configs = sorted({k for r in runs.values() for k in r if keep_cfg(k)},
                     key=lambda k: (not k.startswith("supervised"), -float(m.group(1)) if (m := re.search(r"f=([\d.]+)", k)) else 0, k))

    out = {"seeds": seeds, "configs": {}}
    head = "".join(f"{lab:>16s}" for _, _, lab in COLS)
    print(f"seeds: {seeds}   (mean ± sd over the seeds that have the configuration)\n")
    print(f"{'configuration':40s}{'n':>3s}{head}")
    for k in configs:
        vals = [runs[s][k] for s in seeds if k in runs[s]]
        row, rec = [], {"n_seeds": len(vals)}
        for part, key, lab in COLS:
            m, sd = ms([v[part][key] for v in vals])
            rec[f"{part}.{key}"] = {"mean": m, "sd": sd}
            row.append(f"{m:7.3f}±{sd:.3f}  ")
        out["configs"][k] = rec
        print(f"{k:40s}{len(vals):>3d}" + "".join(f"{c:>16s}" for c in row))

    b = a.dir / "baselines/baselines_summary.json"
    if b.exists():
        print("\nbaselines (deterministic, one run)")
        base = json.loads(b.read_text())
        out["baselines"] = base
        for k, v in base.items():
            if not keep_base(k):
                continue
            print(f"{k:40s}{'':>3s}" + "".join(f"{v[part][key]:>16.3f}" for part, key, _ in COLS))

    if a.out:
        a.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
