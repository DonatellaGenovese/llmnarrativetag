"""The jets the AD narratives are written for, stratified by detector band.

A random sample of the test set would not do: at the 10% working point the
detector flags about half of the test jets (the test set is half top), nearly
all of them true tops, so near-threshold jets and false alarms would be a
handful. Instead, the sample is drawn per band of the detector rarity
(narrative.artefact_anomaly.DETECTOR_BANDS), and within each band half from
true top jets and half from true QCD jets where both exist, so that the
explanations of false alarms (QCD flagged) and of missed tops (top not flagged)
are numerous enough to study.

The truth label is used here, for the design and the analysis, and nowhere
else: the artefact never carries it. Every selected jet is recorded with its
band and label in the output file, which the orchestrator reads as a jets file
(`jets` holds the row indices of the test table, as for top tagging).

Jets with a missing observable are excluded (the artefact cannot report them).
The abstention gate is applied later, by the orchestrator, so part of the
sample (~10-15%) ends up abstained, as in top tagging.

    python -m narrative.select_anomaly_jets
    python -m narrative.select_anomaly_jets --out narrative/outputs/anomaly/narratives/ad100_jets.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from anomaly.gate import Gate, rarity
from .artefact_anomaly import detector_band

# Jets per band (sum 100): ~70 flagged across the three anomalous bands, ~30
# background-like, most of them near the working point.
DESIGN = {"extreme": 23, "strong": 23, "moderate": 24, "borderline": 20, "typical": 10}
LABELS = {0: "top", 1: "qcd"}       # the feature tables' convention


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=repo / "narrative/configs/narrative_anomaly.yaml")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    out = a.out or repo / cfg["outputs"]["narratives"] / "ad100_jets.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the sample is the unit of comparison, delete it on purpose to redraw")

    gate = Gate(repo / cfg["inputs"]["gate"])
    df = pd.read_parquet(repo / cfg["inputs"]["features"])
    obs = [c for c in ("m", "m_SD", "tau21", "tau32", "C2_double_b1", "D2_double_b1",
                       "C3_double_b1", "N3_b1", "pT", "n_const") if c in df.columns]
    ok = ~df[obs].isna().any(axis=1).to_numpy()
    r_pct = 100.0 * rarity(df["t"].to_numpy(np.float64), gate.ref_det)
    band = np.array([detector_band(max(x, 0.01)) for x in r_pct])
    label = df["label"].to_numpy()

    rng = np.random.default_rng(a.seed)
    rows, info, report = [], {}, {}
    for b, n in DESIGN.items():
        want = {0: n - n // 2, 1: n // 2}             # top gets the odd one
        pools = {lab: np.flatnonzero(ok & (band == b) & (label == lab)) for lab in (0, 1)}
        # If one label is short in this band, the other fills the gap.
        take = {lab: min(want[lab], len(pools[lab])) for lab in (0, 1)}
        short = n - sum(take.values())
        for lab in (0, 1):
            extra = min(short, len(pools[lab]) - take[lab])
            take[lab] += extra
            short -= extra
        report[b] = {LABELS[lab]: {"taken": take[lab], "available": int(len(pools[lab]))} for lab in (0, 1)}
        for lab in (0, 1):
            for i in rng.choice(pools[lab], size=take[lab], replace=False):
                rows.append(int(i))
                info[int(i)] = {"jet_id": int(df["jet_id"].iloc[i]), "band": b,
                                "label": LABELS[lab], "detector_rarity": round(float(r_pct[i]), 4)}

    rows.sort()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": len(rows), "seed": a.seed, "n_rows": int(len(df)),
        "design": DESIGN, "selection": report,
        "source": {"features": cfg["inputs"]["features"], "gate": cfg["inputs"]["gate"]},
        "jets": rows,
        "strata": {str(i): info[i] for i in rows},
    }, indent=1))

    print(f"{len(rows)} jets -> {out}")
    for b, v in report.items():
        print(f"  {b:11s} " + "   ".join(f"{k} {x['taken']:>2d} of {x['available']:>6d}" for k, x in v.items()))


if __name__ == "__main__":
    main()
