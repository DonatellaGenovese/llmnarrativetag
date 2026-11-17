"""Abstention gate for the top-tagging narratives, on decisions and ranks.

Replaces the residual gate |t_clip - g_hat| < 1 / < 2 (logit units). That gate
depends on the teacher's logit scale: its thresholds were set on an earlier
teacher (69.5% full, 7.1% abstain) and on the seed-42 teacher they give 50%
full, 29% caveat, 21% abstain. It also abstains on jets both models are sure
of: 48% of its abstentions have |t| >= 5, where t = 8 against g_hat = 5.5 is a
2.5-logit residual between two confident, agreeing verdicts.

This gate mirrors the anomaly-detection one (anomaly/gate.py):

  abstain   the surrogate takes the other decision: sign(g_hat) != sign(t).
            The phi would explain a decision the surrogate does not take.
  caveat    same decision, but the surrogate places the jet far from where the
            tagger does: |rank(t) - rank(g_hat)| > X percentile points, each
            ranked against its own predictions on the reference jets
            (quantile matching, as eval_surrogates does for AD).
  full      otherwise.

Ranks do not move when the logit scale stretches, so neither does the gate.
X is calibrated on the reference (the official val split, which the
surrogate is not fitted on; the narratives are written for test jets): the smallest
candidate that keeps the caveat share of the non-abstained jets at or below
--target-caveat, 0.30 as for AD. A new teacher needs only a rerun of this.

    python -m narrative.gate_top                 # calibrate, write gate.json + gate_reference.npz
    python -m narrative.gate_top --target-caveat 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

REGIME_FULL, REGIME_CAVEAT, REGIME_ABSTAIN = "full", "caveat", "abstain"
CANDIDATES_PP = (2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0)


def rank_pct(x, ref_sorted: np.ndarray) -> np.ndarray:
    """Percentage of reference values at or below x."""
    return 100.0 * np.searchsorted(ref_sorted, np.asarray(x, dtype=np.float64), side="right") / len(ref_sorted)


def regimes(t, g, rank_t, rank_g, caveat_pp: float) -> np.ndarray:
    t, g = np.asarray(t), np.asarray(g)
    out = np.full(len(t), REGIME_FULL, dtype=object)
    out[np.abs(np.asarray(rank_t) - np.asarray(rank_g)) > caveat_pp] = REGIME_CAVEAT
    out[(t > 0) != (g > 0)] = REGIME_ABSTAIN
    return out


def shares(reg: np.ndarray) -> dict:
    return {k: float(np.mean(reg == k)) for k in (REGIME_FULL, REGIME_CAVEAT, REGIME_ABSTAIN)}


class TopGate:
    """Load a calibrated gate and place jets. `t` is the raw teacher logit."""

    def __init__(self, gate_dir: Path):
        gate_dir = Path(gate_dir)
        self.cfg = json.loads((gate_dir / "gate.json").read_text())
        ref = np.load(gate_dir / "gate_reference.npz")
        self.ref_t, self.ref_g = ref["t"], ref["g"]
        self.caveat_pp = float(self.cfg["caveat_pp"])

    def __call__(self, t: float, g: float) -> dict:
        rt, rg = float(rank_pct([t], self.ref_t)[0]), float(rank_pct([g], self.ref_g)[0])
        reg = str(regimes([t], [g], [rt], [rg], self.caveat_pp)[0])
        return {"regime": reg, "rank_t": rt, "rank_g": rg, "rank_diff": abs(rt - rg)}


def _predict(model: Path, df: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(model)
    return np.asarray(bundle["ebm"].predict(df[list(bundle["obs"])]), dtype=np.float64)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=repo / "narrative/configs/narrative_top.yaml")
    p.add_argument("--target-caveat", type=float, default=0.30,
                   help="Max caveat share among the non-abstained reference jets")
    a = p.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    gcfg = cfg["abstention"]
    out_dir = repo / gcfg["gate"]
    model = repo / cfg["inputs"]["model"]
    ref = pd.read_parquet(repo / gcfg["reference"])
    test = pd.read_parquet(repo / cfg["inputs"]["features"])

    t_ref, g_ref = ref["t"].to_numpy(np.float64), _predict(model, ref)
    ok = np.isfinite(t_ref) & np.isfinite(g_ref)
    t_ref, g_ref = t_ref[ok], g_ref[ok]
    srt_t, srt_g = np.sort(t_ref), np.sort(g_ref)
    rt, rg = rank_pct(t_ref, srt_t), rank_pct(g_ref, srt_g)

    table = {}
    for x in CANDIDATES_PP:
        reg = regimes(t_ref, g_ref, rt, rg, x)
        kept = reg != REGIME_ABSTAIN
        table[x] = {**shares(reg), "caveat_of_kept": float(np.mean(reg[kept] == REGIME_CAVEAT))}
    fitting = [x for x in CANDIDATES_PP if table[x]["caveat_of_kept"] <= a.target_caveat]
    x = fitting[0] if fitting else CANDIDATES_PP[-1]

    t_te, g_te = test["t"].to_numpy(np.float64), _predict(model, test)
    reg_te = regimes(t_te, g_te, rank_pct(t_te, srt_t), rank_pct(g_te, srt_g), x)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "gate_reference.npz", t=srt_t, g=srt_g)
    summary = {
        "caveat_pp": x,
        "target_caveat": a.target_caveat,
        "calibrated_on": str(gcfg["reference"]),
        "model": str(cfg["inputs"]["model"]),
        "n_reference": int(len(t_ref)),
        "reference_by_candidate": {str(k): v for k, v in table.items()},
        "reference": table[x],
        "test": shares(reg_te),
    }
    (out_dir / "gate.json").write_text(json.dumps(summary, indent=2))

    print("candidate X (pct points): full / caveat / abstain   caveat of kept")
    for k, v in table.items():
        print(f"  {k:>5}: {v['full']:.3f} / {v['caveat']:.3f} / {v['abstain']:.3f}   {v['caveat_of_kept']:.3f}")
    print(f"chosen X = {x}  (target caveat <= {a.target_caveat:.0%} of kept)")
    s = summary["test"]
    print(f"test: full {s['full']:.3f}  caveat {s['caveat']:.3f}  abstain {s['abstain']:.3f}")
    print(f"wrote {out_dir}/gate.json, gate_reference.npz")


if __name__ == "__main__":
    main()
