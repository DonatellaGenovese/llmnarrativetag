"""Abstention gate for the AD narratives, on rarity instead of the logit.

The top-tagging gate thresholds |t - g_hat| in logit units (1 and 2). The IAD
logit has no fixed scale (its spread changes several-fold with the signal
fraction and the seed), so here every score is turned into a **rarity**: the
share of reference background (QCD) jets that score higher, i.e. the jet's
background efficiency. The detector's rarity uses its logit against the QCD
logits; the surrogate's uses the EBM prediction against the EBM's predictions
on the same QCD jets (the quantile-matched comparison of eval_surrogates). The
reference QCD jets are those of the selection half of val
(surrogate_inputs/<run>/features_val.parquet), which neither the IAD
classifier nor the surrogate fit saw.

Regimes, at the working point WP (10% background):

  abstain   detector and EBM disagree on the decision (rarity <= WP or not):
            the φ would explain a decision the surrogate does not take. Both
            directions: flagged by the detector only, or by the EBM only
  caveat    the jet is flagged by both, but the EBM's rarity is off by more
            than a factor X from the detector's: the explanation covers only
            part of why the jet is that extreme
  full      otherwise (same decision; for flagged jets, rarity within X)

For jets neither flags the rarity is not checked: the narrative says the jet
is background-like, and both agree on that.

X is calibrated, per signal fraction and pooled over the detector seeds, on the
selection half: the smallest candidate factor that keeps the caveat share of
the detector-flagged jets at or below --target-caveat. The rarity reaches the
artefact only as a band (`band`), since the EBM's rarity is off by a factor ~4
in the median.

    python -m anomaly.gate                      # calibrate, write gate.json per run
    python -m anomaly.gate --target-caveat 0.25 --out anomaly/outputs/summary_gate.json

Writes anomaly/outputs/seed<s>/surrogate/<run>/gate.json and gate_reference.npz
(the sorted reference scores, all a narrative run needs to place a new jet).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from surrogate.train_surrogates_top import prepare_raw_X

REGIME_FULL, REGIME_CAVEAT, REGIME_ABSTAIN = "full", "caveat", "abstain"
FACTORS = (2, 3, 5, 10, 20, 50, 100)
# Band edges in background efficiency; the first band is "not anomalous" at WP 10%.
BANDS = ((0.10, "top 10%-1% of background"), (0.01, "top 1%-0.1% of background"),
         (0.001, "rarer than 0.1% of background"))


# -- the gate, usable from the narrative code ---------------------------------

def rarity(score, ref_sorted: np.ndarray) -> np.ndarray:
    """Share of reference jets scoring strictly higher, floored at 1/n (never 0)."""
    n = len(ref_sorted)
    above = n - np.searchsorted(ref_sorted, np.asarray(score, dtype=np.float64), side="right")
    return np.maximum(above, 1) / n


def band(r: float, wp: float = 0.10) -> str:
    if r > wp:
        return "not anomalous (background-like)"
    label = BANDS[0][1]
    for edge, lab in BANDS:
        if r <= edge:
            label = lab
    return label


def regimes(r_det: np.ndarray, r_sur: np.ndarray, wp: float, factor: float) -> np.ndarray:
    d, s = r_det <= wp, r_sur <= wp
    off = np.abs(np.log10(r_det) - np.log10(r_sur)) > np.log10(factor)
    out = np.full(len(r_det), REGIME_FULL, dtype=object)
    out[d & s & off] = REGIME_CAVEAT
    out[d != s] = REGIME_ABSTAIN
    return out


class Gate:
    """Load a calibrated gate (gate.json + gate_reference.npz) and place jets."""

    def __init__(self, run_dir: Path):
        run_dir = Path(run_dir)
        self.cfg = json.loads((run_dir / "gate.json").read_text())
        ref = np.load(run_dir / "gate_reference.npz")
        self.ref_det, self.ref_sur = ref["detector"], ref["ebm"]
        self.wp, self.factor = float(self.cfg["wp"]), float(self.cfg["caveat_factor"])

    def __call__(self, t, g_hat) -> dict:
        rd, rs = rarity([t], self.ref_det), rarity([g_hat], self.ref_sur)
        return {"regime": str(regimes(rd, rs, self.wp, self.factor)[0]),
                "rarity_detector": float(rd[0]), "rarity_ebm": float(rs[0]),
                "decision": "anomalous" if rd[0] <= self.wp else "not anomalous",
                "band": band(float(rd[0]), self.wp)}


# -- calibration ----------------------------------------------------------------

def ebm_predict(mod: Path, df: pd.DataFrame) -> np.ndarray:
    b = joblib.load(mod / "ebm.joblib")
    X, _ = prepare_raw_X(df, list(b["obs"]), b["raw_medians"])
    return b["ebm"].predict(X)


def shares(reg: np.ndarray, r_det: np.ndarray, r_sur: np.ndarray, wp: float) -> dict:
    d, s = r_det <= wp, r_sur <= wp
    out = {}
    for name, m in (("all", np.ones(len(reg), bool)), ("flagged", d), ("not_flagged", ~d)):
        out[name] = {k: float((reg[m] == k).mean()) for k in (REGIME_FULL, REGIME_CAVEAT, REGIME_ABSTAIN)}
        out[name]["n"] = int(m.sum())
    out["abstain_detector_only"] = float((d & ~s).mean())
    out["abstain_ebm_only"] = float((~d & s).mean())
    return out


def ms(v):
    return {"mean": float(st.mean(v)), "sd": float(st.stdev(v)) if len(v) > 1 else 0.0}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=repo / "anomaly/outputs")
    p.add_argument("--fractions", nargs="+", default=["0.1", "0.05", "0.01"])
    p.add_argument("--wp", type=float, default=0.10)
    p.add_argument("--target-caveat", type=float, default=0.30,
                   help="Max caveat share among detector-flagged jets of the selection half")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    summary = {"wp": a.wp, "target_caveat": a.target_caveat, "factors": list(FACTORS)}
    for f in a.fractions:
        run = f"iad_f{f}_latent"
        dirs = sorted((d for d in a.dir.glob("seed*") if (d / "surrogate" / run / "ebm.joblib").exists()),
                      key=lambda d: int(re.sub(r"\D", "", d.name)))
        if not dirs:
            print(f"f={f}: no fitted surrogate")
            continue
        per = {}
        for d in dirs:
            inp, mod = d / "surrogate_inputs" / run, d / "surrogate" / run
            sel = pd.read_parquet(inp / "features_val.parquet")
            test = pd.read_parquet(inp / "features_test.parquet", columns=["jet_id", "t"])
            pred = pd.read_parquet(mod / "test_predictions.parquet", columns=["jet_id", "t_hat_ebm"])
            if not (pred["jet_id"].to_numpy() == test["jet_id"].to_numpy()).all():
                raise SystemExit(f"{mod}: predictions and test table are not row-aligned")
            g_sel = ebm_predict(mod, sel)
            qcd = sel["label"].to_numpy() == 1
            ref_det, ref_sur = np.sort(sel["t"].to_numpy()[qcd]), np.sort(g_sel[qcd])
            per[d.name] = {
                "mod": mod, "ref_det": ref_det, "ref_sur": ref_sur,
                "sel": (rarity(sel["t"].to_numpy(), ref_det), rarity(g_sel, ref_sur)),
                "test": (rarity(test["t"].to_numpy(), ref_det), rarity(pred["t_hat_ebm"].to_numpy(), ref_sur)),
            }

        # Caveat share of flagged jets per factor, pooled over seeds, on the selection half.
        table = {}
        for x in FACTORS:
            c = [shares(regimes(*v["sel"], a.wp, x), *v["sel"], a.wp)["flagged"][REGIME_CAVEAT] for v in per.values()]
            table[x] = ms(c)
        ok = [x for x in FACTORS if table[x]["mean"] <= a.target_caveat]
        factor = ok[0] if ok else FACTORS[-1]

        res = {"caveat_factor": factor, "seeds": list(per),
               "selection_caveat_by_factor": {str(x): table[x] for x in FACTORS}}
        for split in ("sel", "test"):
            sh = [shares(regimes(*v[split], a.wp, factor), *v[split], a.wp) for v in per.values()]
            res[split] = {pop: {k: ms([s[pop][k] for s in sh]) for k in (REGIME_FULL, REGIME_CAVEAT, REGIME_ABSTAIN)}
                          for pop in ("all", "flagged", "not_flagged")}
            res[split]["abstain_detector_only"] = ms([s["abstain_detector_only"] for s in sh])
            res[split]["abstain_ebm_only"] = ms([s["abstain_ebm_only"] for s in sh])
            # Bands the artefact would show, on the detector-flagged jets.
            bd = [pd.Series([band(r, a.wp) for r in v[split][0][v[split][0] <= a.wp]]).value_counts(normalize=True)
                  for v in per.values()]
            res[split]["bands_flagged"] = {lab: ms([b.get(lab, 0.0) for b in bd]) for _, lab in BANDS}
        summary[run] = res

        for name, v in per.items():
            np.savez_compressed(v["mod"] / "gate_reference.npz", detector=v["ref_det"], ebm=v["ref_sur"])
            (v["mod"] / "gate.json").write_text(json.dumps({
                "wp": a.wp, "caveat_factor": factor, "target_caveat": a.target_caveat,
                "calibrated_on": "selection half of val (features_val.parquet), pooled over seeds "
                                 + ", ".join(per), "n_reference_qcd": int(len(v["ref_det"])),
                "detector_threshold": float(np.quantile(v["ref_det"], 1 - a.wp)),
                "ebm_threshold": float(np.quantile(v["ref_sur"], 1 - a.wp)),
                "bands": [[e, lab] for e, lab in BANDS]}, indent=2))

        print(f"\n===== IAD latent, f = {f}   ({len(per)} seeds)")
        print("caveat share of flagged jets (selection half) by factor:  "
              + "  ".join(f"×{x} {table[x]['mean']:.3f}" for x in FACTORS))
        print(f"chosen factor ×{factor}  (target caveat <= {a.target_caveat:.0%} of flagged)")
        for split, lab in (("sel", "selection"), ("test", "test")):
            r = res[split]
            print(f"  {lab:9s} " + "   ".join(
                f"{pop}: full {r[pop]['full']['mean']:.3f} caveat {r[pop]['caveat']['mean']:.3f} "
                f"abstain {r[pop]['abstain']['mean']:.3f}" for pop in ("flagged", "not_flagged", "all")))
        print("  test bands of flagged jets: " + "   ".join(
            f"{lab} {v['mean']:.3f}" for lab, v in res["test"]["bands_flagged"].items()))

    if a.out:
        a.out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
