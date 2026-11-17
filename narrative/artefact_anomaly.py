"""Per-jet artefact for the anomaly-detection narratives.

The AD counterpart of narrative.artefact, for the IAD detector on the frozen
ParT latent (anomaly/transfer.py) and the EBM fitted to its logit
(anomaly/outputs/seed<s>/surrogate/<run>/). Same two rules as the top-tagging
artefact: every number is pre-rounded so the verifier compares what was sent,
and anything the narrator must not assert is absent.

What differs from top tagging, and why:

  * No logits. score_tagger, score_pct, score_reconstructed, recon_pct and
    intercept are all IAD or EBM logits, whose scale changes several-fold with
    the signal fraction and the seed and means nothing to a reader. They are
    replaced by ranks against the background reference.

  * Detector rarity and band. The detector rarity is the share (in %) of
    background (QCD) reference jets that score higher than this jet, i.e. its
    background efficiency. The reference jets are the QCD jets of the selection
    half of val, which neither the IAD classifier nor the surrogate was fitted
    on (anomaly/gate.py). It is a rank, so it is exact up to the size of the
    reference (~100k jets) and is handed over as a number, [AR], and as a word,
    [A], from DETECTOR_BANDS.

  * Surrogate band. The same reading for the EBM's reconstruction of the score,
    ranked against the EBM's own predictions on the same reference jets. The
    EBM's rarity is off from the detector's by a factor ~4 in the median, so it
    is handed over as a band word only, [AS], never as a number. It replaces
    the reconstruction quality of the top-tagging artefact.

  * Decision. "anomalous" when the detector rarity is at or below the working
    point (10%, gate.json), "background-like" otherwise; the gate uses the same
    rule, so decision and regime always agree.

  * Contributions are NOT oriented. Each phi is the EBM term's contribution to
    the anomaly score: positive makes the jet more anomalous, negative more
    background-like, whatever the decision. The words the narrator must use for
    the two directions travel in the artefact (`direction_words`), and the
    verifier reads them from there.

  * Observable pct and rarity are against the background reference (the same
    QCD jets), not against a class: the detector is built against that
    reference and knows nothing else.

  * The truth label is never read. It stays in the feature table and in the
    jet-selection file, for the analysis only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from anomaly.gate import Gate, REGIME_ABSTAIN, REGIME_CAVEAT, REGIME_FULL  # noqa: F401
from .stats import _grid, _pct_below

# Bands of the detector rarity (and of the surrogate's), in % of background
# reference jets scoring higher. The edges are the narratives' working point
# (10%) and the two secondary working points reported in E1 (1%, 0.1%), i.e.
# background rejections of 10, 100 and 1000; 30% only separates
# near-threshold background-like jets from typical ones. Upper edges are
# inclusive, as for the other word tags. The bands of flagged jets are decades,
# so the gate's caveat rule (rarities off by more than x10) implies "the
# surrogate's band differs from the detector's" everywhere except inside
# `extreme`; there the caveat is dropped (see `build`).
DETECTOR_BANDS = (
    (0.1, "extreme"),            # rarity <= 0.1%        anomalous
    (1.0, "strong"),             # 0.1% < rarity <= 1%   anomalous
    (10.0, "moderate"),          # 1% < rarity <= 10%    anomalous (up to the working point)
    (30.0, "borderline"),        # 10% < rarity <= 30%   background-like, near the working point
    (float("inf"), "typical"),   # rarity > 30%          background-like
)
DECISION_ANOMALOUS = "anomalous"
DECISION_BACKGROUND = "background-like"
# One word each, no hyphen: word tags take letters only.
DIRECTION_WORDS = {"positive": "anomalous", "negative": "background"}


def detector_band(rarity_pct: float) -> str:
    for edge, word in DETECTOR_BANDS:
        if rarity_pct <= edge:
            return word
    return DETECTOR_BANDS[-1][1]


class AnomalyArtefactBuilder:
    """Builds AD narrative artefacts from the EBM, the gate and the background reference."""

    def __init__(self, repo: Path, config: dict):
        inp = config["inputs"]
        self.model_path = str(repo / inp["model"])
        bundle = joblib.load(repo / inp["model"])
        self.ebm = bundle["ebm"]
        self.obs: List[str] = list(bundle["obs"])
        self.intercept = float(self.ebm.intercept_)

        self.gate_dir = repo / inp["gate"]
        self.gate = Gate(self.gate_dir)
        self.wp_pct = 100.0 * self.gate.wp

        # Background reference: the QCD jets of the selection half of val
        # (label 1 = QCD, the feature tables' convention), the same jets the
        # gate ranks the scores against.
        ref = pd.read_parquet(repo / inp["reference"], columns=self.obs + ["label"])
        qcd = ref[ref["label"] == 1]
        self.n_reference = int(len(qcd))
        self.reference_source = str(inp["reference"])
        self.obs_grid: Dict[str, np.ndarray] = {o: _grid(qcd[o].to_numpy()) for o in self.obs}

        a = config["artefact"]
        self.top_k = a["top_k"] if a["top_k"] is not None else len(self.obs)
        self.r_value = int(a["round"]["value"])
        self.r_phi = int(a["round"]["phi"])
        self.r_pct = int(a["round"].get("pct", 2))

    # The orchestrator's interface (shared with narrative.artefact.ArtefactBuilder).

    def abstention_reason(self, meta: dict) -> str:
        return ("the detector and the observable model disagree on whether this jet is "
                f"anomalous at {self.wp_pct:g}% background (detector rarity "
                f"{100 * meta['rarity_detector']:.3g}%, surrogate {100 * meta['rarity_ebm']:.3g}%); "
                "the observables do not account for this decision closely enough to describe it")

    def additivity_error(self, record: dict) -> float:
        """intercept + sum(phi) - g_hat, bounded by rounding (the intercept is in meta)."""
        m, a = record["meta"], record["artefact"]
        return float(m["intercept"] + sum(f["phi"] for f in a["features"]) - m["g_hat"])

    def _floor(self, pct: float) -> float:
        # A value beyond every reference jet would read as "no background jet
        # lies further out", which a finite reference cannot support. Clamp to
        # the reporting precision, as narrative.artefact does.
        floor = 10.0 ** (-self.r_pct)
        return min(max(pct, floor), 100.0 - floor)

    def build(self, df: pd.DataFrame, index: int) -> dict:
        row = df.iloc[[index]]
        x = row[self.obs]
        if x.isna().any(axis=None):
            raise ValueError(f"row {index} (jet {int(row['jet_id'].iloc[0])}) has missing observables; "
                             "the jet selection excludes such jets")

        t = float(row["t"].iloc[0])                  # IAD logit, never shown
        g = float(self.ebm.predict(x)[0])           # EBM reconstruction, never shown
        place = self.gate(t, g)
        rar_det = self._floor(100.0 * place["rarity_detector"])
        rar_ebm = 100.0 * place["rarity_ebm"]

        # Caveat only when the narrator can see the discrepancy. It sees the
        # surrogate as a band word, and the bands of flagged jets are decades,
        # so a gap above x10 always crosses a band edge except inside `extreme`,
        # which has no lower edge (e.g. 0.008% vs 0.098%). There both readings
        # say "rarer than 1 in 1000" and a caveat would contradict the two equal
        # band words in the same paragraph, so the jet is narrated in full. The
        # gate's own verdict is kept in meta (`gate_regime`); the gate and its
        # calibration are unchanged.
        gate_regime = place["regime"]
        regime = gate_regime
        if regime == REGIME_CAVEAT and detector_band(rar_det) == detector_band(rar_ebm):
            regime = REGIME_FULL

        phi = np.asarray(self.ebm.eval_terms(x), dtype=np.float64)[0]   # not oriented
        order = np.argsort(-np.abs(phi))
        shown, rest = order[: self.top_k], order[self.top_k:]

        features = []
        for position, j in enumerate(shown, start=1):
            name = self.obs[j]
            value = float(row[name].iloc[0])
            pct = self._floor(_pct_below(self.obs_grid[name], value))
            features.append({
                "i": f"{position:02d}",
                "name": name,
                "value": round(value, self.r_value),
                "phi": round(float(phi[j]), self.r_phi) + 0.0,   # + 0.0: no "-0.0", whose sign the narrator would read
                "pct": round(pct, self.r_pct),
                "rarity": round(min(pct, 100.0 - pct), self.r_pct),
            })

        artefact = {
            "schema_version": "ad-1",
            "jet_id": int(row["jet_id"].iloc[0]),
            "decision": place["decision"].replace("not anomalous", DECISION_BACKGROUND),
            "detector_rarity": round(rar_det, self.r_pct),
            "detector_band": detector_band(rar_det),
            "surrogate_band": detector_band(rar_ebm),
            "direction_words": dict(DIRECTION_WORDS),
            "features": features,
        }
        if len(rest):
            artefact["n_other"] = int(len(rest))
            artefact["phi_other"] = round(float(phi[rest].sum()), self.r_phi)

        meta = {
            "regime": regime,
            "gate_regime": gate_regime,
            "rarity_detector": place["rarity_detector"],
            "rarity_ebm": place["rarity_ebm"],
            "t_raw": round(t, 4),
            "g_hat": round(g, 4),
            "intercept": round(self.intercept, 4),
            "working_point": self.gate.wp,
            "caveat_factor": self.gate.factor,
            "gate": str(self.gate_dir),
            "model": self.model_path,
            "reference": self.reference_source,
            "n_reference": self.n_reference,
            "top_k": self.top_k,
        }
        return {"artefact": artefact, "meta": meta}


def main():
    import argparse

    import yaml

    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=repo / "narrative/configs/narrative_anomaly.yaml")
    p.add_argument("--jet", type=int, default=0, help="Row index; prints one artefact")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    builder = AnomalyArtefactBuilder(repo, cfg)
    df = pd.read_parquet(repo / cfg["inputs"]["features"])
    print(json.dumps(builder.build(df, args.jet), indent=2))


if __name__ == "__main__":
    main()
