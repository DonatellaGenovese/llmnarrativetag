"""Per-jet artefact: the surrogate's local explanation, ready for the narrator.

The artefact is the only thing the narrator sees, so it is also the only thing
verification can check a claim against. Two rules follow:

  * every number handed over is pre-rounded to a string-stable value, so the
    verifier compares what was sent, not a float it re-derives;
  * anything the narrator must not assert is simply absent, rather than present
    and forbidden by the prompt.

Scores and contributions are oriented towards the decision: positive pushes
towards `decision`, negative towards `other_class`. Orientation flips the sign
of phi, of the intercept and of g_hat together, so the additive identity

    intercept + sum(phi) == score_reconstructed

holds in the oriented frame too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

from .stats import ReferenceStats

# Regimes of the abstention gate, by |r| = |t_clip - g_hat|.
REGIME_FULL = "full"
REGIME_CAVEAT = "caveat"
REGIME_ABSTAIN = "abstain"


class ArtefactBuilder:
    """Builds narrative artefacts from a fitted surrogate and frozen stats."""

    def __init__(
        self,
        model_joblib: Path,
        stats: ReferenceStats,
        config: dict,
    ):
        bundle = joblib.load(model_joblib)
        self.ebm = bundle["ebm"]
        self.obs: List[str] = list(bundle["obs"])
        self.clip: Dict[str, float] = dict(bundle["clip"])
        self.intercept = float(self.ebm.intercept_)
        self.stats = stats
        self.model_path = str(model_joblib)

        a = config["artefact"]
        self.top_k = a["top_k"] if a["top_k"] is not None else len(self.obs)
        self.include_pct = bool(a["include_pct"])
        # Derived, so the version can never drift from the shape it describes.
        self.schema_version = 2 if self.include_pct else 1
        self.r_value = int(a["round"]["value"])
        self.r_phi = int(a["round"]["phi"])
        self.r_score = int(a["round"]["score"])
        self.r_pct = int(a["round"].get("pct", 1))

        g = config["abstention"]
        self.full_threshold = float(g["full_threshold"])
        self.caveat_threshold = float(g["caveat_threshold"])

        if self.obs != self.stats.obs:
            raise ValueError("model and reference stats disagree on the observable basis")

    # -- internals ---------------------------------------------------------

    def regime(self, residual: float) -> str:
        r = abs(residual)
        if r < self.full_threshold:
            return REGIME_FULL
        if r < self.caveat_threshold:
            return REGIME_CAVEAT
        return REGIME_ABSTAIN

    def _terms(self, x: pd.DataFrame) -> np.ndarray:
        """Per-observable contributions, shape (n, K), unoriented."""
        return np.asarray(self.ebm.eval_terms(x), dtype=np.float64)

    # -- public ------------------------------------------------------------

    def build(self, df: pd.DataFrame, index: int) -> dict:
        """Artefact plus provenance for a single row of a feature table."""
        row = df.iloc[[index]]
        x = row[self.obs]

        t = float(row["t"].iloc[0])
        t_clip = float(np.clip(t, self.clip["t_clip_lo"], self.clip["t_clip_hi"]))
        g = float(self.ebm.predict(x)[0])
        residual = t_clip - g

        decision = "top" if t > 0 else "qcd"
        other_class = "qcd" if decision == "top" else "top"
        sign = 1.0 if t > 0 else -1.0

        phi = self._terms(x)[0] * sign
        order = np.argsort(-np.abs(phi))
        shown, rest = order[: self.top_k], order[self.top_k :]

        features = []
        for position, j in enumerate(shown, start=1):
            name = self.obs[j]
            value = float(row[name].iloc[0])
            entry = {
                # Tags key off `i`, not `name`: closing a tag named after a
                # 14-character observable is where narrators slip most often,
                # writing [P_C2_double_b1]...[/C2_double_b1]. An index is one
                # character to copy and the failure mode disappears.
                # Two digits always, even for 1..9: the index 10 was the only
                # tag of a different width, and 45% of mismatched pairs were a
                # closing [/I0] for an opening [I10].
                "i": f"{position:02d}",
                "name": name,
                "value": round(value, self.r_value),
                "phi": round(float(phi[j]), self.r_phi),
            }
            if self.include_pct:
                pct = self.stats.obs_percentile(name, other_class, value)
                # A value beyond every reference jet returns exactly 0 or 100,
                # and a narrative then states that no jet of the other class
                # lies further out. Two hundred thousand jets cannot support
                # that; they can support "at most one in ten thousand". Clamp
                # to the reporting precision so the claim stays within it.
                floor = 10.0 ** (-self.r_pct)
                pct = min(max(pct, floor), 100.0 - floor)
                # Handed over rather than left to be derived: the narrator is
                # told to report and not compute, so any quantity it should
                # cite has to arrive already formed.
                # `pct` is passed but NOT tagged: the narrator needs it to say
                # which side of the other class the value falls on, yet tagging
                # it put two often-identical numbers in adjacent tags and
                # sixteen mismatched pairs were an [R<i>] closed as [/P<i>].
                entry["pct"] = round(pct, self.r_pct)
                entry["rarity"] = round(min(pct, 100.0 - pct), self.r_pct)
            features.append(entry)

        artefact = {
            "schema_version": self.schema_version,
            "jet_id": int(row["jet_id"].iloc[0]),
            "decision": decision,
            "other_class": other_class,
            "score_tagger": round(abs(t_clip), self.r_score),
            "score_pct": round(self.stats.score_percentile(decision, t_clip)),
            "score_reconstructed": round(g * sign, self.r_score),
            "recon_pct": round(self.stats.recon_percentile(residual)),
            "intercept": round(self.intercept * sign, self.r_score),
            "features": features,
        }
        # Omitted entirely when the full basis is reported: a narrator handed
        # `n_other: 0` dutifully writes a sentence about nothing.
        if len(rest):
            artefact["n_other"] = int(len(rest))
            artefact["phi_other"] = round(float(phi[rest].sum()), self.r_phi)

        meta = {
            "regime": self.regime(residual),
            "residual": round(residual, 4),
            "t_raw": round(t, 4),
            "t_clipped": bool(t != t_clip),
            "g_hat": round(g, 4),
            "model": self.model_path,
            "stats_source": self.stats.source,
            "top_k": self.top_k,
            "include_pct": self.include_pct,
        }
        return {"artefact": artefact, "meta": meta}

    def build_many(self, df: pd.DataFrame, indices: Sequence[int]) -> List[dict]:
        return [self.build(df, int(i)) for i in indices]


def additivity_error(record: dict) -> float:
    """How far the reported parts are from the reported whole.

    Bounded by rounding: (K + 1) * 0.5 * 10**-r_phi. Non-zero is expected;
    a large value means the orientation or the term extraction is wrong.
    """
    a = record["artefact"]
    parts = a["intercept"] + sum(f["phi"] for f in a["features"]) + a.get("phi_other", 0.0)
    return float(parts - a["score_reconstructed"])


def load_builder(repo: Path, config_path: Path) -> tuple[ArtefactBuilder, dict, pd.DataFrame]:
    cfg = yaml.safe_load(config_path.read_text())
    stats = ReferenceStats.load(repo / cfg["outputs"]["stats"])
    builder = ArtefactBuilder(repo / cfg["inputs"]["model"], stats, cfg)
    df = pd.read_parquet(repo / cfg["inputs"]["features"])
    return builder, cfg, df


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=repo / "narrative/configs/narrative_top.yaml")
    p.add_argument("--jet", type=int, default=None, help="Row index; prints one artefact")
    p.add_argument("--validate", type=int, default=None, metavar="N",
                   help="Check the additive identity on the first N jets")
    args = p.parse_args()

    builder, cfg, df = load_builder(repo, args.config)

    if args.validate:
        n = min(args.validate, len(df))
        errs, regimes = [], {}
        for i in range(n):
            rec = builder.build(df, i)
            errs.append(abs(additivity_error(rec)))
            regimes[rec["meta"]["regime"]] = regimes.get(rec["meta"]["regime"], 0) + 1
        errs = np.array(errs)
        bound = (len(builder.obs) + 1) * 0.5 * 10 ** (-builder.r_phi)
        print(f"additivity over {n} jets: max={errs.max():.4f}  mean={errs.mean():.4f}  "
              f"rounding bound={bound:.4f}  {'OK' if errs.max() <= bound + 1e-9 else 'FAIL'}")
        print("regimes: " + ", ".join(f"{k}={v} ({v / n:.1%})" for k, v in sorted(regimes.items())))
        return

    idx = args.jet if args.jet is not None else 0
    print(json.dumps(builder.build(df, idx), indent=2))


if __name__ == "__main__":
    main()
