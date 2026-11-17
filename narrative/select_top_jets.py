"""The jets the top-tagging narratives are written for, stratified by the
tagger's decision and whether it is right.

A random sample (run100_jets.json, used up to v17) mirrors the test set: with
the tagger at 94% accuracy, 500 random jets hold ~30 mistakes, too few to say
anything about how the narratives explain a wrong decision. Here the sample is
drawn per cell of decision x correctness:

    top_correct   tagged top, a top jet
    qcd_correct   tagged qcd, a QCD jet
    top_wrong     tagged top, a QCD jet  (false positive)
    qcd_wrong     tagged qcd, a top jet  (missed top)

Mistakes are over-sampled by design (DESIGN). The decision is the sign of the
teacher logit, as in the artefact. The truth label is used here, for the design
and the analysis, and nowhere else: the artefact never carries it.

The gate (narrative/gate_top.py) is applied later by the orchestrator and
abstains far more often on some cells than on others (test set, seed-42
teacher: 2% / 7% / 13% / 49% for the four cells above), so the narrated
sample is less balanced than the drawn one; the expected counts are printed.

Jets already used (the files given with --exclude, by default the v17 sample)
and jets with a missing observable are left out.

    python -m narrative.select_top_jets
    python -m narrative.select_top_jets --n 500 --out narrative/outputs/top/narratives/top500_jets.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .artefact import load_builder

# Shares of the sample per cell: 60% correct decisions, 40% mistakes.
DESIGN = {"top_correct": 0.30, "qcd_correct": 0.30, "top_wrong": 0.20, "qcd_wrong": 0.20}
TRUTH = {0: "top", 1: "qcd"}        # the feature tables' convention (label 1 = QCD)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=repo / "narrative/configs/narrative_top.yaml")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--exclude", type=Path, nargs="*",
                   default=[repo / "narrative/outputs/top/narratives/run100_jets.json"])
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    out = a.out or repo / cfg["outputs"]["narratives"] / f"top{a.n}_jets.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the sample is the unit of comparison, delete it on purpose to redraw")

    builder, cfg, df = load_builder(repo, a.config)
    obs = list(builder.obs)
    ok = ~df[obs].isna().any(axis=1).to_numpy()
    excluded = set()
    for f in a.exclude:
        excluded |= set(json.loads(Path(f).read_text())["jets"])
    ok[list(excluded)] = False

    t = df["t"].to_numpy(np.float64)
    decision = np.where(t > 0, "top", "qcd")
    truth = np.vectorize(TRUTH.get)(df["label"].to_numpy())
    cell = np.char.add(decision.astype(str), np.where(decision == truth, "_correct", "_wrong"))

    # Integer counts summing to n, largest remainder.
    raw = {c: a.n * s for c, s in DESIGN.items()}
    want = {c: int(v) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: raw[c] - want[c], reverse=True)[: a.n - sum(want.values())]:
        want[c] += 1

    rng = np.random.default_rng(a.seed)
    rows, strata, selection = [], {}, {}
    for c, n in want.items():
        pool = np.flatnonzero(ok & (cell == c))
        if len(pool) < n:
            raise SystemExit(f"cell {c}: {len(pool)} jets available, {n} wanted")
        selection[c] = {"taken": n, "available": int(len(pool))}
        for i in sorted(rng.choice(pool, size=n, replace=False).tolist()):
            rows.append(int(i))
            strata[str(i)] = {"row": int(i), "jet_id": int(df["jet_id"].iloc[i]), "cell": c,
                              "decision": str(decision[i]), "truth": str(truth[i]), "t": round(float(t[i]), 4)}

    # Expected narrated jets: the regime of every drawn jet under the current gate.
    regimes = {}
    for i in rows:
        reg = builder.build(df, i)["meta"]["regime"]
        strata[str(i)]["regime"] = reg
        regimes.setdefault(strata[str(i)]["cell"], {}).setdefault(reg, 0)
        regimes[strata[str(i)]["cell"]][reg] += 1

    doc = {
        "n": a.n,
        "seed": a.seed,
        "n_rows": int(len(df)),
        "design": want,
        "selection": selection,
        "excludes": [str(Path(f).relative_to(repo)) if Path(f).is_absolute() else str(f) for f in a.exclude],
        "source": {"features": cfg["inputs"]["features"], "gate": cfg["abstention"].get("gate")},
        "regimes_at_selection": regimes,
        "jets": rows,
        "strata": strata,
    }
    out.write_text(json.dumps(doc, indent=1))

    print(f"{len(rows)} jets -> {out}")
    for c in want:
        r = regimes.get(c, {})
        print(f"  {c:<12} taken {want[c]:>4} of {selection[c]['available']:>6}   "
              f"full {r.get('full', 0):>3}  caveat {r.get('caveat', 0):>3}  abstain {r.get('abstain', 0):>3}")
    narrated = sum(v for r in regimes.values() for k, v in r.items() if k != "abstain")
    print(f"  narrated (not abstained): {narrated}")


if __name__ == "__main__":
    main()
