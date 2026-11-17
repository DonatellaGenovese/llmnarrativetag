"""Stress-test summary: does each judge lower the targeted dimension when one
known defect is injected into a good narrative?

    python evaluation/results/judge_stress/analyze_stress.py

For each judge and perturbation, over the base narratives of both use cases:
  detected   share of bases where the targeted dimension scores LOWER on the
             perturbed copy than on the original
  drop       mean score drop of the targeted dimension
  other      mean drop of the two dimensions NOT targeted (collateral; near 0
             means the judge puts the defect where it belongs)
The contradiction probe has no target: the drop is shown for every dimension.
"""
import json
from collections import defaultdict
from pathlib import Path

D = Path(__file__).parent
DIMS = ["linguistic_realization", "physical_soundness", "overall_satisfaction"]
SHORT = {"linguistic_realization": "ling", "physical_soundness": "phys", "overall_satisfaction": "over"}
JUDGES = ["zai-glm-5-3", "muse-spark-1.3", "mistral-large-4"]
key = json.loads((D / "key.json").read_text())
KINDS = ["ling_garble", "phys_direction", "phys_wrong", "phys_overreach",
         "over_nodecision", "over_nodirection", "contradiction"]

for j in JUDGES:
    scores = {}
    for uc in ("top", "ad"):
        f = D / f"stress_{uc}_{j}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "verdict" in r:
                scores[r["id"]] = {d: r["verdict"][d]["score"] for d in DIMS}
    if not scores:
        continue
    orig = {k["base"]: scores.get(i) for i, k in key.items() if k["perturbation"] == "original"}
    print(f"\n=== {j}  ({len(scores)} verdicts)")
    o = [v for v in orig.values() if v]
    print("  originals mean: " + "  ".join(f"{SHORT[d]} {sum(x[d] for x in o) / len(o):.2f}" for d in DIMS))
    print(f"  {'perturbation':<18}{'target':<6}{'detected':>10}{'drop':>7}{'other':>8}   per-dim drop")
    total = defaultdict(list)
    for kind in KINDS:
        rows = [(k, scores[i], orig[k["base"]]) for i, k in key.items()
                if k["perturbation"] == kind and i in scores and orig.get(k["base"])]
        if not rows:
            continue
        per = {d: sum(o[d] - s[d] for _, s, o in rows) / len(rows) for d in DIMS}
        tgt = rows[0][0]["target"]
        if tgt:
            det = sum(s[tgt] < o[tgt] for _, s, o in rows) / len(rows)
            oth = sum(per[d] for d in DIMS if d != tgt) / 2
            total[tgt].append(det)
            print(f"  {kind:<18}{SHORT[tgt]:<6}{det:>9.0%}{per[tgt]:>7.2f}{oth:>8.2f}   "
                  + " ".join(f"{SHORT[d]} {per[d]:+.2f}" for d in DIMS) + f"   (n={len(rows)})")
        else:
            anyd = sum(any(s[d] < o[d] for d in DIMS) for _, s, o in rows) / len(rows)
            print(f"  {kind:<18}{'-':<6}{anyd:>9.0%}{'':>7}{'':>8}   "
                  + " ".join(f"{SHORT[d]} {per[d]:+.2f}" for d in DIMS) + f"   (n={len(rows)}, any dim lower)")
    print("  detection by dimension: " + "  ".join(
        f"{SHORT[d]} {sum(v) / len(v):.0%}" for d, v in total.items()))
