"""Summary of the judge test (judge-v3 / judge-ad-v1, 42 narratives, top-3 generators).

    python evaluation/results/judge_test/analyze.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).parent
DIMS = ["linguistic_realization", "physical_soundness", "overall_satisfaction"]
JUDGES = ["zai-glm-5-3", "mistral-large-4", "muse-spark-1.3"]
sel = json.loads((D / "selection.json").read_text())

for uc in ("top", "ad"):
    print(f"\n==================== {uc.upper()} ====================")
    scored = {}
    for j in JUDGES:
        f = D / f"{uc}_{j}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.read_text().splitlines()]
        ok = [r for r in recs if "verdict" in r]
        scored[j] = {r["id"]: r for r in ok}
        print(f"\n--- {j}: {len(ok)} verdicts, {len(recs) - len(ok)} failed")
        for d in DIMS:
            s = [r["verdict"][d]["score"] for r in ok]
            c = Counter(s)
            print(f"  {d:<24} mean {sum(s) / len(s):.2f}   share of 3: {c[3] / len(s):.0%}   "
                  f"dist 0/1/2/3: {c[0]}/{c[1]}/{c[2]}/{c[3]}")
        by = defaultdict(lambda: defaultdict(list))
        for r in ok:
            for d in DIMS:
                by[r["model"]][d].append(r["verdict"][d]["score"])
        for m, v in by.items():
            print(f"    {m:<22}" + "".join(f"  {d[:4]} {sum(x) / len(x):.2f}" for d, x in v.items()))
        toks = [r.get("usage", {}) for r in ok]
        tin = sum(t.get("in") or 0 for t in toks) / len(ok)
        tout = sum(t.get("out") or 0 for t in toks) / len(ok)
        print(f"    tokens per narrative: in {tin:.0f}, out {tout:.0f}")
    from itertools import combinations
    for a, b in combinations(list(scored), 2):
        ids = sorted(set(scored[a]) & set(scored[b]))
        print(f"\n--- agreement {a} vs {b} on {len(ids)} narratives")
        for d in DIMS:
            sa = [scored[a][i]["verdict"][d]["score"] for i in ids]
            sb = [scored[b][i]["verdict"][d]["score"] for i in ids]
            exact = sum(x == y for x, y in zip(sa, sb)) / len(ids)
            within1 = sum(abs(x - y) <= 1 for x, y in zip(sa, sb)) / len(ids)
            print(f"  {d:<24} exact {exact:.0%}  within 1: {within1:.0%}  "
                  f"mean diff ({a[:4]} - {b[:4]}) {(sum(sa) - sum(sb)) / len(ids):+.2f}")
    print("\n--- scores below 2, with evidence")
    for j, rs in scored.items():
        for i, r in sorted(rs.items()):
            for d in DIMS:
                v = r["verdict"][d]
                if v["score"] < 2:
                    print(f"  [{j[:8]}] {i} {d[:4]}={v['score']}: {v.get('reason', '')[:220]}")
                    if v.get("evidence"):
                        print(f"        evidence: {v['evidence'][:220]}")
