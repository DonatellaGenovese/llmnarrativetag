"""Judge verdicts for the human-annotation sample, as a spreadsheet.

The twenty items in `annotation_set/` were drawn from the same pool the judge
scored — the narratives of block v16 that passed verification — so every one of
them already has a machine verdict. This writes those verdicts out beside the
provenance the annotators do not see, ready to be joined against their sheets
once they are filled in.

Two files, because they answer different questions and one of them is
unreadable in a spreadsheet:

    judge_scores.csv    one row per item, one column per dimension
    judge_reasons.csv   the same verdicts with the justification text

Neither aggregates over generators: five narratives per generator is far too few
to compare generators, and a column that invited that comparison would be read
as one.

`--by-model` is a separate report over a different population — every narrative
the judge scored, not the twenty sampled for the annotators — where the counts
are large enough to mean something. It writes nothing; it prints the table, and
`--latex` emits the body.

    python -m evaluation.judge_report              # the two CSVs
    python -m evaluation.judge_report --by-model   # mean +- SD per generator
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path
from typing import Dict, List

DIMENSIONS = ("linguistic_realization", "internal_consistency", "overall_satisfaction")
SHORT = {"linguistic_realization": "ling", "internal_consistency": "inte",
         "overall_satisfaction": "over"}
LABEL = {"linguistic_realization": "Ling. Realization",
         "internal_consistency": "Int. Consistency",
         "overall_satisfaction": "Overall Sat."}
# Delivery-rate order, so this table can be read against the pass-rate table.
MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "deepseek-v4-flash", "gpt-oss-120b"]


def judge_key(model: str, jet_id) -> str:
    """The id `judge.py --from-runs` assigns, rebuilt from the annotation key."""
    short = model.replace("gemini-3.5-", "g").replace("deepseek-v4-", "d").replace("gpt-oss-", "o")
    return f"{short}-{jet_id}"


def by_model(verdicts: Dict[str, dict], latex: bool) -> None:
    """Mean +- SD per generator over every narrative the judge scored.

    Conditional on passing verification, since only accepted narratives are
    judged, so the N differ by an order of magnitude between generators and the
    columns are not directly comparable. Printing N beside them is what stops
    the table being read as a ranking.
    """
    scores: Dict[str, Dict[str, List[int]]] = {
        m: {d: [] for d in DIMENSIONS} for m in MODELS
    }
    for r in verdicts.values():
        m = r.get("model")
        if m not in scores:          # a judged item outside the four generators
            continue
        for d in DIMENSIONS:
            scores[m][d].append(r["verdict"][d]["score"])

    def cell(v: List[int]) -> str:
        return f"{st.mean(v):.2f} ± {st.stdev(v) if len(v) > 1 else 0.0:.2f}"

    print(f"\n{'Model':<24} {'N':>3}  " + "  ".join(f"{LABEL[d]:<17}" for d in DIMENSIONS))
    for m in MODELS:
        v = scores[m][DIMENSIONS[0]]
        if not v:
            print(f"{m:<24} {'-':>3}   (no verdicts)")
            continue
        print(f"{m:<24} {len(v):>3}  "
              + "  ".join(f"{cell(scores[m][d]):<17}" for d in DIMENSIONS))

    if latex:
        print("\n% LaTeX body")
        for m in MODELS:
            v = scores[m][DIMENSIONS[0]]
            if not v:
                continue
            cells = " & ".join(cell(scores[m][d]).replace("±", "$\\pm$") for d in DIMENSIONS)
            print(f"{m} & {len(v)} & {cells} \\\\")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", type=Path, default=repo / "evaluation/annotation_set")
    p.add_argument("--block", default="v16")
    p.add_argument("--judge", default="claude-sonnet-5")
    p.add_argument("--out", type=Path, default=repo / "evaluation/annotation_set")
    p.add_argument("--by-model", action="store_true",
                   help="Per-generator means over every judged narrative, not the 20-item set")
    p.add_argument("--latex", action="store_true", help="With --by-model, emit the LaTeX body too")
    args = p.parse_args()

    f = repo / "evaluation" / f"judgments_{args.block}_{args.judge}.jsonl"
    if not f.exists():
        raise SystemExit(f"not found: {f}")
    verdicts: Dict[str, dict] = {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if "verdict" in r:
            verdicts[r["id"]] = r

    if args.by_model:
        print(f"judge:  {args.judge}   block: {args.block}   "
              f"{len(verdicts)} narratives judged")
        by_model(verdicts, args.latex)
        return

    key_rows = list(csv.DictReader(open(args.set / "KEY.csv")))
    missing = []

    scores_path = args.out / "judge_scores.csv"
    reasons_path = args.out / "judge_reasons.csv"
    with open(scores_path, "w", newline="") as fh, open(reasons_path, "w", newline="") as rh:
        w = csv.writer(fh)
        w.writerow(["id", "model", "jet_id", "decision", "regime", "words"]
                   + [SHORT[d] for d in DIMENSIONS] + ["n_below_3"])

        rw = csv.writer(rh)
        rw.writerow(["id", "model", "dimension", "score", "reason", "evidence"])

        for k in key_rows:
            row = [k["id"], k["model"], k["jet_id"], k["decision"], k["regime"], k.get("words", "")]
            rec = verdicts.get(judge_key(k["model"], k["jet_id"]))
            if rec is None:
                missing.append(k["id"])
                w.writerow(row + ["", "", "", ""])
                continue
            scores = []
            for d in DIMENSIONS:
                s = rec["verdict"][d]["score"]
                scores.append(s)
                rw.writerow([k["id"], k["model"], SHORT[d], s,
                             rec["verdict"][d].get("reason", ""),
                             rec["verdict"][d].get("evidence", "")])
            w.writerow(row + scores + [sum(1 for s in scores if s < 3)])

    print(f"judge:  {args.judge}")
    print(f"items:  {len(key_rows)}" + (f"   MISSING: {missing}" if missing else ""))
    print(f"\nwrote {scores_path}")
    print(f"wrote {reasons_path}")


if __name__ == "__main__":
    main()
