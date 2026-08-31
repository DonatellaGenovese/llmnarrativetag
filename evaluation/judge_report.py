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

Nothing here is an aggregate over models: five narratives per generator is far
too few to compare generators, and a column that invites that comparison would
be read as one.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

DIMENSIONS = ("linguistic_realization", "internal_consistency", "overall_satisfaction")
SHORT = {"linguistic_realization": "ling", "internal_consistency": "inte",
         "overall_satisfaction": "over"}


def judge_key(model: str, jet_id) -> str:
    """The id `judge.py --from-runs` assigns, rebuilt from the annotation key."""
    short = model.replace("gemini-3.5-", "g").replace("deepseek-v4-", "d").replace("gpt-oss-", "o")
    return f"{short}-{jet_id}"


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", type=Path, default=repo / "evaluation/annotation_set")
    p.add_argument("--block", default="v16")
    p.add_argument("--judge", default="claude-sonnet-5")
    p.add_argument("--out", type=Path, default=repo / "evaluation/annotation_set")
    args = p.parse_args()

    f = repo / "evaluation" / f"judgments_{args.block}_{args.judge}.jsonl"
    if not f.exists():
        raise SystemExit(f"not found: {f}")
    verdicts: Dict[str, dict] = {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if "verdict" in r:
            verdicts[r["id"]] = r

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
