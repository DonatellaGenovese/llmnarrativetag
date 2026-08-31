"""Agreement between the human annotators and the LLM judge, on the 20-item set.

Three tables, because the advice was to report the human scores for transparency
and a single aggregate agreement figure in the main text, and those are not the
same number:

    raters      mean +- SD per rater over the same 20 narratives, laid out like
                Table 2 of the main text. This is the transparency table.
    agreement   per dimension and pooled: raw agreement, quadratic-weighted
                Cohen's kappa, and negative recall. This is where the main-text
                figure comes from.
    models      the human scores split by generator. Five narratives per model:
                enough to cover the range of the scale, far too few to compare
                generators on it, and the caption has to say so.

Why kappa alone would misreport this set. Chance-corrected agreement needs
disagreement to correct against, and this scale is saturated: 95% of the human
ratings are 2 or 3, and Internal Consistency is 3 on all 40 human judgements, so
its kappa is 0/0 and is reported as n/a rather than as a number. The two
annotators score kappa_w 0.86 against each other and the judge scores 0.25
against them, but the judge also agrees with them on 95% of the positive/negative
collapse: both facts are true and only reporting both is honest. The pairing
follows Cohen and treats the two annotators as separate rating pairs (20 items x
2 annotators = 40 per dimension) rather than averaging them, since averaging two
integer raters produces half-point scores that no kappa is defined on.

Negative recall is the column that carries the finding: of the human ratings at
or below 1, the share the judge also placed at or below 1. It is 0 of 6, and it
is the reason the judge is usable as an aggregate baseline and not as a
substitute for reading an individual narrative.

    python -m evaluation.agreement            # all three tables
    python -m evaluation.agreement --latex    # the same, as LaTeX bodies
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# The only judge that ships. `--judge` remains so a re-scored set under another
# name can be read without editing this file.
MAIN_JUDGE = "claude-sonnet-5"
DIMENSIONS = [
    ("ling", "Linguistic-Realization", "Ling. Realization"),
    ("inte", "Internal-Consistency", "Int. Consistency"),
    ("over", "Overall-Satisfaction", "Overall Sat."),
]
# Order for the per-model table: the delivery-rate order of Table 1, so the two
# tables can be read against each other.
MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "deepseek-v4-flash", "gpt-oss-120b"]
POSITIVE = 2  # a rating of 2 or 3 is "positive"; 0 or 1 is "negative"


def kappa(a: Sequence[int], b: Sequence[int], quadratic: bool = False) -> Optional[float]:
    """Cohen's kappa, weighted or not. None when it is undefined.

    Undefined means the expected disagreement is zero — which happens here for
    real, not as an edge case: when both raters give 3 to every item there is
    nothing for chance correction to correct, and a kappa of 0 would read as
    "no better than chance" for a pair of raters that agreed perfectly.
    """
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return None
    idx = {c: i for i, c in enumerate(cats)}
    k, n = len(cats), len(a)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    rows = [sum(r) for r in obs]
    cols = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    span = cats[-1] - cats[0]
    if quadratic:
        w = [[((cats[i] - cats[j]) / span) ** 2 for j in range(k)] for i in range(k)]
    else:
        w = [[0 if i == j else 1 for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * rows[i] * cols[j] / n for i in range(k) for j in range(k))
    return None if den == 0 else 1 - num / den


def agree(human: List[int], other: List[int]) -> dict:
    n = len(human)
    hb = [int(x >= POSITIVE) for x in human]
    ob = [int(x >= POSITIVE) for x in other]
    neg = [i for i, x in enumerate(human) if x < POSITIVE]
    return {
        "n": n,
        "pos_h": 100 * sum(hb) / n,
        "pos_o": 100 * sum(ob) / n,
        "agr": 100 * sum(x == y for x, y in zip(hb, ob)) / n,
        "exact": 100 * sum(x == y for x, y in zip(human, other)) / n,
        "within1": 100 * sum(abs(x - y) <= 1 for x, y in zip(human, other)) / n,
        "kw": kappa(human, other, quadratic=True),
        "neg_n": len(neg),
        "neg_rec": (100 * sum(other[i] < POSITIVE for i in neg) / len(neg)) if neg else None,
        "delta": st.mean(o - h for h, o in zip(human, other)),
    }


def score_column(fields: Sequence[str], judge: str, dim: str) -> str:
    """The column holding one judge's score for one dimension.

    `judge_report.py` prefixes the columns with the judge name when it writes
    more than one judge and leaves them bare when there is only one. Both
    layouts appear on disk, and guessing wrong is a KeyError rather than a wrong
    number, so the choice is made from the header instead of from a convention.
    """
    if f"{judge}_{dim}" in fields:
        return f"{judge}_{dim}"
    if dim in fields:
        return dim
    raise SystemExit(
        f"judge_scores.csv has no column for judge {judge!r}, dimension {dim!r}; "
        f"columns are: {', '.join(fields)}"
    )


def load(root: Path):
    reader = csv.DictReader(open(root / "judge_scores.csv"))
    scores = {r["id"]: r for r in reader}
    fields = reader.fieldnames or []
    key = {r["id"]: r for r in csv.DictReader(open(root / "KEY.csv"))}
    humans = {
        f.stem.replace("human_", ""): {r["id"]: r for r in csv.DictReader(open(f))}
        for f in sorted(root.glob("human_*.csv"))
    }
    if not humans:
        raise SystemExit(f"no human_*.csv in {root}")
    return scores, key, humans, fields


def fmt(v: Optional[float], nd: int = 2, dagger: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}" + ("†" if dagger else "")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", type=Path, default=repo / "evaluation/annotation_set")
    p.add_argument("--judge", default=MAIN_JUDGE)
    p.add_argument("--latex", action="store_true", help="Emit LaTeX table bodies too")
    args = p.parse_args()

    scores, key, humans, fields = load(args.set)
    ids = sorted(scores)
    names = sorted(humans)
    cols = {d: score_column(fields, args.judge, d) for d, _, _ in DIMENSIONS}
    jcol = lambda i, d: int(scores[i][cols[d]])

    # ---------------------------------------------------------------- raters
    print(f"RATERS — mean ± SD over the same {len(ids)} narratives  (judge: {args.judge})")
    print(f"  {'Rater':<22}{'N':>4}  " + "".join(f"{lab:<18}" for _, _, lab in DIMENSIONS))
    rater_rows = []
    for lab, get in (
        *[(f"Annotator {chr(65 + k)}", (lambda n: lambda i, d, c: int(humans[n][i][c]))(n))
          for k, n in enumerate(names)],
        ("Human mean", lambda i, d, c: st.mean(int(humans[n][i][c]) for n in names)),
        (f"LLM judge", lambda i, d, c: jcol(i, d)),
    ):
        cells = []
        for d, col, _ in DIMENSIONS:
            v = [get(i, d, col) for i in ids]
            cells.append(f"{st.mean(v):.2f} ± {st.stdev(v):.2f}")
        rater_rows.append((lab, cells))
        print(f"  {lab:<22}{len(ids):>4}  " + "".join(f"{c:<18}" for c in cells))

    # ------------------------------------------------------------- agreement
    # Pairs, not averages: each annotator against the judge on each item.
    print(f"\nAGREEMENT — judge vs the {len(names)} annotators, "
          f"{len(ids)}x{len(names)} rating pairs per dimension")
    head = (f"  {'Comparison':<30}{'n':>4}{'%pos H':>8}{'%pos J':>8}"
            f"{'Agr':>6}{'exact':>7}{'±1':>6}{'κ_w':>8}{'Neg.rec.':>12}{'Δ':>7}")
    print(head)
    agr_rows = []

    def line(lab: str, human: List[int], other: List[int]) -> None:
        s = agree(human, other)
        # The dagger marks a row with fewer than three negative human ratings:
        # kappa there has almost no disagreement to correct against and must not
        # be read as evidence of poor agreement.
        dag = s["neg_n"] < 3
        nr = "n/a" if s["neg_rec"] is None else f"{s['neg_rec']:.0f}% ({round(s['neg_rec']*s['neg_n']/100)}/{s['neg_n']})"
        agr_rows.append((lab, s, dag, nr))
        print(f"  {lab:<30}{s['n']:>4}{s['pos_h']:>8.0f}{s['pos_o']:>8.0f}{s['agr']:>6.0f}"
              f"{s['exact']:>7.0f}{s['within1']:>6.0f}{fmt(s['kw'], 2, dag):>8}{nr:>12}{s['delta']:>+7.2f}")

    pooled_h, pooled_j = [], []
    for d, col, lab in DIMENSIONS:
        h = [int(humans[n][i][col]) for i in ids for n in names]
        j = [jcol(i, d) for i in ids for _ in names]
        pooled_h += h
        pooled_j += j
        line(lab, h, j)
    line("ALL (3 dimensions pooled)", pooled_h, pooled_j)

    # The ceiling. Judge-human agreement means nothing without the number two
    # people reach on the same items with the same rubric.
    if len(names) == 2:
        print()
        ph, pj = [], []
        for d, col, lab in DIMENSIONS:
            a = [int(humans[names[0]][i][col]) for i in ids]
            b = [int(humans[names[1]][i][col]) for i in ids]
            ph += a
            pj += b
            line(f"human–human · {lab}", a, b)
        line("human–human · ALL (ceiling)", ph, pj)

    # -------------------------------------------------------------- by model
    print(f"\nBY GENERATOR — human scores, {len(ids)//len(MODELS)} narratives per model"
          f"  (covers the scale; too few to rank generators)")
    print(f"  {'Model':<24}{'N':>3}  " + "".join(f"{lab:<18}" for _, _, lab in DIMENSIONS))
    model_rows = []
    for m in MODELS:
        sub = [i for i in ids if key[i]["model"] == m]
        if not sub:
            continue
        cells = []
        for d, col, _ in DIMENSIONS:
            v = [int(humans[n][i][col]) for i in sub for n in names]
            cells.append(f"{st.mean(v):.2f} ± {st.stdev(v):.2f}")
        model_rows.append((m, len(sub), cells))
        print(f"  {m:<24}{len(sub):>3}  " + "".join(f"{c:<18}" for c in cells))

    # --------------------------------------------------------- divergences
    print("\nRATING PAIRS DIVERGING BY 2 OR MORE")
    big = [(i, lab, n, int(humans[n][i][col]), jcol(i, d))
           for i in ids for d, col, lab in DIMENSIONS for n in names
           if abs(jcol(i, d) - int(humans[n][i][col])) >= 2]
    for i, lab, n, h, j in big:
        print(f"  {i}  {lab:<20}{key[i]['model']:<24}human({n[:3]})={h}  judge={j}")
    print(f"  {len(big)} of {len(ids)*len(DIMENSIONS)*len(names)} pairs "
          f"({100*len(big)/(len(ids)*len(DIMENSIONS)*len(names)):.0f}%)")

    if args.latex:
        # Built outside the f-strings: Python 3.10 rejects a backslash inside an
        # f-string expression, and every one of these substitutions needs one.
        pm = lambda cells: " & ".join("$" + c.replace("±", r"\pm") + "$" for c in cells)
        print("\n% ---- raters ----")
        for lab, cells in rater_rows:
            print(f"{lab} & {len(ids)} & " + pm(cells) + r" \\")
        print("\n% ---- agreement ----")
        for lab, s, dag, nr in agr_rows:
            kw = fmt(s["kw"], 2).replace("n/a", "--")
            if dag and s["kw"] is not None:
                kw += r"$^\dagger$"
            print(f"{lab} & {s['n']} & {s['pos_h']:.0f} & {s['pos_o']:.0f} & {s['agr']:.0f} & "
                  f"{s['within1']:.0f} & {kw} & {nr.replace('%', chr(92) + '%')} & "
                  f"{s['delta']:+.2f}" + r" \\")
        print("\n% ---- by generator ----")
        for m, n, cells in model_rows:
            print(f"{m} & {n} & " + pm(cells) + r" \\")


if __name__ == "__main__":
    main()
