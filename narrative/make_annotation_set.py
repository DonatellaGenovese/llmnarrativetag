"""Build a blinded annotation set: narratives, a scoring sheet, and a key.

What the annotators receive is a directory of plain-text narratives named by an
opaque id, the glossary they are asked to check statements against, and a sheet
with one row per narrative in shuffled order. What they must not receive is
`KEY.csv`, which maps each id back to its model, block and jet.

Three choices worth stating, because each of them changes what the numbers mean.

  * Only narratives that passed verification are sampled. The pipeline delivers
    those and withholds the rest, so the quality of the delivered set is the
    operationally relevant quantity. Rates are therefore conditional on
    delivery, and should be reported that way.
  * Tags are stripped. They mark exactly the spans the pipeline already
    guarantees, so leaving them in would tell the annotator where to look and
    contaminate both the fluency and the licence judgements.
  * The sample is stratified across models, not drawn from the strongest one.
    A validation set that only contains good narratives cannot show whether a
    judge agrees with a human at the bottom of the scale, which is the half that
    matters. Five per model is enough to cover the range and far too few to
    compare models on it.

Each narrative is shipped with the artefact it was written from — the same JSON
the model received. It is there so the annotator can see what the writer had to
work with when judging whether the account hangs together and whether it earns
its conclusion. It is NOT there to be checked number by number: every tagged
number was already compared against this artefact deterministically, and asking
a human to redo that spends their attention on the one thing that is already
certain.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml

from .verifier import direction_word, rarity_band, recon_word, side_band

TAG_RE = re.compile(r"\[/?[A-Za-z][A-Za-z0-9_]*\]")
MODELS = ["gemini-3.5-flash", "deepseek-v4-flash", "gemini-3.5-flash-lite", "gpt-oss-120b"]
BLOCKS = ["v16", "b1", "b2", "b3", "b4", "b5"]
ITEMS = ["linguistic", "consistency", "glossary", "glossary_n_bad", "satisfaction"]


def strip_tags(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", TAG_RE.sub("", text)).strip()


def collect(runs: Path, blocks: List[str]) -> Dict[str, List[dict]]:
    """Accepted narratives per model, with the fields needed for stratifying."""
    out: Dict[str, List[dict]] = defaultdict(list)
    for model in MODELS:
        for blk in blocks:
            f = runs / f"{blk}_{model}.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["status"] != "ok":
                    continue
                out[model].append(
                    {
                        "model": model,
                        "block": blk,
                        "jet_id": r["jet_id"],
                        "decision": r["artefact"]["decision"],
                        "regime": r["regime"],
                        "residual": abs(r["meta"]["residual"]),
                        "artefact": r["artefact"],
                        "tagged": r["narrative"],
                        "text": strip_tags(r["narrative"]),
                    }
                )
    return out


FIELDS_LEGEND = """\
READING THE XAI OUTPUT
======================

Each explanation was written from one of these records, and from nothing else.

  decision             the class the classifier assigned to this jet
  other_class          the class it did not assign
  score_tagger         the classifier's score for this jet
  score_pct            % of jets in the same class that scored below this one
  score_reconstructed  the score the interpretable model reproduces
  recon_pct            % of jets reproduced LESS closely than this one, so a
                       high value means this jet is reproduced well
  intercept            the fixed baseline the contributions are added to

  features             one entry per observable, largest contribution first
    name               the observable
    value              its measured value on this jet
    phi                its contribution. Positive pushes towards `decision`,
                       negative towards `other_class`. Contributions pointing
                       both ways in the same jet are normal: the decision is the
                       sum, not a majority.
    pct                % of jets of `other_class` whose value is at or below
                       this one. It says WHERE the value sits, not how odd it is
    rarity             % of `other_class` lying further out in the same
                       direction. Small means unusual for that class. This is
                       the number that says HOW UNUSUAL

FOUR FIELDS ARE ALSO READ AS A WORD

The writer had to attach a word to four of these numbers, chosen from a closed
list by a rule it was given. It had no discretion: the value fixes the word.

  from `rarity`   2 or less -> extreme | above 2 up to 10 -> unusual
                  above 10 -> ordinary
  from `pct`      25 or less -> low | above 25 up to 75 -> mid
                  above 75 -> high
  from `phi`      zero or positive -> the class the classifier chose
                  negative -> the other class
  from `recon_pct`  50 or above -> better | below 50 -> worse

Each record below shows, beside the number, the word that value forces. Those
words have also already been checked and they are all correct.

The numbers in the explanation have already been checked against this record
automatically, and they match. Do not re-check either the numbers or these four
words: use the record to see what the writer had available, not to hunt for
errors that a program has already ruled out.
"""


SCALAR_OF = {"Z": "score_tagger", "ZP": "score_pct", "G": "score_reconstructed",
             "GP": "recon_pct", "B": "intercept"}
NUM_OF = {"V": "value", "I": "phi", "R": "rarity"}
WORD_OF = {"Q": "how unusual, from rarity", "S": "which side, from pct",
           "D": "which class, from the sign of phi"}

# field -> the tag that carries it, for the scalars. `recon_pct` earns two: the
# number itself and the word read off it.
SCALAR_TAG = {"score_tagger": "[Z]", "score_pct": "[ZP]",
              "score_reconstructed": "[G]",
              "recon_pct": "[GP]   and [GQ] better/worse",
              "intercept": "[B]"}


def annotate_record(art: dict, col: int = 34) -> str:
    """The record, with the tag that carries each field, and the word it forces.

    Four fields are read twice over: once as a number and once as a word from a
    closed vocabulary, fixed by a threshold the writer was given. Showing the
    word each value forces — computed here by the verifier's own functions, so
    it cannot drift from what the pipeline checked — makes it plain that the
    reading is not the writer's opinion and is not the annotator's to re-judge.

    `pct` is the one field with no tag of its own: it reaches the writer as the
    basis for a judgement, not as a quantity to quote, which is why an annotator
    will never find it in the prose.
    """
    dec, other = art["decision"], art["other_class"]
    feats = {f["i"]: f for f in art["features"]}
    lines = []
    for raw in json.dumps(art, indent=2).splitlines():
        key = re.match(r'\s*"([a-z_]+)":', raw)
        note = ""
        if key:
            k = key.group(1)
            if k == "decision":
                note = "no tag — appears in prose, and inside [D..]"
            elif k == "other_class":
                note = "no tag — but pct, rarity, [S..] and [Q..] are all"
            elif k == "recon_pct":
                note = f'[GP]   and [GQ] = "{recon_word(float(art[k]))}"'
            elif k in SCALAR_TAG:
                note = SCALAR_TAG[k]
            elif k in ("value", "phi", "rarity", "pct"):
                idx = next((re.search(r'"(\d+)"', l).group(1)
                            for l in reversed(lines) if '"i":' in l), "")
                f = feats.get(idx, {})
                if k == "value":
                    note = f"[V{idx}]"
                elif k == "phi":
                    w = direction_word(float(f["phi"]), dec, other)
                    note = f'[I{idx}]  and [D{idx}] = "{w}"'
                elif k == "rarity":
                    note = f'[R{idx}]  and [Q{idx}] = "{rarity_band(float(f[k]))}"'
                else:
                    note = f'no tag — fixes [S{idx}] = "{side_band(float(f[k]))}"'
        lines.append(raw if not note else f"{raw:<{col}}{note}")
        if key and key.group(1) == "other_class":
            lines.append(f"{'':<{col}}measured against THIS class, never the chosen one")
    return "\n".join(lines)


def worked_example(record: dict) -> str:
    """One explanation shown twice: as it will be scored, and as the checker saw it.

    The tagged form is the pipeline's own output, not a reconstruction, so the
    annotator can see exactly which spans were machine-verified. It appears once
    here and never on the items themselves: at one tag pair per ten words it is
    unreadable as prose, and fluency cannot be judged on a text no reader gets.
    """
    art = record["artefact"]
    clean = record["text"]
    tagged = re.sub(r"[ \t]{2,}", " ", record["tagged"]).strip()
    record_json = annotate_record(art)
    return f"""\
WORKED EXAMPLE — how to read an item
====================================

The same explanation, shown three ways: the record it was written from, the
text as you will score it, and the text as the automatic checker sees it.
Read this once before starting; the items themselves look like section 2.


1. THE RECORD THE WRITER WAS GIVEN
----------------------------------

{record_json}


2. THE EXPLANATION, AS YOU WILL SEE IT
--------------------------------------

{clean}


3. THE SAME EXPLANATION, AS THE CHECKER SEES IT
-----------------------------------------------

The writer was required to wrap every number from the record in a tag naming
the field it came from. A program then compares each tagged number against the
record. This is why you can take the numbers on trust.

  [Z] score_tagger        [ZP] score_pct        [B] intercept
  [G] score_reconstructed [GP] recon_pct        [GQ] better / worse
  [V<i>] value of observable i      [I<i>] its contribution
  [R<i>] its rarity                 [Q<i>] how unusual that rarity is
  [S<i>] which side of the other class      [D<i>] which class it points to

{tagged}

Notice what is NOT tagged: every sentence saying what an observable means. No
program checks those. They are yours.


4. WHAT THE WRITER WAS ALLOWED TO ADD
-------------------------------------

Everything in the explanation falls into exactly three kinds, and it is worth
knowing which is which before you score.

  Numbers        Copied from the record. Every one of them has already been
                 checked automatically and they all match. Ignore them.

  Judgement      The single words inside [Q], [S], [D] and [GQ] above: how
                 unusual a value is, which side of the other class it falls on,
                 which class a contribution points to, whether the jet is
                 reproduced better or worse than most. Each follows a fixed rule
                 the writer was given, and each has also already been checked.
                 Ignore them too.

  Physics        Everything else — the sentences saying what an observable
                 means. "the invariant mass of the whole jet, which concentrates
                 near the top mass for a hadronically decaying top quark".
                 THIS is the part no automatic check can verify, and the part
                 your glossary judgement is about. The writer was given a short
                 glossary entry per observable and told that it is the only
                 licence it has. Deciding whether it stayed inside
                 that licence is the glossary item.

The writer was also told what it must NOT do: attribute the decision to the
interpretable model rather than the classifier, compute anything, say what value
would have changed the outcome, mention the true label or whether the
classification was right, or describe a value with respect to the class that was
chosen — it was only told how the other class is distributed.
"""


def stratify(pool: List[dict], k: int, rng: random.Random) -> List[dict]:
    """Balance the decided class, and prefer to include the rarer regime.

    The sign convention behaves differently on the two classes and the caveat
    block changes the prose, so a sample that happens to miss either is not a
    sample of what the pipeline delivers.
    """
    by = defaultdict(list)
    for r in pool:
        by[(r["decision"], r["regime"])].append(r)
    for v in by.values():
        rng.shuffle(v)

    picked: List[dict] = []
    # One caveat narrative per class first, if any exist, then fill on class.
    for dec in ("top", "qcd"):
        if by[(dec, "caveat")]:
            picked.append(by[(dec, "caveat")].pop())
    for dec in ("top", "qcd"):
        want = k // 2 - sum(1 for r in picked if r["decision"] == dec)
        rest = by[(dec, "full")] + by[(dec, "caveat")]
        picked += rest[:max(0, want)]
    if len(picked) < k:  # a model with a thin pool
        seen = {(r["model"], r["jet_id"]) for r in picked}
        picked += [r for r in pool if (r["model"], r["jet_id"]) not in seen][: k - len(picked)]
    return picked[:k]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=repo / "narrative/configs/narrative_top.yaml")
    p.add_argument("--per-model", type=int, default=5)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--blocks", nargs="+", default=["v16"],
                   help="Blocks to draw from; default the original 100-jet sample only")
    p.add_argument("--out", type=Path, default=repo / "evaluation/annotation_set")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    runs = repo / cfg["outputs"]["narratives"]
    rng = random.Random(args.seed)

    pools = collect(runs, args.blocks)
    chosen: List[dict] = []
    for model in MODELS:
        if not pools[model]:
            print(f"  warning: no accepted narratives for {model}")
            continue
        got = stratify(pools[model], args.per_model, rng)
        if len(got) < args.per_model:
            print(f"  warning: {model} yielded {len(got)}/{args.per_model}")
        chosen += got

    rng.shuffle(chosen)
    ids = [f"N{i:02d}" for i in range(1, len(chosen) + 1)]

    out = args.out
    (out / "items").mkdir(parents=True, exist_ok=True)
    # The same twenty items a second time, tags intact. Same ids, so a row of the
    # scoring sheet refers to the same explanation in either directory. Hand out
    # one or the other, not both: whichever the annotators see, the LLM judge
    # must see too, or their agreement measures the difference in input as well
    # as the difference in judge.
    (out / "items_tagged").mkdir(parents=True, exist_ok=True)
    for nid, rec in zip(ids, chosen):
        art = annotate_record(rec["artefact"])
        head = (f"ITEM {nid}\n{'=' * (5 + len(nid))}\n\n"
                f"--- XAI OUTPUT (what the writer was given; see FIELDS.txt) ---\n"
                f"    the tag on the right is where that field appears in the text\n\n"
                f"{art}\n\n"
                f"--- EXPLANATION (what the writer produced) ---\n\n")
        (out / "items" / f"{nid}.txt").write_text(head + rec["text"] + "\n")
        tagged = re.sub(r"[ \t]{2,}", " ", rec["tagged"]).strip()
        (out / "items_tagged" / f"{nid}.txt").write_text(head + tagged + "\n")
    (out / "FIELDS.txt").write_text(FIELDS_LEGEND)

    # The instructions the writer worked under, verbatim. Reference material, not
    # required reading: an annotator who wants to know whether a guardrail held
    # can look it up, and the version id ties the set to a specific contract.
    pr = yaml.safe_load((repo / cfg["prompt"]["path"]).read_text())
    blocks = [f"PROMPT {pr['id']}", "=" * (7 + len(pr["id"])), "",
              "The instructions every explanation in this set was written under.",
              "Reference only — you do not need to read it to score.",
              "{glossary} was replaced by GLOSSARY.txt and {artefact_json} by the",
              "record at the top of each item.", "",
              "--- SYSTEM ---", "", pr["system"].strip(), "",
              "--- TASK ---", "", pr["task"].strip()]
    for name in ("remainder", "caveat"):
        if pr.get(name):
            blocks += ["", f"--- {name.upper()} (appended only when it applies) ---",
                       "", pr[name].strip()]
    (out / "PROMPT.txt").write_text("\n".join(blocks) + "\n")

    # Drawn from the sample itself so it cannot drift from what is being scored,
    # and from the model with the cleanest output so the example does not teach
    # a defect as if it were the norm.
    demo = next((r for r in chosen if r["model"] == "gemini-3.5-flash"), chosen[0])
    (out / "WORKED_EXAMPLE.txt").write_text(worked_example(demo))

    # The glossary the annotator checks statements against, as prose.
    gl = yaml.safe_load((repo / cfg["glossary"]["path"]).read_text())
    entries = gl.get("observables", gl)
    lines = ["OBSERVABLE GLOSSARY", "=" * 19, ""]
    for name, e in entries.items():
        lines += [f"{name}  —  {e['label']}" + (f" ({e['units']})" if e.get("units") else ""),
                  f"  Definition: {e['definition']}",
                  f"  Meaning:    {' '.join(e['meaning'].split())}", ""]
    (out / "GLOSSARY.txt").write_text("\n".join(lines))

    with open(out / "scoring_sheet.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", *ITEMS, "evidence_span", "notes"])
        for nid in ids:
            w.writerow([nid] + [""] * (len(ITEMS) + 2))

    # `regime` is a property of the jet, not of the model: it comes from the
    # residual |t - g_hat|, the part of the classifier's score the interpretable
    # model fails to reproduce, and is decided before any model is called. It is
    # spelled out here because it changes what the explanation was asked to do.
    REGIME_MEANING = {
        "full": "|r| < 1: the interpretable model reproduces the decision closely",
        "caveat": ("1 <= |r| < 2: it reproduces the decision only partly, and the "
                   "writer was told to say so up front and keep the rest brief"),
    }
    with open(out / "KEY.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "model", "block", "jet_id", "decision", "regime",
                    "residual", "regime_meaning", "words"])
        for nid, r in zip(ids, chosen):
            w.writerow([nid, r["model"], r["block"], r["jet_id"], r["decision"],
                        r["regime"], f"{r['residual']:.3f}",
                        REGIME_MEANING.get(r["regime"], ""), len(r["text"].split())])

    print(f"\n{len(chosen)} items from block(s) {', '.join(args.blocks)} -> {out}")
    print(f"  items/        one .txt per item: XAI record + explanation, blinded and shuffled")
    print(f"  items_tagged/ the same twenty, tags intact — same ids")
    print(f"  FIELDS.txt    how to read the XAI record")
    print(f"  PROMPT.txt    the instructions the writer worked under ({pr['id']})")
    print(f"  GLOSSARY.txt  the {len(entries)} entries statements are checked against")
    print(f"  scoring_sheet.csv   one row per item, to be filled per annotator")
    print(f"  KEY.csv       model/jet mapping — DO NOT give this to annotators")
    comp = defaultdict(int)
    for r in chosen:
        comp[(r["decision"], r["regime"])] += 1
    print("  composition: " + ", ".join(f"{d}/{g}={n}" for (d, g), n in sorted(comp.items())))


if __name__ == "__main__":
    main()
