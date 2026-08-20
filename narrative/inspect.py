"""Render a run's JSONL as something a person can read.

Manual review is where the interpretive layer gets checked — whether the prose
around a number reads it correctly, whether the physics stays inside the
glossary, whether the guardrails held. None of that is visible in a 6 kB JSON
object on one line, so this prints the artefact as a table with the narrative
beside it, which is the layout that makes a misreading obvious.

    python -m narrative.inspect run100_gemini-3.5-flash-lite.jsonl
    python -m narrative.inspect run100_*.jsonl --jet 11133     # same jet, every model
    python -m narrative.inspect run100_deepseek-v4-flash.jsonl --failed
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Dict, List

RULE = "=" * 78


def load(path: Path) -> List[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def render_surrogate(rec: dict) -> str:
    """What the surrogate produced, before anything was hidden or rounded.

    Everything here comes from the record itself, so the model does not have to
    be reloaded. The three quantities the artefact deliberately withholds — the
    teacher score, the reconstruction and the residual between them — are the
    reason this block exists: they are what decides whether the narrative was
    entitled to be written at all.
    """
    a, meta = rec["artefact"], rec.get("meta", {})
    sign = "+1 (as fitted)" if a["decision"] == "top" else "-1 (flipped towards qcd)"
    parts = a["intercept"] + sum(f["phi"] for f in a["features"])
    lines = [
        f"  t      (ParT logit)        {meta.get('t_raw', float('nan')):+8.4f}"
        + ("   [clipped]" if meta.get("t_clipped") else ""),
        f"  g_hat  (surrogate)         {meta.get('g_hat', float('nan')):+8.4f}",
        f"  r = t - g_hat              {meta.get('residual', float('nan')):+8.4f}"
        f"   -> regime '{meta.get('regime')}'",
        "",
        f"  orientation towards `{a['decision']}`: {sign}",
        f"  additive identity: intercept {a['intercept']:+.2f} + sum(phi) "
        f"= {parts:+.2f}  vs score_reconstructed {a['score_reconstructed']:+.2f}"
        + ("" if abs(parts - a["score_reconstructed"]) < 0.06 else "   <-- MISMATCH"),
    ]
    return "\n".join(lines)


def render_artefact(a: dict) -> str:
    """The numbers the model was given, in the order it was told to use them."""
    head = (
        f"decision {a['decision']} (other: {a['other_class']})   "
        f"score {a['score_tagger']}   median {a['score_median']}   pct {a['score_pct']}\n"
        f"reconstructed {a['score_reconstructed']}   recon_pct {a['recon_pct']}   "
        f"intercept {a['intercept']}"
    )
    rows = ["", f"  {'i':>2}  {'name':<14} {'value':>10} {'phi':>8} {'pct':>7} {'rarity':>7}"]
    for f in a["features"]:
        rows.append(
            f"  {f.get('i', '?'):>2}  {f['name']:<14} {f['value']:>10} {f['phi']:>8}"
            f" {f.get('pct', '-'):>7} {f.get('rarity', '-'):>7}"
        )
    if "n_other" in a:
        rows.append(f"  remainder: n_other {a['n_other']}, phi_other {a['phi_other']}")
    return head + "\n" + "\n".join(rows)


def render_record(rec: dict, source: str, show_failed: bool) -> str:
    out = [RULE, f"jet {rec['jet_id']}   regime {rec['regime']}   status {rec['status']}   [{source}]"]

    if rec["status"] == "abstained":
        out.append("\n" + textwrap.fill(rec.get("abstention_reason", ""), 78))
        return "\n".join(out)

    out.append("")
    out.append("-- [1] surrogate, not shown to the model " + "-" * 36)
    out.append(render_surrogate(rec))
    out.append("")
    out.append("-- [2] artefact, the model's entire input " + "-" * 35)
    out.append(render_artefact(rec["artefact"]))
    out.append("")

    if rec.get("narrative"):
        out.append("-- [3] narrative " + "-" * 60)
        out.append("\n".join(textwrap.fill(p, 78) for p in rec["narrative"].split("\n")))
    elif show_failed:
        for att in rec["attempts"]:
            kinds: Dict[str, int] = {}
            for v in att.get("violations", []):
                kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
            out.append(f"-- attempt {att['attempt']} REJECTED: {kinds} " + "-" * 30)
            for v in att.get("violations", [])[:8]:
                out.append(f"     {v['kind']}: {v['detail'][:100]}")
            text = (att.get("completion") or {}).get("text", "")
            out.append("")
            out.append("\n".join(textwrap.fill(p, 78) for p in text.split("\n")))
    else:
        n = len(rec["attempts"])
        out.append(f"(rejected after {n} attempt{'s' if n != 1 else ''}; --failed to see them)")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="One or more run JSONL files")
    p.add_argument("--jet", type=int, default=None, help="Only this jet_id, across every run given")
    p.add_argument("--failed", action="store_true", help="Show rejected attempts and their violations")
    p.add_argument("--limit", type=int, default=None, help="Stop after this many records per run")
    p.add_argument("--out", type=Path, default=None, help="Write to a file instead of stdout")
    args = p.parse_args()

    blocks: List[str] = []
    for path in args.runs:
        records = load(path)
        if args.jet is not None:
            records = [r for r in records if r["jet_id"] == args.jet]
        elif not args.failed:
            records = [r for r in records if r["status"] == "ok"]
        else:
            records = [r for r in records if r["status"] == "failed_verification"]
        if args.limit:
            records = records[: args.limit]
        for rec in records:
            blocks.append(render_record(rec, path.stem.replace("run100_", ""), args.failed))

    text = "\n\n".join(blocks) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"{len(blocks)} records → {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
