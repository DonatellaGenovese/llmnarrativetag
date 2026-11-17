"""Pool the per-block runs into one table, and show the spread across blocks.

Every narrative is re-verified here rather than read from the `violations` field
the run recorded. The recorded field is whatever the verifier was on disk during
that run, and blocks generated weeks apart must be scored under one contract or
the pooled rate is not a rate of anything.

Two views of the same data, answering two different questions:

  pooled      one rate per model over all blocks, with a Wilson interval.
              This is the estimate.
  per block   the same rate computed inside each block of 100 jets. This is not
              a better estimate — disjoint blocks pooled are one larger sample —
              but it answers "is 100 enough?" by showing rather than arguing.

Blocks scatter by design: at a true rate of 0.96 with ~93 narrated jets per
block, the block-to-block standard deviation is 2 points, so blocks landing
between 0.91 and 1.00 is what agreement looks like, not disagreement.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Dict, List

import yaml

from .generator import load_glossary
from .verifier import verify

# Filled from the config's `generators` (or --models) in main().
MODELS: List[str] = []
FAMILIES = {
    "Faith": ("wrong_value",),
    "Comp": ("missing_tag",),
    "Form": ("malformed_tag", "unknown_tag"),
    "NoBare": ("untagged_number",),
    "Read": ("wrong_reading",),
}
# Anomaly detection only (`verifier.banned_terms` in the config): no particle,
# process or signal named. Added to FAMILIES in main() when the config has terms.
IDENT_FAMILY = ("Ident", ("identity_claim",))
BANNED: List[str] = []


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def pick_attempt(record: dict, which: str = "final"):
    """The attempt a report reads: `first`, what the model wrote unaided, or
    `final`, the last one made, which under the retry loop is the delivered
    narrative when one passed (the loop stops there) and the last rewrite when
    none did. With max_attempts 1 the two are the same. None if no attempt
    produced text."""
    atts = [a for a in record.get("attempts") or [] if "completion" in a]
    if not atts:
        return None
    if which == "first":
        return atts[0] if record["attempts"][0] is atts[0] else None
    return atts[-1]


def score_file(path: Path, glossary: dict, which: str = "final") -> Dict[int, dict]:
    """Per-jet family verdicts for one run, from a fresh verification."""
    out: Dict[int, dict] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if not r["attempts"]:
            continue  # abstained: no narrative was requested
        att = pick_attempt(r, which) or {}
        if "completion" not in att:
            # The call itself failed — a 429, a timeout — so there is no text to
            # judge. Counting it as a failure would charge the model for the
            # provider's quota; counting it as a pass is worse. It is excluded,
            # and `n` in the report drops accordingly.
            continue
        rep = verify(att["completion"]["text"], r["artefact"], glossary=glossary, banned_terms=BANNED)
        kinds = {v.kind for v in rep.violations}
        out[r["jet_id"]] = {f: not (kinds & set(ks)) for f, ks in FAMILIES.items()}
        out[r["jet_id"]]["TOTAL"] = rep.ok
    return out


def attempts_to_pass(path: Path, glossary: dict) -> Dict[int, int]:
    """Per narrated jet: the number of the first attempt that passes a fresh
    verification (1 = unaided), or 0 if none of the attempts made does. Jets
    whose call produced no text at all are left out, as in `score_file`."""
    out: Dict[int, int] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        atts = [a for a in r.get("attempts") or [] if "completion" in a]
        if not atts:
            continue
        out[r["jet_id"]] = 0
        for k, att in enumerate(atts, start=1):
            if verify(att["completion"]["text"], r["artefact"], glossary=glossary,
                      banned_terms=BANNED).ok:
                out[r["jet_id"]] = k
                break
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=repo / "narrative/configs/narrative_top.yaml")
    p.add_argument("--blocks", nargs="+", default=["v16"],
                   help="File prefixes to pool; v16 is the 100-jet sample the paper reports")
    # The extra blocks b1-b5 are kept outside git, in extra_blocks/, so pooling
    # them with v16 means reading two directories. First match wins.
    p.add_argument("--runs", type=Path, nargs="+", default=None,
                   help="Directories to search for <block>_<model>.jsonl "
                        "(default: the narratives dir from the config)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Run names to report (default: `generators` in the config). "
                        "The published v16: gemini-3.5-flash deepseek-v4-flash "
                        "gemini-3.5-flash-lite gpt-oss-120b")
    p.add_argument("--attempt", choices=["first", "final"], default="final",
                   help="first: the model unaided (the published v16 contract); "
                        "final: the delivered narrative after the retry loop")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    MODELS[:] = args.models or [g["name"] for g in cfg["generators"]]
    BANNED[:] = list((cfg.get("verifier") or {}).get("banned_terms") or [])
    if BANNED:
        FAMILIES[IDENT_FAMILY[0]] = IDENT_FAMILY[1]
    glossary = load_glossary(repo / cfg["glossary"]["path"], require_reviewed=True)
    run_dirs = args.runs or [repo / cfg["outputs"]["narratives"]]

    # model -> block -> {jet_id: verdicts}
    data: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for model in MODELS:
        for blk in args.blocks:
            for d in run_dirs:
                f = d / f"{blk}_{model}.jsonl"
                if f.exists():
                    data[model][blk] = score_file(f, glossary, args.attempt)
                    break

    present = [b for b in args.blocks if any(b in data[m] for m in MODELS)]
    print(f"blocks pooled: {', '.join(present)}   attempt: {args.attempt}\n")

    print("POOLED")
    head = "".join(f"{f:>8}" for f in list(FAMILIES) + ["TOTAL"])
    print(f"  {'model':<24}{head}{'95% CI':>16}{'n':>6}")
    pooled: Dict[str, dict] = {}
    for model in MODELS:
        merged = {}
        for blk, jets in data[model].items():
            merged.update({(blk, j): v for j, v in jets.items()})
        if not merged:
            continue
        pooled[model] = merged
        n = len(merged)
        row = "".join(
            f"{sum(v[f] for v in merged.values()) / n:>8.2f}" for f in list(FAMILIES) + ["TOTAL"]
        )
        k = sum(v["TOTAL"] for v in merged.values())
        lo, hi = wilson(k, n)
        print(f"  {model:<24}{row}   [{lo:.2f}, {hi:.2f}]{n:>6}")

    print("\nTOTAL PER BLOCK  (spread here is sampling noise, not disagreement)")
    print(f"  {'model':<24}" + "".join(f"{b:>8}" for b in present) + f"{'sd':>8}")
    for model in MODELS:
        if model not in pooled:
            continue
        rates = []
        cells = ""
        for blk in present:
            jets = data[model].get(blk)
            if not jets:
                cells += f"{'--':>8}"
                continue
            r = sum(v["TOTAL"] for v in jets.values()) / len(jets)
            rates.append(r)
            cells += f"{r:>8.2f}"
        sd = (
            math.sqrt(sum((x - sum(rates) / len(rates)) ** 2 for x in rates) / (len(rates) - 1))
            if len(rates) > 1
            else float("nan")
        )
        print(f"  {model:<24}{cells}{sd:>8.3f}")

    print("\nPAIRED COMPARISONS ON TOTAL  (exact McNemar over the shared jets)")
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            if a not in pooled or b not in pooled:
                continue
            keys = sorted(set(pooled[a]) & set(pooled[b]))
            x = sum(1 for k in keys if pooled[a][k]["TOTAL"] and not pooled[b][k]["TOTAL"])
            y = sum(1 for k in keys if not pooled[a][k]["TOTAL"] and pooled[b][k]["TOTAL"])
            pv = mcnemar(x, y)
            verdict = "significant" if pv < 0.05 else "not significant"
            print(f"  {a:<24} vs {b:<24}{x:>4}/{y:<4} n={len(keys):<4} p={pv:.4f}  {verdict}")

    # The cost of the refine loop: how many attempts a delivered narrative took.
    # Only under --attempt final, so the first-attempt report stays as it was.
    if args.attempt != "final":
        return
    max_att = int((cfg.get("llm") or {}).get("max_attempts", 5))
    print(f"\nATTEMPTS TO DELIVERY  (first attempt that passes, re-verified; "
          f"mean over the jets delivered within {max_att})")
    hist_head = "".join(f"{k:>5}" for k in range(1, max_att + 1))
    print(f"  {'model':<24}{'mean':>6}{'sd':>6}{'deliv':>7}{hist_head}{'fail':>6}")
    for model in MODELS:
        ks: List[int] = []
        for blk in present:
            for d in run_dirs:
                f = d / f"{blk}_{model}.jsonl"
                if f.exists():
                    ks += list(attempts_to_pass(f, glossary).values())
                    break
        if not ks:
            continue
        ok = [k for k in ks if k > 0]
        mean = sum(ok) / len(ok) if ok else float("nan")
        sd = (math.sqrt(sum((k - mean) ** 2 for k in ok) / (len(ok) - 1))
              if len(ok) > 1 else float("nan"))
        hist = "".join(f"{sum(1 for k in ok if k == i):>5}" for i in range(1, max_att + 1))
        print(f"  {model:<24}{mean:>6.2f}{sd:>6.2f}{len(ok):>4}/{len(ks):<3}{hist}"
              f"{sum(1 for k in ks if k == 0):>6}")


if __name__ == "__main__":
    main()
