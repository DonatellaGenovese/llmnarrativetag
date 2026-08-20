"""Aggregate a narrative run into the verification metrics table.

Reads the JSONL written by `narrative.orchestrator` and reports pass rates per
constraint family, so a run is summarised the same way every time instead of by
whatever ad-hoc script was to hand.

Two columns, both over the jets that were actually narrated (abstentions are
reported separately, since no narrative was requested for them):

  single-pass  the FIRST attempt only — what a run without rejection sampling
               would have produced, i.e. the ablation baseline.
  final        the accepted narrative, or, for jets that never passed, the LAST
               attempt tried — what the pipeline would actually have handed over.

`final` is not "the best attempt": taking the minimum-violation attempt would
flatter the loop by picking with hindsight.

Only the regex layer is measured here. Whether the prose reads its own numbers
correctly is not a regex question and is not counted — see `narrative/README.md`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

STATUS_OK = "ok"
STATUS_ABSTAINED = "abstained"

# USD per 1M tokens, list price. Reasoning tokens bill as output on all three,
# which is what makes gemini-2.5-pro expensive here: it cannot be told to stop
# thinking, and its output rate is 25x flash-lite's.
PRICES = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.6-flash": (1.50, 7.50),
}


def estimate_cost(completions: List[dict]) -> Optional[float]:
    """USD for a run, or None if any model has no rate on file."""
    total = 0.0
    for c in completions:
        rate = PRICES.get(c["model"])
        if rate is None:
            return None
        u = c["usage"]
        out = (u.get("candidates_token_count", 0) or 0) + (u.get("thoughts_token_count", 0) or 0)
        total += (u.get("prompt_token_count", 0) or 0) / 1e6 * rate[0] + out / 1e6 * rate[1]
    return total

# Constraint families, as sets of violation kinds that must be absent.
FAMILIES: Dict[str, tuple] = {
    "Faith": ("wrong_value",),
    "Comp": ("missing_tag",),
    "Form": ("malformed_tag", "unknown_tag"),
    "NoBare": ("untagged_number",),
    # The reading, not the number: whether the word attached to a value says
    # what the value says. Empty on runs made before prompt top-v8.
    "Read": ("wrong_reading",),
}


def load_run(path: Path) -> List[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _kinds(attempt: dict) -> set:
    return {v["kind"] for v in attempt.get("violations", [])}


def _satisfies(attempt: dict, family: Optional[str]) -> bool:
    """True if the attempt violates nothing in `family` (or nothing at all)."""
    kinds = _kinds(attempt)
    if "error" in attempt:  # generation failed; nothing to check
        return False
    if family is None:
        return not kinds
    return not (set(FAMILIES[family]) & kinds)


def _first(record: dict) -> Optional[dict]:
    return record["attempts"][0] if record["attempts"] else None


def _final(record: dict) -> Optional[dict]:
    """The attempt the pipeline would have handed over."""
    for attempt in record["attempts"]:
        if attempt.get("ok"):
            return attempt
    return record["attempts"][-1] if record["attempts"] else None


def summarise(records: List[dict]) -> dict:
    narrated = [r for r in records if r["status"] != STATUS_ABSTAINED]
    abstained = [r for r in records if r["status"] == STATUS_ABSTAINED]

    def rate(pick: Callable[[dict], Optional[dict]], family: Optional[str]) -> Optional[float]:
        if not narrated:
            return None
        hits = sum(1 for r in narrated if (a := pick(r)) and _satisfies(a, family))
        return hits / len(narrated)

    table = {}
    for name in list(FAMILIES) + ["TOTAL"]:
        family = None if name == "TOTAL" else name
        table[name] = {
            "single_pass": rate(_first, family),
            "final": rate(_final, family),
        }

    accepted = [r for r in narrated if r["status"] == STATUS_OK]
    steps = [len(r["attempts"]) for r in accepted]
    violations = Counter(
        v["kind"] for r in records for a in r["attempts"] for v in a.get("violations", [])
    )
    tags = Counter(
        v["tag"]
        for r in records
        for a in r["attempts"]
        for v in a.get("violations", [])
        if v.get("tag")
    )
    completions = [a["completion"] for r in records for a in r["attempts"] if "completion" in a]
    tokens = sum(c["usage"].get("total_token_count", 0) for c in completions)
    thoughts = sum(c["usage"].get("thoughts_token_count", 0) or 0 for c in completions)
    # Gemini reports "STOP", the OpenAI-compatible providers "stop".
    truncated = sum(
        1
        for c in completions
        if (c.get("finish_reason") or "STOP").upper() != "STOP"
    )

    return {
        "n_jets": len(records),
        "n_narrated": len(narrated),
        "n_abstained": len(abstained),
        "abstention_rate": len(abstained) / len(records) if records else None,
        "regimes": dict(Counter(r["regime"] for r in records)),
        "statuses": dict(Counter(r["status"] for r in records)),
        "table": table,
        "steps_mean": sum(steps) / len(steps) if steps else None,
        "steps_max": max(steps) if steps else None,
        "fail_rate": 1 - len(accepted) / len(narrated) if narrated else None,
        "violations_by_kind": dict(violations.most_common()),
        "violations_by_tag": dict(tags.most_common(10)),
        "calls": sum(len(r["attempts"]) for r in records),
        "total_tokens": tokens,
        "thought_tokens": thoughts,
        "prompt_tokens": sum(c["usage"].get("prompt_token_count", 0) or 0 for c in completions),
        "output_tokens": sum(c["usage"].get("candidates_token_count", 0) or 0 for c in completions),
        "cost_usd": estimate_cost(completions),
        "cost_usd_per_narrated": (
            (estimate_cost(completions) or 0) / len(narrated) if narrated else None
        ),
        # Anything other than STOP means the model ran out of room mid-narrative,
        # which fails verification for reasons that are ours, not the model's.
        "unclean_finish": truncated,
        "thinking_control": dict(Counter(c.get("thinking_control", "?") for c in completions)),
        "prompt_ids": sorted({r.get("prompt_id", "?") for r in records}),
        "models": sorted(
            {a["completion"]["model"] for r in records for a in r["attempts"] if "completion" in a}
        ),
    }


def format_table(s: dict) -> str:
    def pct(x) -> str:
        return "  —  " if x is None else f"{x:5.2f}"

    lines = []
    lines.append(
        f"{', '.join(s['models']) or '?'}  |  prompt {', '.join(s['prompt_ids'])}  |  "
        f"{s['n_jets']} jets ({s['n_narrated']} narrated, {s['n_abstained']} abstained)"
    )
    lines.append("")
    lines.append(f"{'':<10}{'single-pass':>13}{'final':>9}")
    for name, row in s["table"].items():
        mark = "  <-- accept criterion" if name == "TOTAL" else ""
        lines.append(f"{name:<10}{pct(row['single_pass']):>13}{pct(row['final']):>9}{mark}")
    lines.append("")
    steps = "—" if s["steps_mean"] is None else f"{s['steps_mean']:.2f} (max {s['steps_max']})"
    lines.append(f"Steps (attempts per accepted narrative) : {steps}")
    lines.append(f"Fail  (narrated but never accepted)     : {pct(s['fail_rate']).strip()}")
    lines.append(f"Abstention (gate, no model call)        : {pct(s['abstention_rate']).strip()}")
    lines.append("")
    lines.append("regimes  : " + ", ".join(f"{k}={v}" for k, v in sorted(s["regimes"].items())))
    lines.append("statuses : " + ", ".join(f"{k}={v}" for k, v in sorted(s["statuses"].items())))
    if s["violations_by_kind"]:
        lines.append(
            "violations (all attempts): "
            + ", ".join(f"{k}={v}" for k, v in s["violations_by_kind"].items())
        )
    if s["violations_by_tag"]:
        lines.append(
            "worst tags: " + ", ".join(f"{k}={v}" for k, v in list(s["violations_by_tag"].items())[:6])
        )
    lines.append(
        f"tokens   : {s['prompt_tokens']} in, {s['output_tokens']} out"
        + (f", {s['thought_tokens']} reasoning" if s["thought_tokens"] else "")
        + f"  ({s['total_tokens']} total over {s['calls']} calls)"
    )
    if s["cost_usd"] is not None:
        lines.append(
            f"cost     : ${s['cost_usd']:.4f} total, "
            f"${s['cost_usd_per_narrated']:.4f} per narrated jet  (list price)"
        )
    if s["thinking_control"]:
        lines.append(
            "thinking : " + ", ".join(f"{k}={v}" for k, v in sorted(s["thinking_control"].items()))
        )
    if s["unclean_finish"]:
        lines.append(
            f"WARNING  : {s['unclean_finish']} completions did not finish cleanly "
            "— raise llm.max_output_tokens, these failures are not the model's"
        )
    return "\n".join(lines)


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, nargs="?",
                   default=repo / "narrative/outputs/top/narratives/run_v4.jsonl",
                   help="JSONL written by narrative.orchestrator")
    p.add_argument("--json", type=Path, default=None, help="Also write the summary as JSON")
    args = p.parse_args()

    records = load_run(args.run)
    s = summarise(records)
    print(format_table(s))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(s, indent=2))
        print(f"\nWrote → {args.json}")


if __name__ == "__main__":
    main()
