"""Verifier failures as correction lines for the refiner.

The PONTE loop (hercolelab/personalized_xai) hands the refiner one line per
failed check, `FAITHFULNESS ERROR: ...`, `COMPLETENESS ERROR: ...`, plus the
de-duplicated history of every earlier line on the same case. This builds the
same lines from narrative.verifier's violations.

No model is involved: the verifier is deterministic and already knows what the
correct text is, so each line says what was wrong and what to write instead.

    FAITHFULNESS   wrong_value, wrong_reading, unknown_tag
    COMPLETENESS   missing_tag
    FORMAT         malformed_tag, untagged_number   (the tag contract)
    IDENTITY       identity_claim                    (anomaly detection only)
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from .verifier import Violation

CATEGORY = {
    "wrong_value": "FAITHFULNESS",
    "wrong_reading": "FAITHFULNESS",
    "unknown_tag": "FAITHFULNESS",
    "missing_tag": "COMPLETENESS",
    "malformed_tag": "FORMAT",
    "untagged_number": "FORMAT",
    "identity_claim": "IDENTITY",
}


def _line(v: Violation) -> str:
    cat = CATEGORY.get(v.kind, "VERIFICATION")
    if v.kind == "wrong_value":
        fix = f"write [{v.tag}]{_num(v.expected)}[/{v.tag}]"
    elif v.kind == "wrong_reading":
        m = re.search(r"the rule gives '([^']+)'", v.detail)
        fix = f"write [{v.tag}]{m.group(1)}[/{v.tag}]" if m else "use the word the rule gives"
    elif v.kind == "unknown_tag":
        fix = "remove it: use only the tags listed in the instructions"
    elif v.kind == "missing_tag":
        m = re.match(r"\S+ \((.*)\) never appears", v.detail)
        fix = f"include it as [{v.tag}]{m.group(1)}[/{v.tag}]" if m else f"include the [{v.tag}] tag"
    elif v.kind == "malformed_tag":
        fix = "every tag must be a pair [X]value[/X] with the same name in both brackets"
    elif v.kind == "untagged_number":
        fix = ("every number must sit inside its tag; drop a number that has no tag, "
               "and write names without loose digits")
    elif v.kind == "identity_claim":
        fix = ("describe how the jet differs from the background reference without naming "
               "any particle, process or signal; do not use that word at all, even in its "
               "everyday sense")
    else:
        fix = ""
    return f"{cat} ERROR: {v.detail}" + (f" — {fix}" if fix else "")


def _num(x) -> str:
    if x is None:
        return "?"
    # str, not :g, which rounds to six significant digits.
    return str(x)


def failure_lines(violations: Sequence[Violation]) -> List[str]:
    """One correction line per violation, in the verifier's order, no repeats."""
    out: List[str] = []
    for v in violations:
        line = _line(v)
        if line not in out:
            out.append(line)
    return out


def update_history(history: List[str], new_lines: Iterable[str]) -> List[str]:
    """Append the lines not already in the history, as PONTE does."""
    for line in new_lines:
        if line not in history:
            history.append(line)
    return history


def render(lines: Sequence[str], empty: str) -> str:
    return "\n".join(f"- {l}" for l in lines) if lines else empty
