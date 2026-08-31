"""Verification of a narrative against the artefact it was written from.

The narrator is required to wrap every artefact-derived number in a tag naming
its field, which reduces checking a quantitative claim to a regular expression
and an equality test. No second model is in the loop, so the check is exact,
deterministic and free.

What this establishes and what it does not:

  * ESTABLISHED — every number in the narrative comes from the artefact, is
    attributed to the right field, and is reproduced with the right sign and
    magnitude; no artefact number was silently dropped.
  * NOT ESTABLISHED — that the prose around a number reads it correctly, that
    the physical interpretation respects the glossary, or that the guardrails
    were honoured. Those are interpretive, they need Part 4, and no regex
    settles them.

Numbers are compared numerically, not as strings: "38" for 38.0 is the same
claim, while 2.4 for 2.37 is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# [TAG]number[/TAG], where the closing tag must match the opening one.
#
# A generic closer, [TAG]number[/], was tried and rejected on measurement. The
# argument for it was sound in the abstract — the field is named by the opening
# tag, so requiring the name twice is a cost with no informational return — and
# it did help the models that could not keep a pair straight: on
# gemini-3.5-flash-lite it moved Form 0.93 -> 0.99 and NoBare 0.80 -> 0.90, on
# gpt-oss-120b Comp 0.13 -> 0.24. But it cost Comp on the models that had no
# mismatched pairs to fix (gemini-3.5-flash 1.00 -> 0.93), because a closer that
# is free to write is also free to invent: [pct], [decision], [ZS] and
# [other_class] all appeared as tags for the first time. Net on the paired jets,
# gemini-3.5-flash fell 0.94 -> 0.84, p = 0.022 — the only significant contrast
# in the ablation, and a regression. The gain landed where it was not needed.
TAG_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_]*)\]\s*([+-]?\d+(?:\.\d+)?)\s*\[/\1\]")
ANY_BRACKET_RE = re.compile(r"\[/?[A-Za-z][A-Za-z0-9_]*\]")
DIGIT_RE = re.compile(r"\d")

SCALAR_TAGS = {
    "Z": "score_tagger",
    "ZM": "score_median",
    "ZP": "score_pct",
    "G": "score_reconstructed",
    "GP": "recon_pct",
    "B": "intercept",
    "N": "n_other",
    "O": "phi_other",
}
# `pct` is deliberately absent: it reaches the narrator as context for [S<i>]
# but is not itself tagged.
PER_FEATURE_TAGS = {"V": "value", "I": "phi", "R": "rarity"}

# Word tags carry a reading rather than a number, and are checked the same way:
# the expected word is a declared function of a value the artefact already
# passed, so the model is applying a rule it was given rather than being judged
# against one invented afterwards. The vocabulary is closed, so a paraphrase is
# a violation rather than a gap in a keyword list.
READING_TOP_K = 5
# Three bands, not four. The fourth boundary at 25 produced a systematic drift
# — six values between 16 and 23 all called `ordinary` — while no value came
# within half a point of it, so it was not a boundary error but a disagreement
# over a whole range. The two that remain sit where the values are sparsest:
# 4.9% of rarities fall within +-0.5 of 2 and 2.7% of 10, against 11.3% of 1.
# Calibrated on the five leading observables, the only ones obliged to carry a
# judgement and far more extreme than the tail: median rarity 4.42 against 18.54.
RARITY_BANDS = ((2.0, "extreme"), (10.0, "unusual"), (float("inf"), "ordinary"))
# Side bands are independent of the rarity bands and do not align with them.
# The two describe different things — how unusual, and on which side — so they
# need not agree.
#
# A two-word side, low/high cut at 50, was tried and not adopted. It is better
# targeted than this rule and it worked exactly as intended: `mid` went from
# being the subject of every S violation to being written zero times in 1800
# tags, and deepseek's S errors fell from 2 to 1. What it did not do is show up
# in the accept rate. Paired against this rule the four models came out
# 0.87/0.61/0.69/0.10 against 0.94/0.57/0.64/0.10, no contrast significant, and
# the families the change cannot touch moved by more than the family it could:
# deepseek's `Read` fell 0.80 -> 0.74 entirely through Q (14 -> 21) and D
# (8 -> 10). On gemini-3.5-flash-lite 50 of 90 jets flipped verdict. TOTAL is a
# conjunction over ~46 tag decisions, so it goes as p^N and a per-tag drift too
# small to see swings it; at n = 90 it cannot resolve a prompt edit, and the
# per-family rates are what carry the signal. Kept as evidence that a targeted
# fix and a measurable improvement are different claims.
SIDE_BANDS = ((25.0, "low"), (75.0, "mid"), (float("inf"), "high"))
WORD_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_]*)\]\s*([A-Za-z]+)\s*\[/\1\]")


def _band(value: float, bands) -> str:
    """Upper edges are inclusive: a value sitting exactly on a boundary belongs
    to the band below it. Stated that way in the prompt too, because the first
    version said "below 1" and most reading errors landed within half a point
    of that edge."""
    for edge, word in bands:
        if value <= edge:
            return word
    return bands[-1][1]


def rarity_band(rarity: float) -> str:
    """The one word the narrative must use for this rarity."""
    return _band(rarity, RARITY_BANDS)


def side_band(pct: float) -> str:
    """Which part of the other class's range the value falls in."""
    return _band(pct, SIDE_BANDS)


def direction_word(phi: float, decision: str, other_class: str) -> str:
    """The class a contribution pushes towards."""
    return decision if phi >= 0 else other_class


def recon_word(recon_pct: float) -> str:
    """`recon_pct` counts jets reconstructed less closely, so high is good."""
    return "better" if recon_pct >= 50 else "worse"


@dataclass
class Violation:
    kind: str
    detail: str
    tag: Optional[str] = None
    expected: Optional[float] = None
    found: Optional[float] = None

    def __str__(self) -> str:
        return f"[{self.kind}] {self.detail}"


@dataclass
class VerificationReport:
    ok: bool
    violations: List[Violation] = field(default_factory=list)
    tags_found: int = 0
    tags_required: int = 0

    def summary(self) -> str:
        if self.ok:
            return f"PASS ({self.tags_found}/{self.tags_required} required tags, no violations)"
        kinds: Dict[str, int] = {}
        for v in self.violations:
            kinds[v.kind] = kinds.get(v.kind, 0) + 1
        return "FAIL: " + ", ".join(f"{k}x{n}" for k, n in sorted(kinds.items()))


def expected_values(artefact: dict) -> Dict[str, float]:
    """Tag name -> the value the narrative must reproduce for it."""
    expected: Dict[str, float] = {}
    for tag, key in SCALAR_TAGS.items():
        if key in artefact:
            expected[tag] = float(artefact[key])
    for position, f in enumerate(artefact.get("features", []), start=1):
        index = f.get("i", position)
        for prefix, key in PER_FEATURE_TAGS.items():
            if key in f:
                expected[f"{prefix}{index}"] = float(f[key])
    return expected


def required_reading_tags(artefact: dict, reading_top_k: int = READING_TOP_K) -> List[str]:
    """Reading tags the narrative must carry, as opposed to may carry.

    Every reading is checked wherever it appears; only the leading observables
    are obliged to supply one. A narrator that volunteers a judgement on the
    tail is doing more than asked and should be verified, not rejected.
    """
    out: List[str] = []
    if "recon_pct" in artefact:
        out.append("GQ")
    for position, f in enumerate(artefact.get("features", []), start=1):
        if position > reading_top_k:
            continue
        index = f.get("i", position)
        out += [f"D{index}", f"R{index}"]
        if "rarity" in f:
            out.append(f"Q{index}")
        if "pct" in f:
            out.append(f"S{index}")
    return out


def expected_words(artefact: dict, reading_top_k: int = READING_TOP_K) -> Dict[str, str]:
    """Tag name -> the word the narrative must use, wherever it appears.

    Computed for every observable, not only the ones obliged to carry a reading:
    see `required_reading_tags` for that. Ranked by |phi| the first five hold 86%
    of the total contribution and the last three hold 4%, so a judgement is only
    demanded of the head — but one offered on the tail is still checked.
    """
    out: Dict[str, str] = {}
    if "recon_pct" in artefact:
        out["GQ"] = recon_word(float(artefact["recon_pct"]))
    decision, other = artefact.get("decision"), artefact.get("other_class")
    for position, f in enumerate(artefact.get("features", []), start=1):
        index = f.get("i", position)
        out[f"D{index}"] = direction_word(float(f["phi"]), decision, other)
        if "rarity" in f:
            out[f"Q{index}"] = rarity_band(float(f["rarity"]))
        if "pct" in f:
            out[f"S{index}"] = side_band(float(f["pct"]))
    return out


def required_tags(artefact: dict) -> List[str]:
    """Tags the narrative must use at least once.

    `n_other` / `phi_other` are exempt when no observable was held back: with
    the full basis reported there is nothing for them to describe.
    """
    exp = expected_values(artefact)
    if int(artefact.get("n_other", 0)) == 0:
        exp.pop("N", None)
        exp.pop("O", None)
    # The rarity number is obliged only where its reading is: quoting 4.7 after
    # having written `unusual` adds nothing, and R was the most omitted tag of
    # all on every model.
    keep = set(required_reading_tags(artefact))
    exp = {k: v for k, v in exp.items() if not k.startswith("R") or k in keep}
    return sorted(list(exp) + [t for t in keep if t not in exp])


# A run of characters that could spell a physics name: alphanumerics plus the
# punctuation that appears inside one. Deliberately includes `.`, so that a
# decimal number is taken whole and judged whole rather than split into digits
# that individually look harmless.
NAME_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_/*^.+-]*")


def _normalise_name(text: str) -> str:
    """A name reduced to its letters and digits, or "" if it has no letter.

    `tau2/tau1`, `tau2/1` and `tau21` do not all reduce to the same string — the
    first keeps its second `tau` — which is why both the observable name and its
    glossary label are normalised and matched against.
    """
    core = re.sub(r"[^A-Za-z0-9]", "", text)
    return core if any(c.isalpha() for c in core) else ""


def _allowed_literals(artefact: dict, glossary: Optional[dict]) -> List[str]:
    """Strings that legitimately contain digits outside a tag.

    Observable names and their glossary labels are unavoidable in prose:
    `tau32` and "energy correlation ratio C2" carry digits that are part of a
    name, not a quantitative claim.
    """
    out: List[str] = []
    for f in artefact.get("features", []):
        out.append(f["name"])
    if glossary:
        for name, entry in glossary.items():
            out.append(name)
            if entry.get("label"):
                out.append(entry["label"])
            # Tokens mixing letters and digits anywhere in the glossary: the
            # correlator formulas the narrator is licensed to quote, `e4*e2/e3^2`
            # or `C2`, are names, not measurements. Purely numeric tokens are
            # deliberately NOT whitelisted — a bare `0.1` in the narrative is a
            # quantitative claim whatever the glossary happens to contain.
            for text in (entry.get("definition"), entry.get("meaning")):
                for token in (text or "").split():
                    token = token.strip(".,;:()[]")
                    if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
                        out.append(token)
    # Longest first, so `C3_double_b1` is consumed before `C3`.
    return sorted(set(out), key=len, reverse=True)


def verify(
    narrative: str,
    artefact: dict,
    glossary: Optional[dict] = None,
    require_complete: bool = True,
    extra_literals: Sequence[str] = (),
) -> VerificationReport:
    expected = expected_values(artefact)
    words = expected_words(artefact)
    violations: List[Violation] = []

    # 1. Well-formed tags, and the values they carry.
    seen: Dict[str, List[float]] = {}
    for match in TAG_RE.finditer(narrative):
        tag, raw = match.group(1), match.group(2)
        value = float(raw)
        seen.setdefault(tag, []).append(value)
        if tag not in expected:
            violations.append(
                Violation("unknown_tag", f"{tag!r} is not a field of this artefact", tag=tag)
            )
        elif value != expected[tag]:
            violations.append(
                Violation(
                    "wrong_value",
                    f"{tag} reported as {raw}, artefact says {expected[tag]}",
                    tag=tag,
                    expected=expected[tag],
                    found=value,
                )
            )

    # 1b. Word tags: a reading of a number, checked against the rule the prompt
    # declared. A wrong word here is a claim that inverts or overstates what the
    # number says — the failure the numeric checks cannot see.
    for match in WORD_RE.finditer(narrative):
        tag, word = match.group(1), match.group(2).lower()
        if tag not in words:
            if tag not in expected:
                violations.append(
                    Violation("unknown_tag", f"{tag!r} is not a field of this artefact", tag=tag)
                )
            continue
        seen.setdefault(tag, [])
        if word != words[tag]:
            violations.append(
                Violation(
                    "wrong_reading",
                    f"{tag} called {word!r}, the rule gives {words[tag]!r}",
                    tag=tag,
                )
            )

    # 2. Every artefact number actually used.
    needed = required_tags(artefact)
    lookup: Dict[str, object] = {**expected, **words}
    if require_complete:
        for tag in needed:
            if tag not in seen:
                violations.append(
                    Violation("missing_tag", f"{tag} ({lookup[tag]}) never appears", tag=tag)
                )

    # 3. Brackets that did not parse as a tag pair.
    residual = WORD_RE.sub(" ", TAG_RE.sub(" ", narrative))
    for stray in ANY_BRACKET_RE.finditer(residual):
        violations.append(
            Violation("malformed_tag", f"unpaired or malformed tag {stray.group(0)!r}")
        )

    # 4. Bare numbers: a quantitative claim the verifier cannot check.
    stripped = ANY_BRACKET_RE.sub(" ", residual)
    literals = list(_allowed_literals(artefact, glossary)) + list(extra_literals)
    for literal in literals:
        # Whole tokens only: an observable named `m` must not eat the m of "median".
        stripped = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(literal)}(?![A-Za-z0-9_])", " ", stripped
        )
    # Second pass, on the spelling rather than the string. A narrator may write a
    # known name a different way — `tau2/1` for `tau21`, whose glossary label is
    # "N-subjettiness ratio tau2/tau1" — and the exact-match pass above misses it,
    # charging a bare number for what is a name.
    #
    # Matching after normalisation, rather than whitelisting fragments, is what
    # keeps this safe. Whitelisting `tau2` and `1` separately would let a real
    # measurement through; requiring the whole token to normalise onto a whole
    # known name cannot, because every name carries a letter and a bare number
    # never does. `175.27` normalises to `17527` and `175GeV` to `175GeV`,
    # neither of which is a name, so both are still caught.
    known = {n for n in (_normalise_name(x) for x in literals) if n}
    stripped = NAME_TOKEN_RE.sub(
        lambda m: " " if _normalise_name(m.group(0)) in known else m.group(0), stripped
    )
    for line in stripped.splitlines():
        if DIGIT_RE.search(line):
            snippet = line.strip()
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            violations.append(Violation("untagged_number", f"bare digits in: {snippet!r}"))

    return VerificationReport(
        ok=not violations,
        violations=violations,
        tags_found=len(seen),
        tags_required=len(needed),
    )
