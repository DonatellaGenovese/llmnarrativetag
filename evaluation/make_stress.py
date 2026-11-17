"""Stress test for the judges: good narratives with one known defect injected.

Each base narrative (a delivered pilot100 narrative of the top-3 generators,
tags stripped, exactly as the judge sees it) is written out unchanged plus once
per perturbation below. Each perturbation targets one dimension of judge-v3 /
judge-ad-v1, so a judge that works should lower that dimension on the
perturbed copy against the original.

    ling_garble     two sentences with their middle words shuffled       linguistic_realization
    phys_direction  a prong observable whose value is said to be high is
                    read with its LOW-value meaning, applied to this jet  physical_soundness
    phys_wrong      constituent multiplicity read as a count of prongs   physical_soundness
    phys_overreach  the jet asserted to be a top decaying to bW (top);
                    identified as a top decay (AD)                       physical_soundness
    over_nodecision the opening paragraph (decision, score or band,
                    adequacy) removed                                    overall_satisfaction
    over_nodirection every "towards X" / "X direction" phrase removed    overall_satisfaction
    contradiction   a closing sentence saying the other side outweighs   (probe: none since
                    the chosen one                                        Internal Consistency
                                                                          was dropped)

Writes evaluation/results/judge_stress/{top,ad}_items/<id>.txt (the layout
judge.py reads with --set/--items) and key.json (id -> base, perturbation,
target dimension, generator, jet).

    python -m evaluation.make_stress
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import yaml

from evaluation.judge import load_from_runs
from narrative.generator import load_glossary

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "evaluation/results/judge_stress"
MODELS = ["gemini-3.5-flash", "deepseek-flash", "qwen3.8-27b"]
# Two base narratives per generator and task, from the jets of the judge test.
PER_MODEL = 2

# Low-value reading of each prong observable, and the names narrators use.
LOW_READING = {"tau21": "two-pronged", "tau32": "three-pronged", "C2_double_b1": "two-pronged",
               "D2_double_b1": "two-pronged", "C3_double_b1": "three-pronged", "N3_b1": "three-pronged"}
ALIASES = {"tau21": ["tau21", "tau2/tau1", "tau2/1"], "tau32": ["tau32", "tau3/tau2", "tau3/2"],
           "C2_double_b1": ["C2_double_b1", "C2"], "D2_double_b1": ["D2_double_b1", "D2"],
           "C3_double_b1": ["C3_double_b1", "C3"], "N3_b1": ["N3_b1", "N3"],
           "n_const": ["n_const", "multiplicity"]}
SHORT = {"tau21": "tau21", "tau32": "tau32", "C2_double_b1": "C2", "D2_double_b1": "D2",
         "C3_double_b1": "C3", "N3_b1": "N3"}

SENT = re.compile(r"(?<=[.!?])\s+")


def sentences(par: str):
    return [s for s in SENT.split(par) if s]


def insert_after_mention(text: str, aliases, new: str):
    """Insert `new` after the first sentence that mentions one of `aliases`."""
    pars = text.split("\n\n")
    for pi, par in enumerate(pars):
        ss = sentences(par)
        for si, s in enumerate(ss):
            if any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(a)}(?![A-Za-z0-9_])", s) for a in aliases):
                ss.insert(si + 1, new)
                pars[pi] = " ".join(ss)
                return "\n\n".join(pars)
    return None


def garble(text: str, rng) -> str:
    pars = text.split("\n\n")
    pi = 1 if len(pars) > 1 else 0
    ss = sentences(pars[pi])
    for si in range(min(2, len(ss))):
        w = ss[si].split()
        if len(w) > 6:
            mid = w[2:-2]
            rng.shuffle(mid)
            ss[si] = " ".join(w[:2] + mid + w[-2:])
    pars[pi] = " ".join(ss)
    return "\n\n".join(pars)


def no_direction(text: str) -> str:
    t = re.sub(r",?\s*(?:pushing|pulling|contributing|adding|that pushes|which pushes)?\s*"
               r"(?:it |the jet |the score |the decision )?(?:towards|toward) (?:the )?"
               r"(?:top|qcd|QCD|anomalous|background)(?: class| side| direction)?", "", text)
    t = re.sub(r",?\s*(?:in|towards) the (?:anomalous|background|top|qcd) direction", "", t)
    return re.sub(r"[ \t]{2,}", " ", t)


def main() -> None:
    rng = np.random.default_rng(11)
    sel = json.loads((REPO / "evaluation/results/judge_test/selection.json").read_text())
    key = {}
    for uc, cfgp in (("top", "narrative/configs/narrative_top.yaml"),
                     ("ad", "narrative/configs/narrative_anomaly.yaml")):
        cfg = yaml.safe_load((REPO / cfgp).read_text())
        gl = load_glossary(REPO / cfg["glossary"]["path"], require_reviewed=True)
        banned = tuple((cfg.get("verifier") or {}).get("banned_terms") or ())
        meta = load_from_runs(REPO / cfg["outputs"]["narratives"], "pilot100", MODELS, True, gl,
                              banned_terms=banned)
        arte = {}
        for m in MODELS:
            for line in (REPO / cfg["outputs"]["narratives"] / f"pilot100_{m}.jsonl").read_text().splitlines():
                r = json.loads(line)
                arte[(m, r["jet_id"])] = r["artefact"]
        d = OUT / f"{uc}_items"
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.txt"):
            f.unlink()

        jets = sel[uc]["jets"]
        for mi, m in enumerate(MODELS):
            taken = 0
            # Each generator starts further down the jet list, so the bases span
            # different jets rather than the same two for every generator.
            for jet in jets[2 * mi:] + jets[:2 * mi]:
                if taken == PER_MODEL:
                    break
                v = next((x for x in meta.values() if x["model"] == m and x["jet_id"] == jet), None)
                if v is None:
                    continue
                a = arte[(m, jet)]
                # The direction probe needs a prong observable the artefact places high.
                high = [f for f in a["features"] if f["name"] in LOW_READING and f["pct"] > 75]
                if not high:
                    continue
                text, base = v["text"], f"{uc}-{m.split('-')[0][:6]}-{jet}"
                dec = a["decision"]
                if uc == "top":
                    other = a["other_class"]
                    over = ("Taken together, this establishes that the jet is a top quark that "
                            "decayed into a b quark and a W boson.")
                    contra = (f"Overall, the contributions pushing towards {other} outweigh those "
                              f"towards {dec}, which is why the tagger chose {dec}.")
                else:
                    opp = "background" if dec == "anomalous" else "anomalous"
                    over = ("Taken together, this pattern identifies the jet as the decay of a top "
                            "quark, which is what makes it anomalous.")
                    contra = (f"Overall, the contributions towards {opp} outweigh all the others, "
                              f"which is why the detector found this jet {dec}.")
                f = high[0]
                obs = SHORT[f["name"]]
                variants = {
                    "original": (text, None),
                    "ling_garble": (garble(text, rng), "linguistic_realization"),
                    "phys_direction": (insert_after_mention(
                        text, ALIASES[f["name"]],
                        f"With {obs} this high, the jet is clearly {LOW_READING[f['name']]}."),
                        "physical_soundness"),
                    "phys_wrong": (insert_after_mention(
                        text, ALIASES["n_const"],
                        "Constituent multiplicity counts the hard prongs of the jet, so this value "
                        "means the jet has that many separate prongs."), "physical_soundness"),
                    "phys_overreach": (text.rstrip() + " " + over, "physical_soundness"),
                    "over_nodecision": ("\n\n".join(text.split("\n\n")[1:]), "overall_satisfaction"),
                    "over_nodirection": (no_direction(text), "overall_satisfaction"),
                    "contradiction": (text.rstrip() + " " + contra, None),
                }
                for kind, (t, target) in variants.items():
                    if not t or (kind != "original" and t == text):
                        print(f"  skip {base} {kind}: perturbation did not apply")
                        continue
                    iid = f"{base}-{kind}"
                    (d / f"{iid}.txt").write_text(t)
                    key[iid] = {"use_case": uc, "base": base, "perturbation": kind, "target": target,
                                "model": m, "jet_id": jet, "observable": f["name"] if kind == "phys_direction" else None}
                taken += 1
            print(f"{uc} {m}: {taken} base narratives")
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    n = {uc: sum(1 for k in key.values() if k["use_case"] == uc) for uc in ("top", "ad")}
    print(f"items: {n}  -> {OUT}")


if __name__ == "__main__":
    main()
