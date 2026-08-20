"""Abstention gate, generation, verification and retry, for a set of jets.

    build artefact -> regime?
        abstain -> record, no model call
        full / caveat -> generate -> verify -> pass? keep : retry

Retries advance the seed. At temperature 0 a fixed seed would reproduce the
failing narrative verbatim and the loop would spin without ever recovering.

Every attempt is written out, failures included: the rate at which narratives
survive verification is itself a Part 4 measurement, and it cannot be recovered
from the accepted ones alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import yaml

from .artefact import ArtefactBuilder, REGIME_ABSTAIN, REGIME_CAVEAT, additivity_error
from .generator import NarrativeGenerator, load_glossary
from .llm_client import LLMClient
from .stats import ReferenceStats
from .verifier import verify

STATUS_OK = "ok"
STATUS_ABSTAINED = "abstained"
STATUS_FAILED = "failed_verification"
STATUS_ERROR = "error"


def sample_jets(n: int, n_rows: int, seed: int, path: Path) -> List[int]:
    """Row indices for a run, drawn once and reused.

    The file is the unit of comparison: every model must narrate the *same*
    jets, or half of any difference between them is sampling noise. If `path`
    exists it is loaded verbatim and `n`/`seed` are ignored.

    Sampling is not optional. The TopLandscape test file is ordered in blocks of
    up to 100 identical labels, so the first N rows are not a sample of anything
    — the first 8 are all QCD.
    """
    if path.exists():
        return json.loads(path.read_text())["jets"]
    rng = np.random.default_rng(seed)
    jets = sorted(int(i) for i in rng.choice(n_rows, size=min(n, n_rows), replace=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n": len(jets), "seed": seed, "n_rows": n_rows, "jets": jets}))
    return jets


class Orchestrator:
    def __init__(
        self,
        repo: Path,
        config_path: Path,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.repo = repo
        self.cfg = yaml.safe_load(config_path.read_text())
        # CLI overrides, so one config serves every model in a sweep across
        # providers without being edited between runs.
        if model:
            self.cfg["llm"] = {**self.cfg["llm"], "model": model}
        if backend:
            self.cfg["llm"] = {**self.cfg["llm"], "backend": backend}
        if reasoning_effort:
            self.cfg["llm"] = {**self.cfg["llm"], "reasoning_effort": reasoning_effort}

        stats = ReferenceStats.load(repo / self.cfg["outputs"]["stats"])
        self.builder = ArtefactBuilder(repo / self.cfg["inputs"]["model"], stats, self.cfg)
        self.glossary = load_glossary(
            repo / self.cfg["glossary"]["path"],
            require_reviewed=bool(self.cfg["glossary"]["require_reviewed"]),
        )
        self.generator = NarrativeGenerator(
            repo / self.cfg["prompt"]["path"],
            self.glossary,
            LLMClient.from_config(self.cfg),
        )
        self.max_attempts = int(self.cfg["llm"]["max_attempts"])
        self.base_seed = self.cfg["llm"].get("seed") or 0
        self.base_temperature = float(self.cfg["llm"]["temperature"])
        self.retry_temp_step = float(self.cfg["llm"].get("retry_temperature_step", 0.0))

    def run_one(self, df, index: int) -> dict:
        record = self.builder.build(df, index)
        artefact, meta = record["artefact"], record["meta"]
        meta["additivity_error"] = round(additivity_error(record), 4)

        out = {
            "jet_id": artefact["jet_id"],
            "row": int(index),
            "status": STATUS_ABSTAINED,
            "regime": meta["regime"],
            "artefact": artefact,
            "meta": meta,
            "prompt_id": self.generator.prompt_id,
            "narrative": None,
            "attempts": [],
        }

        if meta["regime"] == REGIME_ABSTAIN:
            out["abstention_reason"] = (
                f"|residual| = {abs(meta['residual']):.3f} exceeds "
                f"{self.builder.caveat_threshold}; the observable basis does not "
                "account for this decision closely enough to describe it"
            )
            return out

        caveat = meta["regime"] == REGIME_CAVEAT
        out["prompt_hash"] = self.generator.prompt_hash(artefact, caveat=caveat)

        for attempt in range(self.max_attempts):
            seed = self.base_seed + attempt
            temperature = self.base_temperature + attempt * self.retry_temp_step
            try:
                completion, _ = self.generator.generate(
                    artefact, caveat=caveat, seed=seed, temperature=temperature
                )
            except Exception as exc:  # network, quota, auth
                out["attempts"].append({"attempt": attempt, "seed": seed, "error": repr(exc)})
                out["status"] = STATUS_ERROR
                return out

            report = verify(completion.text, artefact, glossary=self.glossary)
            out["attempts"].append(
                {
                    "attempt": attempt,
                    "seed": seed,
                    "completion": completion.to_dict(),
                    "ok": report.ok,
                    "violations": [
                        {"kind": v.kind, "detail": v.detail, "tag": v.tag} for v in report.violations
                    ],
                }
            )
            if report.ok:
                out["status"] = STATUS_OK
                out["narrative"] = completion.text
                return out

        out["status"] = STATUS_FAILED
        return out

    def run(self, indices: Sequence[int], out_path: Path) -> List[dict]:
        import pandas as pd

        df = pd.read_parquet(self.repo / self.cfg["inputs"]["features"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        with open(out_path, "w") as fh:
            for i in indices:
                rec = self.run_one(df, int(i))
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_att = len(rec["attempts"])
                print(f"  jet {rec['jet_id']:>7}  {rec['regime']:<7}  {rec['status']:<20} "
                      f"({n_att} attempt{'s' if n_att != 1 else ''})")
        return records


def summarise(records: List[dict]) -> str:
    n = len(records)
    by_status, by_regime = {}, {}
    attempts = []
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        by_regime[r["regime"]] = by_regime.get(r["regime"], 0) + 1
        if r["status"] in (STATUS_OK, STATUS_FAILED):
            attempts.append(len(r["attempts"]))
    lines = [f"{n} jets"]
    lines.append("  status: " + ", ".join(f"{k}={v} ({v / n:.0%})" for k, v in sorted(by_status.items())))
    lines.append("  regime: " + ", ".join(f"{k}={v} ({v / n:.0%})" for k, v in sorted(by_regime.items())))
    if attempts:
        lines.append(f"  attempts per narrated jet: mean={np.mean(attempts):.2f} max={max(attempts)}")
    return "\n".join(lines)


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=repo / "narrative/configs/narrative_top.yaml")
    p.add_argument("--jets", type=int, nargs="+", default=None, help="Explicit row indices")
    p.add_argument("--n", type=int, default=5, help="First N rows, if --jets is not given")
    p.add_argument("--model", default=None, help="Override llm.model from the config")
    p.add_argument("--backend", default=None, help="Override llm.backend from the config")
    p.add_argument("--reasoning-effort", default=None,
                   help="none/low/medium/high; for models that cannot switch reasoning off")
    p.add_argument("--jets-file", type=Path, default=None,
                   help="Shared random sample; created on first use, reused after")
    p.add_argument("--sample", type=int, default=None, help="Sample size for --jets-file")
    p.add_argument("--sample-seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    orch = Orchestrator(repo, args.config, model=args.model, backend=args.backend,
                        reasoning_effort=args.reasoning_effort)

    if args.jets_file:
        import pandas as pd

        n_rows = pd.read_parquet(
            repo / orch.cfg["inputs"]["features"], columns=["jet_id"]
        ).shape[0]
        indices = sample_jets(args.sample or args.n, n_rows, args.sample_seed, args.jets_file)
        print(f"{len(indices)} jets from {args.jets_file}")
    elif args.jets:
        indices = args.jets
    else:
        indices = list(range(args.n))

    print(f"model: {orch.cfg['llm']['model']}")
    out = args.out or (repo / orch.cfg["outputs"]["narratives"] / "narratives.jsonl")

    records = orch.run(indices, out)
    print()
    print(summarise(records))
    print(f"Wrote → {out}")


if __name__ == "__main__":
    main()
