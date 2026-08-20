"""Frozen reference statistics for the narrative artefact.

Every "how does this jet compare" number in the artefact is a percentile over
the test set. Recomputing those per jet would mean a full scan of ~400k rows
per narrative, so they are computed once here and stored as quantile grids.

A grid holds the distribution's quantiles at N_GRID evenly spaced levels, which
resolves a percentile to 100/N_GRID. Grids are a few tens of kB each instead of
the ~3 MB a sorted column would take.

Classes are the *tagger's* decision (sign of the teacher logit), never the truth
label: the narrative describes what ParT did, and truth is out of scope for it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

# 10001 levels resolve a percentile to 0.01, which two-decimal `pct` needs and
# ~200k reference jets per class amply support. At 1001 the finest rarity was
# 0.1 and 6% of values saturated to a literal 0.0.
N_GRID = 10001
CLASSES = ("top", "qcd")


def _grid(x: np.ndarray) -> np.ndarray:
    """Quantile grid of a 1-D sample, at N_GRID evenly spaced levels."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return np.quantile(x, np.linspace(0.0, 1.0, N_GRID))


def _pct_below(grid: np.ndarray, value: float) -> float:
    """Percentage of the distribution at or below `value`, from its grid."""
    if not np.isfinite(value):
        return float("nan")
    return float(np.searchsorted(grid, value, side="right") / len(grid) * 100.0)


@dataclass
class ReferenceStats:
    """Percentile lookups over the reference split."""

    obs: List[str]
    intercept: float
    clip: Dict[str, float]
    n: int
    n_by_class: Dict[str, int]
    score_median: Dict[str, float]
    score_grid: Dict[str, np.ndarray]
    resid_grid: np.ndarray
    obs_grid: Dict[str, np.ndarray]
    source: str

    # -- lookups -----------------------------------------------------------

    def score_percentile(self, decision: str, score: float) -> float:
        """% of jets assigned the same class scored at or below this one."""
        return _pct_below(self.score_grid[decision], abs(score))

    def recon_percentile(self, residual: float) -> float:
        """% of jets reconstructed *less* closely than this one.

        Inverted on purpose: a large value means this jet is well reconstructed,
        which is the direction the narrative reads it in.
        """
        return 100.0 - _pct_below(self.resid_grid, abs(residual))

    def obs_percentile(self, name: str, other_class: str, value: float) -> float:
        """% of jets of `other_class` whose value of `name` is at or below this.

        One-sided by construction: both tails are unusual and the middle is
        typical. Unused while `artefact.include_pct` is false.
        """
        return _pct_below(self.obs_grid[f"{name}|{other_class}"], value)

    # -- persistence -------------------------------------------------------

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = {f"score|{c}": self.score_grid[c] for c in self.score_grid}
        arrays["resid"] = self.resid_grid
        arrays.update({f"obs|{k}": v for k, v in self.obs_grid.items()})
        np.savez_compressed(out_dir / "grids.npz", **arrays)
        meta = {
            "obs": self.obs,
            "intercept": self.intercept,
            "clip": self.clip,
            "n": self.n,
            "n_by_class": self.n_by_class,
            "score_median": self.score_median,
            "n_grid": N_GRID,
            "source": self.source,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return out_dir

    @classmethod
    def load(cls, out_dir: Path) -> "ReferenceStats":
        meta = json.loads((out_dir / "meta.json").read_text())
        z = np.load(out_dir / "grids.npz")
        score_grid, obs_grid = {}, {}
        for key in z.files:
            if key.startswith("score|"):
                score_grid[key.split("|", 1)[1]] = z[key]
            elif key.startswith("obs|"):
                obs_grid[key.split("|", 1)[1]] = z[key]
        return cls(
            obs=meta["obs"],
            intercept=meta["intercept"],
            clip=meta["clip"],
            n=meta["n"],
            n_by_class=meta["n_by_class"],
            score_median=meta["score_median"],
            score_grid=score_grid,
            resid_grid=z["resid"],
            obs_grid=obs_grid,
            source=meta["source"],
        )


def decisions(t: np.ndarray) -> np.ndarray:
    """Tagger decision per jet, from the sign of the oriented logit."""
    return np.where(t > 0, "top", "qcd")


def build_stats(
    features_parquet: Path,
    model_joblib: Path,
    obs: Sequence[str] | None = None,
) -> ReferenceStats:
    bundle = joblib.load(model_joblib)
    ebm, obs_cols, clip = bundle["ebm"], list(bundle["obs"]), dict(bundle["clip"])
    if obs is not None and list(obs) != obs_cols:
        raise ValueError(f"config observables {list(obs)} != model observables {obs_cols}")

    df = pd.read_parquet(features_parquet)
    x = df[obs_cols]
    g = np.asarray(ebm.predict(x), dtype=np.float64)
    t = df["t"].to_numpy(dtype=np.float64)
    t_clip = np.clip(t, clip["t_clip_lo"], clip["t_clip_hi"])
    resid = t_clip - g
    dec = decisions(t)

    score_grid, score_median, n_by_class = {}, {}, {}
    for c in CLASSES:
        sel = dec == c
        score_grid[c] = _grid(np.abs(t_clip[sel]))
        score_median[c] = float(np.median(np.abs(t_clip[sel])))
        n_by_class[c] = int(sel.sum())

    # Frozen even while include_pct is false: one pass now, no recompute later.
    obs_grid = {}
    for name in obs_cols:
        col = df[name].to_numpy(dtype=np.float64)
        for c in CLASSES:
            obs_grid[f"{name}|{c}"] = _grid(col[dec == c])

    return ReferenceStats(
        obs=obs_cols,
        intercept=float(ebm.intercept_),
        clip=clip,
        n=int(len(df)),
        n_by_class=n_by_class,
        score_median=score_median,
        score_grid=score_grid,
        resid_grid=_grid(np.abs(resid)),
        obs_grid=obs_grid,
        source=str(features_parquet),
    )


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=repo / "narrative/configs/narrative_top.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    stats = build_stats(
        repo / cfg["inputs"]["features"],
        repo / cfg["inputs"]["model"],
    )
    out = stats.save(repo / cfg["outputs"]["stats"])

    print(f"n = {stats.n}  ({', '.join(f'{k}={v}' for k, v in stats.n_by_class.items())})")
    print(f"intercept = {stats.intercept:+.6f}")
    print("median |score|: " + ", ".join(f"{k}={v:.3f}" for k, v in stats.score_median.items()))
    q = stats.resid_grid
    print(f"|r| median={q[N_GRID // 2]:.3f}  p90={q[int(0.90 * N_GRID)]:.3f}  p99={q[int(0.99 * N_GRID)]:.3f}")
    print(f"Wrote reference stats → {out}")


if __name__ == "__main__":
    main()
