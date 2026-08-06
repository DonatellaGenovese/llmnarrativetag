"""Build QuarkGluon feature table: observables + ParT teacher targets.

Supports one or many data parquet files (concatenated in sorted order),
matching weaver's glob iteration used for logit dumps.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Optional, Sequence

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .observables_qg import OBS_COLUMNS, compute_observables_table


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _expand_paths(patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        matched = sorted(glob.glob(str(pattern)))
        if matched:
            files.extend(Path(p) for p in matched)
        else:
            p = Path(pattern)
            if p.exists():
                files.append(p)
    # unique preserve order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


RAW_COLS = [
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_deta",
    "part_dphi",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "jet_pt",
    "jet_mass",
    "jet_eta",
    "jet_phi",
    "jet_nparticles",
    "label",
]


def _read_raw(files: List[Path], n: int) -> dict:
    """Read up to n jets from parquet files in order."""
    remaining = n
    chunks = {c: [] for c in RAW_COLS}
    for f in files:
        if remaining <= 0:
            break
        table = pq.read_table(f, columns=RAW_COLS)
        take = min(remaining, table.num_rows)
        if take < table.num_rows:
            table = table.slice(0, take)
        for c in RAW_COLS:
            col = table.column(c)
            if c.startswith("part_"):
                chunks[c].extend(col.to_pylist())
            else:
                chunks[c].append(np.asarray(col))
        remaining -= take
    raw = {}
    for c in RAW_COLS:
        if c.startswith("part_"):
            raw[c] = chunks[c]
        else:
            raw[c] = np.concatenate(chunks[c], axis=0) if chunks[c] else np.array([])
    return raw


def build_features(
    data_patterns: Sequence[str],
    pred_parquet: Path,
    out_parquet: Path,
    config_path: Path,
    max_jets: Optional[int] = None,
) -> Path:
    cfg = _load_yaml(config_path)
    files = _expand_paths(data_patterns)
    if not files:
        raise FileNotFoundError(f"No data files matched: {data_patterns}")

    pred = ak.from_parquet(str(pred_parquet))
    n_pred = len(pred)
    n = n_pred if max_jets is None else min(max_jets, n_pred)

    raw = _read_raw(files, n)
    if len(raw["jet_pt"]) != n:
        raise RuntimeError(
            f"Raw jets ({len(raw['jet_pt'])}) != requested n={n} from pred ({n_pred}). "
            f"Files: {[str(f) for f in files]}"
        )

    obs = compute_observables_table(
        raw["part_px"],
        raw["part_py"],
        raw["part_pz"],
        raw["part_energy"],
        raw["part_deta"],
        raw["part_dphi"],
        raw["part_charge"],
        raw["part_isChargedHadron"],
        raw["part_isNeutralHadron"],
        raw["part_isPhoton"],
        raw["part_isElectron"],
        raw["part_isMuon"],
        max_jets=n,
        show_progress=True,
    )

    tgt = cfg["targets"]
    df = pd.DataFrame(obs)
    df["jet_id"] = np.arange(n, dtype=np.int64)
    df["jet_eta"] = np.asarray(raw["jet_eta"][:n], dtype=np.float64)
    df["jet_phi"] = np.asarray(raw["jet_phi"][:n], dtype=np.float64)
    df["jet_nparticles"] = np.asarray(raw["jet_nparticles"][:n], dtype=np.int64)
    df["label_file"] = np.asarray(raw["label"][:n], dtype=np.int64)  # 1=Q, 0=G in file

    df["z_q"] = np.asarray(pred[tgt["z_q"]][:n], dtype=np.float64)
    df["z_g"] = np.asarray(pred[tgt["z_g"]][:n], dtype=np.float64)
    df["t"] = np.asarray(pred[tgt["t"]][:n], dtype=np.float64)
    df["s"] = np.asarray(pred[tgt["s"]][:n], dtype=np.float64)
    df["label"] = np.asarray(pred[tgt["label"]][:n], dtype=np.int64)  # weaver: 0=Q, 1=G
    df["pred_label"] = np.asarray(pred["pred_label"][:n], dtype=np.int64)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"Wrote {len(df)} jets → {out_parquet}")
    show = [c for c in cfg["observables"] if c in df.columns] + ["t", "s"]
    print(df[show].describe().T[["mean", "std", "min", "max"]])
    return out_parquet


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-parquet",
        nargs="+",
        required=True,
        help="One or more parquet paths / globs (sorted)",
    )
    p.add_argument("--pred-parquet", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=repo / "surrogate/configs/qg_basis.yaml")
    p.add_argument("--max-jets", type=int, default=None)
    args = p.parse_args()
    build_features(args.data_parquet, args.pred_parquet, args.out, args.config, args.max_jets)


if __name__ == "__main__":
    main()
