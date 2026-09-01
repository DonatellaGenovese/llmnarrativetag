"""Build TopLandscape feature table: observables + ParT teacher targets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .observables_top import TopObsConfig, compute_observables_table


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_features(
    data_parquet: Path,
    pred_parquet: Path,
    out_parquet: Path,
    config_path: Path,
    max_jets: Optional[int] = None,
) -> Path:
    cfg = _load_yaml(config_path)
    obs_cfg = TopObsConfig(
        R0=float(cfg["R0"]),
        sd_z_cut=float(cfg["softdrop"]["z_cut"]),
        sd_beta=float(cfg["softdrop"]["beta"]),
        tau_beta=float(cfg["nsubjettiness"]["beta"]),
        ecf_beta=float(cfg["ecf"]["beta"]),
        truncate_C3=int(cfg["ecf"]["truncate_C3"]),
        truncate_N3=int(cfg["ecf"]["truncate_N3"]),
    )

    # Predictions (one row per jet, same order as weaver test iteration over the file)
    pred = ak.from_parquet(str(pred_parquet))
    n_pred = len(pred)
    n = n_pred if max_jets is None else min(max_jets, n_pred)

    # Raw constituents from the dataset parquet (same jet order as stored)
    cols = [
        "part_px",
        "part_py",
        "part_pz",
        "part_energy",
        "jet_pt",
        "jet_mass",
        "jet_eta",
        "jet_phi",
        "jet_nparticles",
        "label",
    ]
    # Read only needed rows when possible
    table = pq.read_table(data_parquet, columns=cols)
    if n < table.num_rows:
        table = table.slice(0, n)
    raw = {c: table.column(c).to_pylist() if c.startswith("part_") else np.asarray(table.column(c)) for c in cols}

    # Awkward lists for particles
    part_px = ak.Array(raw["part_px"])
    part_py = ak.Array(raw["part_py"])
    part_pz = ak.Array(raw["part_pz"])
    part_E = ak.Array(raw["part_energy"])

    obs = compute_observables_table(
        part_px,
        part_py,
        part_pz,
        part_E,
        jet_pt=np.asarray(raw["jet_pt"], dtype=np.float64),
        jet_mass=np.asarray(raw["jet_mass"], dtype=np.float64),
        cfg=obs_cfg,
        max_jets=n,
        show_progress=True,
    )

    tgt = cfg["targets"]
    df = pd.DataFrame(obs)
    df["jet_id"] = np.arange(n, dtype=np.int64)
    df["jet_eta"] = np.asarray(raw["jet_eta"][:n], dtype=np.float64)
    df["jet_phi"] = np.asarray(raw["jet_phi"][:n], dtype=np.float64)
    df["jet_nparticles"] = np.asarray(raw["jet_nparticles"][:n], dtype=np.int64)
    # file label: 1=top, 0=QCD (physics label)
    df["label_file"] = np.asarray(raw["label"][:n], dtype=np.int64)

    # Teacher outputs (weaver class index: 0=top, 1=QCD in our dump naming)
    df["z_top"] = np.asarray(pred[tgt["z_top"]][:n], dtype=np.float64)
    df["z_qcd"] = np.asarray(pred[tgt["z_qcd"]][:n], dtype=np.float64)
    df["t"] = np.asarray(pred[tgt["t"]][:n], dtype=np.float64)
    df["s"] = np.asarray(pred[tgt["s"]][:n], dtype=np.float64)
    df["label"] = np.asarray(pred[tgt["label"]][:n], dtype=np.int64)  # class index from weaver
    df["pred_label"] = np.asarray(pred["pred_label"][:n], dtype=np.int64)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"Wrote {len(df)} jets → {out_parquet}")
    print(df[cfg["observables"] + ["t", "s"]].describe().T[["mean", "std", "min", "max"]])
    return out_parquet


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-parquet",
        type=Path,
        required=True,
        help="TopLandscape split, e.g. particle_transformer/datasets/TopLandscape/test_file.parquet",
    )
    # No default: weaver names each training run after its start time, so the
    # path differs on every machine. run_official_top.sh discovers it.
    p.add_argument(
        "--pred-parquet",
        type=Path,
        required=True,
        help="ParT logits for the same split, from that run's predict_output/",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=repo / "surrogate/outputs/top/features_test.parquet",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=repo / "surrogate/configs/top_basis.yaml",
    )
    p.add_argument("--max-jets", type=int, default=None, help="Prototype subset size")
    args = p.parse_args()
    build_features(args.data_parquet, args.pred_parquet, args.out, args.config, args.max_jets)


if __name__ == "__main__":
    main()
