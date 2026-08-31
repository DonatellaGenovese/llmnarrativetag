"""TopLandscape jet observables via FastJet + fjcontrib."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .fjcontrib_loader import load_fjcontrib


@dataclass(frozen=True)
class TopObsConfig:
    R0: float = 0.8
    sd_z_cut: float = 0.1
    sd_beta: float = 0.0
    tau_beta: float = 1.0
    ecf_beta: float = 1.0
    truncate_C3: int = 50
    truncate_N3: int = 40


def _as_std_vector_pseudojet(fj, std, px, py, pz, E):
    parts = std.vector[fj.PseudoJet]()
    n = len(px)
    for i in range(n):
        # skip null/invalid
        if not (math.isfinite(E[i]) and E[i] > 0):
            continue
        parts.push_back(fj.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(E[i])))
    return parts


def _leading_pt_jet(fj, std, jet, n_keep: int):
    """Return a PseudoJet built from the n_keep hardest constituents."""
    consts = jet.constituents()
    n = int(consts.size())
    if n == 0:
        return jet
    order = sorted(range(n), key=lambda i: consts[i].pt(), reverse=True)
    keep = order[: min(n_keep, n)]
    parts = std.vector[fj.PseudoJet]()
    for i in keep:
        parts.push_back(consts[i])
    if parts.size() == 0:
        return jet
    # join without reclustering: vector sum via join helper
    return fj.join(parts)


def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    if b == 0.0 or not math.isfinite(b) or not math.isfinite(a):
        return default
    return a / b


class TopObservableComputer:
    """Compute the pre-registered TopLandscape Tier A+B basis for one jet."""

    def __init__(self, cfg: Optional[TopObsConfig] = None):
        self.cfg = cfg or TopObsConfig()
        self.fj, self.std = load_fjcontrib()
        fj = self.fj
        beta = self.cfg.tau_beta
        R0 = self.cfg.R0
        ecf_beta = self.cfg.ecf_beta

        self._jetdef_ca = fj.JetDefinition(fj.cambridge_algorithm, R0)
        self._softdrop = fj.contrib.SoftDrop(self.cfg.sd_beta, self.cfg.sd_z_cut, R0)

        axes = fj.contrib.OnePass_KT_Axes()
        measure = fj.contrib.NormalizedMeasure(beta, R0)
        self._tau1 = fj.contrib.Nsubjettiness(1, axes, measure)
        self._tau2 = fj.contrib.Nsubjettiness(2, axes, measure)
        self._tau3 = fj.contrib.Nsubjettiness(3, axes, measure)
        self._tau21 = fj.contrib.NsubjettinessRatio(2, 1, axes, measure)
        self._tau32 = fj.contrib.NsubjettinessRatio(3, 2, axes, measure)

        measure_ecf = fj.contrib.EnergyCorrelator.pt_R
        self._C2 = fj.contrib.EnergyCorrelatorC2(ecf_beta, measure_ecf)
        self._D2 = fj.contrib.EnergyCorrelatorD2(ecf_beta, measure_ecf)
        self._C3 = fj.contrib.EnergyCorrelatorCseries(3, ecf_beta, measure_ecf)
        self._N3 = fj.contrib.EnergyCorrelatorN3(ecf_beta, measure_ecf)

    def compute_from_constituents(
        self,
        px: Sequence[float],
        py: Sequence[float],
        pz: Sequence[float],
        E: Sequence[float],
        jet_pt: Optional[float] = None,
        jet_mass: Optional[float] = None,
    ) -> Dict[str, float]:
        fj, std = self.fj, self.std
        parts = _as_std_vector_pseudojet(fj, std, px, py, pz, E)
        n_const = int(parts.size())
        out = {
            "m": np.nan,
            "m_SD": np.nan,
            "tau21": np.nan,
            "tau32": np.nan,
            "C2_double_b1": np.nan,
            "D2_double_b1": np.nan,
            "C3_double_b1": np.nan,
            "N3_b1": np.nan,
            "pT": np.nan if jet_pt is None else float(jet_pt),
            "n_const": float(n_const),
        }
        if n_const == 0:
            return out

        # Recluster with C/A (SoftDrop requirement); take hardest inclusive jet
        cs = fj.ClusterSequence(parts, self._jetdef_ca)
        jets = cs.inclusive_jets(0.0)
        if jets.size() == 0:
            return out
        # pick the jet with largest pT
        jet = jets[0]
        for i in range(1, int(jets.size())):
            if jets[i].pt() > jet.pt():
                jet = jets[i]

        out["pT"] = float(jet.pt()) if jet_pt is None else float(jet_pt)
        out["m"] = float(jet.m()) if jet_mass is None else float(jet_mass)

        try:
            groomed = self._softdrop(jet)
            out["m_SD"] = float(groomed.m())
        except Exception:
            out["m_SD"] = np.nan

        try:
            out["tau21"] = float(self._tau21(jet))
            out["tau32"] = float(self._tau32(jet))
        except Exception:
            # fallback to explicit ratio
            t1 = float(self._tau1(jet))
            t2 = float(self._tau2(jet))
            t3 = float(self._tau3(jet))
            out["tau21"] = _safe_div(t2, t1)
            out["tau32"] = _safe_div(t3, t2)

        try:
            out["C2_double_b1"] = float(self._C2(jet))
            out["D2_double_b1"] = float(self._D2(jet))
        except Exception:
            pass

        try:
            jet50 = _leading_pt_jet(fj, std, jet, self.cfg.truncate_C3)
            out["C3_double_b1"] = float(self._C3(jet50))
        except Exception:
            pass

        try:
            jet40 = _leading_pt_jet(fj, std, jet, self.cfg.truncate_N3)
            out["N3_b1"] = float(self._N3(jet40))
        except Exception:
            pass

        return out


def compute_observables_table(
    part_px: Iterable,
    part_py: Iterable,
    part_pz: Iterable,
    part_E: Iterable,
    jet_pt: Optional[np.ndarray] = None,
    jet_mass: Optional[np.ndarray] = None,
    cfg: Optional[TopObsConfig] = None,
    max_jets: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """Vector of jagged constituent arrays → dict of observable columns."""
    computer = TopObservableComputer(cfg)
    keys = [
        "m",
        "m_SD",
        "tau21",
        "tau32",
        "C2_double_b1",
        "D2_double_b1",
        "C3_double_b1",
        "N3_b1",
        "pT",
        "n_const",
    ]
    cols: Dict[str, List[float]] = {k: [] for k in keys}

    n = len(part_px)
    if max_jets is not None:
        n = min(n, max_jets)
    iterator = range(n)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="observables", unit="jet")

    for i in iterator:
        px = np.asarray(part_px[i], dtype=np.float64)
        py = np.asarray(part_py[i], dtype=np.float64)
        pz = np.asarray(part_pz[i], dtype=np.float64)
        E = np.asarray(part_E[i], dtype=np.float64)
        pt_i = None if jet_pt is None else float(jet_pt[i])
        m_i = None if jet_mass is None else float(jet_mass[i])
        obs = computer.compute_from_constituents(px, py, pz, E, jet_pt=pt_i, jet_mass=m_i)
        for k in keys:
            cols[k].append(obs[k])

    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
