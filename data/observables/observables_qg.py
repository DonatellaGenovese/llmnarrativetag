"""QuarkGluon jet observables (Tier A+B+C) from constituents.

Precompute once per jet: pT_jet, z_i, ΔR_i, and pairwise ΔR_ij.
Uses parquet fields: part_deta, part_dphi, part_pt (via px/py), part_charge,
and the five PID one-hot flags.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
from tqdm import tqdm


def _wrap_dphi(dphi: np.ndarray) -> np.ndarray:
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def _safe_entropy(f: np.ndarray) -> float:
    """−∑ f log f with 0·log0 = 0."""
    f = f[f > 0]
    if f.size == 0:
        return 0.0
    return float(-np.sum(f * np.log(f)))


def compute_jet_observables(
    pt: np.ndarray,
    deta: np.ndarray,
    dphi: np.ndarray,
    charge: np.ndarray,
    is_ch: np.ndarray,
    is_nh: np.ndarray,
    is_ph: np.ndarray,
    is_el: np.ndarray,
    is_mu: np.ndarray,
    px: Optional[np.ndarray] = None,
    py: Optional[np.ndarray] = None,
    pz: Optional[np.ndarray] = None,
    energy: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute the full QG basis for one jet. Arrays are per-constituent."""
    n = int(len(pt))
    out = {
        "n_pf": float(n),
        "n_Q": np.nan,
        "w_pf": np.nan,
        "pTD": np.nan,
        "C_02": np.nan,
        "C_1_b1": np.nan,
        "lambda_LHA": np.nan,
        "lambda_21": np.nan,
        "r_lambda": np.nan,
        "S_frag": np.nan,
        "ellipticity": np.nan,
        "mass": np.nan,
        "pT": np.nan,
        "S_PID": np.nan,
        "E_Q": np.nan,
        "Q_05": np.nan,
        "f_ch": np.nan,
        "f_gamma": np.nan,
        "f_nh": np.nan,
        "z_max": np.nan,
    }
    if n == 0:
        return out

    pt = np.asarray(pt, dtype=np.float64)
    deta = np.asarray(deta, dtype=np.float64)
    dphi = _wrap_dphi(np.asarray(dphi, dtype=np.float64))
    charge = np.asarray(charge, dtype=np.float64)

    # mask invalid
    ok = np.isfinite(pt) & (pt > 0) & np.isfinite(deta) & np.isfinite(dphi)
    if not np.any(ok):
        return out
    pt, deta, dphi, charge = pt[ok], deta[ok], dphi[ok], charge[ok]
    is_ch = np.asarray(is_ch, dtype=np.float64)[ok]
    is_nh = np.asarray(is_nh, dtype=np.float64)[ok]
    is_ph = np.asarray(is_ph, dtype=np.float64)[ok]
    is_el = np.asarray(is_el, dtype=np.float64)[ok]
    is_mu = np.asarray(is_mu, dtype=np.float64)[ok]
    n = int(len(pt))

    pT_jet = float(np.sum(pt))
    out["pT"] = pT_jet
    out["n_pf"] = float(n)
    if pT_jet <= 0:
        return out

    z = pt / pT_jet
    dR = np.sqrt(deta * deta + dphi * dphi)

    out["n_Q"] = float(np.sum(np.abs(charge) > 0))
    out["w_pf"] = float(np.sum(z * dR))
    out["pTD"] = float(np.sqrt(np.sum(z * z)))
    out["lambda_LHA"] = float(np.sum(z * np.sqrt(dR)))  # ΔR^0.5
    out["lambda_21"] = float(np.sum(z * z * dR))
    out["r_lambda"] = (
        out["lambda_LHA"] / out["lambda_21"] if out["lambda_21"] > 0 else np.nan
    )
    # Shannon entropy of z (0 log 0 = 0)
    z_pos = z[z > 0]
    out["S_frag"] = float(-np.sum(z_pos * np.log(z_pos))) if z_pos.size else 0.0

    # Jet charge κ=0.5: ∑ q_i z_i^{0.5}
    out["Q_05"] = float(np.sum(charge * np.sqrt(z)))
    out["z_max"] = float(np.max(z))

    # C_0.2 and C_1^{β=1}: ∑_{i<j} pTi pTj (ΔRij)^β / pT_jet^2
    if n >= 2:
        deta_i = deta[:, None]
        deta_j = deta[None, :]
        dphi_ij = _wrap_dphi(dphi[:, None] - dphi[None, :])
        dR_ij = np.sqrt((deta_i - deta_j) ** 2 + dphi_ij**2)
        iu = np.triu_indices(n, k=1)
        pt_ij = pt[iu[0]] * pt[iu[1]]
        dR_u = dR_ij[iu]
        out["C_02"] = float(np.sum(pt_ij * (dR_u ** 0.2)) / (pT_jet**2))
        out["C_1_b1"] = float(np.sum(pt_ij * dR_u) / (pT_jet**2))
    else:
        out["C_02"] = 0.0
        out["C_1_b1"] = 0.0

    # Ellipticity from transverse inertia tensor
    r2 = dR * dR
    mask_r = r2 > 1e-24
    if np.any(mask_r):
        w = pt[mask_r]
        de = deta[mask_r]
        dp = dphi[mask_r]
        r2m = r2[mask_r]
        # I_ij = ∑ pT * r_i r_j / r^2
        I_ee = np.sum(w * (de * de) / r2m)
        I_pp = np.sum(w * (dp * dp) / r2m)
        I_ep = np.sum(w * (de * dp) / r2m)
        # eigenvalues of [[I_ee, I_ep],[I_ep, I_pp]]
        tr = I_ee + I_pp
        det = I_ee * I_pp - I_ep * I_ep
        disc = max(tr * tr - 4.0 * det, 0.0)
        sqrt_disc = math.sqrt(disc)
        chi_max = 0.5 * (tr + sqrt_disc)
        chi_min = 0.5 * (tr - sqrt_disc)
        out["ellipticity"] = float(chi_min / chi_max) if chi_max > 0 else np.nan
    else:
        out["ellipticity"] = np.nan

    # Mass from massless 4-vectors using absolute eta/phi if px,py,pz,E given;
    # else reconstruct from pT, deta, dphi relative to jet — need jet eta/phi.
    # Prefer cartesian 4-momenta when provided.
    if px is not None and py is not None and pz is not None and energy is not None:
        px_ = np.asarray(px, dtype=np.float64)[ok]
        py_ = np.asarray(py, dtype=np.float64)[ok]
        pz_ = np.asarray(pz, dtype=np.float64)[ok]
        E_ = np.asarray(energy, dtype=np.float64)[ok]
        E_sum = float(np.sum(E_))
        px_sum = float(np.sum(px_))
        py_sum = float(np.sum(py_))
        pz_sum = float(np.sum(pz_))
        m2 = E_sum * E_sum - px_sum * px_sum - py_sum * py_sum - pz_sum * pz_sum
        out["mass"] = float(math.sqrt(max(m2, 0.0)))
    else:
        out["mass"] = np.nan

    # S_PID + explicit PID fractions (number fractions)
    n_el = float(np.sum(is_el > 0.5))
    n_mu = float(np.sum(is_mu > 0.5))
    n_ch = float(np.sum(is_ch > 0.5))
    n_nh = float(np.sum(is_nh > 0.5))
    n_ph = float(np.sum(is_ph > 0.5))
    counts = np.array([n_el, n_mu, n_ch, n_nh, n_ph], dtype=np.float64)
    out["S_PID"] = _safe_entropy(counts / n) if n > 0 else 0.0
    out["f_ch"] = n_ch / n
    out["f_gamma"] = n_ph / n
    out["f_nh"] = n_nh / n

    # E_Q: charged energy fraction; E_i = pT cosh(eta_abs) ≈ energy if available
    if energy is not None:
        E_ = np.asarray(energy, dtype=np.float64)[ok]
        E_jet = float(np.sum(E_))
        charged = np.abs(charge) > 0
        out["E_Q"] = float(np.sum(E_[charged]) / E_jet) if E_jet > 0 else np.nan
    else:
        # fallback using pT only (underestimates for forward particles)
        charged = np.abs(charge) > 0
        out["E_Q"] = float(np.sum(pt[charged]) / pT_jet)

    return out


OBS_COLUMNS = [
    "n_pf",
    "n_Q",
    "w_pf",
    "pTD",
    "C_02",
    "C_1_b1",
    "lambda_LHA",
    "lambda_21",
    "r_lambda",
    "S_frag",
    "ellipticity",
    "mass",
    "pT",
    "S_PID",
    "E_Q",
    "Q_05",
    "f_ch",
    "f_gamma",
    "f_nh",
    "z_max",
]


def compute_observables_table(
    part_px,
    part_py,
    part_pz,
    part_energy,
    part_deta,
    part_dphi,
    part_charge,
    part_isChargedHadron,
    part_isNeutralHadron,
    part_isPhoton,
    part_isElectron,
    part_isMuon,
    max_jets: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """Vector of jets (awkward or list-of-arrays) → column dict."""
    import awkward as ak

    n = len(part_px) if max_jets is None else min(max_jets, len(part_px))
    cols = {k: np.empty(n, dtype=np.float64) for k in OBS_COLUMNS}

    it = range(n)
    if show_progress:
        it = tqdm(it, desc="observables", unit="jet")

    for i in it:
        px = np.asarray(part_px[i], dtype=np.float64)
        py = np.asarray(part_py[i], dtype=np.float64)
        pz = np.asarray(part_pz[i], dtype=np.float64)
        E = np.asarray(part_energy[i], dtype=np.float64)
        pt = np.sqrt(px * px + py * py)
        obs = compute_jet_observables(
            pt=pt,
            deta=np.asarray(part_deta[i], dtype=np.float64),
            dphi=np.asarray(part_dphi[i], dtype=np.float64),
            charge=np.asarray(part_charge[i], dtype=np.float64),
            is_ch=np.asarray(part_isChargedHadron[i], dtype=np.float64),
            is_nh=np.asarray(part_isNeutralHadron[i], dtype=np.float64),
            is_ph=np.asarray(part_isPhoton[i], dtype=np.float64),
            is_el=np.asarray(part_isElectron[i], dtype=np.float64),
            is_mu=np.asarray(part_isMuon[i], dtype=np.float64),
            px=px,
            py=py,
            pz=pz,
            energy=E,
        )
        for k in OBS_COLUMNS:
            cols[k][i] = obs[k]
    return cols
