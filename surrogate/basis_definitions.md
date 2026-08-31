# TopLandscape observable basis — pre-registered definitions

Dataset: TopLandscape (top vs QCD), anti-kT R = 0.8, pT ∈ [550, 650] GeV.
Teacher target: `t = z_top - z_qcd` (= `logit_diff` from ParT dump).
Constituents: original parquet 4-vectors (not weaver-padded `pf_*`).

## SoftDrop (`m_SD`)

- Recluster with Cambridge/Aachen.
- SoftDrop parameters: **z_cut = 0.1**, **β_SD = 0**, **R₀ = 0.8**.
- Implementation: `fastjet::contrib::SoftDrop` via cppyy.
- Also keep ungroomed mass `m` (= `jet_mass` from file / PseudoJet mass).

## N-subjettiness

- β = **1**, R₀ = **0.8**.
- Axes: **OnePass_KT_Axes**.
- Measure: `NormalizedMeasure(β=1, R0=0.8)`.
- Ratios: `tau21 = τ₂/τ₁`, `tau32 = τ₃/τ₂`.
- Implementation: `fastjet::contrib::Nsubjettiness` / `NsubjettinessRatio`.

## Energy correlators (β = 1, pt_R measure)

Unambiguous names (avoid C₂ naming collision with QuarkGluon papers):

| Internal name | Definition | fjcontrib class |
|---|---|---|
| `C2_double_b1` | ECF(3) ECF(1) / ECF(2)² | `EnergyCorrelatorC2(1)` |
| `D2_double_b1` | ECF(3) ECF(1)³ / ECF(2)³ | `EnergyCorrelatorD2(1)` |
| `C3_double_b1` | ECF(4) ECF(2) / ECF(3)² | `EnergyCorrelatorCseries(3, 1)` |
| `N3_b1` | ECFG(2,4)/ECFG(1,3)² | `EnergyCorrelatorN3(1)` |

### Truncation (cost control, following [D])

- `C3_double_b1`: leading **50** constituents by pT.
- `N3_b1`: leading **40** constituents by pT.
- SoftDrop, τ, C2, D2, m, m_SD: **all** constituents.
- Recorded as an approximation.

## Other

- `pT`: jet transverse momentum (nearly flat in this window; kept for stratification).
- `n_const`: constituent multiplicity (Tier B, IRC-unsafe).

## Tier labels

- Tier A (IRC-safe core): `m`, `m_SD`, `tau21`, `tau32`, `C2_double_b1`, `D2_double_b1`, `C3_double_b1`, `N3_b1`, `pT`
- Tier B: `n_const`
