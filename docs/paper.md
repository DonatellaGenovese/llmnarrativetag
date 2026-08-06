# A Verified Natural Language Explanation of Jet Tagger Behavior

## Project description

In the context of Jet tagging, common Deep taggers that operating directly on jet constituents (ParticleNet [1], Particle Transformer (ParT) [2]) substantially outperform classifiers built on a small set of high-level substructure observables, as established by the common benchmarks of the top-tagging task [3,4]. This gain comes at a cost in transparency. The input is an unordered set of O(100) four-vectors, and an attribution assigned to an individual constituent carries no standalone physical meaning: explaining such a model in terms of its own inputs is of limited use to a physicist. The response within the field has been to project the model's behaviour onto expert observables via augmented inputs and layerwise relevance propagation [5], decision-ordering agreement with human-readable variables [6], or feature selection against a physics basis [7,8].

Given that the tagger is trained on constituents, the strategy presented in this project is to re-express its behaviour in a physical space. In particulaer we fit an **interpretable surrogate** defined on substructure observables and trained to reproduce ParT's output which aim to characterise what the tagger's decisions track.

As a second step, we translate the explanation given by the tagger into natural language. LLMs have been used as a post-hoc layer turning XAI artefacts into readable narratives [9,10], but in domains with no ground truth against which claims can be checked [11]. Here the observable basis provides one, and the narrative is constrained accordingly, such that quantitative claims are verified against the surrogate, and the system abstains where the surrogate fails.

## Part 1 (Dataset and Observables)

We use the two benchmarks on which ParT is evaluated in [2], chosen because they probe physically distinct discrimination mechanisms.

**TopLandscape** [12] is the top quark tagging reference dataset assembled for the comparative study of [3]: hadronically decaying top jets against a QCD dijet background. Discrimination here is driven by the presence of a hard three-prong structure at a fixed mass scale.

**QuarkGluon** [4,14] contains light-quark and gluon jets from Z(→νν)+jet events, with the jet flavour set by the parton in the hard process. Discrimination here is driven not by a hard substructure but by the overall radiation profile of the jet: the Casimir ratio C_A/C_F makes gluon jets broader and higher in multiplicity than quark jets [15].

The consequence for this work is that **the two observable bases must differ**, and are not interchangeable: prong counting for top, fragmentation and radiation pattern for quark/gluon.

**TopLandscape observables**

| Name | Definition | Tier | Ref. |
| --- | --- | --- | --- |
| `m` | Ungroomed jet mass | A | [3,5,7] |
| `m_SD` | Soft-drop mass (`z_cut` = 0.1, β = 0, R₀ = 0.8) | A | [3,5,8] |
| `tau21` | N-subjettiness ratio τ₂/τ₁ (β = 1, one-pass kT axes) | A | [5,6,16] |
| `tau32` | N-subjettiness ratio τ₃/τ₂ (β = 1, one-pass kT axes) | A | [3,5,7] |
| `C2_b1` | Energy correlation ratio C₂ (β = 1) | A | [5,8,19] |
| `D2_b1` | Energy correlation ratio D₂ (β = 1) | A | [5,6,8] |
| `C3_b1` | Energy correlation ratio C₃ (β = 1, truncated at 50 constituents) | A | [5,8] |
| `N3_b1` | Generalized correlator N₃ (β = 1, truncated at 40 constituents) | A | [8,20] |
| `pT` | Jet transverse momentum | A | [3,5] |
| `n_const` | Constituent multiplicity | B | [3,21] |

**Quark Gluon observables**

| Name | Definition | Tier | Ref. |
| --- | --- | --- | --- |
| `n_pf` | Constituent multiplicity | B | [21,23,25] |
| `n_Q` | Charged constituent multiplicity | B | [23,25] |
| `w_pf` | Girth / jet width, λ¹₁ = Σ zᵢ ΔRᵢ | A | [8,23,25] |
| `pTD` | Σ zᵢ² (equivalently λ²₀) | A | [23,25] |
| `C_02` | Two-point energy correlator (β = 0.2) | A | [8,24] |
| `lambda_LHA` | Les Houches Angularity λ¹₀.₅ (κ = 1, β = 0.5) | A | [8,28] |
| `lambda_21` | Generalized angularity λ²₁ (κ = 2, β = 1) | A | [8] |
| `r_lambda` | Ratio λ_LHA / λ₂₁ (width-decorrelated) | A | derived — defined in text |
| `S_frag` | Shannon entropy of the momentum fractions zᵢ | A | defined in text |
| `ellipticity` | χ_min / χ_max of the constituent distribution | A | defined in text |
| `mass` | Jet mass from constituent four-vectors | A | [23,25] |
| `pT` | Jet transverse momentum | A | [25] |
| `S_PID` | Shannon entropy of the PID fractions (5 categories) | C | [4] |
| `E_Q` | Charged energy fraction | C | [4,22] |

| Name | Definition | Ref. |
| --- | --- | --- |
| `Q_05` | Jet charge, κ = 0.5 | [26,27] |
| `f_ch` | Number fraction of charged constituents | [4,22] |
| `f_gamma` | Number fraction of photon constituents | [4,22] |
| `f_nh` | Number fraction of neutral-hadron constituents | [4,22] |
| `z_max` | Largest constituent momentum fraction | defined in text |
| `C_1_b1` | Two-point energy correlator (β = 1) | [8,24] |

Description of datasets

## Part 2 (XAI Techniques)

Post-hoc attribution methods are confined to the space the model reads, and for ParT that space is the constituent list, where a single particle is a stochastic product of hadronisation, carries no correspondence across jets, and matches none of the questions physicists actually ask. So we need an explanation in the observable space. To achieve this we use a surrogate model trained with distillation [29]: ParT is the teacher, and an interpretable student is trained to reproduce its logit rather than the truth label.

#### 2.1 The surrogate: GAMs and EBMs

A generalized additive model (GAM) [30] keeps the additive structure of a linear model while freeing the functional form:

$$
\hat{g}(\mathbf{x}) = \beta_0 + \sum_{k=1}^{K} f_k(x_k)
$$

where each shape function $f_k$ is arbitrary. Interpretability follows from dimensionality: each $f_k$ is a function of one variable and can be plotted exactly, and that plot is the complete account of the model's dependence on observable $k$, with no hidden conditioning on the others. The price is the absence of interactions, which are pushed into the residual.

Variants differ in how the shape functions are estimated. Spline-based GAMs enforce smoothness, a poor match here since a physical threshold (the edge of the top mass window) would be smoothed into a transition that does not exist. Tree-based GAMs [31] produce step functions and can represent sharp thresholds. We use the **Explainable Boosting Machine** (EBM) [32,33], fit by cyclic round-robin boosting. EBMs reach accuracy comparable to unconstrained models on tabular problems while remaining exactly decomposable [33].

### 2.2 Definition of the explanation

Let $z(j)$ be ParT's logit for jet $j$ and $\mathbf{x}(j)$ its vector of $k$ observables. We can provide a global explanaton by plotting the shape functions over its range; $f_k(x)$ is the contribution to ParT's logit when observable $k$ takes value $x$.

We can also provide a **local explanation:** for a single jet, the vector of contributions $\varphi_k(J) = f_k(x_k(j))$ together with $\beta_0$, satisfying

$$
\beta_0 + \sum_{k=1}^{K} \varphi_k(j) = \hat{g}(\mathbf{x}(j))
$$

i.e. the local explanation is the global one evaluated at a point. Finally we can also compute the **residual:**

$$
r(j) = z(j) - \hat{g}(\mathbf{x}(j))
$$

which can be seen as the part of the decision the observable basis does not express.

We measure the performance of the surrogate model against the teacher using: the $R^2$ of the surrogate against the ParT logit, decision agreement, and the average decision ordering with respect to ParT [6], which compares rankings and connects to the existing HEP interpretability literature.

To investigate the usage of linear surrogaye we fit a pre-registered ladder of models on the same observable basis:

1. A linear regressor as a lower bound
2. the EBM;
3. a GA²M with a limited number of pairwise interactions [32], purified by functional ANOVA to keep the terms identifiable [34];
4. an unconstrained gradient-boosted model, not interpretable, as an upper bound.

The gap between (2) and (4) measures how much of ParT's behaviour requires non-additivity.

## Part 3 (LLM Narrative)

This part is deticated on define the LLM narrative. We structured the prompt as [TODO] and we use the model [TODO]. [TODO]

## Part 4 (LLM Narrative evaluation)

## Part 5 (Additional challenges)

Additional tests could be conducted on the JetClass dataset and also we want to explore what happens on the quality of narration if we also add a RAG.

## References

[1] H. Qu and L. Gouskos, *ParticleNet: Jet Tagging via Particle Clouds*, Phys. Rev. D **101** (2020) 056019, arXiv:1902.08570.

[2] H. Qu, C. Li and S. Qian, *Particle Transformer for Jet Tagging*, Proc. 39th ICML, PMLR **162** (2022) 18281, arXiv:2202.03772.

[3] G. Kasieczka, T. Plehn, A. Butter et al., *The Machine Learning Landscape of Top Taggers*, SciPost Phys. **7** (2019) 014, arXiv:1902.09914.

[4] P. T. Komiske, E. M. Metodiev and J. Thaler, *Energy Flow Networks: Deep Sets for Particle Jets*, JHEP **01** (2019) 121, arXiv:1810.05165.

[5] G. Agarwal, L. Hay, I. Iashvili et al., *Explainable AI for ML jet taggers using expert variables and layerwise relevance propagation*, JHEP **05** (2021) 208, arXiv:2011.13466.

[6] T. Faucett, J. Thaler and D. Whiteson, *Mapping machine-learned physics into a human-readable space*, Phys. Rev. D **103** (2021) 036020, arXiv:2010.11998.

[7] A. Khot, M. S. Neubauer and A. Roy, *A detailed study of interpretability of deep neural network based top taggers*, Mach. Learn. Sci. Tech. **4** (2023) 035003, arXiv:2210.04371.

[8] R. Das, G. Kasieczka and D. Shih, *Feature selection with distance correlation*, Phys. Rev. D **109** (2024) 054009, arXiv:2212.00046.

[9] D. Martens, J. Hinns, C. Dams, M. Vergouwen and T. Evgeniou, *Tell me a story! Narrative-driven XAI with Large Language Models*, Decision Support Systems **191** (2025) 114218, arXiv:2309.17057.

[10] A. Zytek, S. Pidò, S. Alnegheimish et al., *Explingo: Explaining AI Predictions using Large Language Models*, IEEE BigData (2024), arXiv:2412.05145.

[11] T. Ichmoukhamedov, J. Hinns and D. Martens, *How good is my story? Towards quantitative metrics for evaluating LLM-generated XAI narratives*, arXiv:2412.10220.

[12] G. Kasieczka, T. Plehn, J. Thompson and M. Russell, *Top Quark Tagging Reference Dataset*, Zenodo (2019).

[13] M. Cacciari, G. P. Salam and G. Soyez, *The anti-k_t jet clustering algorithm*, JHEP **04** (2008) 063, arXiv:0802.1189.

[14] P. T. Komiske, E. M. Metodiev and J. Thaler, *Pythia8 Quark and Gluon Jets for Energy Flow*, Zenodo (2019).

[15] M. Cacciari, G. P. Salam and G. Soyez, *FastJet user manual*, Eur. Phys. J. C **72** (2012) 1896, arXiv:1111.6097.

[16] K. Datta and A. Larkoski, *How Much Information is in a Jet?*, JHEP **06** (2017) 073, arXiv:1704.08249.

[17] K. Datta and A. J. Larkoski, *Novel Jet Observables from Machine Learning*, JHEP **03** (2018) 086, arXiv:1710.01305.

[18] K. Datta, A. Larkoski and B. Nachman, *Automating the Construction of Jet Observables with Machine Learning*, Phys. Rev. D **100** (2019) 095016, arXiv:1902.07180.

[19] G. Kasieczka, S. Marzani, G. Soyez and G. Stagnitto, *Towards machine learning analytics for jet substructure*, JHEP **09** (2020) 195, arXiv:2007.04319.

[20] A. Chakraborty, S. H. Lim, M. M. Nojiri and M. Takeuchi, *Neural network-based top tagger with two-point energy correlations and geometry of soft emissiki, J. Mulligan, M. Płoskoń and F. Ringer, *Is infrared-collinear safe information all you need for jet classification?*, arXiv:2305.08979.

[22] P. T. Komiske, E. M. Metodiev and M. D. Schwartz, *Deep learning in color: towards automated quark/gluon jet discrimination*, JHEP **01** (2017) 110, arXiv:1612.01551.

[23] H. Luo, M.-X. Luo, K. Wang, T. Xu and G. Zhu, *Quark jet versus gluon jet: deep neural networks with high-level features*, arXiv:1712.03634.

[24] P. T. Komiske, E. M. Metodiev and J. Thaler, *Energy Flow Polynomials: a complete linear basis for jet substructure*, JHEP **04** (2018) 013, arXiv:1712.07124.

[25] G. Kasieczka, N. Kiefer, T. Plehn and J. M. Thompson, *Quark-Gluon Tagging: Machine Learning vs Detector*, SciPost Phys. **6** (2019) 069, arXiv:1812.09223.

[26] K. Fraser and M. D. Schwartz, *Jet Charge and Machine Learning*, JHEP **10** (2018) 093, arXiv:1803.08066.

[27] Y.-C. J. Chen, C.-W. Chiang, G. Cottin and D. Shih, *Boosted W and Z tagging with jet charge and deep learning*, Phys. Rev. D **101** (2020) 053001, arXiv:1908.08256.

[28] K. Lee, J. Mulligan, M. Płoskoń, F. Ringer and F. Yuan, *Machine learning-based jet and event classification at the Electron-Ion Collider*, JHEP **03** (2023) 085, arXiv:2210.06450.

[29] G. Hinton, O. Vinyals and J. Dean, *Distilling the Knowledge in a Neural Network*, arXiv:1503.02531.

[30] T. Hastie and R. Tibshirani, *Generalized Additive Models*, Statistical Science **1** (1986) 297.

[31] Y. Lou, R. Caruana and J. Gehrke, *Intelligible Models for Classification and Regression*, KDD (2012) 150.

[32] Y. Lou, R. Caruana, J. Gehrke and G. Hooker, *Accurate Intelligible Models with Pairwise Interactions*, KDD (2013) 623.

[33] H. Nori, S. Jenkins, P. Koch and R. Caruana, *InterpretML: A Unified Framework for Machine Learning Interpretability*, arXiv:1909.09223.

[34] B. Lengerich, S. Tan, C.-H. Chang, G. Hooker, and R. Caruana, *Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models*, Proc. AISTATS, PMLR **108** (2020) 2582, arXiv:1911.04974.
