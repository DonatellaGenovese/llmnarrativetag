## 2.3 Setup

Teacher: ParT fine-tuned on TopLandscape. The distillation target is the logit difference t(j) = z_top(j) − z_QCD(j), clipped to the [0.5, 99.5] percentiles of the training distribution, [−9.70, +7.40]. Official file splits, no random resampling:

| Split | Jets |
| --- | --- |
| train | 1 210 999 |
| val | 402 999 |
| test | 403 999 |

The surrogate is fitted on train, selected on val, reported on test. The EBM is purely additive (no pairwise terms), 256 bins per feature, learning rate 0.01, up to 5000 boosting rounds. The linear lower bound is a ridge regression on the same basis with `m`, `m_SD` and `pT` entered logarithmically. The upper bound is an unconstrained gradient-boosted tree ensemble on the raw features.

In the code the three energy correlation ratios carry the names `C2_double_b1`, `D2_double_b1`, `C3_double_b1`, to avoid the collision with the different C₂ used in the quark/gluon literature.

## 2.4 Metrics

Four quantities, measuring agreement with the teacher rather than accuracy on the truth label:

- **R²** of the surrogate against the clipped ParT logit. How much of the teacher's output the surrogate reproduces.
- **Decision agreement**: the fraction of jets on which surrogate and teacher assign the same class, sign(ĝ) = sign(t).
- **Average Decision Ordering (ADO)** with respect to ParT [6]: over pairs of jets drawn one from each truth class, the fraction on which surrogate and teacher order the pair the same way. It is insensitive to any monotone reparametrisation of the score, and so isolates ranking behaviour from calibration.
- **MAE** between surrogate and teacher probabilities.

Accuracy and AUC against the truth label are also reported, but only to locate the surrogate relative to ParT itself: the object being explained is the tagger, not the physics.

## 2.5 The ladder

Test set, 403 999 jets.

| Model | R² | Spearman | ADO vs ParT | Decision agreement | MAE | Acc | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ridge | 0.886 | 0.938 | 0.962 | 0.932 | 0.076 | 0.901 | 0.955 |
| **EBM** | **0.962** | **0.972** | **0.983** | **0.951** | **0.050** | 0.918 | 0.975 |
| GBDT | 0.980 | 0.984 | 0.988 | 0.967 | 0.035 | 0.927 | 0.979 |
| ParT (teacher) | — | — | — | — | — | 0.935 | 0.983 |

Train, validation and test agree to the third decimal for every model — the EBM scores R² = 0.9621 / 0.9619 / 0.9619 — so none of what follows is a generalisation effect.

For scale, ParT's own ADO against the truth ordering is 0.983. The additive surrogate tracks ParT's ranking as closely as ParT tracks the physics.

## 2.6 Gap decomposition

The ladder splits the distance between a linear model and the teacher into three parts, on the test set:

| Component | Measured as | Value |
| --- | --- | --- |
| Non-linearity used | R²(EBM) − R²(Ridge) | 0.0763 ± 0.0000 |
| Cost of additivity | R²(GBDT) − R²(EBM) | 0.0183 ± 0.0000 |
| Basis insufficiency | 1 − R²(GBDT) | 0.0198 ± 0.0000 |

The uncertainties are the standard deviation over five refits at different random seeds, on identical data. Both stochastic models are extremely stable at this sample size: the EBM bags four outer models and the GBDT subsamples 80% of rows and columns, yet R² moves by 2×10⁻⁵ and 1.4×10⁻⁵ respectively. The cost of additivity, the number the additivity claim rests on, is 0.01832 with a standard deviation of 0.00002 — roughly 760 sigma from zero. Ridge is a closed-form convex fit with no seed and no variance.

**ParT's top-tagging decision is almost entirely additive in this basis.** Free shape functions are necessary — a linear model on the same ten observables leaves 7.6 points of R² on the table — but interactions between observables account for under 2 points, and only 2 more lie outside the basis altogether. The interpretability constraint therefore costs very little here: what the surrogate cannot say, the unconstrained model largely cannot say either.

## 2.7 What the surrogate uses

EBM importances (mean absolute contribution), against the univariate correlation of each observable with the teacher logit:

| Observable | EBM importance | corr with t |
| --- | --- | --- |
| `m` | 2.44 | +0.86 |
| `C2_b1` | 1.36 | +0.48 |
| `D2_b1` | 1.31 | −0.61 |
| `n_const` | 1.00 | +0.43 |
| `m_SD` | 0.53 | +0.85 |
| `N3_b1` | 0.52 | −0.44 |
| `tau32` | 0.46 | −0.68 |
| `C3_b1` | 0.29 | −0.27 |
| `pT` | 0.12 | +0.02 |
| `tau21` | 0.10 | −0.32 |

The two columns rank differently, and the disagreement is the point. `m_SD` matches `m` in marginal correlation (0.85 against 0.86) but carries a fifth of its importance: once the ungroomed mass is in the model, the groomed one adds little. `C2_b1` moves the other way, second in importance on a marginal correlation of 0.48. Marginal relevance and actual use are not the same quantity, and only the surrogate reports the second.

## 2.8 Conditioning of the basis

Condition number of the standardised design matrix: **11.6**. Variance inflation factors:

| Observable | VIF |
| --- | --- |
| `m` | 22.1 |
| `m_SD` | 12.0 |
| `C2_b1` | 7.3 |
| `n_const` | 3.5 |
| `D2_b1` | 3.4 |
| `C3_b1` | 2.9 |
| `tau21` | 2.8 |
| `tau32` | 2.6 |
| `N3_b1` | 2.2 |
| `pT` | 1.1 |

The mass pair is collinear, as expected. This does not affect predictive performance, but it does limit how firmly credit can be attributed *between* `m` and `m_SD`, and any statement about their individual contributions should be read with that in mind.

## 2.9 Control: what does distillation buy?

An interpretable model on ten good observables might reach these numbers on its own, without tracking the teacher at all. To test this, the same three models were refitted on the truth label instead of the ParT logit, on identical splits and features. Accuracy / AUC against truth, test set:

| Model | Distilled from ParT | Trained on labels | Δ |
| --- | --- | --- | --- |
| Ridge | 0.901 / 0.955 | 0.911 / 0.965 | +0.010 / +0.010 |
| EBM | 0.918 / 0.975 | 0.922 / 0.977 | +0.004 / +0.002 |
| GBDT | 0.927 / 0.979 | 0.928 / 0.980 | +0.001 / +0.001 |

All numbers carry a binomial standard error of ±0.0004 on 403 999 jets, and the comparison is paired: for the EBM, 3658 jets are called correctly only by the distilled model against 5339 only by the label-trained one, McNemar p = 1.4×10⁻⁷⁰. The gaps are small but not noise.

**This control does not separate the two hypotheses, and it is worth being explicit about why.** The label-trained EBM does not merely match the distilled one on truth: it also agrees with ParT slightly *more* (0.956 against 0.951), and on the 26 125 jets where ParT is wrong it follows ParT into the error marginally more often (0.761 against 0.755). Nothing here shows that distillation is what makes the surrogate track the tagger.

The reason is that ParT is accurate. At 0.935 on truth, its decisions and the truth labels are nearly the same supervision signal once projected onto this observable basis, so fitting either produces almost the same classifier. A control built on binary decisions cannot distinguish targets that nearly coincide.

What distillation does buy is the continuous object the explanation is built from. The local explanation is a decomposition of ĝ, and ĝ is fitted to reproduce ParT's logit — R² = 0.962, ADO = 0.983, decision agreement = 0.951 (§2.5). A classifier trained on labels does not target that logit at all, and its shape functions would decompose a different quantity. The claim that the explanation describes ParT rests on those agreement metrics, not on this table.

Note also that the surrogate agrees with ParT (0.951) more than it is accurate on truth (0.918): it reproduces a substantial share of the tagger's mistakes, which is the behaviour an explanation of ParT should have and an independently good classifier should not.

## 2.10 The residual, and where the explanation should abstain

Intercept β₀ = −0.947. The residual r(j) = t(j) − ĝ(x(j)) on the test set, with t clipped to the same bounds the surrogate was fitted against — the clip touches 1.03% of test jets and moves none of the figures below beyond the third decimal:

| Quantity | Value |
| --- | --- |
| mean | −0.001 |
| std | 1.099 |
| median \|r\| | 0.627 |
| 90th percentile | 1.773 |
| 99th percentile | 3.352 |
| \|r\| < 1 | 69.5% of jets |
| \|r\| < 2 | 92.9% |
| \|r\| > 3 | 1.6% |

The residual is not noise: it identifies the jets the basis fails to describe, and it is calibrated. Restricting to jets below a residual threshold raises the agreement between surrogate and teacher monotonically:

| Threshold | Coverage | Agreement with ParT | Surrogate acc | ParT acc |
| --- | --- | --- | --- | --- |
| none | 100% | 0.951 | 0.918 | 0.935 |
| \|r\| < 2 | 92.9% | 0.970 | 0.933 | 0.940 |
| \|r\| < 1.5 | 85.1% | 0.978 | 0.939 | 0.943 |
| \|r\| < 1 | 69.5% | 0.986 | 0.946 | 0.948 |
| \|r\| < 0.75 | 57.4% | 0.991 | 0.949 | 0.950 |
| \|r\| < 0.5 | 41.5% | 0.994 | 0.952 | 0.952 |
| \|r\| < 0.25 | 21.9% | 0.997 | 0.953 | 0.953 |

The last two columns are given together because the surrogate's rising accuracy is a selection effect — low-residual jets are easier for ParT too. The meaningful quantity is the *gap*: 1.7 accuracy points over the whole test set, 0.1 points at \|r\| < 0.25. Where the residual is small the surrogate is not more accurate than ParT, it is indistinguishable from it, and that is the condition under which an explanation may speak for the tagger.

This table sets the abstention policy used in Part 3. The thresholds are round numbers read off a smooth curve, not an optimum; the bands, measured separately rather than cumulatively:

| Regime | Condition | Coverage | Agreement with ParT | Surrogate acc | ParT acc |
| --- | --- | --- | --- | --- | --- |
| full | \|r\| < 1 | 69.5% | 0.986 | 0.946 | 0.948 |
| caveat | 1 ≤ \|r\| < 2 | 23.3% | 0.921 | 0.894 | 0.915 |
| abstain | \|r\| ≥ 2 | 7.1% | 0.707 | 0.726 | 0.880 |

The abstention band is where the surrogate stops being a stand-in for the tagger: agreement falls to 0.707, and while ParT still classifies those jets at 0.880 the surrogate manages 0.726. These are jets ParT decides on grounds the ten observables do not carry, and no narrative built from the decomposition can describe why.
