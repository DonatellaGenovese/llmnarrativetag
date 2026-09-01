# Generating Reliable XAI Narratives for Deep Classifiers in High-Energy Physics

This repository turns the decision of a jet tagger into a natural-language explanation. The tagger is distilled into an interpretable surrogate over physical observables, an LLM narrates that surrogate under a tagging contract, and a deterministic parser delivers the narrative only if every tag matches.

## Pipeline

![Verified narrative generation pipeline](figures/pipeline.png)

ParT is distilled into an EBM surrogate; local artifacts are formatted into a tagged prompt and passed to an LLM. A deterministic parser checks each tag against the artifact, and the narrative is delivered only if all checks pass. The system abstains before generation when the surrogate fails to reproduce the tagger locally.

## Case Study

Jet tagging on [TopLandscape](https://zenodo.org/record/2603256), hadronically decaying top jets against a QCD dijet background, anti-kT R = 0.8, pT in [550, 650] GeV, on the official splits of 1.2 M train, 400 k validation
and 400 k test jets.

The tagger being explained is the [Particle Transformer](https://arxiv.org/abs/2202.03772), pretrained on JetClass
and fine-tuned on TopLandscape.

Four LLMs are benchmarked as narrators (Gemini 3.5 Flash, Gemini 3.5 Flash-Lite, DeepSeek-v4-Flash, GPT-OSS-120B) and Claude Sonnet 5 serves as an LLM-as-a-judge, validated against two human annotators.

## Getting Started

Two environments, because the teacher and the rest have incompatible dependency stacks. Everything is run from the repository root.

```bash
# Observables, surrogate, narrative, evaluation
conda env create -f environment.yml
conda activate part-surrogate
export PYTHONPATH="$PWD"

# ParT inference only
python -m venv particle_transformer/.venv
source particle_transformer/.venv/bin/activate
pip install torch weaver-core awkward
```

FastJet and fjcontrib come from conda-forge rather than pip, because the observables are computed through cppyy against the C++ headers and shared libraries, which `fjcontrib_loader.py` resolves under `sys.prefix`.

### 1) Fine-tuning ParT

`particle_transformer/` vendors the upstream toolkit at commit `2925bdb`: the network definitions, the dataloader, the weaver configs and the train scripts, copied verbatim. Fetch the dataset and fine-tune the teacher there.

```bash
cd particle_transformer
./get_datasets.py TopLandscape -d datasets
./train_TopLandscape.sh ParT-FineTune kin
cd ..
```

The fine-tune starts from upstream's JetClass-pretrained `models/ParT_kin.pt`, so that file has to be in place first. This repository never trains on JetClass itself.

Then dump the teacher logits. `predict_TopLandscape_logits.py` exists because weaver's own `--predict` applies a softmax before writing and never exports the raw logits, which are the quantity the surrogate is fit to.

```bash
source particle_transformer/.venv/bin/activate
./scripts/infer_TopLandscape_logits.sh ParT-FineTune <checkpoint>   # test split
./scripts/dump_toplandscape_train_val_logits.sh                     # train and validation
```

Both scripts step into `particle_transformer/` on their own, since weaver resolves its configs relative to that tree, and both discover the timestamped run directory rather than assume one.

### 2) Fitting the surrogate

Ten substructure observables, pre-registered in [`surrogate/basis_definitions.md`](surrogate/basis_definitions.md), are computed with FastJet and joined to the ParT logits. Three models are then fit on the logit difference `t(j) = z_top(j) - z_qcd(j)`: Ridge as an additive-linear lower bound, an EBM at zero interactions as the interpretable surrogate, and GBDT unconstrained as an upper bound.

```bash
conda activate part-surrogate
./surrogate/scripts/run_official_top.sh        # features, then Ridge / EBM / GBDT
python -m surrogate.eval_classifier_metrics --predictions surrogate/outputs/top/models/test_predictions.parquet
python -m surrogate.seed_variance              # ADO, and each gap's stability across seeds
./surrogate/scripts/run_label_classifiers.sh   # control: the same models fit on truth labels
```

Feature tables and fitted models stay local under `surrogate/outputs/`. The numbers they produce are copied into [`surrogate/metrics/top/`](surrogate/metrics/top/) and tracked.

## Generating and verifying narratives

The EBM decomposition of one jet becomes an artifact, and the narrator must wrap every artifact-derived number and every threshold-governed word in a labelled tag. Checking a claim is then a regular expression and an equality test, with no second model in the loop.

```bash
python -m narrative.stats                      # reference percentiles, once per surrogate fit
python -m narrative.artefact --validate 3000   # offline check, no API key needed

python -m narrative.orchestrator \
    --backend vertex --model gemini-3.5-flash-lite \
    --jets-file narrative/outputs/top/narratives/run100_jets.json \
    --out narrative/outputs/top/narratives/scratch_gemini-3.5-flash-lite.jsonl

python -m narrative.blocks_report              # the pass-rate table
```

`run100_jets.json` fixes which hundred jets a run covers. Every model must narrate the same jets or half of any difference between them is sampling noise, so the sample is drawn once into that file and loaded verbatim afterwards.

`blocks_report.py` re-verifies every narrative rather than trusting the verdict each run recorded, since a rate is only a rate if it comes from one contract.

| `--backend` | protocol | credential |
|---|---|---|
| `aistudio` | google-genai | `GOOGLE_API_KEY` |
| `vertex` | google-genai | application default credentials |
| `deepseek` | OpenAI-compatible | `DEEPSEEK_API_KEY` |
| `groq` | OpenAI-compatible | `GROQ_API_KEY` |
| `ollama` | OpenAI-compatible | `OLLAMA_API_KEY` |

## LLM-as-a-judge

What verification cannot settle is scored 0 to 3 on three dimensions by Claude Sonnet 5, and by two human annotators on a twenty-item subsample. Both see the narrative alone, without the artifact.

```bash
./run_judges.sh                                # scores a block, resumable
python -m evaluation.judge_report              # scored CSVs into evaluation/results/
python -m evaluation.judge_report --by-model   # mean per generator, --latex for the table body
python -m evaluation.agreement                 # human and judge agreement, --latex
```

`evaluation/annotation_set/` holds only what defines the set: the twenty items as handed out, and `KEY.csv`, the provenance the annotators never saw. Everything a run produced is in `evaluation/results/`.

## Results

**Surrogate fidelity** on the test set. ADO is average decision ordering; the teacher's is measured against the physical truth ordering rather than against itself.

| model | R² | ADO | decision agreement | AUC |
|---|---|---|---|---|
| Ridge | 0.886 | 0.962 | 0.932 | 0.955 |
| **EBM (ours)** | **0.962** | **0.983** | **0.951** | **0.975** |
| GBDT | 0.980 | 0.988 | 0.967 | 0.979 |
| ParT (teacher) | | 0.983 | | 0.983 |

**Delivery under verification**, over the 90 narrated jets of the hundred-jet sample; the gate withholds the other ten before any model is called. `TOTAL` is the strict conjunction of `Faith` (every tagged number matches), `Comp` (all 46 tags present), `Form` (no malformed tag), `NoBare` (no digit outside a tag) and `Read` (every tagged word is the one its number implies).

| model | Faith | Comp | Form | NoBare | Read | TOTAL |
|---|---|---|---|---|---|---|
| Gemini 3.5 Flash-Lite | 0.99 | 0.86 | 0.93 | 0.80 | 0.83 | 0.57 |
| Gemini 3.5 Flash | 0.99 | 1.00 | 1.00 | 1.00 | 0.97 | **0.96** |
| DeepSeek v4 Flash | 0.93 | 0.96 | 0.99 | 0.87 | 0.80 | 0.64 |
| GPT-OSS 120B | 0.97 | 0.13 | 0.37 | 0.24 | 0.90 | 0.10 |

All four report artifact numbers accurately, yet delivery ranges from 0.96 to 0.10: faithfulness alone is a misleading criterion, and the models fail in different ways.

The judge is an aggregate baseline, not a substitute for reading a narrative. It reproduces the annotators' positive or negative verdict on 95% of judgments, but its quadratic-weighted kappa against them is 0.25 where the two annotators score 0.86 against each other, and of the six human ratings at or below 1 it caught none.

Every number above regenerates from a command in this repository. Not tracked here: the datasets, the checkpoints, the teacher logits, the feature tables and the narrative runs, all large and all reproducible from the steps above.

## Citation

H. Qu, C. Li and S. Qian, *Particle Transformer for Jet Tagging*, ICML 2022, PMLR **162** (2022) 18281. [arXiv:2202.03772](https://arxiv.org/abs/2202.03772)

H. Qu, C. Li and S. Qian, *JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics*, Zenodo (2022). [doi:10.5281/zenodo.6601445](https://doi.org/10.5281/zenodo.6601445)

G. Kasieczka, T. Plehn, J. Thompson and M. Russell, *Top Quark Tagging Reference Dataset*, Zenodo (2019). [doi:10.5281/zenodo.2603256](https://doi.org/10.5281/zenodo.2603256)

G. Kasieczka, T. Plehn, A. Butter et al., *The Machine Learning Landscape of Top Taggers*, SciPost Phys. **7** (2019) 014. [arXiv:1902.09914](https://arxiv.org/abs/1902.09914)

Y. Lou, R. Caruana and J. Gehrke, *Intelligible Models for Classification and Regression*, KDD 2012, 150. [doi:10.1145/2339530.2339556](https://doi.org/10.1145/2339530.2339556)

H. Nori, S. Jenkins, P. Koch and R. Caruana, *InterpretML: A Unified Framework for Machine Learning Interpretability*, (2019). [arXiv:1909.09223](https://arxiv.org/abs/1909.09223)

M. Cacciari, G. P. Salam and G. Soyez, *FastJet User Manual*, Eur. Phys. J. C **72** (2012) 1896. [arXiv:1111.6097](https://arxiv.org/abs/1111.6097)

[fastjet-contrib](https://fastjet.hepforge.org/contrib/) and [hqucms/weaver-core](https://github.com/hqucms/weaver-core)
