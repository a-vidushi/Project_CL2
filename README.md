# Sense entropy and WSD difficulty

Result tables and figures are in `results/`; the report is `report.md`. `third_party/DMRST_Parser/` is the original code of the DMRST parser (Liu et al., 2021), not ours.

## Setup

```bash
uv sync
```

For RST features, put the DMRST checkpoint (linked from `third_party/DMRST_Parser/depth_mode/Savings/README.md`) at `data/models/multi_all_checkpoint.torchsave`.

## Running

```bash
uv run --group rst python -m wsd_difficulty parse-rst   # DMRST trees for every SemCor document (~15 min on Apple GPU)
uv run python -m wsd_difficulty prepare                 # corpora, lemmas, spaCy parses, sentence + discourse features
uv run python -m wsd_difficulty run --workers 8         # sentence-level 10-fold CV (+ external test sets)
uv run python -m wsd_difficulty run-discourse --workers 8   # discourse feature sets on top, saved per word so it can resume
uv run python -m wsd_difficulty analyse                 # tables in results/
uv run python -m wsd_difficulty.report_figures          # figures in results/figures/
```

`uv run python -m wsd_difficulty all --max-target-words 50 --workers 8` does a quick run on a sample.

## Directory structure

| File | What it does |
|---|---|
| `corpus.py` | Reads the Raganato XML + gold keys |
| `lemma_selection.py` | ≥30 instances, ≥2 senses with ≥5 instances |
| `sense_statistics.py` | Entropy, skew (MFS probability), information content, terciles |
| `folds.py` | Document-level 10-fold split, size-balanced |
| `parsing.py` | spaCy tags and dependencies on gold tokens |
| `features.py` | Window words, collocations, POS, dependencies |
| `models/` | MFS, simplified Lesk, Naive Bayes (k and calibration temperature tuned on dev docs), logistic regression (SGD + L2, L2 and epochs tuned on dev docs), decision lists |
| `relatedness.py` | Path similarity (nouns, verbs), extended gloss overlap (adjectives, adverbs) |
| `experiments.py` | Cross-validation and external validation, one row per prediction |
| `information.py` | H(S\|w) − H(S\|w,C) as held-out cross-entropy; feature mutual information |
| `statistics.py` | Spearman, partial Spearman, VIF, bootstrap CIs, McNemar, calibration |
| `discourse.py` | Other mentions, previous-mention and document-sense features, lexical chains |
| `rst.py`, `rst_parser.py` | DMRST runner and RST tree features (relation, nuclearity, depth, tree distance) |
| `discourse_analysis.py` | One sense per discourse, H(S\|w,d), RQ4 gains and significance |
| `correctness_regression.py` | Logistic regression of instance correctness on word, context and discourse predictors, standard errors clustered by lemma |
| `analysis.py` | Result tables |
| `report_figures.py` | Report figures |
