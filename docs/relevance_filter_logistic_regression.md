# Relevance Filter: Logistic Regression Prototype

This prototype tests whether a lightweight learning-based filter can decide
which top RRF search results are actually relevant. The goal is not to replace
BM25, FAISS, RRF, or the cross-encoder. Instead, the model uses their scores and
ranks as features and learns how to combine those signals.

The current binary target is:

```text
is_relevant = relevance >= 2
```

The model is evaluated with query-grouped cross-validation. This means all rows
for the same query stay together in either train or test, so the evaluation asks:

```text
Can the model generalize to queries it has not seen before?
```

## Data

Current CE-augmented training file:

```text
data/text/results/relevance_training_data_all_ce.csv
```

It contains 300 labeled rows from 60 queries, with 5 RRF results per query.

## Model

The evaluator trains temporary logistic regression models inside each
cross-validation fold. It does not save a production model.

The implementation lives in:

```text
scripts/hybrid/logistic_regression/relevance_features.py
scripts/hybrid/logistic_regression/evaluate_relevance_model.py
```

## Named Feature Sets

```text
retrieval
retrieval_ce
retrieval_ce_text
retrieval_ce_metadata
retrieval_ce_text_metadata
```

### `retrieval`

Uses only retrieval and fusion signals:

```text
rrf_rank
rrf_score
bm25_rank
bm25_score
faiss_rank
faiss_score
```

### `retrieval_ce`

Adds cross-encoder features:

```text
ce_rank
ce_score
```

### `retrieval_ce_text`

Adds lightweight text-overlap features:

```text
title_overlap_ratio
chunk_overlap_ratio
exact_query_in_title
exact_query_in_chunk
```

### `retrieval_ce_metadata`

Adds metadata-overlap features:

```text
agency_overlap_ratio
subject_overlap_ratio
```

### `retrieval_ce_text_metadata`

Uses every feature listed above.

## Commands

Run one feature set:

```bash
python -m scripts.hybrid.logistic_regression.evaluate_relevance_model \
  --data data/text/results/relevance_training_data_all_ce.csv \
  --feature_set retrieval_ce
```

Run full ablation:

```bash
python -m scripts.hybrid.logistic_regression.evaluate_relevance_model \
  --data data/text/results/relevance_training_data_all_ce.csv \
  --ablate
```

## Outputs

Evaluation reports are written to:

```text
data/evals/eval_logistic_regression/
```

Full ablation summary:

```text
data/evals/eval_logistic_regression/ablation_summary.csv
```

Each feature set also gets its own folder containing:

```text
summary.csv
threshold_metrics.csv
feature_weights.csv
per_query_analysis_threshold_0.40.csv
```

## Current Finding

The current best feature set is:

```text
retrieval_ce
```

At threshold `0.40`, it currently performs best overall in the ablation:

```text
roc_auc ~= 0.812
precision ~= 0.553
recall ~= 0.854
f1 ~= 0.672
```

Adding the current text-overlap and metadata-overlap features did not improve
the model in the latest run, so they should remain experimental.
