"""
Feature preparation for the first relevance-classifier prototype.

This module intentionally uses only retrieval-derived features:
- RRF rank/score
- BM25 rank/score
- FAISS rank/score

The group for evaluation is the query string, so cross-validation can hold out
entire queries rather than splitting rows from the same query across train/test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_TRAINING_DATA = Path("data/text/results/relevance_training_data_all.csv")

FEATURE_COLUMNS: List[str] = [
    "rrf_rank",
    "rrf_score",
    "bm25_rank",
    "bm25_score",
    "faiss_rank",
    "faiss_score",
]


@dataclass(frozen=True)
class RelevanceDataset:
    frame: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    group_column: str

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.feature_columns]

    @property
    def y(self) -> pd.Series:
        return self.frame[self.target_column]

    @property
    def groups(self) -> pd.Series:
        return self.frame[self.group_column]


def load_training_frame(path: Path = DEFAULT_TRAINING_DATA) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing training data: {path}")
    return pd.read_csv(path)


def _coerce_numeric_column(frame: pd.DataFrame, column: str, fill_value: float) -> None:
    frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill_value)


def build_relevance_dataset(
    path: Path = DEFAULT_TRAINING_DATA,
    *,
    relevant_threshold: int = 2,
    missing_rank_value: float = 999.0,
    missing_score_value: float = 0.0,
) -> RelevanceDataset:
    frame = load_training_frame(path).copy()

    required = {"relevance", "query", *FEATURE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Training data missing required columns: {missing}")

    frame["query"] = frame["query"].fillna("").astype(str).str.strip()
    if not frame["query"].any():
        raise RuntimeError("Training data has no non-empty query values")

    rank_columns = ["rrf_rank", "bm25_rank", "faiss_rank"]
    score_columns = ["rrf_score", "bm25_score", "faiss_score"]

    for column in rank_columns:
        _coerce_numeric_column(frame, column, fill_value=missing_rank_value)
    for column in score_columns:
        _coerce_numeric_column(frame, column, fill_value=missing_score_value)

    frame["relevance"] = pd.to_numeric(frame["relevance"], errors="coerce")
    if frame["relevance"].isna().any():
        bad_count = int(frame["relevance"].isna().sum())
        raise RuntimeError(f"Training data has {bad_count} rows with invalid relevance labels")

    frame["is_relevant"] = (frame["relevance"] >= int(relevant_threshold)).astype(int)

    return RelevanceDataset(
        frame=frame,
        feature_columns=list(FEATURE_COLUMNS),
        target_column="is_relevant",
        group_column="query",
    )


def describe_dataset(dataset: RelevanceDataset) -> str:
    frame = dataset.frame
    total_rows = len(frame)
    total_queries = int(frame[dataset.group_column].nunique())
    positives = int(frame[dataset.target_column].sum())
    negatives = int(total_rows - positives)
    label_counts = frame["relevance"].value_counts().sort_index().to_dict()
    return "\n".join(
        [
            f"rows: {total_rows}",
            f"queries: {total_queries}",
            f"positive(label>=2): {positives}",
            f"negative(label<2): {negatives}",
            f"label_counts: {label_counts}",
            f"features: {', '.join(dataset.feature_columns)}",
        ]
    )
