"""
Feature preparation for the first relevance-classifier prototype.

This module intentionally uses only retrieval-derived features:
- RRF rank/score
- BM25 rank/score
- FAISS rank/score
- Cross-encoder rank/score
- Lightweight text overlap features

The group for evaluation is the query string, so cross-validation can hold out
entire queries rather than splitting rows from the same query across train/test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional

import pandas as pd


DEFAULT_TRAINING_DATA = Path("data/text/results/relevance_training_data_all_ce.csv")

BASE_FEATURE_COLUMNS: List[str] = [
    "rrf_rank",
    "rrf_score",
    "bm25_rank",
    "bm25_score",
    "faiss_rank",
    "faiss_score",
    "ce_rank",
    "ce_score",
]

DERIVED_TEXT_FEATURE_COLUMNS: List[str] = [
    "title_overlap_ratio",
    "chunk_overlap_ratio",
    "exact_query_in_title",
    "exact_query_in_chunk",
    "agency_overlap_ratio",
    "subject_overlap_ratio",
]

FEATURE_COLUMNS: List[str] = [
    *BASE_FEATURE_COLUMNS,
    *DERIVED_TEXT_FEATURE_COLUMNS,
]

FEATURE_SETS: Dict[str, List[str]] = {
    "retrieval": [
        "rrf_rank",
        "rrf_score",
        "bm25_rank",
        "bm25_score",
        "faiss_rank",
        "faiss_score",
    ],
    "retrieval_ce": [
        "rrf_rank",
        "rrf_score",
        "bm25_rank",
        "bm25_score",
        "faiss_rank",
        "faiss_score",
        "ce_rank",
        "ce_score",
    ],
    "retrieval_ce_text": [
        "rrf_rank",
        "rrf_score",
        "bm25_rank",
        "bm25_score",
        "faiss_rank",
        "faiss_score",
        "ce_rank",
        "ce_score",
        "title_overlap_ratio",
        "chunk_overlap_ratio",
        "exact_query_in_title",
        "exact_query_in_chunk",
    ],
    "retrieval_ce_metadata": [
        "rrf_rank",
        "rrf_score",
        "bm25_rank",
        "bm25_score",
        "faiss_rank",
        "faiss_score",
        "ce_rank",
        "ce_score",
        "agency_overlap_ratio",
        "subject_overlap_ratio",
    ],
    "retrieval_ce_text_metadata": list(FEATURE_COLUMNS),
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "under",
    "with",
}


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


def _tokens(value: object) -> List[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    return [token for token in raw_tokens if token not in STOPWORDS and len(token) > 1]


def _token_overlap_ratio(query: object, text: object) -> float:
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(_tokens(text))
    return len(query_tokens & text_tokens) / len(query_tokens)


def _exact_query_in_text(query: object, text: object) -> int:
    query_s = " ".join(str(query or "").lower().split())
    text_s = " ".join(str(text or "").lower().split())
    if not query_s or not text_s:
        return 0
    return int(query_s in text_s)


def _optional_text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna("").astype(str)
    return pd.Series([""] * len(frame), index=frame.index)


def add_text_overlap_features(frame: pd.DataFrame) -> None:
    query = frame["query"].fillna("").astype(str)
    title = _optional_text_column(frame, "title")
    chunk = _optional_text_column(frame, "full_chunk_text")
    agencies = _optional_text_column(frame, "agencies_affected")
    subjects = _optional_text_column(frame, "subject_matter")

    frame["title_overlap_ratio"] = [
        _token_overlap_ratio(q, t) for q, t in zip(query, title)
    ]
    frame["chunk_overlap_ratio"] = [
        _token_overlap_ratio(q, t) for q, t in zip(query, chunk)
    ]
    frame["exact_query_in_title"] = [
        _exact_query_in_text(q, t) for q, t in zip(query, title)
    ]
    frame["exact_query_in_chunk"] = [
        _exact_query_in_text(q, t) for q, t in zip(query, chunk)
    ]
    frame["agency_overlap_ratio"] = [
        _token_overlap_ratio(q, a) for q, a in zip(query, agencies)
    ]
    frame["subject_overlap_ratio"] = [
        _token_overlap_ratio(q, s) for q, s in zip(query, subjects)
    ]


def build_relevance_dataset(
    path: Path = DEFAULT_TRAINING_DATA,
    *,
    relevant_threshold: int = 2,
    missing_rank_value: float = 999.0,
    missing_score_value: float = 0.0,
    feature_columns: Optional[List[str]] = None,
) -> RelevanceDataset:
    frame = load_training_frame(path).copy()

    required = {"relevance", "query", *BASE_FEATURE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Training data missing required columns: {missing}")

    frame["query"] = frame["query"].fillna("").astype(str).str.strip()
    if not frame["query"].any():
        raise RuntimeError("Training data has no non-empty query values")

    add_text_overlap_features(frame)

    rank_columns = ["rrf_rank", "bm25_rank", "faiss_rank", "ce_rank"]
    score_columns = [
        "rrf_score",
        "bm25_score",
        "faiss_score",
        "ce_score",
        *DERIVED_TEXT_FEATURE_COLUMNS,
    ]

    for column in rank_columns:
        _coerce_numeric_column(frame, column, fill_value=missing_rank_value)
    for column in score_columns:
        _coerce_numeric_column(frame, column, fill_value=missing_score_value)

    frame["relevance"] = pd.to_numeric(frame["relevance"], errors="coerce")
    if frame["relevance"].isna().any():
        bad_count = int(frame["relevance"].isna().sum())
        raise RuntimeError(f"Training data has {bad_count} rows with invalid relevance labels")

    frame["is_relevant"] = (frame["relevance"] >= int(relevant_threshold)).astype(int)

    selected_features = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
    missing_features = sorted(set(selected_features) - set(frame.columns))
    if missing_features:
        raise RuntimeError(f"Selected feature columns were not built: {missing_features}")

    return RelevanceDataset(
        frame=frame,
        feature_columns=selected_features,
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
