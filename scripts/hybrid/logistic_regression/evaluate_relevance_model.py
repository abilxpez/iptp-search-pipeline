"""
Evaluate a logistic-regression relevance classifier with query-grouped CV.

The model is trained temporarily inside each fold. This script does not save a
production model; it estimates whether the approach is worth using.

Usage:

python -m scripts.hybrid.logistic_regression.evaluate_relevance_model
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from scripts.hybrid.logistic_regression.relevance_features import (  # type: ignore
    DEFAULT_TRAINING_DATA,
    build_relevance_dataset,
    describe_dataset,
)


DEFAULT_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


@dataclass
class ThresholdMetrics:
    threshold: float
    predicted_positive: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    accuracy: float


def _import_sklearn() -> Dict[str, object]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing scikit-learn. Install it in this environment, for example:\n"
            "  python -m pip install scikit-learn"
        ) from exc

    return {
        "LogisticRegression": LogisticRegression,
        "GroupKFold": GroupKFold,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
        "roc_auc_score": roc_auc_score,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> ThresholdMetrics:
    y_pred = (probabilities >= float(threshold)).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, len(y_true))
    return ThresholdMetrics(
        threshold=float(threshold),
        predicted_positive=int(y_pred.sum()),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
    )


def _parse_thresholds(raw: str) -> List[float]:
    out: List[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise RuntimeError("At least one threshold is required")
    return out


def _format_metrics_table(metrics: Iterable[ThresholdMetrics]) -> str:
    lines = [
        "threshold,predicted_positive,tp,fp,fn,tn,precision,recall,f1,accuracy",
    ]
    for m in metrics:
        lines.append(
            ",".join(
                [
                    f"{m.threshold:.2f}",
                    str(m.predicted_positive),
                    str(m.true_positive),
                    str(m.false_positive),
                    str(m.false_negative),
                    str(m.true_negative),
                    f"{m.precision:.3f}",
                    f"{m.recall:.3f}",
                    f"{m.f1:.3f}",
                    f"{m.accuracy:.3f}",
                ]
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate logistic-regression relevance classifier.")
    p.add_argument("--data", default=str(DEFAULT_TRAINING_DATA), help="Training CSV path")
    p.add_argument("--folds", type=int, default=5, help="Number of query-grouped CV folds")
    p.add_argument(
        "--thresholds",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS),
        help="Comma-separated probability thresholds to evaluate",
    )
    p.add_argument("--positive_label_threshold", type=int, default=2, help="Relevant if relevance >= this value")
    p.add_argument("--random_state", type=int, default=13)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sklearn = _import_sklearn()
    thresholds = _parse_thresholds(args.thresholds)

    dataset = build_relevance_dataset(
        Path(args.data),
        relevant_threshold=int(args.positive_label_threshold),
    )
    print(describe_dataset(dataset))
    print("")

    X = dataset.X.to_numpy(dtype=float)
    y = dataset.y.to_numpy(dtype=int)
    groups = dataset.groups.to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(int(args.folds), len(unique_groups))
    if n_splits < 2:
        raise RuntimeError(f"Need at least 2 query groups for grouped CV; found {len(unique_groups)}")

    GroupKFold = sklearn["GroupKFold"]
    LogisticRegression = sklearn["LogisticRegression"]
    StandardScaler = sklearn["StandardScaler"]
    make_pipeline = sklearn["make_pipeline"]
    roc_auc_score = sklearn["roc_auc_score"]

    probabilities = np.zeros(len(y), dtype=float)
    fold_summaries: List[str] = []
    cv = GroupKFold(n_splits=n_splits)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=int(args.random_state),
            ),
        )
        model.fit(X[train_idx], y[train_idx])
        fold_probabilities = model.predict_proba(X[test_idx])[:, 1]
        probabilities[test_idx] = fold_probabilities

        test_groups = len(np.unique(groups[test_idx]))
        positives = int(y[test_idx].sum())
        fold_summaries.append(
            f"fold {fold_idx}: rows={len(test_idx)} queries={test_groups} positives={positives}"
        )

    print("folds")
    for line in fold_summaries:
        print(line)
    print("")

    try:
        auc = float(roc_auc_score(y, probabilities))
        print(f"roc_auc: {auc:.3f}")
    except Exception:
        print("roc_auc: unavailable")
    print("")

    print("threshold_metrics")
    metrics = [_threshold_metrics(y, probabilities, t) for t in thresholds]
    print(_format_metrics_table(metrics))
    print("")

    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=int(args.random_state),
        ),
    )
    final_model.fit(X, y)
    classifier = final_model.named_steps["logisticregression"]
    weights = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])
    weighted = sorted(
        zip(dataset.feature_columns, weights),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )

    print("final_refit_feature_weights")
    print(f"intercept,{intercept:.6f}")
    print("feature,weight")
    for feature, weight in weighted:
        print(f"{feature},{float(weight):.6f}")


if __name__ == "__main__":
    main()
