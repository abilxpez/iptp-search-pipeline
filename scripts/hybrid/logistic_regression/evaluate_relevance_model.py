"""
Evaluate a logistic-regression relevance classifier with query-grouped CV.

The model is trained temporarily inside each fold. This script does not save a
production model; it estimates whether the approach is worth using.

Usage:

python -m scripts.hybrid.logistic_regression.evaluate_relevance_model
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from scripts.hybrid.logistic_regression.relevance_features import (  # type: ignore
    DEFAULT_TRAINING_DATA,
    FEATURE_SETS,
    build_relevance_dataset,
    describe_dataset,
)


DEFAULT_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
DEFAULT_OUTPUT_DIR = Path("data/evals/eval_logistic_regression")


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


@dataclass
class QueryMetrics:
    query: str
    rows: int
    actual_positive: int
    predicted_positive: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    missed_titles: str
    false_positive_titles: str


@dataclass
class BandMetrics:
    band: str
    rows: int
    relevant_rows: int
    irrelevant_rows: int
    precision: float
    avg_probability: float


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


def _probability_band(probability: float, *, high_threshold: float, medium_threshold: float) -> str:
    if probability >= float(high_threshold):
        return "high"
    if probability >= float(medium_threshold):
        return "medium"
    return "low"


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


def _truncate_titles(titles: Sequence[str], limit: int = 3) -> str:
    cleaned = [str(title).strip() for title in titles if str(title).strip()]
    if not cleaned:
        return "-"
    shown = cleaned[:limit]
    if len(cleaned) > limit:
        shown.append(f"...(+{len(cleaned) - limit} more)")
    return " | ".join(shown)


def _query_metrics(frame, probabilities: np.ndarray, threshold: float) -> List[QueryMetrics]:
    query_values = frame["query"].astype(str)
    y_true = frame["is_relevant"].to_numpy(dtype=int)
    y_pred = (probabilities >= float(threshold)).astype(int)
    results: List[QueryMetrics] = []

    for query in sorted(query_values.unique()):
        mask = query_values == query
        subset = frame.loc[mask]
        subset_true = y_true[mask.to_numpy()]
        subset_pred = y_pred[mask.to_numpy()]

        tp = int(((subset_pred == 1) & (subset_true == 1)).sum())
        fp = int(((subset_pred == 1) & (subset_true == 0)).sum())
        fn = int(((subset_pred == 0) & (subset_true == 1)).sum())
        tn = int(((subset_pred == 0) & (subset_true == 0)).sum())

        missed_titles = _truncate_titles(subset.loc[(subset_pred == 0) & (subset_true == 1), "title"].tolist())
        false_positive_titles = _truncate_titles(
            subset.loc[(subset_pred == 1) & (subset_true == 0), "title"].tolist()
        )

        actual_positive = tp + fn
        predicted_positive = tp + fp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        accuracy = _safe_div(tp + tn, len(subset))

        results.append(
            QueryMetrics(
                query=query,
                rows=len(subset),
                actual_positive=actual_positive,
                predicted_positive=predicted_positive,
                true_positive=tp,
                false_positive=fp,
                false_negative=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                accuracy=accuracy,
                missed_titles=missed_titles,
                false_positive_titles=false_positive_titles,
            )
        )

    return results


def _prediction_rows(
    frame,
    probabilities: np.ndarray,
    *,
    high_threshold: float,
    medium_threshold: float,
) -> List[dict]:
    rows: List[dict] = []
    for idx, (_, row) in enumerate(frame.iterrows()):
        probability = float(probabilities[idx])
        band = _probability_band(
            probability,
            high_threshold=float(high_threshold),
            medium_threshold=float(medium_threshold),
        )
        rows.append(
            {
                "query": row.get("query", ""),
                "rrf_rank": row.get("rrf_rank", ""),
                "title": row.get("title", ""),
                "relevance": row.get("relevance", ""),
                "is_relevant": row.get("is_relevant", ""),
                "relevance_probability": f"{probability:.6f}",
                "relevance_band": band,
                "rrf_score": row.get("rrf_score", ""),
                "bm25_rank": row.get("bm25_rank", ""),
                "bm25_score": row.get("bm25_score", ""),
                "faiss_rank": row.get("faiss_rank", ""),
                "faiss_score": row.get("faiss_score", ""),
                "ce_rank": row.get("ce_rank", ""),
                "ce_score": row.get("ce_score", ""),
            }
        )
    return rows


def _band_metrics(prediction_rows: Sequence[dict]) -> List[BandMetrics]:
    out: List[BandMetrics] = []
    for band in ["high", "medium", "low"]:
        rows = [row for row in prediction_rows if row.get("relevance_band") == band]
        total = len(rows)
        relevant = sum(1 for row in rows if str(row.get("is_relevant")) in {"1", "1.0", "True", "true"})
        probabilities = [
            float(row.get("relevance_probability") or 0.0)
            for row in rows
        ]
        out.append(
            BandMetrics(
                band=band,
                rows=total,
                relevant_rows=relevant,
                irrelevant_rows=total - relevant,
                precision=_safe_div(relevant, total),
                avg_probability=(sum(probabilities) / total) if total else 0.0,
            )
        )
    return out


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate logistic-regression relevance classifier.")
    p.add_argument("--data", default=str(DEFAULT_TRAINING_DATA), help="Training CSV path")
    p.add_argument(
        "--feature_set",
        default="retrieval_ce_text_metadata",
        choices=sorted(FEATURE_SETS.keys()),
        help="Named feature set to evaluate",
    )
    p.add_argument(
        "--ablate",
        action="store_true",
        help="Evaluate all named feature sets and write an ablation summary",
    )
    p.add_argument("--folds", type=int, default=5, help="Number of query-grouped CV folds")
    p.add_argument(
        "--thresholds",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS),
        help="Comma-separated probability thresholds to evaluate",
    )
    p.add_argument("--positive_label_threshold", type=int, default=2, help="Relevant if relevance >= this value")
    p.add_argument(
        "--analysis_threshold",
        type=float,
        default=0.4,
        help="Probability threshold to use for per-query error analysis",
    )
    p.add_argument(
        "--high_threshold",
        type=float,
        default=0.7,
        help="Probability threshold for high-confidence result band",
    )
    p.add_argument(
        "--medium_threshold",
        type=float,
        default=0.4,
        help="Probability threshold for medium-confidence result band",
    )
    p.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where CSV evaluation reports will be written",
    )
    p.add_argument("--random_state", type=int, default=13)
    return p.parse_args()


def _evaluate_feature_set(
    *,
    args: argparse.Namespace,
    sklearn: Dict[str, object],
    thresholds: List[float],
    feature_set_name: str,
    output_dir: Path,
    quiet: bool = False,
) -> Dict[str, object]:
    dataset = build_relevance_dataset(
        Path(args.data),
        relevant_threshold=int(args.positive_label_threshold),
        feature_columns=FEATURE_SETS[feature_set_name],
    )
    if not quiet:
        print(f"feature_set: {feature_set_name}")
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

    if not quiet:
        print("folds")
        for line in fold_summaries:
            print(line)
        print("")

    try:
        auc = float(roc_auc_score(y, probabilities))
        if not quiet:
            print(f"roc_auc: {auc:.3f}")
    except Exception:
        auc = float("nan")
        if not quiet:
            print("roc_auc: unavailable")
    if not quiet:
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

    analysis_threshold = float(args.analysis_threshold)
    high_threshold = float(args.high_threshold)
    medium_threshold = float(args.medium_threshold)
    if high_threshold <= medium_threshold:
        raise RuntimeError(
            f"high_threshold must be greater than medium_threshold; got "
            f"high={high_threshold}, medium={medium_threshold}"
        )
    query_metrics = _query_metrics(dataset.frame, probabilities, analysis_threshold)
    prediction_rows = _prediction_rows(
        dataset.frame,
        probabilities,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
    )
    band_metrics = _band_metrics(prediction_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_metrics = [_threshold_metrics(y, probabilities, t) for t in thresholds]
    analysis_metric = _threshold_metrics(y, probabilities, analysis_threshold)
    summary_rows = [
        {
            "feature_set": feature_set_name,
            "rows": len(dataset.frame),
            "queries": int(dataset.frame[dataset.group_column].nunique()),
            "positive_rows": int(dataset.frame[dataset.target_column].sum()),
            "negative_rows": int(len(dataset.frame) - dataset.frame[dataset.target_column].sum()),
            "positive_label_threshold": int(args.positive_label_threshold),
            "roc_auc": f"{auc:.6f}" if np.isfinite(auc) else "nan",
            "analysis_threshold": f"{analysis_threshold:.2f}",
            "analysis_precision": f"{analysis_metric.precision:.6f}",
            "analysis_recall": f"{analysis_metric.recall:.6f}",
            "analysis_f1": f"{analysis_metric.f1:.6f}",
            "analysis_accuracy": f"{analysis_metric.accuracy:.6f}",
            "high_threshold": f"{high_threshold:.2f}",
            "medium_threshold": f"{medium_threshold:.2f}",
        }
    ]
    threshold_rows = [asdict(metric) for metric in threshold_metrics]
    weight_rows = [{"feature": "intercept", "weight": f"{intercept:.6f}"}] + [
        {"feature": feature, "weight": f"{float(weight):.6f}"} for feature, weight in weighted
    ]
    query_rows = [asdict(metric) for metric in sorted(
        query_metrics,
        key=lambda item: (item.false_negative, item.false_positive, item.recall, item.query),
        reverse=True,
    )]

    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "threshold_metrics.csv", threshold_rows)
    _write_csv(output_dir / "feature_weights.csv", weight_rows)
    _write_csv(output_dir / f"per_query_analysis_threshold_{analysis_threshold:.2f}.csv", query_rows)
    _write_csv(output_dir / "row_predictions.csv", prediction_rows)
    _write_csv(output_dir / "band_metrics.csv", [asdict(metric) for metric in band_metrics])

    if not quiet:
        print(f"wrote_csv_reports: {output_dir}")
        print("files:")
        print(f"- {output_dir / 'summary.csv'}")
        print(f"- {output_dir / 'threshold_metrics.csv'}")
        print(f"- {output_dir / 'feature_weights.csv'}")
        print(f"- {output_dir / f'per_query_analysis_threshold_{analysis_threshold:.2f}.csv'}")
        print(f"- {output_dir / 'row_predictions.csv'}")
        print(f"- {output_dir / 'band_metrics.csv'}")

    return summary_rows[0]


def main() -> None:
    args = parse_args()
    sklearn = _import_sklearn()
    thresholds = _parse_thresholds(args.thresholds)
    base_output_dir = Path(args.output_dir)

    if args.ablate:
        summary_rows: List[Dict[str, object]] = []
        for feature_set_name in sorted(FEATURE_SETS.keys()):
            output_dir = base_output_dir / feature_set_name
            print(f"evaluating feature_set={feature_set_name}")
            summary = _evaluate_feature_set(
                args=args,
                sklearn=sklearn,
                thresholds=thresholds,
                feature_set_name=feature_set_name,
                output_dir=output_dir,
                quiet=True,
            )
            summary_rows.append(summary)
            print(
                "  "
                f"roc_auc={summary['roc_auc']} "
                f"precision@{summary['analysis_threshold']}={summary['analysis_precision']} "
                f"recall@{summary['analysis_threshold']}={summary['analysis_recall']} "
                f"f1@{summary['analysis_threshold']}={summary['analysis_f1']}"
            )
        _write_csv(base_output_dir / "ablation_summary.csv", summary_rows)
        print(f"wrote_ablation_summary: {base_output_dir / 'ablation_summary.csv'}")
        return

    _evaluate_feature_set(
        args=args,
        sklearn=sklearn,
        thresholds=thresholds,
        feature_set_name=str(args.feature_set),
        output_dir=base_output_dir,
        quiet=False,
    )


if __name__ == "__main__":
    main()
