"""
Train and save the final logistic-regression relevance filter.

This is separate from evaluation: evaluation trains temporary fold models to
estimate performance, while this script trains one final model on all labeled
rows and writes artifacts for API/UI integration.

Usage:

python -m scripts.hybrid.logistic_regression.train_relevance_model \
  --data data/text/results/relevance_training_data_all_ce.csv \
  --feature_set retrieval
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from scripts.hybrid.logistic_regression.relevance_features import (  # type: ignore
    DEFAULT_TRAINING_DATA,
    FEATURE_SETS,
    build_relevance_dataset,
    describe_dataset,
)


DEFAULT_OUTPUT_DIR = Path("data/models/relevance_filter_logistic_regression")
DEFAULT_HIGH_THRESHOLD = 0.70
DEFAULT_MEDIUM_THRESHOLD = 0.40


def _import_sklearn() -> Dict[str, object]:
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing model dependency. Install scikit-learn/joblib in this environment, for example:\n"
            "  python -m pip install scikit-learn"
        ) from exc

    return {
        "joblib": joblib,
        "LogisticRegression": LogisticRegression,
        "StandardScaler": StandardScaler,
        "make_pipeline": make_pipeline,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final logistic-regression relevance filter.")
    parser.add_argument("--data", default=str(DEFAULT_TRAINING_DATA), help="Training CSV path")
    parser.add_argument(
        "--feature_set",
        default="retrieval",
        choices=sorted(FEATURE_SETS.keys()),
        help="Named feature set to train",
    )
    parser.add_argument("--positive_label_threshold", type=int, default=2)
    parser.add_argument("--high_threshold", type=float, default=DEFAULT_HIGH_THRESHOLD)
    parser.add_argument("--medium_threshold", type=float, default=DEFAULT_MEDIUM_THRESHOLD)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--random_state", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sklearn = _import_sklearn()

    if float(args.high_threshold) <= float(args.medium_threshold):
        raise RuntimeError(
            f"high_threshold must be greater than medium_threshold; got "
            f"high={args.high_threshold}, medium={args.medium_threshold}"
        )

    dataset = build_relevance_dataset(
        Path(args.data),
        relevant_threshold=int(args.positive_label_threshold),
        feature_columns=FEATURE_SETS[str(args.feature_set)],
    )

    X = dataset.X.to_numpy(dtype=float)
    y = dataset.y.to_numpy(dtype=int)

    model = sklearn["make_pipeline"](
        sklearn["StandardScaler"](),
        sklearn["LogisticRegression"](
            class_weight="balanced",
            max_iter=1000,
            random_state=int(args.random_state),
        ),
    )
    model.fit(X, y)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    metadata_path = output_dir / "metadata.json"

    sklearn["joblib"].dump(model, model_path)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "logistic_regression",
        "feature_set": str(args.feature_set),
        "feature_columns": list(dataset.feature_columns),
        "training_data": str(args.data),
        "rows": int(len(dataset.frame)),
        "queries": int(dataset.frame[dataset.group_column].nunique()),
        "positive_label_threshold": int(args.positive_label_threshold),
        "positive_rows": int(dataset.frame[dataset.target_column].sum()),
        "negative_rows": int(len(dataset.frame) - dataset.frame[dataset.target_column].sum()),
        "high_threshold": float(args.high_threshold),
        "medium_threshold": float(args.medium_threshold),
        "random_state": int(args.random_state),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(describe_dataset(dataset))
    print(f"saved_model: {model_path}")
    print(f"saved_metadata: {metadata_path}")


if __name__ == "__main__":
    main()
