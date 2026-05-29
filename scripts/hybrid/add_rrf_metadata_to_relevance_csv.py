"""
Backfill RRF metadata columns into a relevance training CSV.

This is useful for CE-augmented files that were created before
agencies_affected/subject_matter were included in create_data.py.

Usage:

python -m scripts.hybrid.add_rrf_metadata_to_relevance_csv \
  --input data/text/results/relevance_training_data_all_ce.csv \
  --output data/text/results/relevance_training_data_all_ce.csv \
  --rrf data/text/results/rrf_results_original_full_chunks.csv \
  --rrf data/text/results/rrf_results_full_chunks.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_INPUT = Path("data/text/results/relevance_training_data_all_ce.csv")
DEFAULT_OUTPUT = Path("data/text/results/relevance_training_data_all_ce.csv")
DEFAULT_RRF_FILES = [
    Path("data/text/results/rrf_results_original_full_chunks.csv"),
    Path("data/text/results/rrf_results_full_chunks.csv"),
]


def _norm(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _key(row: Dict[str, str]) -> Tuple[str, str, str]:
    rank = _norm(row.get("rrf_rank") or row.get("rank"))
    return (_norm(row.get("query")), rank, _norm(row.get("title")))


def _build_metadata_index(paths: List[Path]) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    index: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for path in paths:
        for row in _read_csv(path):
            key = _key(row)
            if not key[0] or not key[1] or not key[2]:
                continue
            index[key] = {
                "agencies_affected": _norm(row.get("agencies_affected")),
                "subject_matter": _norm(row.get("subject_matter")),
            }
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add RRF metadata columns to relevance CSV.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input relevance CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV")
    parser.add_argument(
        "--rrf",
        action="append",
        default=None,
        help="RRF full-chunks CSV. May be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rrf_paths = [Path(p) for p in args.rrf] if args.rrf else list(DEFAULT_RRF_FILES)

    rows = _read_csv(input_path)
    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")

    metadata_index = _build_metadata_index(rrf_paths)
    matched = 0
    for row in rows:
        metadata = metadata_index.get(_key(row))
        if not metadata:
            continue
        row["agencies_affected"] = metadata["agencies_affected"]
        row["subject_matter"] = metadata["subject_matter"]
        matched += 1

    fieldnames = list(rows[0].keys())
    for col in ["agencies_affected", "subject_matter"]:
        if col not in fieldnames:
            fieldnames.append(col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Metadata matches: {matched}/{len(rows)}")


if __name__ == "__main__":
    main()
