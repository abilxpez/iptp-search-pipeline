"""
Create a compact relevance-training dataset from labeled RRF results plus
BM25/semantic scores.

Usage:

python3 -m scripts.hybrid.create_data

or:

python3 -m scripts.hybrid.create_data \
  --out data/text/results/relevance_training_data.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_RRF_SCORED = Path("data/text/results/rrf_results_scored_new.csv")
DEFAULT_RRF_CLEANED = Path("data/text/results/rrf_results_full_chunks_cleaned.csv")
DEFAULT_BM25 = Path("data/text/bm25_results_new.csv")
DEFAULT_FAISS = Path("data/text/faiss_results_new.csv")
DEFAULT_OUT = Path("data/text/results/relevance_training_data.csv")


def _norm(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _short_title_match(left: str, right: str) -> bool:
    left_n = _norm(left)
    right_n = _norm(right)
    if left_n == right_n:
        return True
    if len(left_n) >= 20 and right_n.startswith(left_n):
        return True
    if len(right_n) >= 20 and left_n.startswith(right_n):
        return True
    return False


def _validate_rrf_alignment(scored_rows: List[Dict[str, str]], cleaned_rows: List[Dict[str, str]]) -> None:
    if len(scored_rows) != len(cleaned_rows):
        raise RuntimeError(
            f"RRF row count mismatch: scored={len(scored_rows)} cleaned={len(cleaned_rows)}"
        )

    mismatches: List[str] = []
    for idx, (scored, cleaned) in enumerate(zip(scored_rows, cleaned_rows), start=1):
        scored_query = _norm(scored.get("query"))
        cleaned_query = _norm(cleaned.get("query"))
        scored_rank = _norm(scored.get("rank"))
        cleaned_rank = _norm(cleaned.get("rank"))
        scored_title = _norm(scored.get("title"))
        cleaned_title = _norm(cleaned.get("title"))
        if (
            scored_query != cleaned_query
            or scored_rank != cleaned_rank
            or not _short_title_match(scored_title, cleaned_title)
        ):
            mismatches.append(
                f"row {idx}: scored=({scored_query!r}, {scored_rank!r}, {scored_title!r}) "
                f"cleaned=({cleaned_query!r}, {cleaned_rank!r}, {cleaned_title!r})"
            )

    if mismatches:
        preview = "\n".join(mismatches[:10])
        raise RuntimeError(f"RRF scored/cleaned alignment failed:\n{preview}")


def _score_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Map (query, title) to the best-ranked row for that retrieval method.
    If a title appears more than once for a query, keep the smallest rank.
    """
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        key = (_norm(row.get("query")), _norm(row.get("title")))
        if not key[0] or not key[1]:
            continue
        existing = out.get(key)
        if existing is None:
            out[key] = row
            continue
        try:
            current_rank = int(float(_norm(row.get("rank")) or "999999"))
            existing_rank = int(float(_norm(existing.get("rank")) or "999999"))
        except Exception:
            current_rank = 999999
            existing_rank = 999999
        if current_rank < existing_rank:
            out[key] = row
    return out


def _lookup_score(
    index: Dict[Tuple[str, str], Dict[str, str]],
    *,
    query: str,
    title: str,
) -> Tuple[str, str]:
    row = index.get((_norm(query), _norm(title)))
    if row is None:
        return "", ""
    return _norm(row.get("score")), _norm(row.get("rank"))


def _first_value(row: Dict[str, str], keys: List[str]) -> str:
    for key in keys:
        value = _norm(row.get(key))
        if value:
            return value
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create relevance training dataset.")
    p.add_argument("--rrf_scored", default=str(DEFAULT_RRF_SCORED))
    p.add_argument("--rrf_cleaned", default=str(DEFAULT_RRF_CLEANED))
    p.add_argument("--bm25", default=str(DEFAULT_BM25))
    p.add_argument("--faiss", default=str(DEFAULT_FAISS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rrf_scored_path = Path(args.rrf_scored)
    rrf_cleaned_path = Path(args.rrf_cleaned)
    bm25_path = Path(args.bm25)
    faiss_path = Path(args.faiss)
    out_path = Path(args.out)

    scored_rows = _read_csv(rrf_scored_path)
    cleaned_rows = _read_csv(rrf_cleaned_path)
    bm25_rows = _read_csv(bm25_path)
    faiss_rows = _read_csv(faiss_path)

    _validate_rrf_alignment(scored_rows, cleaned_rows)

    bm25_index = _score_index(bm25_rows)
    faiss_index = _score_index(faiss_rows)

    fieldnames = [
        "relevance",
        "reason",
        "query",
        "rrf_rank",
        "title",
        "full_chunk_text",
        "rrf_score",
        "bm25_score",
        "bm25_rank",
        "faiss_score",
        "faiss_rank",
        "agencies_affected",
        "subject_matter",
    ]

    missing_bm25 = 0
    missing_faiss = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for scored, cleaned in zip(scored_rows, cleaned_rows):
            query = _norm(cleaned.get("query"))
            title = _norm(cleaned.get("title"))
            bm25_score, bm25_rank = _lookup_score(bm25_index, query=query, title=title)
            faiss_score, faiss_rank = _lookup_score(faiss_index, query=query, title=title)
            if not bm25_score:
                missing_bm25 += 1
            if not faiss_score:
                missing_faiss += 1

            writer.writerow(
                {
                    "relevance": _norm(scored.get("relevance")),
                    "reason": _norm(scored.get("reason")),
                    "query": query,
                    "rrf_rank": _norm(cleaned.get("rank")),
                    "title": title,
                    "full_chunk_text": _norm(cleaned.get("full_chunk_text")),
                    "rrf_score": _first_value(cleaned, ["rrf_score", "score"]),
                    "bm25_score": bm25_score,
                    "bm25_rank": bm25_rank,
                    "faiss_score": faiss_score,
                    "faiss_rank": faiss_rank,
                    "agencies_affected": _norm(cleaned.get("agencies_affected")),
                    "subject_matter": _norm(cleaned.get("subject_matter")),
                }
            )

    print(f"Wrote {len(scored_rows)} rows to {out_path}")
    print(f"Missing BM25 matches: {missing_bm25}")
    print(f"Missing FAISS matches: {missing_faiss}")


if __name__ == "__main__":
    main()
