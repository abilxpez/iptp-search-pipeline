# Export BM25 search results to a CSV for Google Sheets.
#
# Reads queries (one per line) and runs scripts.bm25.search_bm25 (seek-based),
# then writes one CSV row per (query, result-rank).
#
"""
Usage:

python -m scripts.bm25.export_bm25_results_csv \
    --config config.json \
    --queries data/text/queries.txt \
    --top_k 5 \
    --out data/text/bm25_results.csv \
    --snippet_chars 1000 \
    --oversample 30 \
    --max_candidates 300 


Optional filters (same as search_bm25):
  --entry_id 1505 --agency "U.S. Citizenship and Immigration Services" ...

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.bm25.search_bm25 import (  # type: ignore
    search_bm25,
)


def _read_queries(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing queries file: {path}")
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if not q:
                continue
            if q.startswith("#"):
                continue
            out.append(q)
    return out


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export BM25 results to CSV (Google Sheets friendly).")
    p.add_argument("--config", default="config.json", help="Path to config.json")
    p.add_argument("--queries", required=True, help="Text file with one query per line")
    p.add_argument("--top_k", type=int, default=5, help="Top K results per query")
    p.add_argument("--out", required=True, help="Output CSV path")

    # BM25 params (optional)
    p.add_argument("--k1", type=float, default=None)
    p.add_argument("--b", type=float, default=None)

    # Optional filters (same as search_bm25)
    p.add_argument("--entry_id", default=None)
    p.add_argument("--administration", default=None)
    p.add_argument("--agency", default=None)
    p.add_argument("--subject", default=None)

    # Output behavior
    p.add_argument(
        "--include_empty_queries",
        action="store_true",
        help="If a query returns 0 results, still write one row with rank=0.",
    )
    p.add_argument(
        "--snippet_chars",
        type=int,
        default=440,
        help="Max chars in snippet column",
    )
    p.add_argument(
        "--max_candidates",
        type=int,
        default=None,
        help="Number of raw hits to retrieve before deduping (defaults to max(top_k*10, top_k))",
    )
    p.add_argument(
        "--oversample",
        type=int,
        default=10,
        help="Candidate multiplier before dedupe (default 10× top_k)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    filters: Dict[str, Optional[str]] = {
        "entry_id": args.entry_id,
        "administration": args.administration,
        "agency": args.agency,
        "subject": args.subject,
    }

    queries_path = Path(args.queries)
    queries = _read_queries(queries_path)
    if not queries:
        raise RuntimeError(f"No queries found in {queries_path}")

    out_path = Path(args.out)
    _ensure_parent_dir(out_path)

    # CSV header (one row per result)
    fieldnames = [
        "query",
        "rank",
        "score",
        "source",
        "doc_id",
        "chunk_id",
        "entry_id",
        "attachment_id",
        "page_start",
        "page_end",
        "title",
        "administration",
        "agencies_affected",
        "subject_matter",
        "source_path",
        "matched_attachment_ids",
        "matched_doc_ids",
        "matched_chunk_ids",
        "matched_count",
        "best_rank",
        "snippet",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in queries:
            results = search_bm25(
                config_path=Path(args.config),
                query=q,
                top_k=int(args.top_k),
                filters=filters,
                k1=args.k1,
                b=args.b,
                snippet_chars=int(args.snippet_chars),
                max_candidates=args.max_candidates,
                oversample=int(args.oversample),
            )

            if not results and args.include_empty_queries:
                writer.writerow(
                    {
                        "query": q,
                        "rank": 0,
                        "score": "",
                        "doc_id": "",
                        "chunk_id": "",
                        "entry_id": "",
                        "attachment_id": "",
                        "page_start": "",
                        "page_end": "",
                        "title": "",
                        "administration": "",
                        "agencies_affected": "",
                        "subject_matter": "",
                        "source_path": "",
                        "source": "",
                        "snippet": "",
                    }
                )
                continue

            for rank, r in enumerate(results, start=1):
                doc = r.doc or {}
                meta = doc.get("meta") or {}

                # Meta fields can be scalars or lists; store list-ish as JSON strings for Sheets.
                agencies = meta.get("agencies_affected")
                subjects = meta.get("subject_matter")
                agencies_s = json.dumps(agencies, ensure_ascii=False) if isinstance(agencies, list) else (agencies or "")
                subjects_s = json.dumps(subjects, ensure_ascii=False) if isinstance(subjects, list) else (subjects or "")

                matched_attachment_ids = json.dumps(
                    r.matched_attachment_ids, ensure_ascii=False
                )
                matched_doc_ids = json.dumps(r.matched_doc_ids, ensure_ascii=False)
                matched_chunk_ids = json.dumps(
                    r.matched_chunk_ids, ensure_ascii=False
                )

                writer.writerow(
                    {
                        "query": q,
                        "rank": rank,
                        "score": f"{float(r.score):.6f}",
                        "doc_id": int(r.doc_id),
                        "chunk_id": doc.get("chunk_id") or "",
                        "entry_id": doc.get("entry_id") or "",
                        "attachment_id": doc.get("attachment_id") or "",
                        "page_start": doc.get("page_start") or "",
                        "page_end": doc.get("page_end") or "",
                        "title": meta.get("title") or "",
                        "administration": meta.get("administration") or "",
                        "agencies_affected": agencies_s,
                        "subject_matter": subjects_s,
                        "source_path": doc.get("source_path") or "",
                        "source": r.source,
                        "matched_attachment_ids": matched_attachment_ids,
                        "matched_doc_ids": matched_doc_ids,
                        "matched_chunk_ids": matched_chunk_ids,
                        "matched_count": r.matched_count,
                        "best_rank": r.best_rank,
                        "snippet": r.snippet or "",
                    }
                )

    print(f"Wrote CSV: {out_path}")
    print(f"Queries: {len(queries)} | top_k: {int(args.top_k)} | k1={args.k1} b={args.b}")
    if any(v for v in filters.values()):
        print(f"Filters: { {k: v for k, v in filters.items() if v} }")


if __name__ == "__main__":
    main()
