"""
How to use

python -m scripts.semantic.export_faiss_results_csv \
  --run_dir data/embeddings/bge_mean_norm \
  --chunks data/sample_100/chunks/chunks.jsonl \
  --queries data/text/queries.txt \
  --out data/text/faiss_results.csv \
  --top_k 5 \
  --oversample 10 \
  --snippet_chars 1000 
  
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from scripts.semantic.search_faiss import search_faiss  # type: ignore


def _read_queries(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing queries file: {path}")
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if not q or q.startswith("#"):
                continue
            out.append(q)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export FAISS results to CSV with BM25-compatible columns.")
    p.add_argument("--run_dir", default="data/embeddings/bge_mean_norm", help="Embedding run/manifest directory")
    p.add_argument("--chunks", default="data/sample_100/chunks/chunks.jsonl", help="Path to chunks.jsonl")
    p.add_argument("--queries", required=True, help="Text file with one query per line")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--top_k", type=int, default=5, help="Top K results per query")
    p.add_argument("--oversample", type=int, default=10, help="FAISS oversample factor (top_k * oversample candidates)")
    p.add_argument("--snippet_chars", type=int, default=220, help="Max chars for snippet column")
    p.add_argument("--max_candidates", type=int,default=None,help="Max raw FAISS hits to consider before deduping (default=top_k*oversample)")
    p.add_argument("--entry_id", default=None)
    p.add_argument("--administration", default=None)
    p.add_argument("--agency", default=None)
    p.add_argument("--subject", default=None)
    p.add_argument("--device", default=None, help='Override embedding device ("cuda"/"mps"/"cpu")')
    p.add_argument("--batch_size", type=int, default=16, help="Embedding batch size for query encoding")
    p.add_argument("--ef_search", type=int, default=None, help="Override HNSW efSearch (if supported)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    queries_path = Path(args.queries)
    queries = _read_queries(queries_path)
    if not queries:
        raise RuntimeError(f"No queries found in {queries_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filters: Dict[str, Optional[str]] = {
        "entry_id": args.entry_id,
        "administration": args.administration,
        "agency": args.agency,
        "subject": args.subject,
    }

    fieldnames = [
        "query",
        "rank",
        "score",
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
            results = search_faiss(
                run_dir=Path(args.run_dir),
                chunks_path=Path(args.chunks),
                query=q,
                top_k=int(args.top_k),
                oversample=int(args.oversample),
                filters=filters,
                device=args.device if args.device else None,
                batch_size=int(args.batch_size),
                ef_search=int(args.ef_search) if args.ef_search is not None else None,
                snippet_chars=int(args.snippet_chars),
                max_candidates=args.max_candidates,
            )

            for rank, r in enumerate(results, start=1):
                doc = r.doc or {}
                meta = doc.get("meta") or {}
                policy_meta = meta.get("policy") or {}
                attachment_meta = meta.get("attachment") or {}
                flat_meta = doc.get("meta_flat") or {}
                agencies = flat_meta.get("agencies_affected") or []
                subjects = flat_meta.get("subject_matter") or []
                agencies_s = json.dumps(agencies, ensure_ascii=False) if isinstance(agencies, list) else (agencies or "")
                subjects_s = json.dumps(subjects, ensure_ascii=False) if isinstance(subjects, list) else (subjects or "")

                matched_attachment_ids = json.dumps(r.matched_attachment_ids, ensure_ascii=False)
                matched_doc_ids = json.dumps(r.matched_doc_ids, ensure_ascii=False)
                matched_chunk_ids = json.dumps(r.matched_chunk_ids, ensure_ascii=False)

                doc_id_value = doc.get("policydocument_id") or doc.get("doc_id") or ""
                doc_id_str = str(doc_id_value) if doc_id_value is not None else ""

                writer.writerow(
                    {
                        "query": q,
                        "rank": rank,
                        "score": f"{float(r.score):.6f}",
                        "chunk_id": doc.get("chunk_id") or "",
                        "entry_id": doc.get("page_ptr_id") or "",
                        "attachment_id": doc.get("attachment_id") or attachment_meta.get("policydocument_id") or "",
                        "page_start": doc.get("page_start") or "",
                        "page_end": doc.get("page_end") or "",
                        "title": flat_meta.get("title") or policy_meta.get("title") or "",
                        "administration": flat_meta.get("administration") or "",
                        "agencies_affected": agencies_s,
                        "subject_matter": subjects_s,
                        "source_path": doc.get("source_path") or "",
                        "matched_attachment_ids": matched_attachment_ids,
                        "matched_doc_ids": matched_doc_ids,
                        "matched_chunk_ids": matched_chunk_ids,
                        "matched_count": r.matched_count,
                        "best_rank": r.best_rank,
                        "snippet": r.snippet or "",
                    }
                )


if __name__ == "__main__":
    main()
