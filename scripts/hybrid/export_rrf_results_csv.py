"""
Export RRF (BM25 + FAISS fusion) results to CSV (Google Sheets friendly).

Usage:

python -m scripts.hybrid.export_rrf_results_csv \
  --queries data/text/queries.txt \
  --top_k 5 \
  --out data/text/rrf_results.csv \
  --snippet_chars 1000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from scripts.hybrid.search_rrf import run_rrf  # type: ignore
from scripts.semantic.search_faiss import (  # type: ignore
    build_chunks_sqlite_if_missing,
    chunks_sqlite_path_for,
    format_snippet,
    get_chunk_offset,
    init_chunks_sqlite,
    load_chunk_by_offset,
    open_sqlite,
)


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
    p = argparse.ArgumentParser(description="Export RRF results to CSV.")
    p.add_argument("--queries", required=True, help="Text file with one query per line")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--top_k", type=int, default=5, help="Top K results per query")
    p.add_argument("--snippet_chars", type=int, default=220, help="Max chars in snippet column")

    # Inputs
    p.add_argument("--chunks", default="data/sample_100/chunks/chunks.jsonl", help="Path to chunks.jsonl")

    # RRF generation
    p.add_argument("--config", default="config.json", help="Path to config.json")
    p.add_argument("--run_dir", default="data/embeddings/bge_mean_norm", help="Embedding run dir (manifest.json)")
    p.add_argument("--bm25_top_k", type=int, default=100, help="BM25 top_k to retrieve (cap 100)")
    p.add_argument("--bm25_oversample", type=int, default=10, help="BM25 oversample factor")
    p.add_argument("--bm25_max_candidates", type=int, default=None, help="BM25 max raw hits before dedupe")
    p.add_argument("--k1", type=float, default=None, help="BM25 k1 override")
    p.add_argument("--b", type=float, default=None, help="BM25 b override")
    p.add_argument("--faiss_top_k", type=int, default=100, help="FAISS top_k to retrieve (cap 100)")
    p.add_argument("--faiss_oversample", type=int, default=10, help="FAISS oversample factor")
    p.add_argument("--faiss_max_candidates", type=int, default=None, help="FAISS max raw hits before dedupe")
    p.add_argument("--device", default=None, help='Override embedding device: "mps", "cuda", or "cpu"')
    p.add_argument("--batch_size", type=int, default=16, help="Embedding batch size for query embedding")
    p.add_argument("--ef_search", type=int, default=None, help="Override HNSW efSearch (if applicable)")
    p.add_argument("--rrf_k", type=int, default=60, help="RRF k constant")
    p.add_argument(
        "--sort_by_announced_date",
        action="store_true",
        help="Sort base BM25/FAISS results by announced_date before fusion",
    )

    # Optional filters (shared)
    p.add_argument("--entry_id", default=None)
    p.add_argument("--administration", default=None)
    p.add_argument("--agency", default=None)
    p.add_argument("--subject", default=None)

    return p.parse_args()


def _load_doc_for_hit(hit, *, chunks_path: Path, chunks_conn):
    if hit.chunk_id:
        off = get_chunk_offset(chunks_conn, hit.chunk_id)
        if off is not None:
            try:
                return load_chunk_by_offset(chunks_path, off)
            except Exception:
                return {}
    return {}


def _snippet_for_hit(*, doc: dict, hit, snippet_chars: int) -> str:
    text = doc.get("text") if isinstance(doc, dict) else None
    if isinstance(text, str) and text.strip():
        if snippet_chars and snippet_chars > 0:
            return format_snippet(text, max_chars=int(snippet_chars))
        return text.strip()

    if hit.snippet:
        if snippet_chars and snippet_chars > 0:
            return format_snippet(hit.snippet, max_chars=int(snippet_chars))
        return str(hit.snippet)

    return ""


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
        "bm25_best_rank",
        "faiss_best_rank",
        "snippet",
    ]

    chunks_path = Path(args.chunks)
    sqlite_path = chunks_sqlite_path_for(chunks_path)
    build_chunks_sqlite_if_missing(chunks_path, sqlite_path)
    conn = open_sqlite(sqlite_path)
    init_chunks_sqlite(conn)

    try:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for q in queries:
                fused = run_rrf(
                    query=str(q),
                    config_path=Path(args.config),
                    run_dir=Path(args.run_dir),
                    chunks_path=chunks_path,
                    bm25_top_k=int(args.bm25_top_k),
                    bm25_oversample=int(args.bm25_oversample),
                    bm25_max_candidates=args.bm25_max_candidates,
                    k1=args.k1,
                    b=args.b,
                    faiss_top_k=int(args.faiss_top_k),
                    faiss_oversample=int(args.faiss_oversample),
                    faiss_max_candidates=args.faiss_max_candidates,
                    device=args.device if args.device else None,
                    batch_size=int(args.batch_size),
                    ef_search=int(args.ef_search) if args.ef_search is not None else None,
                    snippet_chars=int(args.snippet_chars),
                    rrf_k=int(args.rrf_k),
                    filters=filters,
                    sort_by_announced_date=bool(args.sort_by_announced_date),
                )

                fused = fused[: int(args.top_k)]

                for rank, h in enumerate(fused, start=1):
                    doc = _load_doc_for_hit(h, chunks_path=chunks_path, chunks_conn=conn)
                    meta = doc.get("meta") or {}
                    policy_meta = meta.get("policy") or {}
                    attachment_meta = meta.get("attachment") or {}
                    flat_meta = doc.get("meta_flat") or {}

                    agencies = flat_meta.get("agencies_affected") or []
                    subjects = flat_meta.get("subject_matter") or []
                    agencies_s = json.dumps(agencies, ensure_ascii=False) if isinstance(agencies, list) else (agencies or "")
                    subjects_s = json.dumps(subjects, ensure_ascii=False) if isinstance(subjects, list) else (subjects or "")

                    doc_id_str = str(h.doc_id) if getattr(h, "doc_id", None) is not None else ""
                    snippet = _snippet_for_hit(doc=doc, hit=h, snippet_chars=int(args.snippet_chars))

                    writer.writerow(
                        {
                            "query": q,
                            "rank": rank,
                            "score": f"{float(h.rrf_score):.6f}",
                            "doc_id": doc_id_str,
                            "chunk_id": doc.get("chunk_id") or h.chunk_id or "",
                            "entry_id": h.entry_id or doc.get("page_ptr_id") or "",
                            "attachment_id": doc.get("attachment_id") or attachment_meta.get("policydocument_id") or "",
                            "page_start": doc.get("page_start") or "",
                            "page_end": doc.get("page_end") or "",
                            "title": flat_meta.get("title") or policy_meta.get("title") or "",
                            "administration": flat_meta.get("administration") or "",
                            "agencies_affected": agencies_s,
                            "subject_matter": subjects_s,
                            "source_path": doc.get("source_path") or "",
                            "matched_attachment_ids": json.dumps(getattr(h, "matched_attachment_ids", []) or [], ensure_ascii=False),
                            "matched_doc_ids": json.dumps(getattr(h, "matched_doc_ids", []) or [], ensure_ascii=False),
                            "matched_chunk_ids": json.dumps(getattr(h, "matched_chunk_ids", []) or [], ensure_ascii=False),
                            "matched_count": getattr(h, "matched_count", 0) or 0,
                            "bm25_best_rank": getattr(h, "bm25_best_rank", 0) or 0,
                            "faiss_best_rank": getattr(h, "faiss_best_rank", 0) or 0,
                            "snippet": snippet or "",
                        }
                    )
    finally:
        try:
            conn.close()
        except Exception:
            pass

    print(f"Wrote CSV: {out_path}")
    print(f"Queries: {len(queries)} | top_k: {int(args.top_k)}")
    if any(v for v in filters.values()):
        print(f"Filters: { {k: v for k, v in filters.items() if v} }")


if __name__ == "__main__":
    main()
