"""
Hybrid RRF search over BM25 + FAISS results.

Example:

python -m scripts.hybrid.rrf \
  --q "temporary protected status" \
  --config config.json \
  --run_dir data/embeddings/bge_mean_norm \
  --chunks data/sample_100/chunks/chunks.jsonl \
--sort_by_announced_date

alternative:

python -m scripts.hybrid.rrf \
  --q "temporary protected status" \
  --config config.json \
  --run_dir data/embeddings/bge_mean_norm \
  --chunks data/sample_100/chunks/chunks.jsonl \
  --bm25_top_k 100 \
  --faiss_top_k 100 \
  --final_k 50 \
  --out_jsonl data/text/rrf_results_temporary_protected_status.jsonl


"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from scripts.bm25.search_bm25 import search_bm25  # type: ignore
from scripts.semantic.search_faiss import search_faiss  # type: ignore
from scripts.bm25.search_bm25 import format_excerpt  # type: ignore


@dataclass
class RRFHit:
    entry_id: str
    rrf_score: float = 0.0
    sources: Set[str] = field(default_factory=set)
    bm25_rank: Optional[int] = None
    faiss_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    faiss_score: Optional[float] = None
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    attachment_id: Optional[str] = None
    snippet: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    announced_date: Optional[str] = None
    relevant_chunk: Optional[Dict[str, Any]] = None
    matched_attachment_ids: List[str] = field(default_factory=list)
    matched_doc_ids: List[str] = field(default_factory=list)
    matched_chunk_ids: List[str] = field(default_factory=list)
    matched_count: int = 0
    bm25_best_rank: int = 0
    faiss_best_rank: int = 0


def _norm_entry_id(doc: Dict[str, Any]) -> Optional[str]:
    entry_id = doc.get("entry_id")
    if entry_id is None:
        entry_id = doc.get("page_ptr_id")
    if entry_id is None:
        return None
    return str(entry_id)


def _norm_doc_id(doc: Dict[str, Any]) -> Optional[str]:
    doc_id = doc.get("doc_id")
    if doc_id is None:
        return None
    return str(doc_id)


def _norm_attachment_id(doc: Dict[str, Any]) -> Optional[str]:
    attachment_id = doc.get("attachment_id")
    if attachment_id is None:
        attachment_id = doc.get("policydocument_id")
    if attachment_id is None:
        return None
    return str(attachment_id)


def _merge_hit(
    hits: Dict[str, RRFHit],
    *,
    source: str,
    rank: int,
    score: float,
    doc: Dict[str, Any],
    title: Optional[str],
    summary: Optional[str],
    announced_date: Optional[str],
    relevant_chunk: Optional[Dict[str, Any]],
    snippet: Optional[str],
    matched_attachment_ids: Optional[List[str]],
    matched_doc_ids: Optional[List[str]],
    matched_chunk_ids: Optional[List[str]],
    matched_count: Optional[int],
    best_rank: Optional[int],
    rrf_k: int,
) -> None:
    entry_id = _norm_entry_id(doc)
    if not entry_id:
        return

    hit = hits.get(entry_id)
    if hit is None:
        hit = RRFHit(entry_id=entry_id)
        hits[entry_id] = hit

    hit.rrf_score += 1.0 / (float(rrf_k) + float(rank))
    hit.sources.add(source)

    if source == "bm25":
        if hit.bm25_rank is None or rank < hit.bm25_rank:
            hit.bm25_rank = rank
            hit.bm25_score = float(score)
    elif source == "semantic":
        if hit.faiss_rank is None or rank < hit.faiss_rank:
            hit.faiss_rank = rank
            hit.faiss_score = float(score)

    if hit.chunk_id is None:
        chunk_id = doc.get("chunk_id")
        if chunk_id is not None:
            hit.chunk_id = str(chunk_id)

    if hit.doc_id is None:
        if source == "bm25":
            hit.doc_id = _norm_doc_id(doc)
    if hit.attachment_id is None:
        hit.attachment_id = _norm_attachment_id(doc)
    if hit.snippet is None and snippet:
        hit.snippet = snippet
    if hit.title is None and title:
        hit.title = title
    if hit.summary is None and summary:
        hit.summary = summary
    if hit.announced_date is None and announced_date:
        hit.announced_date = announced_date
    if hit.relevant_chunk is None and isinstance(relevant_chunk, dict):
        hit.relevant_chunk = relevant_chunk

    if matched_attachment_ids:
        for mid in matched_attachment_ids:
            if mid and mid not in hit.matched_attachment_ids:
                hit.matched_attachment_ids.append(mid)
    if matched_doc_ids:
        for mid in matched_doc_ids:
            if mid and mid not in hit.matched_doc_ids:
                hit.matched_doc_ids.append(mid)
    if matched_chunk_ids:
        for mid in matched_chunk_ids:
            if mid and mid not in hit.matched_chunk_ids:
                hit.matched_chunk_ids.append(mid)

    if best_rank is not None and int(best_rank) > 0:
        if source == "bm25":
            if hit.bm25_best_rank <= 0 or int(best_rank) < hit.bm25_best_rank:
                hit.bm25_best_rank = int(best_rank)
        elif source == "semantic":
            if hit.faiss_best_rank <= 0 or int(best_rank) < hit.faiss_best_rank:
                hit.faiss_best_rank = int(best_rank)

    # matched_count is derived from unique matched_chunk_ids across sources
    if hit.matched_chunk_ids:
        hit.matched_count = len(hit.matched_chunk_ids)
    elif matched_count is not None:
        hit.matched_count = max(hit.matched_count, int(matched_count))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid RRF search over BM25 + FAISS results.")
    p.add_argument("--q", required=True, help="Query string")

    # BM25 config
    p.add_argument("--config", default="config.json", help="Path to config.json")
    p.add_argument("--bm25_top_k", type=int, default=100, help="BM25 top_k to retrieve (cap 100)")
    p.add_argument("--bm25_oversample", type=int, default=10, help="BM25 oversample factor")
    p.add_argument("--bm25_max_candidates", type=int, default=None, help="BM25 max raw hits before dedupe")
    p.add_argument("--k1", type=float, default=None, help="BM25 k1 override")
    p.add_argument("--b", type=float, default=None, help="BM25 b override")

    # FAISS config
    p.add_argument("--run_dir", default="data/embeddings/bge_mean_norm", help="Embedding run dir (manifest.json)")
    p.add_argument("--chunks", default="data/sample_100/chunks/chunks.jsonl", help="Path to chunks.jsonl")
    p.add_argument("--faiss_top_k", type=int, default=100, help="FAISS top_k to retrieve (cap 100)")
    p.add_argument("--faiss_oversample", type=int, default=10, help="FAISS oversample factor")
    p.add_argument("--faiss_max_candidates", type=int, default=None, help="FAISS max raw hits before dedupe")
    p.add_argument("--device", default=None, help='Override embedding device: "mps", "cuda", or "cpu"')
    p.add_argument("--batch_size", type=int, default=16, help="Embedding batch size for query embedding")
    p.add_argument("--ef_search", type=int, default=None, help="Override HNSW efSearch (if applicable)")

    # Output / fusion
    p.add_argument("--final_k", type=int, default=50, help="Number of fused results to return")
    p.add_argument("--rrf_k", type=int, default=60, help="RRF k constant")
    p.add_argument("--snippet_chars", type=int, default=220, help="Max chars in snippet output")
    p.add_argument(
        "--sort_by_announced_date",
        action="store_true",
        help="Sort base BM25/FAISS results by announced_date before fusion",
    )
    p.add_argument("--out_jsonl", default=None, help="Optional path to write fused results as JSONL")

    # Optional filters (shared)
    p.add_argument("--entry_id", default=None)
    p.add_argument("--administration", default=None)
    p.add_argument("--agency", default=None)
    p.add_argument("--subject", default=None)
    return p.parse_args()


def rrf_fuse(
    *,
    bm25_results: List[Any],
    faiss_results: List[Any],
    rrf_k: int,
) -> List[RRFHit]:
    hits: Dict[str, RRFHit] = {}

    for rank, r in enumerate(bm25_results, start=1):
        _merge_hit(
            hits,
            source="bm25",
            rank=rank,
            score=float(r.score),
            doc=r.doc or {},
            title=getattr(r, "title", None),
            summary=getattr(r, "summary", None),
            announced_date=getattr(r, "announced_date", None),
            relevant_chunk=getattr(r, "relevant_chunk", None),
            snippet=r.snippet,
            matched_attachment_ids=getattr(r, "matched_attachment_ids", None),
            matched_doc_ids=getattr(r, "matched_doc_ids", None),
            matched_chunk_ids=getattr(r, "matched_chunk_ids", None),
            matched_count=getattr(r, "matched_count", None),
            best_rank=getattr(r, "best_rank", None),
            rrf_k=int(rrf_k),
        )

    for rank, r in enumerate(faiss_results, start=1):
        _merge_hit(
            hits,
            source="semantic",
            rank=rank,
            score=float(r.score),
            doc=r.doc or {},
            title=getattr(r, "title", None),
            summary=getattr(r, "summary", None),
            announced_date=getattr(r, "announced_date", None),
            relevant_chunk=getattr(r, "relevant_chunk", None),
            snippet=r.snippet,
            matched_attachment_ids=getattr(r, "matched_attachment_ids", None),
            matched_doc_ids=getattr(r, "matched_doc_ids", None),
            matched_chunk_ids=getattr(r, "matched_chunk_ids", None),
            matched_count=getattr(r, "matched_count", None),
            best_rank=getattr(r, "best_rank", None),
            rrf_k=int(rrf_k),
        )

    return sorted(hits.values(), key=lambda h: (-h.rrf_score, h.entry_id))


def run_rrf(
    *,
    query: str,
    config_path: Path,
    run_dir: Path,
    chunks_path: Path,
    bm25_top_k: int,
    bm25_oversample: int,
    bm25_max_candidates: Optional[int],
    k1: Optional[float],
    b: Optional[float],
    faiss_top_k: int,
    faiss_oversample: int,
    faiss_max_candidates: Optional[int],
    device: Optional[str],
    batch_size: int,
    ef_search: Optional[int],
    snippet_chars: int,
    rrf_k: int,
    filters: Dict[str, Optional[str]],
    sort_by_announced_date: bool = False,
) -> List[RRFHit]:
    bm25_results = search_bm25(
        config_path=config_path,
        query=str(query),
        top_k=bm25_top_k,
        filters=filters,
        k1=k1,
        b=b,
        snippet_chars=int(snippet_chars),
        max_candidates=bm25_max_candidates,
        oversample=int(bm25_oversample),
        sort_by_announced_date=sort_by_announced_date,
    )

    faiss_results = search_faiss(
        run_dir=run_dir,
        chunks_path=chunks_path,
        query=str(query),
        top_k=faiss_top_k,
        oversample=int(faiss_oversample),
        filters=filters,
        device=device if device else None,
        batch_size=int(batch_size),
        ef_search=int(ef_search) if ef_search is not None else None,
        snippet_chars=int(snippet_chars),
        max_candidates=faiss_max_candidates,
        sort_by_announced_date=sort_by_announced_date,
    )

    return rrf_fuse(
        bm25_results=bm25_results,
        faiss_results=faiss_results,
        rrf_k=int(rrf_k),
    )


def write_rrf_jsonl(path: Path, hits: List[RRFHit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for h in hits:
            f.write(
                (
                    "{"
                    f"\"entry_id\":{json.dumps(h.entry_id)},"
                    f"\"rrf_score\":{h.rrf_score:.6f},"
                    f"\"sources\":{json.dumps(sorted(h.sources))},"
                    f"\"bm25_rank\":{json.dumps(h.bm25_rank)},"
                    f"\"faiss_rank\":{json.dumps(h.faiss_rank)},"
                    f"\"bm25_score\":{json.dumps(h.bm25_score)},"
                    f"\"faiss_score\":{json.dumps(h.faiss_score)},"
                    f"\"chunk_id\":{json.dumps(h.chunk_id)},"
                    f"\"doc_id\":{json.dumps(h.doc_id)},"
                    f"\"attachment_id\":{json.dumps(h.attachment_id)},"
                    f"\"snippet\":{json.dumps(h.snippet)},"
                    f"\"matched_attachment_ids\":{json.dumps(h.matched_attachment_ids)},"
                    f"\"matched_doc_ids\":{json.dumps(h.matched_doc_ids)},"
                    f"\"matched_chunk_ids\":{json.dumps(h.matched_chunk_ids)},"
                    f"\"matched_count\":{json.dumps(h.matched_count)},"
                    f"\"bm25_best_rank\":{json.dumps(h.bm25_best_rank)},"
                    f"\"faiss_best_rank\":{json.dumps(h.faiss_best_rank)}"
                    "}\n"
                )
            )


def main() -> None:
    args = parse_args()

    bm25_top_k = min(int(args.bm25_top_k), 100)
    faiss_top_k = min(int(args.faiss_top_k), 100)
    final_k = min(int(args.final_k), 50)

    filters: Dict[str, Optional[str]] = {
        "entry_id": args.entry_id,
        "administration": args.administration,
        "agency": args.agency,
        "subject": args.subject,
    }

    # -----------------------------
    # BM25 search
    # -----------------------------
    fused = run_rrf(
        query=str(args.q),
        config_path=Path(args.config),
        run_dir=Path(args.run_dir),
        chunks_path=Path(args.chunks),
        bm25_top_k=bm25_top_k,
        bm25_oversample=int(args.bm25_oversample),
        bm25_max_candidates=args.bm25_max_candidates,
        k1=args.k1,
        b=args.b,
        faiss_top_k=faiss_top_k,
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
    fused = fused[:final_k]

    # -----------------------------
    # Print results (debug-friendly)
    # -----------------------------
    print(f"Query: {args.q}")
    if any(v for v in filters.values()):
        print(f"Filters: { {k: v for k, v in filters.items() if v} }")
    print(f"RRF k={int(args.rrf_k)} | final_k={final_k}\n")

    for rank, h in enumerate(fused, start=1):
        sources = ",".join(sorted(h.sources)) if h.sources else ""
        print(f"{rank:02d}. rrf={h.rrf_score:.6f}  policy_id={h.entry_id}  sources={sources}")
        if h.title:
            print(f"    title={format_excerpt(h.title, max_chars=75)}")
        if h.announced_date:
            print(f"    announced_date={h.announced_date}")
        if h.summary:
            print(f"    summary={format_excerpt(h.summary, max_chars=75)}")
        if h.relevant_chunk:
            ch = h.relevant_chunk
            print(f"    chunk:")
            print(f"      text={format_excerpt(ch.get('text'), max_chars=75)}")
            print(f"      attachment_date={ch.get('attachment_date') or ''}")
            print(f"      type={ch.get('document_type') or ''}")
            print(f"      document_id={ch.get('document_id') or ''}")
            page_start = ch.get("page_start") or ""
            page_end = ch.get("page_end") or ""
            print(f"      pages={page_start}-{page_end}")
            print(f"      id={ch.get('chunk_id') or ''}")
        else:
            print(f"    chunk=None")
        print("")

    if args.out_jsonl:
        write_rrf_jsonl(Path(args.out_jsonl), fused)


if __name__ == "__main__":
    main()
