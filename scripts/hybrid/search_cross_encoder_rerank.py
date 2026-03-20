"""
Cross-encoder rerank on top of BM25 + FAISS RRF fusion.

Example:

python -m scripts.hybrid.search_cross_encoder_rerank \
    --q "temporary protected status" \
    --auto_rrf \
    --cross_encoder_model "cross-encoder/ms-marco-MiniLM-L-6-v2"

python -m scripts.hybrid.search_cross_encoder_rerank \
    --q "temporary protected status" \
    --chunks data/sample_100/chunks/chunks.jsonl \
    --auto_rrf \
    --cross_encoder_model "cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --ce_top_k 20 \
    --final_k 20 \
    --sort_by_announced_date

other options: 
    --ce_batch_size 16 \
    --ce_max_chars 4000
    --sort_by_announced_date
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.semantic.search_faiss import (  # type: ignore
    build_chunks_sqlite_if_missing,
    chunks_sqlite_path_for,
    get_chunk_offset,
    init_chunks_sqlite,
    load_chunk_by_offset,
    open_sqlite,
)
from scripts.bm25.search_bm25 import format_excerpt  # type: ignore
from scripts.hybrid.search_rrf import RRFHit, run_rrf  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-encoder rerank on top of RRF fusion.")
    p.add_argument("--q", required=True, help="Query string")

    # Inputs
    p.add_argument("--rrf_results", default=None, help="Path to RRF JSONL results (from scripts.hybrid.rrf)")
    p.add_argument(
        "--auto_rrf",
        action="store_true",
        help="Generate RRF results automatically (no file is written)",
    )
    p.add_argument("--chunks", default="data/sample_100/chunks/chunks.jsonl", help="Path to chunks.jsonl")

    # RRF generation (used only with --auto_rrf)
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
    p.add_argument("--snippet_chars", type=int, default=220, help="Max chars in snippet output")
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

    # Cross-encoder
    p.add_argument(
        "--cross_encoder_model",
        required=True,
        help='HF model id for cross-encoder (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2")',
    )
    p.add_argument("--ce_batch_size", type=int, default=16, help="Cross-encoder batch size")
    p.add_argument("--ce_max_chars", type=int, default=4000, help="Max chars per chunk before encoding")
    p.add_argument("--ce_top_k", type=int, default=20, help="How many fused hits to rerank (default=all)")
    p.add_argument("--ce_device", default=None, help='Override CE device: "cuda", "mps", or "cpu"')
    p.add_argument("--final_k", type=int, default=20, help="How many results to print (default=all)")

    return p.parse_args()


def _resolve_device(override: Optional[str]) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def _date_to_int(date_str: Optional[str]) -> int:
    if not isinstance(date_str, str):
        return 0
    s = date_str.strip()
    if len(s) < 10:
        return 0
    try:
        y = int(s[0:4])
        m = int(s[5:7])
        d = int(s[8:10])
        return y * 10000 + m * 100 + d
    except Exception:
        return 0


def _get_text_for_hit(
    hit: RRFHit,
    *,
    chunks_path: Path,
    chunks_conn,
    max_chars: int,
) -> str:
    if hit.chunk_id:
        off = get_chunk_offset(chunks_conn, hit.chunk_id)
        if off is not None:
            try:
                doc = load_chunk_by_offset(chunks_path, off)
                text = doc.get("text")
                if isinstance(text, str) and text.strip():
                    text = text.strip()
                    if max_chars and len(text) > max_chars:
                        return text[: max_chars - 1] + "…"
                    return text
            except Exception:
                pass
    return hit.snippet or ""


def _cross_encoder_rerank(
    *,
    query: str,
    hits: List[RRFHit],
    model_id: str,
    chunks_path: Path,
    batch_size: int,
    max_chars: int,
    device: torch.device,
) -> List[RRFHit]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    model.to(device)

    sqlite_path = chunks_sqlite_path_for(chunks_path)
    build_chunks_sqlite_if_missing(chunks_path, sqlite_path)
    conn = open_sqlite(sqlite_path)
    init_chunks_sqlite(conn)

    try:
        pairs: List[Tuple[str, str]] = []
        for h in hits:
            text = _get_text_for_hit(h, chunks_path=chunks_path, chunks_conn=conn, max_chars=max_chars)
            pairs.append((query, text))

        scores: List[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), max(1, int(batch_size))):
                batch = pairs[i : i + int(batch_size)]
                qs, ts = zip(*batch)
                enc = tokenizer(
                    list(qs),
                    list(ts),
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                logits = out.logits
                if logits.ndim == 2 and logits.shape[-1] > 1:
                    logits = logits[:, -1]
                else:
                    logits = logits.squeeze(-1)
                scores.extend(logits.detach().float().cpu().tolist())

        for h, s in zip(hits, scores):
            h.ce_score = float(s)

        return sorted(
            hits,
            key=lambda h: (-(getattr(h, "ce_score", 0.0) or 0.0), -h.rrf_score, h.entry_id),
        )
    finally:
        conn.close()


def _load_rrf_jsonl(path: Path) -> List[RRFHit]:
    hits: List[RRFHit] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            entry_id = obj.get("entry_id")
            if entry_id is None:
                continue
            hit = RRFHit(entry_id=str(entry_id))
            hit.rrf_score = float(obj.get("rrf_score", 0.0))
            sources = obj.get("sources") or []
            if isinstance(sources, list):
                hit.sources = set(str(x) for x in sources)
            hit.bm25_rank = obj.get("bm25_rank")
            hit.faiss_rank = obj.get("faiss_rank")
            hit.bm25_score = obj.get("bm25_score")
            hit.faiss_score = obj.get("faiss_score")
            hit.chunk_id = obj.get("chunk_id")
            hit.doc_id = obj.get("doc_id")
            hit.attachment_id = obj.get("attachment_id")
            hit.snippet = obj.get("snippet")
            hit.matched_attachment_ids = obj.get("matched_attachment_ids") or []
            hit.matched_doc_ids = obj.get("matched_doc_ids") or []
            hit.matched_chunk_ids = obj.get("matched_chunk_ids") or []
            hit.matched_count = int(obj.get("matched_count") or 0)
            hit.bm25_best_rank = int(obj.get("bm25_best_rank") or 0)
            hit.faiss_best_rank = int(obj.get("faiss_best_rank") or 0)
            hits.append(hit)
    return hits


def run_cross_encoder(
    *,
    query: str,
    rrf_results: Optional[Path],
    auto_rrf: bool,
    chunks_path: Path,
    config_path: Path,
    run_dir: Path,
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
    rrf_k: int,
    snippet_chars: int,
    filters: Dict[str, Optional[str]],
    cross_encoder_model: str,
    ce_batch_size: int,
    ce_max_chars: int,
    ce_top_k: Optional[int],
    ce_device: Optional[str],
    final_k: Optional[int],
    sort_by_announced_date: bool = False,
) -> Tuple[List[RRFHit], Optional[Path]]:
    if auto_rrf:
        bm25_top_k = min(int(bm25_top_k), 100)
        faiss_top_k = min(int(faiss_top_k), 100)

        fused = run_rrf(
            query=str(query),
            config_path=Path(config_path),
            run_dir=Path(run_dir),
            chunks_path=Path(chunks_path),
            bm25_top_k=bm25_top_k,
            bm25_oversample=int(bm25_oversample),
            bm25_max_candidates=bm25_max_candidates,
            k1=k1,
            b=b,
            faiss_top_k=faiss_top_k,
            faiss_oversample=int(faiss_oversample),
            faiss_max_candidates=faiss_max_candidates,
            device=device if device else None,
            batch_size=int(batch_size),
            ef_search=int(ef_search) if ef_search is not None else None,
            snippet_chars=int(snippet_chars),
            rrf_k=int(rrf_k),
            filters=filters,
            sort_by_announced_date=False,
        )
        rrf_path = None
    else:
        if not rrf_results:
            raise RuntimeError("Provide --rrf_results or use --auto_rrf")
        rrf_path = Path(rrf_results)
        fused = _load_rrf_jsonl(rrf_path)

    rerank_k = int(ce_top_k) if ce_top_k is not None else len(fused)
    rerank_k = min(rerank_k, len(fused))
    device = _resolve_device(ce_device)

    reranked_scored = _cross_encoder_rerank(
        query=str(query),
        hits=fused[:rerank_k],
        model_id=str(cross_encoder_model),
        chunks_path=Path(chunks_path),
        batch_size=int(ce_batch_size),
        max_chars=int(ce_max_chars),
        device=device,
    )

    reranked = reranked_scored + fused[rerank_k:]

    if sort_by_announced_date:
        reranked = sorted(
            reranked_scored,
            key=lambda h: (-_date_to_int(h.announced_date), -float(getattr(h, "ce_score", 0.0)), -h.rrf_score, h.entry_id),
        )

    out_k = int(final_k) if final_k is not None else len(reranked)
    out_k = min(out_k, len(reranked))
    reranked = reranked[:out_k]

    return reranked, rrf_path


def main() -> None:
    args = parse_args()

    filters = {
        "entry_id": args.entry_id,
        "administration": args.administration,
        "agency": args.agency,
        "subject": args.subject,
    }

    reranked, rrf_path = run_cross_encoder(
        query=str(args.q),
        rrf_results=Path(args.rrf_results) if args.rrf_results else None,
        auto_rrf=bool(args.auto_rrf),
        chunks_path=Path(args.chunks),
        config_path=Path(args.config),
        run_dir=Path(args.run_dir),
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
        rrf_k=int(args.rrf_k),
        snippet_chars=int(args.snippet_chars),
        filters=filters,
        cross_encoder_model=str(args.cross_encoder_model),
        ce_batch_size=int(args.ce_batch_size),
        ce_max_chars=int(args.ce_max_chars),
        ce_top_k=int(args.ce_top_k) if args.ce_top_k is not None else None,
        ce_device=args.ce_device if args.ce_device else None,
        final_k=int(args.final_k) if args.final_k is not None else None,
        sort_by_announced_date=bool(args.sort_by_announced_date),
    )

    print(f"Query: {args.q}")
    print(f"CE model={args.cross_encoder_model} | final_k={len(reranked)}\n")

    for rank, h in enumerate(reranked, start=1):
        sources = ",".join(sorted(h.sources)) if h.sources else ""
        ce_val = getattr(h, "ce_score", None)
        print(
            f"{rank:02d}. ce={'' if ce_val is None else f'{ce_val:.4f}'}  "
            f"rrf={h.rrf_score:.6f}  policy_id={h.entry_id}  "
            f"sources={sources}"
        )
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


if __name__ == "__main__":
    main()
