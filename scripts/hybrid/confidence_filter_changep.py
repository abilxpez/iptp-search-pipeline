"""
Confidence filter on top of cross-encoder rerank.

Filter rule:
  kept = { d : ce(d) >= max_ce - delta and ce(d) >= T_abs }

By default, we run cross-encoder on the top 20 and then filter.

How to run:

python -m scripts.hybrid.confidence_filter \
  --q "temporary protected status" \
  --auto_rrf \
  --cross_encoder_model "cross-encoder/ms-marco-MiniLM-L-6-v2" \
  --ce_top_k 100 \
  --a_best 0.0 \
  --a_tail -3.5 \
  --cap 20 \
  --min_return 3 \
  --lambda_penalty 5


python -m scripts.hybrid.confidence_filter \
  --q "temporary protected status" \
  --auto_rrf \
  --cross_encoder_model "cross-encoder/ms-marco-MiniLM-L-6-v2" \
  --ce_top_k 100

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from scripts.hybrid.search_cross_encoder_rerank import run_cross_encoder  # type: ignore
from scripts.hybrid.search_rrf import RRFHit  # type: ignore


def confidence_filter(
    reranked: List[RRFHit],
    *,
    a_best: float,
    a_tail: float,
    cap: int,
    min_return: int,
    lambda_penalty: float,
) -> List[RRFHit]:
    # Constants
    A_best = float(a_best)
    A_tail = float(a_tail)
    CAP = int(cap)
    MIN_RETURN = int(min_return)

    scores = [getattr(r, "ce_score", None) for r in reranked]
    scores = [float(s) for s in scores if s is not None]
    M = len(scores)
    if M == 0:
        print("[WARN] No candidates after rerank.")
        return []

    if scores[0] < A_best:
        print(
            f"[WARN] Low-confidence query: best ce_score={scores[0]:.3f} < A_best={A_best}. "
            f"Will still return MIN_RETURN={MIN_RETURN} (if available)."
        )

    K_max = min(CAP, M - 1)
    if K_max < 1:
        K_star = 1
    else:
        best_sep = float("-inf")
        K_star = 1
        for k in range(1, K_max + 1):
            mean_top = sum(scores[0:k]) / float(k)
            mean_bottom = sum(scores[k:M]) / float(M - k)
            sep = mean_top - mean_bottom
            score_k = sep - (float(lambda_penalty) / float(k))
            if score_k > best_sep:
                best_sep = score_k
                K_star = k

    K_clean = 0
    for i in range(0, K_star):
        if scores[i] >= A_tail:
            K_clean += 1
        else:
            break

    K_final = K_clean
    if K_final < MIN_RETURN:
        forced = min(MIN_RETURN, CAP, M)
        if forced > K_final:
            print(
                f"[WARN] Forced minimum results: K_clean={K_clean} -> K_final={forced}. "
                f"(K_star={K_star}, best={scores[0]:.3f}, A_best={A_best}, A_tail={A_tail})"
            )
        K_final = forced

    K_final = min(K_final, CAP)
    return reranked[0:K_final]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confidence filter on top of cross-encoder rerank.")
    p.add_argument("--q", required=True, help="Query string")

    # Inputs
    p.add_argument("--rrf_results", default=None, help="Path to RRF JSONL results (from scripts.hybrid.rrf)")
    p.add_argument(
        "--auto_rrf",
        action="store_true",
        help="Generate RRF results automatically and save as rrf_results_{query}.jsonl",
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
    p.add_argument("--ce_top_k", type=int, default=20, help="How many fused hits to rerank (default=20)")
    p.add_argument("--ce_device", default=None, help='Override CE device: "cuda", "mps", or "cpu"')

    # Confidence filter (design flags)
    p.add_argument("--a_best", type=float, default=0.0, help="Viability threshold for top ce_score")
    p.add_argument("--a_tail", type=float, default=-2.5, help="Tail floor threshold for ce_score")
    p.add_argument("--cap", type=int, default=20, help="Max results returned")
    p.add_argument("--min_return", type=int, default=3, help="Minimum results returned")
    p.add_argument("--lambda_penalty", type=float, default=5.0, help="Linear penalty lambda for change-point score")
    p.add_argument("--conf_out_dir", default="data/text", help="Directory to save confidence JSONL")

    return p.parse_args()


def _slugify_query(q: str) -> str:
    s = q.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    if not s:
        s = "query"
    return s[:80]


def _print_hits(title: str, hits: List[RRFHit]) -> None:
    print(title)
    if not hits:
        print("  (no results)\n")
        return
    print("")
    for rank, h in enumerate(hits, start=1):
        sources = ",".join(sorted(h.sources)) if h.sources else ""
        ce_val = getattr(h, "ce_score", None)
        print(
            f"{rank:02d}. ce={'' if ce_val is None else f'{ce_val:.4f}'}  "
            f"rrf={h.rrf_score:.6f}  entry_id={h.entry_id}  "
            f"doc_id={h.doc_id or ''}  attachment_id={h.attachment_id or ''}  "
            f"sources={sources}"
        )
        print(f"    bm25_rank={h.bm25_rank or ''}  faiss_rank={h.faiss_rank or ''}")
        print(f"    chunk_id={h.chunk_id or ''}")
        if h.snippet:
            print(f"    snippet={h.snippet}")
        print("")


def _write_conf_jsonl(path: Path, hits: List[RRFHit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for h in hits:
            f.write(
                (
                    "{"
                    f"\"entry_id\":{json.dumps(h.entry_id)},"
                    f"\"ce_score\":{json.dumps(getattr(h, 'ce_score', None))},"
                    f"\"rrf_score\":{h.rrf_score:.6f},"
                    f"\"sources\":{json.dumps(sorted(h.sources))},"
                    f"\"bm25_rank\":{json.dumps(h.bm25_rank)},"
                    f"\"faiss_rank\":{json.dumps(h.faiss_rank)},"
                    f"\"bm25_score\":{json.dumps(h.bm25_score)},"
                    f"\"faiss_score\":{json.dumps(h.faiss_score)},"
                    f"\"chunk_id\":{json.dumps(h.chunk_id)},"
                    f"\"doc_id\":{json.dumps(h.doc_id)},"
                    f"\"attachment_id\":{json.dumps(h.attachment_id)},"
                    f"\"snippet\":{json.dumps(h.snippet)}"
                    "}\n"
                )
            )


def main() -> None:
    args = parse_args()

    filters: Dict[str, Optional[str]] = {
        "entry_id": args.entry_id,
        "administration": args.administration,
        "agency": args.agency,
        "subject": args.subject,
    }

    reranked, _rrf_path = run_cross_encoder(
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
        ce_top_k=int(args.ce_top_k),
        ce_device=args.ce_device if args.ce_device else None,
        final_k=int(args.ce_top_k),
    )

    reranked = reranked[: int(args.ce_top_k)]
    max_ce = getattr(reranked[0], "ce_score", None) if reranked else None
    filtered = confidence_filter(
        reranked,
        a_best=float(args.a_best),
        a_tail=float(args.a_tail),
        cap=int(args.cap),
        min_return=int(args.min_return),
        lambda_penalty=float(args.lambda_penalty),
    )

    slug = _slugify_query(str(args.q))
    out_path = Path(args.conf_out_dir) / f"conf_results_{slug}.jsonl"

    print(f"Query: {args.q}")
    print(
        f"CE results: {len(reranked)} | max_ce={'' if max_ce is None else f'{float(max_ce):.4f}'} | "
        f"kept={len(filtered)}\n"
    )

    _print_hits("Cross-encoder (top 20):", reranked)
    _print_hits("After confidence filter:", filtered)

    _write_conf_jsonl(out_path, filtered)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
