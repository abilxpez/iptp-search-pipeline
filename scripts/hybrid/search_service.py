from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.bm25.search_bm25 import search_bm25  # type: ignore
from scripts.hybrid.search_rrf import RRFHit, rrf_fuse  # type: ignore
from scripts.semantic.search_faiss import (  # type: ignore
    SearchResult as SemanticSearchResult,
    aggregate_candidates,
    build_chunks_sqlite_if_missing,
    build_embedder_from_manifest,
    build_row_offsets_if_missing,
    chunks_sqlite_path_for,
    doc_passes_filters,
    faiss_search_rows,
    get_chunk_id_for_row,
    get_chunk_offset,
    get_index_semantics,
    init_chunks_sqlite,
    load_chunk_by_offset,
    load_faiss_index,
    load_manifest,
    load_row_offsets,
    maybe_set_hnsw_ef_search,
    open_sqlite,
    resolve_faiss_artifacts,
    row_offsets_path_for,
)


def _record_timing(timings: Optional[Dict[str, float]], key: str, started: float) -> None:
    if timings is not None:
        timings[key] = round((time.perf_counter() - started) * 1000.0, 2)


class SearchService:
    def __init__(
        self,
        *,
        config_path: Path,
        run_dir: Path,
        chunks_path: Path,
        device: Optional[str] = "cpu",
        batch_size: int = 16,
        ef_search: Optional[int] = None,
        warmup_query: Optional[str] = "temporary protected status",
        timings: Optional[Dict[str, float]] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.run_dir = Path(run_dir)
        self.chunks_path = Path(chunks_path)
        self.device = device
        self.batch_size = int(batch_size)
        self.ef_search = ef_search

        total_started = time.perf_counter()

        phase_started = time.perf_counter()
        self.manifest = load_manifest(self.run_dir)
        self.index_path, self.mapping_path, self.faiss_meta = resolve_faiss_artifacts(self.run_dir, self.manifest)
        self.dim, self.metric, self.normalize = get_index_semantics(self.manifest, self.faiss_meta)
        _record_timing(timings, "service.semantic_manifest", phase_started)

        phase_started = time.perf_counter()
        self.row_offsets_path = row_offsets_path_for(self.mapping_path)
        build_row_offsets_if_missing(self.mapping_path, self.row_offsets_path)
        self.row_offsets = load_row_offsets(self.row_offsets_path, mmap=True)
        _record_timing(timings, "service.semantic_row_offsets", phase_started)

        phase_started = time.perf_counter()
        self.chunks_sqlite_path = chunks_sqlite_path_for(self.chunks_path)
        build_chunks_sqlite_if_missing(self.chunks_path, self.chunks_sqlite_path)
        _record_timing(timings, "service.semantic_chunks_sqlite", phase_started)

        phase_started = time.perf_counter()
        self.index = load_faiss_index(self.index_path)
        maybe_set_hnsw_ef_search(self.index, self.ef_search)
        if int(getattr(self.index, "d", self.dim)) != int(self.dim):
            raise RuntimeError(
                f"FAISS index dim mismatch: index.d={getattr(self.index,'d',None)} manifest.dim={self.dim}"
            )
        self.ntotal = int(getattr(self.index, "ntotal", 0))
        if self.ntotal <= 0:
            raise RuntimeError("FAISS index is empty")
        _record_timing(timings, "service.semantic_load_index", phase_started)

        phase_started = time.perf_counter()
        self.embedder = build_embedder_from_manifest(
            self.manifest,
            device=self.device,
            batch_size=self.batch_size,
        )
        _record_timing(timings, "service.semantic_build_embedder", phase_started)

        if warmup_query:
            phase_started = time.perf_counter()
            qvec = self.embedder.embed_texts([str(warmup_query)])
            if qvec.shape != (1, self.dim):
                raise RuntimeError(f"Warmup embedding shape mismatch: got {qvec.shape}, expected (1, {self.dim})")
            _record_timing(timings, "service.semantic_warm_embed", phase_started)

        _record_timing(timings, "service.init_total", total_started)

    def search_semantic(
        self,
        *,
        query: str,
        top_k: int,
        oversample: int,
        filters: Dict[str, Optional[str]],
        snippet_chars: int,
        max_candidates: Optional[int],
        sort_by_announced_date: bool = False,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[SemanticSearchResult]:
        total_started = time.perf_counter()

        phase_started = time.perf_counter()
        qvec = self.embedder.embed_texts([query])
        if qvec.shape != (1, self.dim):
            raise RuntimeError(f"Query embedding shape mismatch: got {qvec.shape}, expected (1, {self.dim})")
        _record_timing(timings, "service.faiss.embed_query", phase_started)

        k0 = max(int(top_k), 1) * max(int(oversample), 1)
        k0 = min(k0, self.ntotal)

        max_expand_factor = 5
        tried: List[int] = []
        candidates: List[Dict[str, Any]] = []

        phase_started = time.perf_counter()
        conn = open_sqlite(self.chunks_sqlite_path)
        try:
            init_chunks_sqlite(conn)
            _record_timing(timings, "service.faiss.open_chunks_sqlite", phase_started)

            for attempt in range(2):
                k_candidate = k0 if attempt == 0 else min(self.ntotal, k0 * max_expand_factor)
                if k_candidate in tried:
                    continue
                tried.append(k_candidate)

                phase_started = time.perf_counter()
                distances, rows = faiss_search_rows(self.index, qvec, k_candidate)
                _record_timing(timings, f"service.faiss.search_rows_attempt_{attempt + 1}", phase_started)

                scores = -distances if self.metric == "l2" else distances

                phase_started = time.perf_counter()
                reached_candidate_limit = False
                for score, row in zip(scores.tolist(), rows.tolist()):
                    if row is None:
                        continue
                    try:
                        row_i = int(row)
                    except Exception:
                        continue
                    if row_i < 0 or row_i >= int(self.row_offsets.shape[0]):
                        continue

                    try:
                        chunk_id = get_chunk_id_for_row(self.mapping_path, self.row_offsets, row_i)
                    except Exception:
                        continue

                    offset = get_chunk_offset(conn, chunk_id)
                    if offset is None:
                        continue

                    try:
                        doc = load_chunk_by_offset(self.chunks_path, offset)
                    except Exception:
                        continue

                    if isinstance(doc, dict):
                        attachment_id_val = doc.get("attachment_id") or doc.get("policydocument_id")
                        if attachment_id_val is not None:
                            doc["attachment_id"] = str(attachment_id_val)

                    if not doc_passes_filters(doc, filters):
                        continue

                    candidates.append({"chunk_id": chunk_id, "score": float(score), "doc": doc})
                    if max_candidates and len(candidates) >= int(max_candidates):
                        reached_candidate_limit = True
                        break
                _record_timing(timings, f"service.faiss.hydrate_candidates_attempt_{attempt + 1}", phase_started)

                phase_started = time.perf_counter()
                aggregated = aggregate_candidates(
                    candidates,
                    snippet_chars,
                    self.chunks_path,
                    sort_by_announced_date=sort_by_announced_date,
                )
                _record_timing(timings, f"service.faiss.aggregate_attempt_{attempt + 1}", phase_started)
                if len(aggregated) >= int(top_k) or reached_candidate_limit:
                    break

            phase_started = time.perf_counter()
            aggregated = aggregate_candidates(
                candidates,
                snippet_chars,
                self.chunks_path,
                sort_by_announced_date=sort_by_announced_date,
            )
            _record_timing(timings, "service.faiss.aggregate_final", phase_started)
            return aggregated[: int(top_k)]
        finally:
            conn.close()
            _record_timing(timings, "service.faiss.total", total_started)

    def search(
        self,
        *,
        query: str,
        bm25_top_k: int,
        bm25_oversample: int,
        bm25_max_candidates: Optional[int],
        k1: Optional[float],
        b: Optional[float],
        faiss_top_k: int,
        faiss_oversample: int,
        faiss_max_candidates: Optional[int],
        snippet_chars: int,
        rrf_k: int,
        filters: Dict[str, Optional[str]],
        sort_by_announced_date: bool = False,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[RRFHit]:
        total_started = time.perf_counter()

        phase_started = time.perf_counter()
        bm25_results = search_bm25(
            config_path=self.config_path,
            query=str(query),
            top_k=bm25_top_k,
            filters=filters,
            k1=k1,
            b=b,
            snippet_chars=int(snippet_chars),
            max_candidates=bm25_max_candidates,
            oversample=int(bm25_oversample),
            sort_by_announced_date=sort_by_announced_date,
            timings=timings,
        )
        _record_timing(timings, "service.rrf.bm25_total", phase_started)

        phase_started = time.perf_counter()
        faiss_results = self.search_semantic(
            query=str(query),
            top_k=faiss_top_k,
            oversample=int(faiss_oversample),
            filters=filters,
            snippet_chars=int(snippet_chars),
            max_candidates=faiss_max_candidates,
            sort_by_announced_date=sort_by_announced_date,
            timings=timings,
        )
        _record_timing(timings, "service.rrf.faiss_total", phase_started)

        phase_started = time.perf_counter()
        fused = rrf_fuse(
            bm25_results=bm25_results,
            faiss_results=faiss_results,
            rrf_k=int(rrf_k),
        )
        _record_timing(timings, "service.rrf.fuse", phase_started)
        _record_timing(timings, "service.rrf.total", total_started)
        return fused
