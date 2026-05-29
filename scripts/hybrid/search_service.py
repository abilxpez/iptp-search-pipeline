from __future__ import annotations

import time
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.bm25.search_bm25 import (  # type: ignore
    BM25_B_DEFAULT,
    BM25_K1_DEFAULT,
    _sqlite_ro,
    read_json,
    score_query,
)
from scripts.common.config import get_cfg_value, get_path, load_config  # type: ignore
from scripts.common.text_processing import init_text_processing_from_config  # type: ignore
from scripts.hybrid.search_rrf import RRFHit, rrf_fuse  # type: ignore
from scripts.semantic.search_faiss import (  # type: ignore
    SearchResult as SemanticSearchResult,
    aggregate_candidates,
    build_chunks_sqlite_if_missing,
    build_embedder_from_manifest,
    build_row_offsets_if_missing,
    build_title_summary_maps,
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


@dataclass
class CrossEncoderState:
    model_id: str
    tokenizer: Any
    model: Any
    device: Any
    batch_size: int
    max_chars: int


@dataclass
class RelevanceModelState:
    model: Any
    feature_columns: List[str]
    high_threshold: float
    medium_threshold: float


def load_relevance_model_state(
    *,
    model_dir: Path,
    timings: Optional[Dict[str, float]] = None,
) -> RelevanceModelState:
    started = time.perf_counter()
    import joblib

    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing relevance model artifact: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing relevance model metadata: {metadata_path}")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise RuntimeError(f"Invalid relevance model feature_columns in {metadata_path}")
    supported = ["rrf_rank", "rrf_score", "bm25_rank", "bm25_score", "faiss_rank", "faiss_score"]
    if [str(col) for col in feature_columns] != supported:
        raise RuntimeError(
            "Live relevance model currently supports only retrieval feature columns "
            f"{supported}; got {feature_columns}"
        )

    state = RelevanceModelState(
        model=model,
        feature_columns=[str(col) for col in feature_columns],
        high_threshold=float(metadata.get("high_threshold", 0.7)),
        medium_threshold=float(metadata.get("medium_threshold", 0.4)),
    )
    if state.high_threshold <= state.medium_threshold:
        raise RuntimeError(
            f"Invalid relevance band thresholds: high={state.high_threshold}, medium={state.medium_threshold}"
        )
    _record_timing(timings, "service.relevance.load_model", started)
    return state


def _resolve_cross_encoder_device(device_override: Optional[str]) -> Any:
    import torch

    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cross_encoder_state(
    *,
    model_id: str,
    device: Optional[str],
    batch_size: int,
    max_chars: int,
    timings: Optional[Dict[str, float]] = None,
) -> CrossEncoderState:
    started = time.perf_counter()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ce_device = _resolve_cross_encoder_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    model.to(ce_device)
    _record_timing(timings, "service.ce.load_model", started)

    return CrossEncoderState(
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        device=ce_device,
        batch_size=int(batch_size),
        max_chars=int(max_chars),
    )


def warm_cross_encoder_state(
    state: CrossEncoderState,
    *,
    query: str = "temporary protected status",
    text: str = "Temporary Protected Status policy action.",
    timings: Optional[Dict[str, float]] = None,
) -> None:
    started = time.perf_counter()
    import torch

    encoded = state.tokenizer(
        [(query, text)],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(state.device) for k, v in encoded.items()}
    with torch.no_grad():
        state.model(**encoded)
    _record_timing(timings, "service.ce.warm_inference", started)


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
        ce_model_id: Optional[str] = None,
        ce_device: Optional[str] = None,
        ce_batch_size: int = 16,
        ce_max_chars: int = 1500,
        ce_warmup: bool = True,
        relevance_model_dir: Optional[Path] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.run_dir = Path(run_dir)
        self.chunks_path = Path(chunks_path)
        self.device = device
        self.batch_size = int(batch_size)
        self.ef_search = ef_search
        self.cross_encoder: Optional[CrossEncoderState] = None
        self.relevance_model: Optional[RelevanceModelState] = None

        total_started = time.perf_counter()

        phase_started = time.perf_counter()
        self.bm25_config = load_config(self.config_path)
        self.bm25_base_dir = self.config_path.parent.resolve()
        init_text_processing_from_config(self.config_path)
        self.bm25_chunks_path = get_path(self.bm25_config, "paths.chunks_jsonl", base_dir=self.bm25_base_dir)
        self.bm25_docs_path = get_path(self.bm25_config, "paths.bm25_docs_jsonl", base_dir=self.bm25_base_dir)
        self.bm25_inv_path = get_path(
            self.bm25_config,
            "paths.bm25_inverted_index_jsonl",
            base_dir=self.bm25_base_dir,
        )
        self.bm25_offsets_db_path = get_path(
            self.bm25_config,
            "paths.bm25_offsets_sqlite",
            base_dir=self.bm25_base_dir,
        )
        self.bm25_docs_offsets_db_path = get_path(
            self.bm25_config,
            "paths.bm25_docs_offsets_sqlite",
            base_dir=self.bm25_base_dir,
        )
        self.bm25_chunks_offsets_db_path = get_path(
            self.bm25_config,
            "paths.bm25_chunks_offsets_sqlite",
            base_dir=self.bm25_base_dir,
        )
        self.bm25_corpus_stats_path = get_path(
            self.bm25_config,
            "paths.bm25_corpus_stats_json",
            base_dir=self.bm25_base_dir,
        )
        stats = read_json(self.bm25_corpus_stats_path)
        self.bm25_n_docs = int(stats.get("n_docs", 0))
        self.bm25_avgdl = float(stats.get("avgdl", 0.0))
        if self.bm25_n_docs <= 0 or self.bm25_avgdl <= 0.0:
            raise RuntimeError(
                f"Invalid corpus_stats.json (n_docs={self.bm25_n_docs}, avgdl={self.bm25_avgdl})"
            )
        k1_cfg = get_cfg_value(self.bm25_config, "bm25.k1")
        b_cfg = get_cfg_value(self.bm25_config, "bm25.b")
        self.bm25_k1_default = float(k1_cfg) if k1_cfg is not None else BM25_K1_DEFAULT
        self.bm25_b_default = float(b_cfg) if b_cfg is not None else BM25_B_DEFAULT
        _record_timing(timings, "service.bm25_context", phase_started)

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

        phase_started = time.perf_counter()
        self.title_summary_maps = build_title_summary_maps(self.chunks_path)
        _record_timing(timings, "service.title_summary_maps", phase_started)

        if ce_model_id:
            self.cross_encoder = load_cross_encoder_state(
                model_id=str(ce_model_id),
                device=ce_device,
                batch_size=int(ce_batch_size),
                max_chars=int(ce_max_chars),
                timings=timings,
            )
            if ce_warmup:
                warm_cross_encoder_state(self.cross_encoder, timings=timings)

        if relevance_model_dir:
            self.relevance_model = load_relevance_model_state(
                model_dir=Path(relevance_model_dir),
                timings=timings,
            )

        _record_timing(timings, "service.init_total", total_started)

    def _relevance_features_for_hit(self, *, hit: RRFHit, rrf_rank: int) -> List[float]:
        return [
            float(rrf_rank),
            float(hit.rrf_score or 0.0),
            float(hit.bm25_rank if hit.bm25_rank is not None else 999.0),
            float(hit.bm25_score if hit.bm25_score is not None else 0.0),
            float(hit.faiss_rank if hit.faiss_rank is not None else 999.0),
            float(hit.faiss_score if hit.faiss_score is not None else 0.0),
        ]

    def annotate_relevance_hits(
        self,
        hits: List[RRFHit],
        *,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[RRFHit]:
        if self.relevance_model is None or not hits:
            return hits

        started = time.perf_counter()
        phase_started = time.perf_counter()
        features = [
            self._relevance_features_for_hit(hit=hit, rrf_rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]
        _record_timing(timings, "service.relevance.prepare_features", phase_started)

        phase_started = time.perf_counter()
        probabilities = self.relevance_model.model.predict_proba(features)[:, 1]
        _record_timing(timings, "service.relevance.predict", phase_started)

        phase_started = time.perf_counter()
        for hit, probability in zip(hits, probabilities.tolist()):
            prob = float(probability)
            if prob >= self.relevance_model.high_threshold:
                band = "high"
            elif prob >= self.relevance_model.medium_threshold:
                band = "medium"
            else:
                band = "low"
            setattr(hit, "relevance_probability", prob)
            setattr(hit, "relevance_band", band)
        _record_timing(timings, "service.relevance.assign_bands", phase_started)
        _record_timing(timings, "service.relevance.annotate", started)
        return hits

    def search_bm25(
        self,
        *,
        query: str,
        top_k: int,
        filters: Dict[str, Optional[str]],
        k1: Optional[float],
        b: Optional[float],
        snippet_chars: int,
        max_candidates: Optional[int],
        oversample: int,
        sort_by_announced_date: bool = False,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[Any]:
        total_started = time.perf_counter()
        k1_val = float(k1) if k1 is not None else self.bm25_k1_default
        b_val = float(b) if b is not None else self.bm25_b_default

        phase_started = time.perf_counter()
        offsets_conn = _sqlite_ro(self.bm25_offsets_db_path)
        docs_offsets_conn = _sqlite_ro(self.bm25_docs_offsets_db_path)
        chunks_offsets_conn = _sqlite_ro(self.bm25_chunks_offsets_db_path)
        _record_timing(timings, "service.bm25.open_sqlite", phase_started)

        try:
            phase_started = time.perf_counter()
            results = score_query(
                query=str(query),
                top_k=int(top_k),
                k1=k1_val,
                b=b_val,
                n_docs=self.bm25_n_docs,
                avgdl=self.bm25_avgdl,
                docs_path=self.bm25_docs_path,
                inv_path=self.bm25_inv_path,
                chunks_path=self.bm25_chunks_path,
                offsets_conn=offsets_conn,
                docs_offsets_conn=docs_offsets_conn,
                chunks_offsets_conn=chunks_offsets_conn,
                filters=filters,
                snippet_chars=int(snippet_chars),
                max_candidates=max_candidates,
                oversample=int(oversample),
                sort_by_announced_date=sort_by_announced_date,
                title_summary_maps=self.title_summary_maps,
            )
            _record_timing(timings, "service.bm25.score_query", phase_started)
            return results
        finally:
            try:
                offsets_conn.close()
            except Exception:
                pass
            try:
                docs_offsets_conn.close()
            except Exception:
                pass
            try:
                chunks_offsets_conn.close()
            except Exception:
                pass
            _record_timing(timings, "service.bm25.total", total_started)

    # EXPERIMENTAL CE FILTER HELPERS: remove these methods if CE filtering is discarded.
    def _text_for_cross_encoder_hit(self, hit: RRFHit) -> str:
        parts: List[str] = []
        if hit.title:
            parts.append(str(hit.title))
        if hit.summary:
            parts.append(str(hit.summary))
        if hit.full_chunk_text:
            parts.append(str(hit.full_chunk_text))
        elif hit.snippet:
            parts.append(str(hit.snippet))

        text = "\n\n".join(p.strip() for p in parts if p and str(p).strip())
        if self.cross_encoder and self.cross_encoder.max_chars > 0 and len(text) > self.cross_encoder.max_chars:
            return text[: self.cross_encoder.max_chars]
        return text

    def score_cross_encoder_hits(
        self,
        *,
        query: str,
        hits: List[RRFHit],
        top_k: int,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[RRFHit]:
        if self.cross_encoder is None:
            raise RuntimeError("Cross-encoder state is not loaded. Start API with --enable-ce-filter.")

        total_started = time.perf_counter()
        import torch

        state = self.cross_encoder
        scored_hits = list(hits)
        to_score = scored_hits[: max(0, int(top_k))]
        if not to_score:
            _record_timing(timings, "service.ce.total", total_started)
            return scored_hits

        phase_started = time.perf_counter()
        pairs = [(str(query), self._text_for_cross_encoder_hit(hit)) for hit in to_score]
        _record_timing(timings, "service.ce.prepare_pairs", phase_started)
        scores: List[float] = []

        phase_started = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(pairs), state.batch_size):
                batch = pairs[start : start + state.batch_size]
                encoded = state.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(state.device) for k, v in encoded.items()}
                output = state.model(**encoded)
                logits = output.logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    batch_scores = logits[:, 0]
                elif logits.ndim == 2:
                    batch_scores = logits[:, -1]
                else:
                    batch_scores = logits.reshape(-1)
                scores.extend(float(v) for v in batch_scores.detach().cpu().tolist())
        _record_timing(timings, "service.ce.inference", phase_started)

        phase_started = time.perf_counter()
        for hit, score in zip(to_score, scores):
            setattr(hit, "ce_score", float(score))
        _record_timing(timings, "service.ce.attach_scores", phase_started)

        _record_timing(timings, "service.ce.total", total_started)
        return scored_hits

    def filter_cross_encoder_hits(
        self,
        *,
        hits: List[RRFHit],
        threshold: float,
        min_return: int,
        top_k: int,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[RRFHit]:
        started = time.perf_counter()
        if not hits:
            _record_timing(timings, "service.ce.filter", started)
            return []

        score_limit = max(0, int(top_k))
        scored_window = hits[:score_limit] if score_limit else []
        unscored_tail = hits[score_limit:] if score_limit else hits

        kept = [
            hit
            for hit in scored_window
            if getattr(hit, "ce_score", None) is not None and float(getattr(hit, "ce_score")) >= float(threshold)
        ]

        min_needed = max(0, int(min_return))
        if min_needed and len(kept) < min_needed:
            for hit in scored_window:
                if hit not in kept:
                    kept.append(hit)
                if len(kept) >= min_needed:
                    break

        # For the prototype, only the CE-scored window is eligible for display.
        # Keep this assignment so removing CE filtering later is a single-method deletion.
        _unused_tail = unscored_tail

        _record_timing(timings, "service.ce.filter", started)
        return kept

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
                    title_summary_maps=self.title_summary_maps,
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
                title_summary_maps=self.title_summary_maps,
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
        ce_filter: bool = False,
        ce_top_k: int = 10,
        ce_threshold: float = 2.5,
        ce_min_return: int = 1,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[RRFHit]:
        total_started = time.perf_counter()

        phase_started = time.perf_counter()
        bm25_results = self.search_bm25(
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

        if ce_filter:
            if self.cross_encoder is None:
                raise RuntimeError("CE filter requested, but cross-encoder is not loaded. Start API with --enable-ce-filter.")
            fused = self.score_cross_encoder_hits(
                query=str(query),
                hits=fused,
                top_k=int(ce_top_k),
                timings=timings,
            )
            fused = self.filter_cross_encoder_hits(
                hits=fused,
                threshold=float(ce_threshold),
                min_return=int(ce_min_return),
                top_k=int(ce_top_k),
                timings=timings,
            )

        fused = self.annotate_relevance_hits(fused, timings=timings)

        _record_timing(timings, "service.rrf.total", total_started)
        return fused
