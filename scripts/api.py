from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

# Stability defaults for local macOS development with ML backends.
# Keep tokenizer threading off and cap CPU thread pools unless caller overrides.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# macOS local-dev workaround when torch/faiss pull duplicate OpenMP runtimes.
# This is not ideal for production, but prevents hard aborts in mixed wheel setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _parse_int(value: Any, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _parse_float(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _first_qs_value(qs: Dict[str, Any], key: str, default: Any = None) -> Any:
    raw = qs.get(key, default)
    if isinstance(raw, list):
        return raw[0] if raw else default
    return raw


def _hit_to_dict(hit: Any) -> Dict[str, Any]:
    data = asdict(hit)
    sources = getattr(hit, "sources", [])
    data["sources"] = sorted(list(sources))
    summary = data.get("summary")
    snippet = data.get("snippet")
    data["summary"] = str(summary).strip() if summary else ""
    data["snippet"] = str(snippet).strip() if snippet else ""
    # Explicit alias for UI consumers that need the matched chunk snippet.
    data["chunk_snippet"] = data["snippet"]
    full_chunk_text = data.get("full_chunk_text")
    data["full_chunk_text"] = str(full_chunk_text) if full_chunk_text else ""
    source_path = data.get("source_path")
    data["source_path"] = str(source_path) if source_path else ""
    data["source_file_name"] = os.path.basename(data["source_path"]) if data["source_path"] else ""
    return data


@dataclass
class APIConfig:
    config_path: Path
    run_dir: Path
    chunks_path: Path
    allow_origin: str
    bm25_top_k: int = 100
    bm25_oversample: int = 10
    bm25_max_candidates: Optional[int] = None
    k1: Optional[float] = None
    b: Optional[float] = None
    faiss_top_k: int = 100
    faiss_oversample: int = 10
    faiss_max_candidates: Optional[int] = None
    device: Optional[str] = "cpu"
    batch_size: int = 16
    ef_search: Optional[int] = None
    snippet_chars: int = 220
    rrf_k: int = 60
    final_k: int = 50


class SearchAPIHandler(BaseHTTPRequestHandler):
    api_config: APIConfig

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", self.api_config.allow_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.api_config.allow_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "iptp_search_api",
                    "config": str(self.api_config.config_path),
                    "run_dir": str(self.api_config.run_dir),
                    "chunks": str(self.api_config.chunks_path),
                },
            )
            return

        if parsed.path == "/search":
            qs = parse_qs(parsed.query, keep_blank_values=False)
            payload = {
                "q": _first_qs_value(qs, "q"),
                "final_k": _first_qs_value(qs, "final_k"),
                "entry_id": _first_qs_value(qs, "entry_id"),
                "administration": _first_qs_value(qs, "administration"),
                "agency": _first_qs_value(qs, "agency"),
                "subject": _first_qs_value(qs, "subject"),
                "sort_by_announced_date": _first_qs_value(qs, "sort_by_announced_date"),
            }
            self._handle_search(payload)
            return

        self._not_found()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/search":
            self._not_found()
            return

        content_length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"ok": False, "error": "JSON body must be an object"})
            return

        self._handle_search(payload)

    def _handle_search(self, payload: Dict[str, Any]) -> None:
        query = str(payload.get("q") or "").strip()
        if not query:
            self._send_json(400, {"ok": False, "error": "Missing required field: q"})
            return

        cfg = self.api_config
        final_k = _parse_int(payload.get("final_k"), cfg.final_k, min_value=1, max_value=50)

        filters: Dict[str, Optional[str]] = {
            "entry_id": (str(payload.get("entry_id")).strip() or None) if payload.get("entry_id") is not None else None,
            "administration": (str(payload.get("administration")).strip() or None) if payload.get("administration") is not None else None,
            "agency": (str(payload.get("agency")).strip() or None) if payload.get("agency") is not None else None,
            "subject": (str(payload.get("subject")).strip() or None) if payload.get("subject") is not None else None,
        }

        bm25_top_k = _parse_int(payload.get("bm25_top_k"), cfg.bm25_top_k, min_value=1, max_value=100)
        faiss_top_k = _parse_int(payload.get("faiss_top_k"), cfg.faiss_top_k, min_value=1, max_value=100)
        bm25_oversample = _parse_int(payload.get("bm25_oversample"), cfg.bm25_oversample, min_value=1, max_value=100)
        faiss_oversample = _parse_int(payload.get("faiss_oversample"), cfg.faiss_oversample, min_value=1, max_value=100)
        snippet_chars = _parse_int(payload.get("snippet_chars"), cfg.snippet_chars, min_value=50, max_value=500)
        batch_size = _parse_int(payload.get("batch_size"), cfg.batch_size, min_value=1, max_value=128)
        rrf_k = _parse_int(payload.get("rrf_k"), cfg.rrf_k, min_value=1, max_value=1000)
        sort_by_announced_date = _parse_bool(payload.get("sort_by_announced_date"), default=False)

        bm25_max_candidates = payload.get("bm25_max_candidates", cfg.bm25_max_candidates)
        if bm25_max_candidates is not None:
            bm25_max_candidates = _parse_int(bm25_max_candidates, cfg.bm25_top_k, min_value=1)

        faiss_max_candidates = payload.get("faiss_max_candidates", cfg.faiss_max_candidates)
        if faiss_max_candidates is not None:
            faiss_max_candidates = _parse_int(faiss_max_candidates, cfg.faiss_top_k, min_value=1)

        k1 = _parse_float(payload.get("k1"), cfg.k1)
        b = _parse_float(payload.get("b"), cfg.b)

        device = payload.get("device", cfg.device)
        if device is not None:
            device = str(device).strip() or None

        ef_search = payload.get("ef_search", cfg.ef_search)
        if ef_search is not None:
            ef_search = _parse_int(ef_search, cfg.ef_search or 64, min_value=1)

        started = time.perf_counter()
        timings: Dict[str, float] = {}
        try:
            fused = _run_search(
                cfg=cfg,
                query=query,
                bm25_top_k=bm25_top_k,
                bm25_oversample=bm25_oversample,
                bm25_max_candidates=bm25_max_candidates,
                k1=k1,
                b=b,
                faiss_top_k=faiss_top_k,
                faiss_oversample=faiss_oversample,
                faiss_max_candidates=faiss_max_candidates,
                device=device,
                batch_size=batch_size,
                ef_search=ef_search,
                snippet_chars=snippet_chars,
                rrf_k=rrf_k,
                filters=filters,
                sort_by_announced_date=sort_by_announced_date,
                timings=timings,
            )
        except Exception as e:
            self._send_json(500, {"ok": False, "error": str(e)})
            return

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        timings["api.search_handler_total"] = elapsed_ms
        result_mapping_started = time.perf_counter()
        results = [_hit_to_dict(hit) for hit in fused[:final_k]]
        timings["api.result_mapping"] = round((time.perf_counter() - result_mapping_started) * 1000.0, 2)
        self._send_json(
            200,
            {
                "ok": True,
                "query": query,
                "count": len(results),
                "elapsed_ms": elapsed_ms,
                "timings_ms": timings,
                "results": results,
            },
        )


def _run_search(
    *,
    cfg: APIConfig,
    query: str,
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
    sort_by_announced_date: bool,
    timings: Optional[Dict[str, float]] = None,
) -> Any:
    import_started = time.perf_counter()
    from scripts.hybrid.search_rrf import run_rrf  # type: ignore
    if timings is not None:
        timings["api.import_run_rrf"] = round((time.perf_counter() - import_started) * 1000.0, 2)

    return run_rrf(
        query=query,
        config_path=cfg.config_path,
        run_dir=cfg.run_dir,
        chunks_path=cfg.chunks_path,
        bm25_top_k=bm25_top_k,
        bm25_oversample=bm25_oversample,
        bm25_max_candidates=bm25_max_candidates,
        k1=k1,
        b=b,
        faiss_top_k=faiss_top_k,
        faiss_oversample=faiss_oversample,
        faiss_max_candidates=faiss_max_candidates,
        device=device,
        batch_size=batch_size,
        ef_search=ef_search,
        snippet_chars=snippet_chars,
        rrf_k=rrf_k,
        filters=filters,
        sort_by_announced_date=sort_by_announced_date,
        timings=timings,
    )


def warm_search_backend(cfg: APIConfig) -> None:
    started = time.perf_counter()
    timings: Dict[str, float] = {}
    try:
        import_started = time.perf_counter()
        from scripts.hybrid.search_rrf import run_rrf as _run_rrf  # type: ignore  # noqa: F401

        timings["api.import_run_rrf"] = round((time.perf_counter() - import_started) * 1000.0, 2)

        bm25_started = time.perf_counter()
        from scripts.common.text_processing import init_text_processing_from_config, tokenize  # type: ignore

        init_text_processing_from_config(cfg.config_path)
        tokenize("temporary protected status")
        timings["warmup.bm25_text_processing"] = round((time.perf_counter() - bm25_started) * 1000.0, 2)

        semantic_started = time.perf_counter()
        from scripts.semantic.search_faiss import (  # type: ignore
            build_chunks_sqlite_if_missing,
            build_embedder_from_manifest,
            build_row_offsets_if_missing,
            chunks_sqlite_path_for,
            get_index_semantics,
            load_faiss_index,
            load_manifest,
            load_row_offsets,
            maybe_set_hnsw_ef_search,
            resolve_faiss_artifacts,
            row_offsets_path_for,
        )

        mf = load_manifest(cfg.run_dir)
        index_path, mapping_path, faiss_meta = resolve_faiss_artifacts(cfg.run_dir, mf)
        dim, _metric, _normalize_flag = get_index_semantics(mf, faiss_meta)
        timings["warmup.semantic_manifest"] = round((time.perf_counter() - semantic_started) * 1000.0, 2)

        offsets_started = time.perf_counter()
        offsets_path = row_offsets_path_for(mapping_path)
        build_row_offsets_if_missing(mapping_path, offsets_path)
        load_row_offsets(offsets_path, mmap=True)
        timings["warmup.semantic_row_offsets"] = round((time.perf_counter() - offsets_started) * 1000.0, 2)

        chunks_started = time.perf_counter()
        sqlite_path = chunks_sqlite_path_for(cfg.chunks_path)
        build_chunks_sqlite_if_missing(cfg.chunks_path, sqlite_path)
        timings["warmup.semantic_chunks_sqlite"] = round((time.perf_counter() - chunks_started) * 1000.0, 2)

        index_started = time.perf_counter()
        index = load_faiss_index(index_path)
        maybe_set_hnsw_ef_search(index, cfg.ef_search)
        if int(getattr(index, "d", dim)) != int(dim):
            raise RuntimeError(f"FAISS index dim mismatch: index.d={getattr(index,'d',None)} manifest.dim={dim}")
        timings["warmup.semantic_load_index"] = round((time.perf_counter() - index_started) * 1000.0, 2)

        embedder_started = time.perf_counter()
        embedder = build_embedder_from_manifest(mf, device=cfg.device, batch_size=cfg.batch_size)
        timings["warmup.semantic_build_embedder"] = round((time.perf_counter() - embedder_started) * 1000.0, 2)

        embed_started = time.perf_counter()
        qvec = embedder.embed_texts(["temporary protected status"])
        if qvec.shape != (1, dim):
            raise RuntimeError(f"Warmup embedding shape mismatch: got {qvec.shape}, expected (1, {dim})")
        timings["warmup.semantic_embed_query"] = round((time.perf_counter() - embed_started) * 1000.0, 2)
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        print(f"Search warmup failed after {elapsed_ms} ms: {e}")
        return

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    timings["warmup.total"] = elapsed_ms
    print(f"Search warmup completed in {elapsed_ms} ms")
    print(f"Search warmup timings: {json.dumps(timings, sort_keys=True)}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="IPTP Search HTTP API (RRF hybrid search).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--config", default=str(repo_root / "config.json"), help="Path to config.json")
    parser.add_argument(
        "--run_dir",
        default=str(repo_root / "data/embeddings/bge_mean_norm"),
        help="Path to embeddings run dir (contains manifest + faiss artifacts)",
    )
    parser.add_argument(
        "--chunks",
        default=str(repo_root / "data/sample_100/chunks/chunks.jsonl"),
        help="Path to chunks.jsonl",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Embedding device for FAISS query embedding: "cpu", "mps", or "cuda" (default: cpu)',
    )
    parser.add_argument("--allow-origin", default="*", help='CORS origin, e.g. "*" or "http://localhost:4173"')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = APIConfig(
        config_path=Path(args.config).resolve(),
        run_dir=Path(args.run_dir).resolve(),
        chunks_path=Path(args.chunks).resolve(),
        device=str(args.device).strip() if args.device is not None else "cpu",
        allow_origin=str(args.allow_origin),
    )

    handler = SearchAPIHandler
    handler.api_config = config
    server = ThreadingHTTPServer((str(args.host), int(args.port)), handler)

    print(f"Starting IPTP Search API on http://{args.host}:{args.port}")
    print(f"  config={config.config_path}")
    print(f"  run_dir={config.run_dir}")
    print(f"  chunks={config.chunks_path}")
    print("  endpoints: GET /health, GET/POST /search")
    warm_search_backend(config)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    main()
