# scripts/search_faiss.py
# Search FAISS artifacts produced by scripts/build_faiss.py, with streaming / seek-based lookups:
# - Never loads row_to_chunk_id.jsonl fully: builds row offsets once (BM25-style), then seeks per hit.
# - Never loads chunks.jsonl fully: builds an on-disk SQLite map chunk_id -> byte offset, then seeks per hit.
# - Oversamples FAISS results and applies metadata filters post-retrieval (same filters + output format as search_bm25.py).

"""
How to run
-----------
python3 -m scripts.semantic.search_faiss --q "temporary protected status"

other flags:

python -m scripts.semantic.search_faiss \
  --run_dir data/embeddings/bge_mean_norm \
  --chunks data/sample_100/chunks/chunks.jsonl \
  --q "temporary protected status" \
  --top_k 10 \
  --sort_by_announced_date


// old, same flags still work, but call as a module

- python3 scripts.search_faiss.py --q "ice raids" --top_k 10
- python3 scripts.search_faiss.py \
  --q "ice raids" \
  --top_k 10 \
  --administration Biden \
  --agency ICE

- python3 scripts.search_faiss.py \
  --q "ice raids" \
  --subject "Immigration Enforcement"
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.semantic.bge_embedder import BGEConfig, BGEEmbedder


DEFAULT_RUN_DIR = Path("data/embeddings/bge_mean_norm")
DEFAULT_CHUNKS_PATH = Path("data/sample_100/chunks/chunks.jsonl")


# -----------------------------
# Result type (mirrors BM25 style)
# -----------------------------

@dataclass(frozen=True)
class SearchResult:
    score: float
    doc: Dict[str, Any]
    snippet: str
    chunk_date: str
    title: str
    summary: str
    announced_date: str
    relevant_chunk: Optional[Dict[str, Any]]
    matched_attachment_ids: List[str]
    matched_doc_ids: List[str]
    matched_chunk_ids: List[str]
    matched_count: int
    best_rank: int


# -----------------------------
# Small utilities
# -----------------------------

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_snippet(text: str, max_chars: int = 220) -> str:
    t = " ".join(str(text).split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def format_excerpt(text: Any, max_chars: int = 75) -> str:
    t = " ".join(str(text or "").split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def _date_to_int(date_str: Any) -> int:
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


def build_title_summary_maps(chunks_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    title_by_entry: Dict[str, str] = {}
    summary_by_entry: Dict[str, str] = {}
    summary_par_idx: Dict[str, int] = {}

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            st = obj.get("source_type")
            if st not in {"policy_title", "policy_summary"}:
                continue

            entry_id = obj.get("page_ptr_id")
            if entry_id is None:
                meta = obj.get("meta") or {}
                policy_meta = meta.get("policy") if isinstance(meta, dict) else None
                if isinstance(policy_meta, dict):
                    entry_id = policy_meta.get("page_ptr_id")
            if entry_id is None:
                continue
            entry_key = str(entry_id)

            text = obj.get("text") or ""
            if st == "policy_title":
                if entry_key not in title_by_entry:
                    title_by_entry[entry_key] = str(text)
            elif st == "policy_summary":
                meta = obj.get("meta") or {}
                summary_meta = meta.get("summary") if isinstance(meta, dict) else None
                par_idx = 0
                if isinstance(summary_meta, dict):
                    try:
                        par_idx = int(summary_meta.get("paragraph_index", 0))
                    except Exception:
                        par_idx = 0
                prev_idx = summary_par_idx.get(entry_key)
                if prev_idx is None or par_idx < prev_idx:
                    summary_par_idx[entry_key] = par_idx
                    summary_by_entry[entry_key] = str(text)

    return title_by_entry, summary_by_entry


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    ensure_parent_dir(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def seek_jsonl_line_by_offset_bytes(path: Path, offset: int) -> Dict[str, Any]:
    # Use binary mode so offsets are true byte offsets.
    with path.open("rb") as f:
        f.seek(int(offset))
        line = f.readline()
        if not line:
            raise RuntimeError(f"Failed to read JSONL line at offset {offset} in {path}")
        return json.loads(line.decode("utf-8"))


# -----------------------------
# Filters (copy of BM25 logic)
# -----------------------------

def doc_passes_filters(doc: Dict[str, Any], filters: Dict[str, Optional[str]]) -> bool:
    entry_id = filters.get("entry_id")
    administration = filters.get("administration")
    agency = filters.get("agency")
    subject = filters.get("subject")

    # Entry filter uses the top-level doc field.
    if entry_id and doc.get("entry_id") != entry_id:
        return False

    # Everything else is stored in meta (as built by chunking / db export).
    meta = doc.get("meta") or {}

    if administration and meta.get("administration") != administration:
        return False

    if agency:
        agencies = meta.get("agencies_affected") or []
        if agency not in agencies:
            return False

    if subject:
        subjects = meta.get("subject_matter") or []
        if subject not in subjects:
            return False

    return True


# -----------------------------
# FAISS + manifest loading
# -----------------------------

def import_faiss_or_raise() -> Any:
    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        raise RuntimeError(
            "FAISS is not installed or failed to import.\n"
            "Try one of:\n"
            "  pip install faiss-cpu\n"
            "  conda install -c pytorch faiss-cpu\n"
            f"Import error: {e}"
        ) from e


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json at {manifest_path}")
    mf = read_json(manifest_path)
    if not isinstance(mf, dict):
        raise RuntimeError(f"manifest.json is not a dict: {manifest_path}")
    return mf


def resolve_faiss_artifacts(run_dir: Path, mf: Dict[str, Any]) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Returns (index_path, mapping_path, faiss_meta)
    Prefers manifest['faiss'] pointers; falls back to run_dir/faiss conventional paths.
    """
    faiss_dir = run_dir / "faiss"

    # Fallback defaults
    index_path = faiss_dir / "index.faiss"
    mapping_path = faiss_dir / "row_to_chunk_id.jsonl"
    meta_path = faiss_dir / "build_faiss_meta.json"

    faiss_block = mf.get("faiss")
    if isinstance(faiss_block, dict):
        ip = faiss_block.get("index_path")
        mp = faiss_block.get("mapping_path")
        meta_rel = faiss_block.get("meta_path")

        if isinstance(ip, str):
            index_path = (run_dir / ip) if not Path(ip).is_absolute() else Path(ip)
        if isinstance(mp, str):
            mapping_path = (run_dir / mp) if not Path(mp).is_absolute() else Path(mp)
        if isinstance(meta_rel, str):
            meta_path = (run_dir / meta_rel) if not Path(meta_rel).is_absolute() else Path(meta_rel)

    # Meta is helpful but not strictly required for querying; still load if present.
    faiss_meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            obj = read_json(meta_path)
            if isinstance(obj, dict):
                faiss_meta = obj
        except Exception:
            faiss_meta = {}

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index at {index_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing mapping JSONL at {mapping_path}")

    return index_path, mapping_path, faiss_meta


def get_index_semantics(mf: Dict[str, Any], faiss_meta: Dict[str, Any]) -> Tuple[int, str, bool]:
    """
    Returns (dim, metric, normalize_flag).
    Metric is "ip" or "l2".
    """
    dim = mf.get("dim")
    if not isinstance(dim, int) or dim <= 0:
        raise RuntimeError("manifest.json missing valid 'dim'")

    cfg = mf.get("config")
    if not isinstance(cfg, dict):
        raise RuntimeError("manifest.json missing 'config' dict")

    normalize_flag = cfg.get("normalize")
    if not isinstance(normalize_flag, bool):
        raise RuntimeError("manifest.json missing boolean config.normalize")

    metric = None

    # Prefer meta if available
    try:
        metric = faiss_meta.get("index", {}).get("metric")
    except Exception:
        metric = None

    # Or manifest['faiss']['metric']
    if metric is None:
        fb = mf.get("faiss")
        if isinstance(fb, dict):
            m2 = fb.get("metric")
            if isinstance(m2, str):
                metric = m2

    # Default derived from normalization
    if metric is None:
        metric = "ip" if normalize_flag else "l2"

    metric = str(metric).lower()
    if metric not in ("ip", "l2"):
        raise RuntimeError(f"Unsupported metric in artifacts: {metric}")

    return dim, metric, normalize_flag


def load_faiss_index(index_path: Path) -> Any:
    faiss = import_faiss_or_raise()
    try:
        idx = faiss.read_index(str(index_path))
        return idx
    except Exception as e:
        raise RuntimeError(f"Failed to read FAISS index: {index_path} ({e})") from e


def maybe_set_hnsw_ef_search(index: Any, ef_search: Optional[int]) -> None:
    # HNSW indices have attribute hnsw.efSearch; other indices will ignore.
    if ef_search is None:
        return
    try:
        if hasattr(index, "hnsw") and hasattr(index.hnsw, "efSearch"):
            index.hnsw.efSearch = int(ef_search)
    except Exception:
        # Non-fatal; just ignore.
        return


# -----------------------------
# Mapping: row -> chunk_id (seek-based offsets)
# -----------------------------

def row_offsets_path_for(mapping_path: Path) -> Path:
    # store alongside the mapping file
    return mapping_path.with_name("row_to_chunk_id_offsets.npy")


def build_row_offsets_if_missing(mapping_path: Path, offsets_path: Path) -> Path:
    """
    Build offsets[row] = byte offset of line in row_to_chunk_id.jsonl
    Requires row_to_chunk_id.jsonl to contain contiguous rows 0..N-1.
    """
    if offsets_path.exists():
        return offsets_path

    offsets: List[int] = []

    with mapping_path.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue

            obj = json.loads(s.decode("utf-8"))
            row = obj.get("row")
            if not isinstance(row, int):
                raise RuntimeError("Mapping JSONL line missing integer 'row'")
            cid = obj.get("chunk_id")
            if not isinstance(cid, str) or not cid:
                raise RuntimeError("Mapping JSONL line missing valid 'chunk_id'")

            # enforce contiguous 0..N-1 (BM25-style)
            if row != len(offsets):
                raise RuntimeError(
                    f"row_to_chunk_id.jsonl row not contiguous "
                    f"(expected {len(offsets)}, got {row})"
                )

            offsets.append(int(pos))

    arr = np.asarray(offsets, dtype=np.int64)
    atomic_save_npy(offsets_path, arr)
    return offsets_path


def load_row_offsets(offsets_path: Path, mmap: bool = True) -> np.ndarray:
    if mmap:
        return np.load(offsets_path, mmap_mode="r")
    return np.load(offsets_path)


def get_chunk_id_for_row(mapping_path: Path, offsets: np.ndarray, row: int) -> str:
    if row < 0 or row >= int(offsets.shape[0]):
        raise IndexError(f"row out of range: {row}")

    obj = seek_jsonl_line_by_offset_bytes(mapping_path, int(offsets[row]))

    # Defensive: verify row matches
    if int(obj.get("row", -1)) != int(row):
        raise RuntimeError(f"Row offset mismatch: expected row={row}, got {obj.get('row')}")

    cid = obj.get("chunk_id")
    if not isinstance(cid, str) or not cid:
        raise RuntimeError(f"Invalid chunk_id for row={row}")
    return cid


# -----------------------------
# Chunk lookup: chunk_id -> offset (SQLite) -> seek chunks.jsonl
# -----------------------------

def chunks_sqlite_path_for(chunks_path: Path) -> Path:
    # Keep alongside chunks.jsonl so it’s reusable across runs.
    return chunks_path.with_name("chunks_lookup.sqlite")


def open_sqlite(sqlite_path: Path) -> sqlite3.Connection:
    ensure_parent_dir(sqlite_path)
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def init_chunks_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks_lookup (
            chunk_id TEXT PRIMARY KEY,
            offset   INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            k TEXT PRIMARY KEY,
            v TEXT
        )
        """
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, k: str) -> Optional[str]:
    cur = conn.execute("SELECT v FROM meta WHERE k = ?", (k,))
    row = cur.fetchone()
    return None if row is None else str(row[0])


def set_meta(conn: sqlite3.Connection, k: str, v: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (k, v))


def build_chunks_sqlite_if_missing(chunks_path: Path, sqlite_path: Path) -> Path:
    """
    Build an on-disk map chunk_id -> byte offset into chunks.jsonl.
    Streaming, resumable-ish (INSERT OR IGNORE), and staleness-aware via meta.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks.jsonl at {chunks_path}")

    # If sqlite exists and matches source size+mtime, reuse.
    src_stat = chunks_path.stat()
    src_sig = f"{src_stat.st_size}:{int(src_stat.st_mtime)}"

    if sqlite_path.exists():
        conn = open_sqlite(sqlite_path)
        try:
            init_chunks_sqlite(conn)
            have = get_meta(conn, "chunks_sig")
            if have == src_sig:
                return sqlite_path
        finally:
            conn.close()
        # Otherwise, rebuild to avoid silent mismatch.
        sqlite_path.unlink(missing_ok=True)

    conn = open_sqlite(sqlite_path)
    try:
        init_chunks_sqlite(conn)

        # Speed knobs for bulk load
        conn.execute("PRAGMA cache_size = -200000;")  # ~200MB if available, best-effort
        conn.execute("BEGIN;")

        batch: List[Tuple[str, int]] = []
        batch_size = 10_000

        with chunks_path.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s.decode("utf-8"))
                except json.JSONDecodeError:
                    # Skip malformed chunk lines; they won't be retrievable anyway.
                    continue

                cid = obj.get("chunk_id")
                if not isinstance(cid, str) or not cid:
                    continue

                batch.append((cid, int(pos)))
                if len(batch) >= batch_size:
                    conn.executemany(
                        "INSERT OR IGNORE INTO chunks_lookup (chunk_id, offset) VALUES (?, ?)",
                        batch,
                    )
                    batch.clear()

        if batch:
            conn.executemany(
                "INSERT OR IGNORE INTO chunks_lookup (chunk_id, offset) VALUES (?, ?)",
                batch,
            )
            batch.clear()

        # Stamp meta and commit
        set_meta(conn, "chunks_path", str(chunks_path))
        set_meta(conn, "chunks_sig", src_sig)
        conn.commit()
        return sqlite_path
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_chunk_offset(conn: sqlite3.Connection, chunk_id: str) -> Optional[int]:
    cur = conn.execute("SELECT offset FROM chunks_lookup WHERE chunk_id = ?", (chunk_id,))
    row = cur.fetchone()
    if row is None:
        return None
    try:
        return int(row[0])
    except Exception:
        return None


def load_chunk_by_offset(chunks_path: Path, offset: int) -> Dict[str, Any]:
    return seek_jsonl_line_by_offset_bytes(chunks_path, int(offset))


# -----------------------------
# Query embedding (consistent with manifest config)
# -----------------------------

def build_embedder_from_manifest(
    mf: Dict[str, Any],
    device: Optional[str],
    batch_size: int,
) -> BGEEmbedder:
    cfg_block = mf.get("config")
    if not isinstance(cfg_block, dict):
        raise RuntimeError("manifest.json missing 'config' dict")

    model_name = str(cfg_block.get("model_name", "BAAI/bge-base-en-v1.5"))
    max_length = int(cfg_block.get("max_length", 512))
    pooling = str(cfg_block.get("pooling", "mean"))
    normalize = bool(cfg_block.get("normalize", True))
    empty_policy = str(cfg_block.get("empty_policy", "replace"))
    empty_replacement = str(cfg_block.get("empty_replacement", "."))
    seed = cfg_block.get("seed", None)
    seed_i = int(seed) if isinstance(seed, int) else None

    # trust_remote_code should remain False for safety unless explicitly set True in artifacts
    trust_remote_code = bool(cfg_block.get("trust_remote_code", False))

    cfg = BGEConfig(
        model_name=model_name,
        device=device,
        batch_size=int(batch_size),
        max_length=max_length,
        pooling=pooling,  # type: ignore[arg-type]
        normalize=normalize,
        empty_policy=empty_policy,  # type: ignore[arg-type]
        empty_replacement=empty_replacement,
        seed=seed_i,
        trust_remote_code=trust_remote_code,
    )
    return BGEEmbedder(cfg)


# -----------------------------
# Core search
# -----------------------------

def faiss_search_rows(
    index: Any,
    qvec: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (D, I) as 1D arrays of length k.
    """
    if qvec.ndim != 2 or qvec.shape[0] != 1:
        raise ValueError("qvec must have shape (1, d)")
    D, I = index.search(qvec.astype(np.float32, copy=False), int(k))
    # FAISS returns shapes (1, k)
    return np.asarray(D[0]), np.asarray(I[0])


def aggregate_candidates(
    candidates: List[Dict[str, Any]],
    snippet_chars: int,
    chunks_path: Path,
    sort_by_announced_date: bool = False,
) -> List[SearchResult]:
    title_by_entry, summary_by_entry = build_title_summary_maps(chunks_path)
    entry_hits: Dict[Any, Dict[str, Any]] = {}
    hit_rank = 0
    for candidate in candidates:
        hit_rank += 1
        doc = candidate["doc"]
        entry_id = doc.get("page_ptr_id")
        chunk_id = str(doc.get("chunk_id") or candidate.get("chunk_id") or "")
        if entry_id is None:
            continue

        score_value = float(candidate["score"])
        attachment_id = str(doc.get("attachment_id") or doc.get("policydocument_id") or "")
        doc_id_val = doc.get("policydocument_id") or doc.get("page_ptr_id") or doc.get("doc_id") or ""
        doc_id_str = str(doc_id_val) if doc_id_val is not None else ""
        snippet_text = format_snippet(doc.get("text", ""), max_chars=snippet_chars)
        meta_obj = doc.get("meta") or {}
        attach_meta = meta_obj.get("attachment") if isinstance(meta_obj, dict) else None
        chunk_date = ""
        chunk_date_int = 0
        chunk_doc_type = ""
        if isinstance(attach_meta, dict):
            raw_date = attach_meta.get("date")
            if raw_date:
                chunk_date = str(raw_date)
                chunk_date_int = _date_to_int(chunk_date)
            dt = attach_meta.get("document_type")
            if dt:
                chunk_doc_type = str(dt)
        chunk_source_type = str(doc.get("source_type") or "")

        entry_key = str(entry_id)
        entry_info = entry_hits.get(entry_key)
        match_info = {
            "chunk_id": chunk_id,
            "score": score_value,
            "chunk_date": chunk_date,
            "chunk_date_int": chunk_date_int,
            "hit_rank": hit_rank,
            "doc": doc,
            "doc_id": doc_id_str,
            "snippet": snippet_text,
            "attachment_id": attachment_id,
            "source_type": chunk_source_type,
            "document_type": chunk_doc_type,
            "text": doc.get("text", ""),
        }
        if entry_info is None:
            entry_info = {
                "score": score_value,
                "best_rank": hit_rank,
                "doc": doc,
                "snippet": snippet_text,
                "best_date_int": chunk_date_int,
                "best_date": chunk_date,
                "title": title_by_entry.get(entry_key, ""),
                "summary": summary_by_entry.get(entry_key, ""),
                "matched_attachment_ids": [],
                "matched_doc_ids": [],
                "matched_chunk_ids": [],
                "matched_chunks": [],
                "match_count": 0,
            }
            entry_hits[entry_key] = entry_info
        elif (
            score_value > entry_info["score"]
            or (
                score_value == entry_info["score"]
                and (
                    chunk_date_int > entry_info["best_date_int"]
                    or (
                        chunk_date_int == entry_info["best_date_int"]
                        and hit_rank < entry_info["best_rank"]
                    )
                )
            )
        ):
            entry_info.update(
                {
                    "score": score_value,
                    "best_rank": hit_rank,
                    "doc": doc,
                    "snippet": snippet_text,
                    "best_date_int": chunk_date_int,
                    "best_date": chunk_date,
                }
            )

        entry_info["match_count"] += 1
        entry_info["matched_chunks"].append(match_info)
        if attachment_id and attachment_id not in entry_info["matched_attachment_ids"]:
            entry_info["matched_attachment_ids"].append(attachment_id)
        if doc_id_str and doc_id_str not in entry_info["matched_doc_ids"]:
            entry_info["matched_doc_ids"].append(doc_id_str)
        if chunk_id and chunk_id not in entry_info["matched_chunk_ids"]:
            entry_info["matched_chunk_ids"].append(chunk_id)

    if sort_by_announced_date:
        def _announced_int(info: Dict[str, Any]) -> int:
            doc = info.get("doc") or {}
            meta_flat = doc.get("meta_flat") or {}
            return _date_to_int(meta_flat.get("announced_date"))

        sorted_entries = sorted(
            entry_hits.values(),
            key=lambda info: (-_announced_int(info), -info["score"], info["best_rank"]),
        )
    else:
        sorted_entries = sorted(
            entry_hits.values(),
            key=lambda info: (-info["score"], -int(info.get("best_date_int", 0)), info["best_rank"]),
        )

    results: List[SearchResult] = []
    for info in sorted_entries:
        matched_sorted = sorted(
            info.get("matched_chunks", []),
            key=lambda m: (-float(m["score"]), -int(m.get("chunk_date_int", 0)), int(m["hit_rank"])),
        )
        info["matched_chunk_ids"] = [m["chunk_id"] for m in matched_sorted if m.get("chunk_id")]

        top_match = matched_sorted[0] if matched_sorted else None
        top_source_type = str((top_match or {}).get("source_type") or "")

        # Display override rules:
        # - top=title: prefer highest-scoring PDF chunk; else summary; else title.
        # - top=summary: display summary.
        # - top=pdf_page: display that top PDF chunk.
        display_match = top_match
        if top_source_type == "policy_title":
            best_pdf_for_title = next((m for m in matched_sorted if m.get("source_type") == "pdf_page"), None)
            best_summary_for_title = next((m for m in matched_sorted if m.get("source_type") == "policy_summary"), None)
            if best_pdf_for_title is not None:
                display_match = best_pdf_for_title
            elif best_summary_for_title is not None:
                display_match = best_summary_for_title

        if display_match is not None:
            info["doc"] = display_match.get("doc") or info.get("doc")
            info["snippet"] = display_match.get("snippet") or info.get("snippet")
            info["best_date_int"] = int(display_match.get("chunk_date_int") or info.get("best_date_int") or 0)
            info["best_date"] = str(display_match.get("chunk_date") or info.get("best_date") or "")
        best_pdf = None
        for m in matched_sorted:
            if m.get("source_type") == "pdf_page":
                best_pdf = m
                break
        if best_pdf is not None:
            info["relevant_chunk"] = {
                "text": format_excerpt(best_pdf.get("text"), max_chars=75),
                "source_type": "pdf_page",
                "attachment_date": best_pdf.get("chunk_date") or "",
                "document_type": best_pdf.get("document_type") or "",
            }
        else:
            info["relevant_chunk"] = None

        results.append(
            SearchResult(
                score=info["score"],
                doc=info["doc"],
                snippet=info["snippet"],
                chunk_date=info.get("best_date", ""),
                title=info.get("title", ""),
                summary=info.get("summary", ""),
                announced_date=str((info.get("doc") or {}).get("meta", {}).get("announced_date") or ""),
                relevant_chunk=info.get("relevant_chunk"),
                matched_attachment_ids=info["matched_attachment_ids"],
                matched_doc_ids=info["matched_doc_ids"],
                matched_chunk_ids=info["matched_chunk_ids"],
                matched_count=info["match_count"],
                best_rank=info["best_rank"],
            )
        )

    return results


def search_faiss(
    run_dir: Path,
    chunks_path: Path,
    query: str,
    top_k: int,
    oversample: int,
    filters: Dict[str, Optional[str]],
    device: Optional[str],
    batch_size: int,
    ef_search: Optional[int],
    snippet_chars: int,
    max_candidates: Optional[int],
    sort_by_announced_date: bool = False,
) -> List[SearchResult]:
    mf = load_manifest(run_dir)
    index_path, mapping_path, faiss_meta = resolve_faiss_artifacts(run_dir, mf)
    dim, metric, _normalize_flag = get_index_semantics(mf, faiss_meta)

    # Ensure row offsets exist (streaming build once)
    offsets_path = row_offsets_path_for(mapping_path)
    build_row_offsets_if_missing(mapping_path, offsets_path)
    row_offsets = load_row_offsets(offsets_path, mmap=True)

    # Ensure chunks sqlite exists (streaming build once)
    sqlite_path = chunks_sqlite_path_for(chunks_path)
    build_chunks_sqlite_if_missing(chunks_path, sqlite_path)

    # Load index
    index = load_faiss_index(index_path)
    maybe_set_hnsw_ef_search(index, ef_search)

    # Basic sanity
    if int(getattr(index, "d", dim)) != int(dim):
        raise RuntimeError(f"FAISS index dim mismatch: index.d={getattr(index,'d',None)} manifest.dim={dim}")

    ntotal = int(getattr(index, "ntotal", 0))
    if ntotal <= 0:
        raise RuntimeError("FAISS index is empty")

    # Embed query
    embedder = build_embedder_from_manifest(mf, device=device, batch_size=batch_size)
    qvec = embedder.embed_texts([query])  # (1, d)
    if qvec.shape != (1, dim):
        raise RuntimeError(f"Query embedding shape mismatch: got {qvec.shape}, expected (1, {dim})")

    # Candidate strategy: oversample, and if filters are strict, retry with larger pool once.
    k0 = max(int(top_k), 1) * max(int(oversample), 1)
    k0 = min(k0, ntotal)

    # For strict filters, allow one expansion (bounded).
    max_expand_factor = 5
    tried: List[int] = []
    candidates: List[Dict[str, Any]] = []

    # Open SQLite once
    conn = open_sqlite(sqlite_path)
    try:
        init_chunks_sqlite(conn)

        for attempt in range(2):
            k_candidate = k0 if attempt == 0 else min(ntotal, k0 * max_expand_factor)
            if k_candidate in tried:
                continue
            tried.append(k_candidate)

            D, I = faiss_search_rows(index, qvec, k_candidate)

            # Convert to a consistent "higher is better" score for ranking/printing.
            # - IP: larger is better already (cosine if normalized).
            # - L2: smaller is better, so use negative distance.
            if metric == "l2":
                scores = -D
            else:
                scores = D

            # Iterate candidates in returned order
            reached_candidate_limit = False
            for score, row in zip(scores.tolist(), I.tolist()):
                # FAISS uses -1 for empty results sometimes; skip
                if row is None:
                    continue
                try:
                    row_i = int(row)
                except Exception:
                    continue
                if row_i < 0:
                    continue
                if row_i >= int(row_offsets.shape[0]):
                    # Mapping shorter than index -> can't resolve
                    continue

                try:
                    chunk_id = get_chunk_id_for_row(mapping_path, row_offsets, row_i)
                except Exception:
                    continue

                off = get_chunk_offset(conn, chunk_id)
                if off is None:
                    continue

                try:
                    doc = load_chunk_by_offset(chunks_path, off)
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

            aggregated = aggregate_candidates(
                candidates, snippet_chars, chunks_path, sort_by_announced_date=sort_by_announced_date
            )
            if len(aggregated) >= int(top_k) or reached_candidate_limit:
                break

        aggregated = aggregate_candidates(
            candidates, snippet_chars, chunks_path, sort_by_announced_date=sort_by_announced_date
        )
        return aggregated[: int(top_k)]
    finally:
        conn.close()


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search FAISS artifacts built from chunks.jsonl (streaming + scalable).")
    p.add_argument("--q", required=True, help="Query string")
    p.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR), help="Embedding run directory (contains manifest.json + faiss/)")
    p.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH), help="Path to chunks.jsonl")

    p.add_argument("--top_k", type=int, default=10, help="Number of results to return")
    p.add_argument("--oversample", type=int, default=10, help="FAISS candidates = top_k * oversample (pre-filter)")
    p.add_argument(
        "--snippet_chars",
        type=int,
        default=220,
        help="Max chars in snippet output",
    )
    p.add_argument(
        "--max_candidates",
        type=int,
        default=None,
        help="Max raw FAISS hits to consider before deduping (default=top_k*oversample)",
    )

    # Optional filters (same as BM25)
    p.add_argument("--entry_id", default=None, help="Filter to a single entry_id")
    p.add_argument("--administration", default=None, help="Filter by administration (meta)")
    p.add_argument("--agency", default=None, help="Filter by agencies_affected (meta)")
    p.add_argument("--subject", default=None, help="Filter by subject_matter (meta)")

    # Runtime knobs (do not change embedding semantics)
    p.add_argument("--device", default=None, help='Override device: "mps", "cuda", or "cpu" (default auto)')
    p.add_argument("--batch_size", type=int, default=16, help="Embedding batch size for query embedding")
    p.add_argument("--ef_search", type=int, default=None, help="Override HNSW efSearch at query time (if applicable)")
    p.add_argument(
        "--sort_by_announced_date",
        action="store_true",
        help="Sort final results by announced_date (desc), then score",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    chunks_path = Path(args.chunks)

    filters: Dict[str, Optional[str]] = {
        "entry_id": args.entry_id,
        "administration": args.administration,
        "agency": args.agency,
        "subject": args.subject,
    }

    results = search_faiss(
        run_dir=run_dir,
        chunks_path=chunks_path,
        query=str(args.q),
        top_k=int(args.top_k),
        oversample=int(args.oversample),
        filters=filters,
        device=args.device if args.device else None,
        batch_size=int(args.batch_size),
        ef_search=int(args.ef_search) if args.ef_search is not None else None,
        snippet_chars=int(args.snippet_chars),
        max_candidates=args.max_candidates,
        sort_by_announced_date=bool(args.sort_by_announced_date),
    )

    print(f"Query: {args.q}")
    if any(v for v in filters.values()):
        print(f"Filters: { {k: v for k, v in filters.items() if v} }")
    print(f"Results: {len(results)}\n")

    for rank, r in enumerate(results, start=1):
        doc = r.doc
        meta = doc.get("meta") or {}

        entry_id = doc.get("page_ptr_id")
        title = r.title or meta.get("title") or ""
        meta_flat = doc.get("meta_flat") or {}
        announced_date = r.announced_date or meta_flat.get("announced_date") or ""
        summary = r.summary or ""

        print(f"{rank:02d}. score={r.score:.4f}  policy_id={entry_id}")
        if title:
            print(f"    title={format_excerpt(title, max_chars=75)}")
        if announced_date:
            print(f"    announced_date={announced_date}")
        if summary:
            print(f"    summary={format_excerpt(summary, max_chars=75)}")
        if r.relevant_chunk:
            ch = r.relevant_chunk
            print(f"    chunk:")
            print(f"      text={format_excerpt(ch.get('text'), max_chars=75)}")
            print(f"      attachment_date={ch.get('attachment_date') or ''}")
            print(f"      type={ch.get('document_type') or ''}")
            print(f"      document_id={doc.get('attachment_id') or ''}")
            page_start = doc.get("page_start") or ""
            page_end = doc.get("page_end") or ""
            print(f"      pages={page_start}-{page_end}")
            print(f"      id={doc.get('chunk_id') or ''}")
        else:
            print(f"    chunk=None")
        print("")


if __name__ == "__main__":
    main()
