# scripts/build_faiss.py
# Build a FAISS index from sharded embedding outputs (from embed_chunks.py).
# Design goals:
# - Stage-separated: reads embedding run_dir/manifest.json, writes FAISS artifacts under run_dir/faiss/
# - Fault-tolerant by default: logs shard errors and continues (skip corrupt/missing shards)
# - Atomic outputs: index, mapping, meta, and manifest updates written atomically
# - Debuggable: JSONL mapping (viewable), plus error-only JSONL log and a single meta JSON snapshot

"""
How to run 

python3 -m scripts.semantic.build_faiss \
  --run_dir data/embeddings/bge_mean_norm \
  --overwrite

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import shutil

import numpy as np


# -----------------------------
# Utilities
# -----------------------------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def atomic_write_bytes(final_path: Path, data: bytes) -> None:
    ensure_dir(final_path.parent)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(final_path)


def atomic_write_json(final_path: Path, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    atomic_write_bytes(final_path, data)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    rec = {"ts": now_iso(), **obj}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def rel_to_run_dir(run_dir: Path, p: Path) -> str:
    try:
        return str(p.relative_to(run_dir))
    except Exception:
        return str(p)


def truncate(s: str, max_len: int = 800) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# -----------------------------
# Error typing
# -----------------------------

@dataclass(frozen=True)
class ShardError:
    error_type: str
    error_class: str
    error_msg: str
    shard_idx: Optional[int] = None
    vec_path: Optional[str] = None
    ids_path: Optional[str] = None
    expected_dim: Optional[int] = None
    observed_dim: Optional[int] = None
    expected_count: Optional[int] = None
    observed_count: Optional[int] = None


# -----------------------------
# Manifest + shard loading
# -----------------------------

def validate_manifest(mf: Dict[str, Any], manifest_path: Path) -> Tuple[int, bool, List[Dict[str, Any]]]:
    # Returns (dim, normalize_flag, shards)
    if not isinstance(mf, dict):
        raise RuntimeError(f"Manifest is not a dict: {manifest_path}")

    dim = mf.get("dim")
    if not isinstance(dim, int) or dim <= 0:
        raise RuntimeError(f"Manifest missing valid 'dim': {manifest_path}")

    cfg = mf.get("config")
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Manifest missing 'config' dict: {manifest_path}")

    normalize = cfg.get("normalize")
    if not isinstance(normalize, bool):
        # In your embed_chunks fingerprint, normalize is present; still handle gracefully
        raise RuntimeError(f"Manifest missing boolean config.normalize: {manifest_path}")

    shards = mf.get("shards")
    if not isinstance(shards, list) or len(shards) == 0:
        raise RuntimeError(f"Manifest missing non-empty 'shards' list: {manifest_path}")

    # Basic shard entry sanity
    for s in shards:
        if not isinstance(s, dict):
            raise RuntimeError("Manifest shards entries must be dicts")
        if "idx" not in s or "vec_path" not in s or "ids_path" not in s or "count" not in s:
            raise RuntimeError("Manifest shards entries must include idx/vec_path/ids_path/count")

    # Ensure sorted by idx for deterministic mapping
    shards_sorted = sorted(shards, key=lambda d: int(d.get("idx", 0)))
    return dim, bool(normalize), shards_sorted


def load_ids_jsonl(ids_path: Path) -> List[str]:
    ids: List[str] = []
    with ids_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                # preserve JSONDecodeError so caller can classify as IDS_PARSE_FAIL
                raise json.JSONDecodeError(
                    f"Bad JSON in ids file at line {line_num}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e
            cid = obj.get("chunk_id")
            if not isinstance(cid, str) or not cid:
                raise ValueError(f"Missing/invalid chunk_id in ids file at line {line_num}")
            ids.append(cid)
    return ids


def load_embeddings_npy(vec_path: Path) -> np.ndarray:
    # allow_pickle=False for safety; embeddings should be numeric ndarray
    try:
        return np.load(vec_path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as e:
        # Wrap so caller can classify reliably as NPY_LOAD_FAIL
        raise RuntimeError(f"NPY_LOAD_FAIL: {e}") from e


def shard_file_paths(run_dir: Path, vec_path_str: str, ids_path_str: str) -> Tuple[Path, Path]:
    def _resolve(p_str: str) -> Path:
        p = Path(p_str)

        # 1) if already absolute, use it
        if p.is_absolute():
            return p

        # 2) try as-is (relative to current working directory)
        if p.exists():
            return p

        # 3) try relative to run_dir
        p2 = run_dir / p
        if p2.exists():
            return p2

        # 4) if string already contains the run_dir path, strip the prefix and retry
        run_s = str(run_dir).rstrip("/") + "/"
        s = str(p).replace("\\", "/")
        if run_s in s:
            # keep suffix after the first occurrence of run_dir
            suffix = s.split(run_s, 1)[1]
            p3 = run_dir / Path(suffix)
            return p3

        # fallback (keeps error messages informative)
        return p2

    return _resolve(vec_path_str), _resolve(ids_path_str)

# -----------------------------
# FAISS construction
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


def make_index(faiss: Any, index_type: str, metric: str, dim: int, params: Dict[str, Any]) -> Any:
    metric = metric.lower()
    index_type = index_type.lower()

    if index_type == "flat":
        if metric == "ip":
            return faiss.IndexFlatIP(dim)
        if metric == "l2":
            return faiss.IndexFlatL2(dim)
        raise ValueError(f"Unsupported metric for flat: {metric}")

    if index_type == "hnsw":
        # FAISS HNSWFlat supports either IP or L2 metric via metric argument in constructor
        # Some builds use faiss.METRIC_INNER_PRODUCT / METRIC_L2
        M = int(params.get("m", 32))
        if metric == "ip":
            idx = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        elif metric == "l2":
            idx = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
        else:
            raise ValueError(f"Unsupported metric for hnsw: {metric}")

        # Construction / search knobs
        efc = int(params.get("ef_construction", 200))
        efs = int(params.get("ef_search", 64))
        idx.hnsw.efConstruction = efc
        idx.hnsw.efSearch = efs  # stored as default; can also be overridden at query time
        return idx

    raise ValueError(f"Unsupported index_type: {index_type}")


# -----------------------------
# Atomic mapping writer (JSONL)
# -----------------------------

class AtomicMappingWriter:
    """
    Writes mapping to a temporary file during build, then atomically renames to final_path on commit().
    Ensures mapping can be written streaming without holding all IDs in memory.
    """
    def __init__(self, final_path: Path):
        self.final_path = final_path
        ensure_dir(final_path.parent)
        self.tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        self._f = self.tmp_path.open("w", encoding="utf-8")
        self.rows_written = 0

    def write_rows(self, start_row: int, chunk_ids: Sequence[str]) -> None:
        # start_row should equal self.rows_written for sanity
        if start_row != self.rows_written:
            raise RuntimeError(
                f"Mapping row misalignment: start_row={start_row}, rows_written={self.rows_written}"
            )
        for i, cid in enumerate(chunk_ids):
            rec = {"row": start_row + i, "chunk_id": cid}
            self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.rows_written += len(chunk_ids)

    def commit(self) -> None:
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        self.tmp_path.replace(self.final_path)

    def abort(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
        try:
            if self.tmp_path.exists():
                self.tmp_path.unlink()
        except Exception:
            pass


# -----------------------------
# Overwrite / cleanup
# -----------------------------

def wipe_dir_contents(dir_path: Path) -> None:
    if dir_path.exists():
        shutil.rmtree(dir_path, ignore_errors=True)

# -----------------------------
# Main build
# -----------------------------

def build_faiss(
    run_dir: Optional[Path],
    manifest_path: Optional[Path],
    index_type: str,
    metric: Optional[str],
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
    overwrite: bool,
    on_shard_error: str,
) -> None:
    if run_dir is None and manifest_path is None:
        raise ValueError("Provide either --run_dir or --manifest")

    if manifest_path is None:
        manifest_path = Path(run_dir) / "manifest.json"  # type: ignore[arg-type]
    manifest_path = Path(manifest_path)

    if run_dir is None:
        run_dir = manifest_path.parent
    run_dir = Path(run_dir)

    mf = safe_read_json(manifest_path)
    if mf is None:
        # fatal
        raise RuntimeError(f"Manifest missing or unreadable: {manifest_path}")

    dim, normalize_flag, shards = validate_manifest(mf, manifest_path)

    # Derive default metric from normalization if not provided
    if metric is None:
        metric = "ip" if normalize_flag else "l2"
    metric = metric.lower()

    # Paths
    faiss_dir = run_dir / "faiss"
    ensure_dir(faiss_dir)

    index_path = faiss_dir / "index.faiss"
    mapping_path = faiss_dir / "row_to_chunk_id.jsonl"
    meta_path = faiss_dir / "build_faiss_meta.json"
    errors_path = faiss_dir / "build_faiss_errors.jsonl"

    # Overwrite handling
    if (index_path.exists() or mapping_path.exists() or meta_path.exists()) and not overwrite:
        raise RuntimeError(
            "FAISS artifacts already exist. Use --overwrite to rebuild.\n"
            f"  {index_path}\n  {mapping_path}\n  {meta_path}"
        )
    if overwrite:
        wipe_dir_contents(faiss_dir)
        ensure_dir(faiss_dir)

    # One-line RUN_START
    append_jsonl(
        errors_path,
        {
            "type": "RUN_START",
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
            "index_type": index_type,
            "metric": metric,
            "dim": dim,
            "on_shard_error": on_shard_error,
            "params": {
                "hnsw_m": hnsw_m,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
            } if index_type.lower() == "hnsw" else {},
        },
    )

    # Import faiss (fatal if missing)
    try:
        faiss = import_faiss_or_raise()
    except Exception as e:
        append_jsonl(
            errors_path,
            {
                "type": "FATAL_FAISS_IMPORT",
                "error_type": "ERROR_FAISS_IMPORT",
                "error_class": e.__class__.__name__,
                "error_msg": truncate(str(e)),
                "run_dir": str(run_dir),
            },
        )
        raise

    # Build index
    idx_params: Dict[str, Any] = {}
    if index_type.lower() == "hnsw":
        idx_params = {"m": int(hnsw_m), "ef_construction": int(ef_construction), "ef_search": int(ef_search)}

    try:
        index = make_index(faiss, index_type=index_type, metric=metric, dim=dim, params=idx_params)
    except Exception as e:
        append_jsonl(
            errors_path,
            {
                "type": "FATAL_INDEX_INIT",
                "error_type": "ERROR_INDEX_INIT",
                "error_class": e.__class__.__name__,
                "error_msg": truncate(str(e)),
            },
        )
        raise

    # Tracking
    skipped: List[Dict[str, Any]] = []
    indexed_shard_indices: List[int] = []
    skipped_shard_indices: List[int] = []
    ntotal_before = 0
    total_vectors_manifest = int(mf.get("total_embedded", 0)) if isinstance(mf.get("total_embedded", 0), int) else 0
    num_shards_total = len(shards)

    # Duplicate detection (across indexed shards)
    seen_chunk_ids: set[str] = set()

    # Mapping writer (atomic)
    mw = AtomicMappingWriter(mapping_path)
    fatal_during_write = False

    def log_shard_error(err: ShardError) -> None:
        rec: Dict[str, Any] = {
            "type": "SHARD_SKIPPED" if err.error_type != "FAISS_ADD_FAIL" else "SHARD_ADD_FAIL",
            "shard_idx": err.shard_idx,
            "vec_path": err.vec_path,
            "ids_path": err.ids_path,
            "error_type": err.error_type,
            "error_class": err.error_class,
            "error_msg": truncate(err.error_msg),
        }
        if err.expected_dim is not None:
            rec["expected_dim"] = err.expected_dim
        if err.observed_dim is not None:
            rec["observed_dim"] = err.observed_dim
        if err.expected_count is not None:
            rec["expected_count"] = err.expected_count
        if err.observed_count is not None:
            rec["observed_count"] = err.observed_count
        append_jsonl(errors_path, rec)

    # Process shards
    for s in shards:
        shard_idx = int(s.get("idx"))
        vec_path_str = str(s.get("vec_path"))
        ids_path_str = str(s.get("ids_path"))
        expected_count = int(s.get("count", 0)) if isinstance(s.get("count", 0), int) else None

        vec_path, ids_path = shard_file_paths(run_dir, vec_path_str, ids_path_str)

        try:
            if not vec_path.exists():
                raise FileNotFoundError(f"Missing vec file: {vec_path}")
            if not ids_path.exists():
                raise FileNotFoundError(f"Missing ids file: {ids_path}")

            # Load
            X = load_embeddings_npy(vec_path)
            ids = load_ids_jsonl(ids_path)

            # Validate shape/dtype
            if X.ndim != 2:
                raise ValueError(f"Embeddings array is not 2D: shape={getattr(X, 'shape', None)}")

            observed_dim = int(X.shape[1])
            if observed_dim != dim:
                raise ValueError(f"Shard dim mismatch: expected {dim}, got {observed_dim}")

            # dtype cast to float32 (warn only in meta; not error)
            if X.dtype != np.float32:
                X = X.astype(np.float32, copy=False)

            n = int(X.shape[0])
            if len(ids) != n:
                raise ValueError(f"IDs count ({len(ids)}) != embeddings rows ({n})")

            if expected_count is not None and expected_count > 0 and expected_count != n:
                # Not necessarily fatal in some systems, but treat as shard error to keep things predictable
                raise ValueError(f"Manifest count ({expected_count}) != embeddings rows ({n})")

            # Finite check
            if not np.isfinite(X).all():
                raise ValueError("Embeddings contain NaN or Inf values")

            # Duplicate check (skip shard if it introduces any dup)
            dup = next((cid for cid in ids if cid in seen_chunk_ids), None)
            if dup is not None:
                raise ValueError(f"Duplicate chunk_id detected across shards: {dup}")

            # Add to index (only after all validations pass)
            ntotal_before = int(index.ntotal)
            try:
                index.add(X)
            except Exception as e_add:
                raise RuntimeError(f"FAISS add failed: {e_add}") from e_add

            # Mapping rows (only after add succeeded)
            start_row = ntotal_before
            mw.write_rows(start_row=start_row, chunk_ids=ids)

            # Update seen IDs only after successful add+mapping write
            seen_chunk_ids.update(ids)
            indexed_shard_indices.append(shard_idx)

        except Exception as e:
            # Classify error type
            msg = str(e)
            etype = "SHARD_ERROR"

            # Missing files
            if isinstance(e, FileNotFoundError):
                if "vec file" in msg.lower():
                    etype = "MISSING_VEC_FILE"
                elif "ids file" in msg.lower():
                    etype = "MISSING_IDS_FILE"
                else:
                    etype = "MISSING_FILE"

            # Corrupt ids JSONL (now preserved as JSONDecodeError by load_ids_jsonl)
            elif isinstance(e, json.JSONDecodeError):
                etype = "IDS_PARSE_FAIL"

            # Corrupt .npy (deterministic if you applied Change 2)
            elif isinstance(e, RuntimeError) and msg.startswith("NPY_LOAD_FAIL:"):
                etype = "NPY_LOAD_FAIL"

            # Validation issues (shape/dim/count/nonfinite/dups)
            elif isinstance(e, ValueError):
                m = msg.lower()
                if "dim mismatch" in m:
                    etype = "DIM_MISMATCH"
                elif "ids count" in m or "manifest count" in m:
                    etype = "COUNT_MISMATCH"
                elif "nan" in m or "inf" in m or "nonfinite" in m:
                    etype = "NONFINITE_VALUES"
                elif "duplicate chunk_id" in m:
                    etype = "DUPLICATE_CHUNK_ID"
                else:
                    etype = "VALIDATION_FAIL"

            # FAISS add failure
            elif isinstance(e, RuntimeError) and "faiss add failed" in msg.lower():
                etype = "FAISS_ADD_FAIL"

            # else: keep SHARD_ERROR (unknown)

            observed_dim = None
            observed_count = None
            try:
                # Try to add context if partials exist
                if "X" in locals() and isinstance(locals().get("X"), np.ndarray):
                    observed_dim = int(locals()["X"].shape[1]) if locals()["X"].ndim == 2 else None
                    observed_count = int(locals()["X"].shape[0]) if locals()["X"].ndim >= 1 else None
            except Exception:
                pass

            err = ShardError(
                error_type=etype,
                error_class=e.__class__.__name__,
                error_msg=msg,
                shard_idx=shard_idx,
                vec_path=str(vec_path),
                ids_path=str(ids_path),
                expected_dim=dim,
                observed_dim=observed_dim,
                expected_count=expected_count,
                observed_count=observed_count,
            )

            log_shard_error(err)
            skipped.append(
                {
                    "ts": now_iso(),
                    "shard_idx": shard_idx,
                    "vec_path": rel_to_run_dir(run_dir, vec_path),
                    "ids_path": rel_to_run_dir(run_dir, ids_path),
                    "error_type": etype,
                    "error_class": e.__class__.__name__,
                    "error_msg": truncate(msg),
                }
            )
            skipped_shard_indices.append(shard_idx)

            if on_shard_error.lower() == "fail":
                # Abort whole run
                fatal_during_write = True
                break

            # Continue to next shard (skip mode)
            continue

    # If we bailed early in fail mode, we should not commit partial outputs
    if fatal_during_write:
        mw.abort()
        append_jsonl(
            errors_path,
            {
                "type": "RUN_END",
                "run_dir": str(run_dir),
                "status": "FAILED",
                "reason": "on_shard_error=fail triggered",
                "num_shards_total": num_shards_total,
                "num_shards_indexed": len(indexed_shard_indices),
                "num_shards_skipped": len(skipped_shard_indices),
                "total_vectors_manifest": total_vectors_manifest,
                "total_vectors_indexed": int(index.ntotal),
                "coverage_pct": (100.0 * float(index.ntotal) / float(total_vectors_manifest)) if total_vectors_manifest > 0 else None,
                "meta_path": str(meta_path),
            },
        )
        raise RuntimeError("Build aborted due to shard error (on_shard_error=fail).")

    # If nothing was indexed, treat as fatal by default (no manifest update)
    ntotal_indexed = int(index.ntotal)
    if ntotal_indexed == 0:
        mw.abort()
        append_jsonl(
            errors_path,
            {
                "type": "RUN_END",
                "run_dir": str(run_dir),
                "status": "FAILED",
                "reason": "no vectors indexed",
                "num_shards_total": num_shards_total,
                "num_shards_indexed": 0,
                "num_shards_skipped": len(skipped_shard_indices),
                "total_vectors_manifest": total_vectors_manifest,
                "total_vectors_indexed": 0,
                "coverage_pct": 0.0,
                "meta_path": str(meta_path),
            },
        )
        raise RuntimeError("No vectors indexed. Check build_faiss_errors.jsonl for shard errors.")

    # Commit mapping
    try:
        mw.commit()
    except Exception as e:
        mw.abort()
        append_jsonl(
            errors_path,
            {
                "type": "FATAL_MAPPING_WRITE",
                "error_type": "MAPPING_WRITE_FAIL",
                "error_class": e.__class__.__name__,
                "error_msg": truncate(str(e)),
                "mapping_path": str(mapping_path),
            },
        )
        raise

    # Write index atomically
    try:
        # faiss.write_index writes to path; to make atomic we write to tmp then rename
        tmp_index_path = index_path.with_suffix(index_path.suffix + ".tmp")
        ensure_dir(index_path.parent)
        faiss.write_index(index, str(tmp_index_path))
        # fsync directory entry not trivial cross-platform; rely on rename semantics + file write
        tmp_index_path.replace(index_path)
    except Exception as e:
        append_jsonl(
            errors_path,
            {
                "type": "FATAL_INDEX_WRITE",
                "error_type": "FAISS_WRITE_FAIL",
                "error_class": e.__class__.__name__,
                "error_msg": truncate(str(e)),
                "index_path": str(index_path),
            },
        )
        raise

    # Build meta JSON
    coverage_pct = (100.0 * float(ntotal_indexed) / float(total_vectors_manifest)) if total_vectors_manifest > 0 else None
    meta: Dict[str, Any] = {
        "created_ts": now_iso(),
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "embedding_config_fingerprint": mf.get("config"),
        "index": {
            "index_type": ("hnsw_flat" if index_type.lower() == "hnsw" else "flat"),
            "metric": metric,
            "dim": dim,
            "params": idx_params if index_type.lower() == "hnsw" else {},
            "ntotal": ntotal_indexed,
        },
        "counts": {
            "total_vectors_manifest": total_vectors_manifest,
            "total_vectors_indexed": ntotal_indexed,
            "coverage_pct": coverage_pct,
            "num_shards_total": num_shards_total,
            "num_shards_indexed": len(indexed_shard_indices),
            "num_shards_skipped": len(skipped_shard_indices),
        },
        "artifacts": {
            "index_path": rel_to_run_dir(run_dir, index_path),
            "mapping_path": rel_to_run_dir(run_dir, mapping_path),
        },
        "shards": {
            "indexed_shard_indices": indexed_shard_indices,
            "skipped_shard_indices": skipped_shard_indices,
            "skipped_shards": skipped,
        },
    }

    # Write meta atomically
    try:
        atomic_write_json(meta_path, meta)
    except Exception as e:
        append_jsonl(
            errors_path,
            {
                "type": "FATAL_META_WRITE",
                "error_type": "META_WRITE_FAIL",
                "error_class": e.__class__.__name__,
                "error_msg": truncate(str(e)),
                "meta_path": str(meta_path),
            },
        )
        raise

    # Update manifest (atomic)
    try:
        mf2 = dict(mf)
        mf2["faiss"] = {
            "created_ts": meta["created_ts"],
            "index_type": meta["index"]["index_type"],
            "metric": metric,
            "dim": dim,
            "ntotal_indexed": ntotal_indexed,
            "meta_path": rel_to_run_dir(run_dir, meta_path),
            "index_path": rel_to_run_dir(run_dir, index_path),
            "mapping_path": rel_to_run_dir(run_dir, mapping_path),
            "params": idx_params if index_type.lower() == "hnsw" else {},
            "num_shards_total": num_shards_total,
            "num_shards_indexed": len(indexed_shard_indices),
            "num_shards_skipped": len(skipped_shard_indices),
        }
        atomic_write_json(manifest_path, mf2)
    except Exception as e:
        append_jsonl(
            errors_path,
            {
                "type": "FATAL_MANIFEST_UPDATE",
                "error_type": "MANIFEST_WRITE_FAIL",
                "error_class": e.__class__.__name__,
                "error_msg": truncate(str(e)),
                "manifest_path": str(manifest_path),
            },
        )
        raise

    # One-line RUN_END
    append_jsonl(
        errors_path,
        {
            "type": "RUN_END",
            "run_dir": str(run_dir),
            "status": "OK",
            "index_type": meta["index"]["index_type"],
            "metric": metric,
            "dim": dim,
            "total_vectors_manifest": total_vectors_manifest,
            "total_vectors_indexed": ntotal_indexed,
            "num_shards_total": num_shards_total,
            "num_shards_indexed": len(indexed_shard_indices),
            "num_shards_skipped": len(skipped_shard_indices),
            "skipped_shard_indices": skipped_shard_indices,
            "coverage_pct": coverage_pct,
            "meta_path": str(meta_path),
            "index_path": str(index_path),
            "mapping_path": str(mapping_path),
        },
    )

    print("FAISS build complete.")
    print(f"Run dir: {run_dir}")
    print(f"Index: {index_path}")
    print(f"Mapping: {mapping_path}")
    print(f"Meta: {meta_path}")
    print(f"Errors/status log: {errors_path}")
    print(f"Vectors indexed: {ntotal_indexed} / {total_vectors_manifest} ({coverage_pct:.2f}% if known)" if coverage_pct is not None else f"Vectors indexed: {ntotal_indexed}")
    print(f"Shards indexed: {len(indexed_shard_indices)} / {num_shards_total} (skipped {len(skipped_shard_indices)})")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS index from embed_chunks run_dir (fault-tolerant, atomic outputs).")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--run_dir", default="data/embeddings/bge_mean_norm",
               help="Embedding run directory (contains manifest.json)")
    g.add_argument("--manifest", default=None, help="Path to embedding manifest.json")

    p.add_argument("--index_type", choices=["flat", "hnsw"], default="hnsw", help="FAISS index family")
    p.add_argument("--metric", choices=["ip", "l2"], default=None, help="Similarity metric (default derived from manifest.normalize)")
    p.add_argument("--hnsw_m", type=int, default=32, help="HNSW M (graph degree)")
    p.add_argument("--ef_construction", type=int, default=200, help="HNSW efConstruction")
    p.add_argument("--ef_search", type=int, default=64, help="HNSW efSearch default (query-time knob)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing run_dir/faiss artifacts")
    p.add_argument("--on_shard_error", choices=["skip", "fail"], default="skip", help="Skip bad shards or fail whole run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else None
    manifest_path = Path(args.manifest) if args.manifest else None

    build_faiss(
        run_dir=run_dir,
        manifest_path=manifest_path,
        index_type=str(args.index_type),
        metric=str(args.metric) if args.metric is not None else None,
        hnsw_m=int(args.hnsw_m),
        ef_construction=int(args.ef_construction),
        ef_search=int(args.ef_search),
        overwrite=bool(args.overwrite),
        on_shard_error=str(args.on_shard_error),
    )


if __name__ == "__main__":
    main()
