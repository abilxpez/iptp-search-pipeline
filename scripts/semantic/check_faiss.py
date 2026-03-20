# scripts/check_faiss.py
#
# Quick smoke-check for a FAISS index built by scripts/build_faiss.py.
#
# Verifies:
# 1) Artifacts exist + index loads
# 2) index.ntotal == #rows in row_to_chunk_id.jsonl
# 3) Self-retrieval sanity: sampled stored vectors retrieve their own chunk_id in top-k
#
# Run:
#   python3 -m scripts.check_faiss --run_dir data/embeddings/bge_mean_norm
#
# Options:
#   --k 10 --num_queries 5 --shard_idx 0 --seed 0
#
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# basic IO helpers
# -----------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def _count_jsonl_rows(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _load_mapping_chunk_ids(mapping_path: Path) -> List[str]:
    out: List[str] = []
    for rec in _iter_jsonl(mapping_path):
        cid = rec.get("chunk_id")
        if not isinstance(cid, str) or not cid:
            raise RuntimeError(f"Bad mapping record (missing chunk_id): {rec}")
        out.append(cid)
    return out


def _load_ids_jsonl(ids_path: Path) -> List[str]:
    out: List[str] = []
    for rec in _iter_jsonl(ids_path):
        cid = rec.get("chunk_id")
        if not isinstance(cid, str) or not cid:
            raise RuntimeError(f"Bad ids record (missing chunk_id): {rec}")
        out.append(cid)
    return out


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (X / norms).astype(np.float32, copy=False)


# -----------------------------
# path resolution
# -----------------------------

def _resolve_shard_paths(run_dir: Path, vec_path_str: str, ids_path_str: str) -> Tuple[Path, Path]:
    """
    Mirrors build_faiss.py path resolution (simplified):
    - absolute: use as-is
    - relative: try relative to run_dir
    - run-prefixed: if string contains run_dir, strip prefix and retry
    """
    def _resolve(p_str: str) -> Path:
        p = Path(p_str)
        if p.is_absolute():
            return p

        p2 = run_dir / p
        if p2.exists():
            return p2

        run_s = str(run_dir).rstrip("/") + "/"
        s = str(p).replace("\\", "/")
        if run_s in s:
            suffix = s.split(run_s, 1)[1]
            return run_dir / Path(suffix)

        return p2  # best-effort fallback for error messages

    return _resolve(vec_path_str), _resolve(ids_path_str)


# -----------------------------
# faiss helpers
# -----------------------------

def _import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        raise RuntimeError(
            "FAISS import failed. Install one of:\n"
            "  pip install faiss-cpu\n"
            "  conda install -c pytorch faiss-cpu\n"
            f"Import error: {e}"
        ) from e


def _load_index(faiss: Any, index_path: Path):
    try:
        return faiss.read_index(str(index_path))
    except Exception as e:
        raise RuntimeError(f"faiss.read_index failed: {e}") from e


# -----------------------------
# check plumbing
# -----------------------------

@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    manifest_path: Path
    faiss_dir: Path
    index_path: Path
    mapping_path: Path
    meta_path: Path


def _artifacts(run_dir: Path) -> RunArtifacts:
    faiss_dir = run_dir / "faiss"
    return RunArtifacts(
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        faiss_dir=faiss_dir,
        index_path=faiss_dir / "index.faiss",
        mapping_path=faiss_dir / "row_to_chunk_id.jsonl",
        meta_path=faiss_dir / "build_faiss_meta.json",
    )


def _require_artifacts(a: RunArtifacts) -> None:
    missing = [p for p in [a.manifest_path, a.index_path, a.mapping_path, a.meta_path] if not p.exists()]
    if missing:
        msg = "Missing required artifacts:\n" + "\n".join(f"  - {p}" for p in missing)
        raise RuntimeError(msg)


def _load_manifest_info(manifest_path: Path) -> Tuple[Dict[str, Any], int, bool]:
    mf = _read_json(manifest_path)
    dim = mf.get("dim")
    if not isinstance(dim, int) or dim <= 0:
        raise RuntimeError(f"Manifest dim invalid: {dim}")
    cfg = mf.get("config", {})
    normalize_flag = bool(cfg.get("normalize", False))
    return mf, dim, normalize_flag


def _pick_shard(mf: Dict[str, Any], shard_idx: Optional[int]) -> Dict[str, Any]:
    shards = mf.get("shards", [])
    if not isinstance(shards, list) or not shards:
        raise RuntimeError("Manifest shards missing/empty")

    shards_sorted = sorted(shards, key=lambda s: int(s.get("idx", 0)))
    if shard_idx is None:
        return shards_sorted[0]

    for s in shards_sorted:
        if int(s.get("idx")) == int(shard_idx):
            return s

    raise RuntimeError(f"shard_idx {shard_idx} not found in manifest")


def _check_mapping_alignment(index_ntotal: int, mapping_path: Path) -> int:
    rows = _count_jsonl_rows(mapping_path)
    if rows != index_ntotal:
        raise RuntimeError(f"Mapping rows != index.ntotal ({rows} != {index_ntotal})")
    return rows


def _self_retrieval_check(
    *,
    run_dir: Path,
    mf: Dict[str, Any],
    dim: int,
    normalize_flag: bool,
    index: Any,
    mapping_path: Path,
    shard_idx: Optional[int],
    k: int,
    num_queries: int,
    seed: int,
) -> Tuple[int, int]:
    shard = _pick_shard(mf, shard_idx)

    vec_path_str = str(shard.get("vec_path"))
    ids_path_str = str(shard.get("ids_path"))
    vec_path, ids_path = _resolve_shard_paths(run_dir, vec_path_str, ids_path_str)

    if not vec_path.exists() or not ids_path.exists():
        raise RuntimeError(f"Shard files not found:\n  vec: {vec_path}\n  ids: {ids_path}")

    X = np.load(vec_path, allow_pickle=False)
    ids = _load_ids_jsonl(ids_path)

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise RuntimeError(f"Shard embeddings not 2D array: shape={getattr(X, 'shape', None)}")
    if X.shape[1] != dim:
        raise RuntimeError(f"Shard dim mismatch (expected {dim}, got {X.shape[1]})")
    if len(ids) != X.shape[0]:
        raise RuntimeError(f"Shard ids count != rows ({len(ids)} != {X.shape[0]})")

    X = X.astype(np.float32, copy=False)
    if normalize_flag:
        X = _l2_normalize_rows(X)

    n = int(X.shape[0])
    if n <= 0:
        raise RuntimeError("Shard has zero rows")

    nq = max(1, min(int(num_queries), n))
    kk = max(1, int(k))

    rng = np.random.default_rng(int(seed))
    q_idx = rng.choice(n, size=nq, replace=False)
    Q = X[q_idx, :]

    D, I = index.search(Q, kk)  # noqa: F841

    row_to_cid = _load_mapping_chunk_ids(mapping_path)

    hits = 0
    for j, src_row in enumerate(q_idx.tolist()):
        target = ids[src_row]
        retrieved = []
        for r in I[j].tolist():
            if 0 <= r < len(row_to_cid):
                retrieved.append(row_to_cid[r])
            else:
                retrieved.append("<out_of_range>")
        if target in retrieved:
            hits += 1

    return hits, nq


# -----------------------------
# main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Smoke-check a built FAISS index (artifacts + self-retrieval).")
    ap.add_argument("--run_dir", required=True, help="Embedding run directory (contains manifest.json and faiss/)")
    ap.add_argument("--k", type=int, default=5, help="Top-k for self-retrieval check")
    ap.add_argument("--num_queries", type=int, default=3, help="How many vectors to test from a shard")
    ap.add_argument("--shard_idx", type=int, default=None, help="Which shard to sample from (default: first shard)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for sampling test vectors")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)

    a = _artifacts(run_dir)
    try:
        _require_artifacts(a)

        mf, dim, normalize_flag = _load_manifest_info(a.manifest_path)

        faiss = _import_faiss()
        index = _load_index(faiss, a.index_path)
        ntotal = int(index.ntotal)

        print(f"OK: index loads (ntotal={ntotal}, dim={dim})")

        mapping_rows = _check_mapping_alignment(ntotal, a.mapping_path)
        print(f"OK: mapping rows={mapping_rows} matches index.ntotal")

        hits, nq = _self_retrieval_check(
            run_dir=run_dir,
            mf=mf,
            dim=dim,
            normalize_flag=normalize_flag,
            index=index,
            mapping_path=a.mapping_path,
            shard_idx=args.shard_idx,
            k=args.k,
            num_queries=args.num_queries,
            seed=args.seed,
        )
        print(f"OK: self-check hits={hits}/{nq} (top-{args.k})")

        if hits == 0:
            print("FAIL: self-check found 0 hits; index likely not aligned with mapping or vectors")
            return 5

        print("DONE")
        return 0

    except Exception as e:
        print(f"FAIL: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
