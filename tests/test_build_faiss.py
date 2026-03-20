# tests/test_build_faiss.py
#
# End-to-end + edge-case tests for scripts/build_faiss.py.
#
# Run:
#   pytest -q tests/test_build_faiss.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # skip whole file if faiss unavailable

from scripts.semantic.build_faiss import build_faiss


# -----------------------------
# helpers
# -----------------------------

def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, objs: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _any_tmp_files(root: Path) -> bool:
    for p in root.rglob("*"):
        if p.is_file() and p.name.endswith(".tmp"):
            return True
    return False


def _stable_vectors(n: int, dim: int, *, seed: int = 0, normalize: bool = True, include_nonfinite: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    if include_nonfinite and n > 0 and dim > 0:
        X[0, 0] = np.nan
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X = (X / norms).astype(np.float32)
    return X


def _run_paths(run_dir: Path) -> Tuple[Path, Path, Path, Path]:
    faiss_dir = run_dir / "faiss"
    return (
        faiss_dir / "index.faiss",
        faiss_dir / "row_to_chunk_id.jsonl",
        faiss_dir / "build_faiss_meta.json",
        faiss_dir / "build_faiss_errors.jsonl",
    )


def _write_shard(
    run_dir: Path,
    idx: int,
    X: Optional[np.ndarray],
    chunk_ids: Optional[Sequence[str]],
    *,
    path_style: str = "relative",  # "relative" | "absolute" | "run_prefixed"
    corrupt_vec_file: bool = False,
    corrupt_ids_file: bool = False,
    omit_vec: bool = False,
    omit_ids: bool = False,
) -> Tuple[str, str, int]:
    """
    Writes shard files under run_dir/shards and run_dir/ids.
    Returns (vec_path_str, ids_path_str, count_for_manifest).
    """
    shards_dir = run_dir / "shards"
    ids_dir = run_dir / "ids"
    shards_dir.mkdir(parents=True, exist_ok=True)
    ids_dir.mkdir(parents=True, exist_ok=True)

    vec_path = shards_dir / f"embeddings_{idx:06d}.npy"
    ids_path = ids_dir / f"chunk_ids_{idx:06d}.jsonl"

    # write vecs
    if not omit_vec:
        if corrupt_vec_file:
            vec_path.write_text("not a numpy file", encoding="utf-8")
        else:
            assert X is not None
            with vec_path.open("wb") as f:
                np.save(f, X.astype(np.float32))
    # write ids
    if not omit_ids:
        if corrupt_ids_file:
            ids_path.write_text('{"chunk_id": "ok"}\n{ this is bad json }\n', encoding="utf-8")
        else:
            assert chunk_ids is not None
            _write_jsonl(ids_path, [{"chunk_id": cid} for cid in chunk_ids])

    # compute manifest count
    count = int(X.shape[0]) if X is not None else (len(chunk_ids) if chunk_ids is not None else 0)

    def _style(p: Path) -> str:
        if path_style == "absolute":
            return str(p.resolve())
        if path_style == "run_prefixed":
            # store a run-dir-prefixed path string (project-ish style)
            return str(run_dir / p.relative_to(run_dir))
        # relative
        return str(p.relative_to(run_dir))

    return _style(vec_path), _style(ids_path), count


def _write_manifest(
    run_dir: Path,
    *,
    dim: int,
    normalize: bool,
    shards: List[Dict[str, Any]],
    total_embedded: int,
) -> Path:
    mf = {
        "created_ts": "TEST",
        "run_dir": str(run_dir),
        "chunks_source": "TEST",
        "config": {
            "model_name": "TEST",
            "max_length": 128,
            "pooling": "mean",
            "normalize": normalize,
            "empty_policy": "replace",
            "empty_replacement": ".",
            "seed": None,
            "trust_remote_code": False,
        },
        "dim": dim,
        "shards": shards,
        "total_embedded": total_embedded,
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, mf)
    return manifest_path


def _call_build(
    *,
    run_dir: Path,
    index_type: str = "hnsw",
    metric: Optional[str] = None,
    overwrite: bool = True,
    on_shard_error: str = "skip",
) -> None:
    build_faiss(
        run_dir=run_dir,
        manifest_path=None,
        index_type=index_type,
        metric=metric,
        hnsw_m=32,
        ef_construction=200,
        ef_search=64,
        overwrite=overwrite,
        on_shard_error=on_shard_error,
    )


def _load_index_ntotal(index_path: Path) -> int:
    idx = faiss.read_index(str(index_path))
    return int(idx.ntotal)


# -----------------------------
# fixtures
# -----------------------------

@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "data" / "embeddings" / "bge_mean_norm"
    rd.mkdir(parents=True, exist_ok=True)
    return rd


# ============================================================
# 1) Success path (HNSW)
# ============================================================

def test_build_faiss_hnsw_success_writes_artifacts_and_updates_manifest(run_dir: Path):
    dim = 8
    X0 = _stable_vectors(3, dim, seed=1, normalize=True)
    X1 = _stable_vectors(3, dim, seed=2, normalize=True)

    vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b", "c"], path_style="relative")
    vec1, ids1, c1 = _write_shard(run_dir, 1, X1, ["d", "e", "f"], path_style="relative")

    shards = [
        {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0},
        {"idx": 1, "vec_path": vec1, "ids_path": ids1, "count": c1},
    ]
    _write_manifest(run_dir, dim=dim, normalize=True, shards=shards, total_embedded=6)

    _call_build(run_dir=run_dir, index_type="hnsw", metric=None, overwrite=True)

    index_path, mapping_path, meta_path, errors_path = _run_paths(run_dir)
    assert index_path.exists()
    assert mapping_path.exists()
    assert meta_path.exists()
    assert errors_path.exists()
    assert not _any_tmp_files(run_dir / "faiss")

    # mapping rows
    rows = _read_jsonl(mapping_path)
    assert len(rows) == 6
    assert [r["row"] for r in rows] == list(range(6))
    assert [r["chunk_id"] for r in rows] == ["a", "b", "c", "d", "e", "f"]

    # index ntotal
    assert _load_index_ntotal(index_path) == 6

    meta = _read_json(meta_path)
    assert meta["index"]["index_type"] == "hnsw_flat"
    assert meta["index"]["metric"] == "ip"
    assert meta["index"]["ntotal"] == 6
    assert meta["counts"]["num_shards_skipped"] == 0

    mf2 = _read_json(run_dir / "manifest.json")
    assert "faiss" in mf2
    assert mf2["faiss"]["ntotal_indexed"] == 6
    assert mf2["faiss"]["index_type"] == "hnsw_flat"


# ============================================================
# 2) Flat + default metric L2 when normalize=False
# ============================================================

def test_build_faiss_flat_defaults_to_l2_when_not_normalized(run_dir: Path):
    dim = 8
    X0 = _stable_vectors(4, dim, seed=3, normalize=False)  # not normalized
    vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b", "c", "d"], path_style="relative")

    shards = [{"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}]
    _write_manifest(run_dir, dim=dim, normalize=False, shards=shards, total_embedded=4)

    _call_build(run_dir=run_dir, index_type="flat", metric=None, overwrite=True)

    index_path, mapping_path, meta_path, _ = _run_paths(run_dir)
    assert _load_index_ntotal(index_path) == 4
    meta = _read_json(meta_path)
    assert meta["index"]["index_type"] == "flat"
    assert meta["index"]["metric"] == "l2"
    assert len(_read_jsonl(mapping_path)) == 4


# ============================================================
# 3) Overwrite behavior
# ============================================================

def test_refuses_to_overwrite_without_flag(run_dir: Path):
    dim = 8
    X0 = _stable_vectors(2, dim, seed=4, normalize=True)
    vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative")
    _write_manifest(run_dir, dim=dim, normalize=True, shards=[{"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}], total_embedded=2)

    _call_build(run_dir=run_dir, overwrite=True)

    with pytest.raises(RuntimeError, match="already exist|Overwrite|overwrite"):
        _call_build(run_dir=run_dir, overwrite=False)


def test_overwrite_rebuilds_cleanly(run_dir: Path):
    dim = 8
    X0 = _stable_vectors(2, dim, seed=5, normalize=True)
    vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative")
    _write_manifest(run_dir, dim=dim, normalize=True, shards=[{"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}], total_embedded=2)

    _call_build(run_dir=run_dir, overwrite=True)
    _, _, meta_path1, _ = _run_paths(run_dir)
    meta1 = _read_json(meta_path1)
    ts1 = meta1["created_ts"]

    # rebuild (same data is fine; we just ensure it succeeds and rewrites meta)
    _call_build(run_dir=run_dir, overwrite=True)
    meta2 = _read_json(meta_path1)
    ts2 = meta2["created_ts"]
    assert ts2 >= ts1  # timestamps should be monotonic-ish in same run


# ============================================================
# 4) Path resolution robustness
# ============================================================

@pytest.mark.parametrize("path_style", ["relative", "absolute", "run_prefixed"])
def test_manifest_paths_supported_styles(run_dir: Path, path_style: str):
    dim = 8
    X0 = _stable_vectors(3, dim, seed=10, normalize=True)
    vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b", "c"], path_style=path_style)

    shards = [{"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}]
    _write_manifest(run_dir, dim=dim, normalize=True, shards=shards, total_embedded=3)

    _call_build(run_dir=run_dir, overwrite=True)
    index_path, mapping_path, meta_path, _ = _run_paths(run_dir)
    assert index_path.exists() and mapping_path.exists() and meta_path.exists()
    assert _load_index_ntotal(index_path) == 3


# ============================================================
# 5) Skip-on-error shard cases (parameterized)
# ============================================================

def _make_two_shard_manifest(run_dir: Path, dim: int, normalize: bool, shard0: Dict[str, Any], shard1: Dict[str, Any], total_embedded: int) -> None:
    _write_manifest(run_dir, dim=dim, normalize=normalize, shards=[shard0, shard1], total_embedded=total_embedded)


def _errors_types(errors_path: Path) -> List[str]:
    recs = _read_jsonl(errors_path)
    return [r.get("error_type") for r in recs if r.get("type") in ("SHARD_SKIPPED", "SHARD_ADD_FAIL")]


@pytest.mark.parametrize(
    "failure_mode,expected_error_type",
    [
        ("missing_vec", "MISSING_VEC_FILE"),
        ("missing_ids", "MISSING_IDS_FILE"),
        ("dim_mismatch", "DIM_MISMATCH"),
        ("ids_count_mismatch", "COUNT_MISMATCH"),
        ("manifest_count_mismatch", "COUNT_MISMATCH"),
        ("nonfinite", "NONFINITE_VALUES"),
        ("duplicate_ids", "DUPLICATE_CHUNK_ID"),
        ("npy_corrupt", "NPY_LOAD_FAIL"),
        ("ids_corrupt", "IDS_PARSE_FAIL"),
    ],
)
def test_shard_errors_are_logged_and_skipped_but_run_completes(run_dir: Path, failure_mode: str, expected_error_type: str):
    dim = 8

    # shard 1: always valid, ensures run can complete
    X1 = _stable_vectors(3, dim, seed=21, normalize=True)
    vec1, ids1, c1 = _write_shard(run_dir, 1, X1, ["v1", "v2", "v3"], path_style="relative")
    shard1 = {"idx": 1, "vec_path": vec1, "ids_path": ids1, "count": c1}

    # shard 0: varies by failure mode
    if failure_mode == "missing_vec":
        X0 = _stable_vectors(2, dim, seed=20, normalize=True)
        vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative", omit_vec=True)
        shard0 = {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}

    elif failure_mode == "missing_ids":
        X0 = _stable_vectors(2, dim, seed=20, normalize=True)
        vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative", omit_ids=True)
        shard0 = {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}

    elif failure_mode == "dim_mismatch":
        X0 = _stable_vectors(2, dim + 1, seed=20, normalize=True)
        vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative")
        shard0 = {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}

    elif failure_mode == "ids_count_mismatch":
        X0 = _stable_vectors(3, dim, seed=20, normalize=True)
        vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative")  # only 2 ids
        shard0 = {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": 3}

    elif failure_mode == "manifest_count_mismatch":
        X0 = _stable_vectors(2, dim, seed=20, normalize=True)
        vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative")
        shard0 = {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": 999}  # mismatch

    elif failure_mode == "nonfinite":
        X0 = _stable_vectors(2, dim, seed=20, normalize=True, include_nonfinite=True)
        vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative")
        shard0 = {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}

    elif failure_mode == "duplicate_ids":
        X0 = _stable_vectors(2, dim, seed=20, normalize=True)
        vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["v1", "x"], path_style="relative")  # v1 duplicates shard1
        shard0 = {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}

    elif failure_mode == "npy_corrupt":
        X0 = _stable_vectors(2, dim, seed=20, normalize=True)
        vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative", corrupt_vec_file=True)
        shard0 = {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}

    elif failure_mode == "ids_corrupt":
        X0 = _stable_vectors(2, dim, seed=20, normalize=True)
        vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative", corrupt_ids_file=True)
        shard0 = {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}

    else:
        raise AssertionError("Unknown failure_mode")

    # total_embedded reflects intended total; coverage will be <100% if shard0 skipped
    _make_two_shard_manifest(run_dir, dim, True, shard0, shard1, total_embedded=int(shard0["count"]) + int(shard1["count"]))

    _call_build(run_dir=run_dir, overwrite=True, on_shard_error="skip")

    index_path, mapping_path, meta_path, errors_path = _run_paths(run_dir)

    assert index_path.exists() and mapping_path.exists() and meta_path.exists()
    assert not _any_tmp_files(run_dir / "faiss")

    rows = _read_jsonl(mapping_path)

    if failure_mode == "duplicate_ids":
        # duplicate policy: first-seen chunk_id wins (shard0 is processed first), so shard1 is skipped
        assert _load_index_ntotal(index_path) == 2
        assert len(rows) == 2
        assert [r["chunk_id"] for r in rows] == ["v1", "x"]
    else:
        # shard0 intentionally broken -> skipped, shard1 indexed
        assert _load_index_ntotal(index_path) == 3
        assert len(rows) == 3
        assert [r["chunk_id"] for r in rows] == ["v1", "v2", "v3"]


    # errors log should have expected error type somewhere
    etypes = _errors_types(errors_path)
    assert expected_error_type in etypes, f"Expected {expected_error_type} in {etypes}"


# ============================================================
# 6) on_shard_error=fail aborts and doesn't produce final artifacts
# ============================================================

def test_on_shard_error_fail_aborts_and_no_final_outputs(run_dir: Path):
    dim = 8
    # shard0 missing vec -> should trigger abort immediately
    X0 = _stable_vectors(2, dim, seed=30, normalize=True)
    vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative", omit_vec=True)
    shard0 = {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}

    _write_manifest(run_dir, dim=dim, normalize=True, shards=[shard0], total_embedded=2)

    with pytest.raises(RuntimeError, match="Build aborted|on_shard_error=fail|aborted"):
        _call_build(run_dir=run_dir, overwrite=True, on_shard_error="fail")

    index_path, mapping_path, meta_path, errors_path = _run_paths(run_dir)
    # final outputs should not exist
    assert not index_path.exists()
    assert not mapping_path.exists()
    assert not meta_path.exists()
    assert errors_path.exists()
    # no leftover tmp mapping file
    assert not _any_tmp_files(run_dir / "faiss")


# ============================================================
# 7) All shards skipped => fatal "no vectors indexed" and no outputs
# ============================================================

def test_all_shards_skipped_results_in_fatal_and_no_outputs(run_dir: Path):
    dim = 8
    X0 = _stable_vectors(2, dim, seed=40, normalize=True)
    vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative", omit_vec=True)
    shard0 = {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}
    _write_manifest(run_dir, dim=dim, normalize=True, shards=[shard0], total_embedded=2)

    with pytest.raises(RuntimeError, match="No vectors indexed"):
        _call_build(run_dir=run_dir, overwrite=True, on_shard_error="skip")

    index_path, mapping_path, meta_path, errors_path = _run_paths(run_dir)
    assert not index_path.exists()
    assert not mapping_path.exists()
    assert not meta_path.exists()
    assert errors_path.exists()

    # manifest should not have faiss section
    mf2 = _read_json(run_dir / "manifest.json")
    assert "faiss" not in mf2


# ============================================================
# 8) Mapping/index alignment + monotonic rows across multiple shard sizes
# ============================================================

def test_mapping_rows_match_index_ntotal_and_are_monotonic(run_dir: Path):
    dim = 8
    # 3 shards with different sizes
    X0 = _stable_vectors(2, dim, seed=50, normalize=True)
    X1 = _stable_vectors(5, dim, seed=51, normalize=True)
    X2 = _stable_vectors(3, dim, seed=52, normalize=True)

    vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a0", "a1"], path_style="relative")
    vec1, ids1, c1 = _write_shard(run_dir, 1, X1, ["b0", "b1", "b2", "b3", "b4"], path_style="relative")
    vec2, ids2, c2 = _write_shard(run_dir, 2, X2, ["c0", "c1", "c2"], path_style="relative")

    shards = [
        {"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0},
        {"idx": 1, "vec_path": vec1, "ids_path": ids1, "count": c1},
        {"idx": 2, "vec_path": vec2, "ids_path": ids2, "count": c2},
    ]
    _write_manifest(run_dir, dim=dim, normalize=True, shards=shards, total_embedded=10)

    _call_build(run_dir=run_dir, overwrite=True)

    index_path, mapping_path, meta_path, _ = _run_paths(run_dir)
    assert _load_index_ntotal(index_path) == 10

    rows = _read_jsonl(mapping_path)
    assert len(rows) == 10
    assert [r["row"] for r in rows] == list(range(10))
    assert [r["chunk_id"] for r in rows] == ["a0", "a1", "b0", "b1", "b2", "b3", "b4", "c0", "c1", "c2"]

    meta = _read_json(meta_path)
    assert meta["counts"]["total_vectors_indexed"] == 10


# ============================================================
# 9) Minimal logging: only RUN_START and RUN_END for clean run
# ============================================================

def test_errors_log_is_minimal_for_successful_run(run_dir: Path):
    dim = 8
    X0 = _stable_vectors(2, dim, seed=60, normalize=True)
    vec0, ids0, c0 = _write_shard(run_dir, 0, X0, ["a", "b"], path_style="relative")
    _write_manifest(run_dir, dim=dim, normalize=True, shards=[{"idx": 0, "vec_path": vec0, "ids_path": ids0, "count": c0}], total_embedded=2)

    _call_build(run_dir=run_dir, overwrite=True)

    _, _, _, errors_path = _run_paths(run_dir)
    recs = _read_jsonl(errors_path)
    assert len(recs) == 2
    assert recs[0]["type"] == "RUN_START"
    assert recs[1]["type"] == "RUN_END"
    assert recs[1]["status"] == "OK"
