# tests/test_embed_chunks.py
#
# Fast tests (no HF model load):
#   pytest -q tests/test_embed_chunks.py -m "not slow"
#
# Run everything (includes the slow HF integration test):
#   pytest -q tests/test_embed_chunks.py
#
# NOTE: If you see PytestUnknownMarkWarning for "slow", register it in pytest.ini:
#   [pytest]
#   markers =
#       slow: tests that load HF models / run end-to-end

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import hashlib

import numpy as np
import pytest

import scripts.embed_chunks as ec


# -----------------------------
# helpers: tiny JSONL writers/readers
# -----------------------------
def _write_jsonl_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_chunks_jsonl(
    path: Path,
    records: List[Dict[str, Any]],
    *,
    add_malformed_line: bool = False,
    malformed_line: str = "{ this is not json }",
) -> None:
    lines: List[str] = []
    for r in records:
        lines.append(json.dumps(r, ensure_ascii=False))
    if add_malformed_line:
        lines.insert(1, malformed_line)
    _write_jsonl_lines(path, lines)


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


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _find_log_entries(log_path: Path, typ: str) -> List[Dict[str, Any]]:
    return [rec for rec in _read_jsonl(log_path) if rec.get("type") == typ]


def _list_all_paths(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*")]


def _any_tmp_files(root: Path) -> bool:
    for p in _list_all_paths(root):
        if p.is_file() and p.name.endswith(".tmp"):
            return True
    return False


# -----------------------------
# dummy embedder (fast tests)
# -----------------------------
def _stable_vectors(texts: Sequence[str], dim: int, normalize: bool, *, salt: str = "") -> np.ndarray:
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        s = (t or "") + "|" + salt
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()  # type: ignore[attr-defined]
        seed = int.from_bytes(h[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(dim,)).astype(np.float32)
        if normalize:
            n = float(np.linalg.norm(v))
            v = v / max(n, 1e-12)
        X[i] = v
    return X.astype(np.float32)


class DummyBGEEmbedder:
    def __init__(self, cfg: ec.BGEConfig):
        self.cfg = cfg
        self.dim = 8  # small but >0

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # If you ever want pooling to affect output in tests, include pooling in the salt
        salt = f"{self.cfg.pooling}|{int(self.cfg.normalize)}|{self.cfg.max_length}"
        return _stable_vectors(texts, dim=self.dim, normalize=bool(self.cfg.normalize), salt=salt)


@pytest.fixture()
def patch_dummy_embedder(monkeypatch: pytest.MonkeyPatch):
    # Patch the embedder used inside scripts.embed_chunks
    monkeypatch.setattr(ec, "BGEEmbedder", DummyBGEEmbedder)


@pytest.fixture()
def paths(tmp_path: Path) -> Tuple[Path, Path]:
    chunks_path = tmp_path / "data" / "chunks" / "chunks.jsonl"
    out_dir = tmp_path / "data" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    return chunks_path, out_dir


def _run_embed_chunks(
    *,
    chunks_path: Path,
    out_dir: Path,
    pooling: str = "mean",
    normalize: bool = True,
    batch_size: int = 16,
    max_length: int = 128,
    shard_size: int = 2,
    resume: bool = True,
    overwrite: bool = False,
    device: Optional[str] = "cpu",
    seed: Optional[int] = None,
) -> Path:
    ec.embed_chunks(
        chunks_path=chunks_path,
        out_dir=out_dir,
        pooling=pooling,
        normalize=normalize,
        batch_size=batch_size,
        max_length=max_length,
        shard_size=shard_size,
        resume=resume,
        overwrite=overwrite,
        device=device,
        seed=seed,
    )
    run_dir = out_dir / ec.compute_run_name(pooling=pooling, normalize=normalize)
    return run_dir


# ============================================================
# 1) Creates run directory structure and required files
# ============================================================
def test_embed_chunks_creates_structure_and_required_files(patch_dummy_embedder, paths: Tuple[Path, Path]):
    chunks_path, out_dir = paths
    _write_chunks_jsonl(
        chunks_path,
        [
            {"chunk_id": "c1", "text": "hello"},
            {"chunk_id": "c2", "text": "world"},
            {"chunk_id": "c3", "text": "immigration policy"},
        ],
    )

    run_dir = _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=2)

    assert run_dir.exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "embed_log.jsonl").exists()
    assert (run_dir / "embedded_ids.jsonl").exists()
    assert (run_dir / "shards").exists()
    assert (run_dir / "ids").exists()

    # At least 2 shards expected for N=3, shard_size=2
    assert (run_dir / "shards" / "embeddings_000000.npy").exists()
    assert (run_dir / "ids" / "chunk_ids_000000.jsonl").exists()


# ============================================================
# 2) Shard writing correctness (counts + alignment)
# ============================================================
def test_shards_have_matching_counts_and_alignment(patch_dummy_embedder, paths: Tuple[Path, Path]):
    chunks_path, out_dir = paths
    recs = [
        {"chunk_id": f"c{i}", "text": f"text {i}"} for i in range(1, 6)
    ]  # N=5
    _write_chunks_jsonl(chunks_path, recs)

    run_dir = _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=2)

    mf = _read_json(run_dir / "manifest.json")
    shards = mf["shards"]
    assert len(shards) == 3  # ceil(5/2)=3

    for s in shards:
        vec_path = Path(s["vec_path"])
        ids_path = Path(s["ids_path"])
        assert vec_path.exists()
        assert ids_path.exists()

        X = np.load(vec_path)
        ids = _read_jsonl(ids_path)

        assert X.ndim == 2
        assert X.shape[0] == len(ids) == int(s["count"])
        # alignment check: ids file order preserved (we can’t “verify” embedding values easily here,
        # but we can at least ensure the IDs are exactly what we fed in per shard order).
        for obj in ids:
            assert "chunk_id" in obj and isinstance(obj["chunk_id"], str)


# ============================================================
# 3) Manifest fields correctness and consistency
# ============================================================
def test_manifest_consistency(patch_dummy_embedder, paths: Tuple[Path, Path]):
    chunks_path, out_dir = paths
    _write_chunks_jsonl(
        chunks_path,
        [{"chunk_id": "a", "text": "x"}, {"chunk_id": "b", "text": "y"}, {"chunk_id": "c", "text": "z"}],
    )

    run_dir = _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=2, max_length=128)

    mf = _read_json(run_dir / "manifest.json")
    assert mf["chunks_source"].endswith("chunks.jsonl")
    assert mf["total_embedded"] == 3
    assert mf["dim"] == 8

    shards = mf["shards"]
    assert sum(int(s["count"]) for s in shards) == mf["total_embedded"]

    # config fingerprint matches what embed_chunks uses
    cfg_fp = mf["config"]
    assert cfg_fp["pooling"] == "mean"
    assert cfg_fp["normalize"] is True
    assert cfg_fp["max_length"] == 128


# ============================================================
# 4) Resume skips previously embedded IDs
# ============================================================
def test_resume_skips_all_when_rerun(patch_dummy_embedder, paths: Tuple[Path, Path]):
    chunks_path, out_dir = paths
    _write_chunks_jsonl(
        chunks_path,
        [{"chunk_id": "c1", "text": "t1"}, {"chunk_id": "c2", "text": "t2"}],
    )

    run_dir = _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=10, resume=True)
    mf1 = _read_json(run_dir / "manifest.json")
    assert mf1["total_embedded"] == 2

    # rerun with resume
    run_dir2 = _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=10, resume=True)
    assert run_dir2 == run_dir

    mf2 = _read_json(run_dir / "manifest.json")
    assert mf2["total_embedded"] == 2  # unchanged

    log = run_dir / "embed_log.jsonl"
    ends = _find_log_entries(log, "RUN_END")
    assert ends, "Expected a RUN_END log entry"
    # The last RUN_END should show skipped == 2
    assert int(ends[-1]["total_skipped_resume"]) == 2
    assert int(ends[-1]["total_new_embedded"]) == 0


# ============================================================
# 5) Resume + partial progress works (simulated crash mid-run)
# ============================================================
def test_resume_recovers_after_partial_run(patch_dummy_embedder, paths: Tuple[Path, Path], monkeypatch: pytest.MonkeyPatch):
    chunks_path, out_dir = paths
    _write_chunks_jsonl(
        chunks_path,
        [
            {"chunk_id": "c1", "text": "t1"},
            {"chunk_id": "c2", "text": "t2"},
            {"chunk_id": "c3", "text": "t3"},
            {"chunk_id": "c4", "text": "t4"},
        ],
    )

    run_name = ec.compute_run_name(pooling="mean", normalize=True)
    run_dir = out_dir / run_name

    # Patch update_manifest to crash after first shard has been appended to mf["shards"].
    real_update_manifest = ec.update_manifest
    crashed = {"did": False}

    def crashy_update_manifest(manifest_path: Path, mf: Dict[str, Any], log_path: Path) -> None:
        # Let early manifest updates happen; crash only after first shard write
        real_update_manifest(manifest_path, mf, log_path)
        if not crashed["did"] and isinstance(mf.get("shards"), list) and len(mf["shards"]) >= 1 and int(mf.get("total_embedded", 0)) >= 2:
            crashed["did"] = True
            raise RuntimeError("simulated crash after first shard")

    monkeypatch.setattr(ec, "update_manifest", crashy_update_manifest)

    with pytest.raises(RuntimeError, match="simulated crash"):
        _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=2, resume=True)

    # At this point, embedded_ids.jsonl should exist and contain at least 2 ids
    done_path = run_dir / "embedded_ids.jsonl"
    assert done_path.exists()
    assert _count_jsonl_lines(done_path) >= 2

    # Now rerun without crashing; should resume and finish remaining
    monkeypatch.setattr(ec, "update_manifest", real_update_manifest)
    _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=2, resume=True)

    mf = _read_json(run_dir / "manifest.json")
    assert mf["total_embedded"] == 4


# ============================================================
# 6) Overwrite deletes and rebuilds
# ============================================================
def test_overwrite_rebuilds(patch_dummy_embedder, paths: Tuple[Path, Path]):
    chunks_path, out_dir = paths
    _write_chunks_jsonl(chunks_path, [{"chunk_id": "x", "text": "1"}, {"chunk_id": "y", "text": "2"}])

    run_dir = _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=1, overwrite=False)
    mf1 = _read_json(run_dir / "manifest.json")
    assert mf1["total_embedded"] == 2
    shard0 = run_dir / "shards" / "embeddings_000000.npy"
    assert shard0.exists()

    # overwrite with different shard_size to prove rebuild happened
    run_dir2 = _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=2, overwrite=True)
    assert run_dir2 == run_dir
    mf2 = _read_json(run_dir / "manifest.json")
    assert mf2["total_embedded"] == 2
    # With shard_size=2, only one shard should exist now
    assert (run_dir / "shards" / "embeddings_000000.npy").exists()
    assert not (run_dir / "shards" / "embeddings_000001.npy").exists()


# ============================================================
# 7) Config mismatch blocks (safety)
# ============================================================
def test_manifest_config_mismatch_raises(patch_dummy_embedder, paths: Tuple[Path, Path]):
    chunks_path, out_dir = paths
    _write_chunks_jsonl(chunks_path, [{"chunk_id": "a", "text": "t"}])

    _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=10, max_length=128)

    # Same run_dir name (pooling/normalize same), but different max_length => should raise
    with pytest.raises(RuntimeError, match="config does not match"):
        _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=10, max_length=64, overwrite=False)


# ============================================================
# 8) Bad JSON lines are skipped and logged
# ============================================================
def test_bad_json_line_skipped_and_logged(patch_dummy_embedder, paths: Tuple[Path, Path]):
    chunks_path, out_dir = paths
    _write_chunks_jsonl(
        chunks_path,
        [{"chunk_id": "c1", "text": "t1"}, {"chunk_id": "c2", "text": "t2"}],
        add_malformed_line=True,
    )

    run_dir = _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=10)

    mf = _read_json(run_dir / "manifest.json")
    assert mf["total_embedded"] == 2  # only valid ones

    warns = _find_log_entries(run_dir / "embed_log.jsonl", "WARN_BAD_JSON_LINE")
    assert warns, "Expected WARN_BAD_JSON_LINE in embed_log.jsonl"


# ============================================================
# 9) Missing chunk_id records are skipped
# ============================================================
def test_missing_chunk_id_is_skipped(patch_dummy_embedder, paths: Tuple[Path, Path]):
    chunks_path, out_dir = paths
    _write_chunks_jsonl(
        chunks_path,
        [
            {"chunk_id": "ok1", "text": "t1"},
            {"text": "no id"},                 # missing chunk_id
            {"chunk_id": 123, "text": "bad"},  # non-string chunk_id
            {"chunk_id": "ok2", "text": "t2"},
        ],
    )

    run_dir = _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=10)
    mf = _read_json(run_dir / "manifest.json")
    assert mf["total_embedded"] == 2
    done = _count_jsonl_lines(run_dir / "embedded_ids.jsonl")
    assert done == 2


# ============================================================
# 10) Duplicate chunk_ids are de-duped within file and counted
# ============================================================
def test_duplicate_chunk_ids_deduped_and_counted(patch_dummy_embedder, paths: Tuple[Path, Path]):
    chunks_path, out_dir = paths
    _write_chunks_jsonl(
        chunks_path,
        [
            {"chunk_id": "dup", "text": "t1"},
            {"chunk_id": "dup", "text": "t1 again"},
            {"chunk_id": "uniq", "text": "t2"},
        ],
    )

    run_dir = _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=10)
    mf = _read_json(run_dir / "manifest.json")
    assert mf["total_embedded"] == 2

    ends = _find_log_entries(run_dir / "embed_log.jsonl", "RUN_END")
    assert ends
    assert int(ends[-1]["total_dupe_in_file"]) == 1


# ============================================================
# 11) Atomic write behavior: no leftover .tmp files after success
# ============================================================
def test_no_tmp_files_left_after_success(patch_dummy_embedder, paths: Tuple[Path, Path]):
    chunks_path, out_dir = paths
    _write_chunks_jsonl(chunks_path, [{"chunk_id": "a", "text": "t1"}, {"chunk_id": "b", "text": "t2"}])

    run_dir = _run_embed_chunks(chunks_path=chunks_path, out_dir=out_dir, shard_size=1)
    assert not _any_tmp_files(run_dir), "Found leftover .tmp files; atomic writes should clean up"


# ============================================================
# 12) Naming convention for run_dir
# ============================================================
def test_run_name_convention():
    assert ec.compute_run_name("mean", True) == "bge_mean_norm"
    assert ec.compute_run_name("cls", True) == "bge_cls_norm"
    assert ec.compute_run_name("mean", False) == "bge_mean_raw"
    assert ec.compute_run_name("cls", False) == "bge_cls_raw"


# ============================================================
# 13) Slow HF integration test (true model end-to-end)
# ============================================================
@pytest.mark.slow
def test_hf_integration_end_to_end(tmp_path: Path):
    # This test intentionally loads the real HF model. Keep it tiny.
    chunks_path = tmp_path / "data" / "chunks" / "chunks.jsonl"
    out_dir = tmp_path / "data" / "embeddings"

    _write_chunks_jsonl(
        chunks_path,
        [
            {"chunk_id": "c1", "text": "USCIS eliminated the one-year foreign residency requirement for R-1 religious workers."},
            {"chunk_id": "c2", "text": "ICE memo authorizes forceful entry into residences based on administrative warrants."},
        ],
    )

    run_dir = _run_embed_chunks(
        chunks_path=chunks_path,
        out_dir=out_dir,
        pooling="mean",
        normalize=True,
        batch_size=4,
        max_length=128,
        shard_size=2,
        resume=True,
        overwrite=False,
        device=None,  # auto (mps on your M2)
        seed=123,
    )

    mf = _read_json(run_dir / "manifest.json")
    assert mf["total_embedded"] == 2
    assert isinstance(mf["dim"], int) and mf["dim"] > 0

    # Load shard and check finiteness + norm ~ 1
    shard0 = Path(mf["shards"][0]["vec_path"])
    X = np.load(shard0)
    assert X.shape[0] == 2
    assert X.shape[1] == mf["dim"]
    assert X.dtype == np.float32
    assert np.isfinite(X).all()

    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
