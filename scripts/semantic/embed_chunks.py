# scripts/embed_chunks.py
# Embed chunk-level JSONL records (from chunk_pages.py) using BGEEmbedder, and persist embeddings.
# Design goals:
# - Stage-separated: reads chunks.jsonl, writes embeddings to a separate output directory
# - Resume: safe to re-run; skips chunk_ids already embedded
# - Scales: streaming read, sharded writes, atomic file writes
# - AWS-ready: shard files are easy to upload to S3 later with minimal code change

"""
How to run 

python3 -m scripts.semantic.embed_chunks \
  --chunks data/sample_100/chunks/chunks.jsonl \
  --normalize \
  --overwrite

"""

from __future__ import annotations

import argparse
import json
import os
import shutil


# ---- perf / responsiveness knobs (must be set before tokenizer/model work) ----
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")  # try 4 for speed; drop to 2 if needed
os.environ.setdefault("MKL_NUM_THREADS", "4")

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scripts.semantic.bge_embedder import BGEConfig, BGEEmbedder

# -----------------------------
# Utilities
# -----------------------------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_bytes(final_path: Path, data: bytes, do_fsync: bool = True) -> None:
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        f.write(data)
        f.flush()
        if do_fsync:
            os.fsync(f.fileno())
    tmp_path.replace(final_path)


def atomic_write_json(final_path: Path, obj: Dict[str, Any], do_fsync: bool = False) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    atomic_write_bytes(final_path, data, do_fsync=do_fsync)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    rec = {"ts": now_iso(), **obj}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compute_run_name(pooling: str, normalize: bool) -> str:
    norm_tag = "norm" if normalize else "raw"
    return f"bge_{pooling}_{norm_tag}"


def config_fingerprint(cfg: BGEConfig) -> Dict[str, Any]:
    # Keep only fields that affect the resulting embeddings
    # device/batch_size are runtime details, not semantic identity
    d = asdict(cfg)
    keep = {
        "model_name",
        "max_length",
        "pooling",
        "normalize",
        "empty_policy",
        "empty_replacement",
        "seed",
        "trust_remote_code",
    }
    return {k: d.get(k) for k in keep}


# -----------------------------
# Streaming chunk reader
# -----------------------------

def iter_chunks_jsonl(chunks_path: Path) -> Iterable[Tuple[str, str]]:
    # Yields (chunk_id, text) for each record
    with chunks_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed JSON lines
                yield (f"__BAD_JSON_LINE_{line_num}", "")
                continue

            chunk_id = obj.get("chunk_id")
            text = obj.get("text")
            if not chunk_id or not isinstance(chunk_id, str):
                # Skip records missing IDs (cannot resume reliably)
                continue
            if not isinstance(text, str):
                text = "" if text is None else str(text)

            yield (chunk_id, text)


# -----------------------------
# Resume tracking
# -----------------------------

def load_done_set(done_ids_path: Path, log_path: Path) -> set[str]:
    done: set[str] = set()
    if not done_ids_path.exists():
        return done

    # MVP: load into memory for simplicity. Works fine for thousands–hundreds of thousands.
    # # MVP
    with done_ids_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                append_jsonl(
                    log_path,
                    {
                        "type": "WARN_BAD_DONE_ID_LINE",
                        "line_num": line_num,
                        "path": str(done_ids_path),
                    },
                )
                continue

            cid = obj.get("chunk_id")
            if isinstance(cid, str) and cid:
                done.add(cid)

    append_jsonl(
        log_path,
        {
            "type": "RESUME_DONE_SET_LOADED",
            "path": str(done_ids_path),
            "count": len(done),
        },
    )
    return done


def append_done_ids(done_ids_path: Path, chunk_ids: Sequence[str]) -> None:
    ensure_dir(done_ids_path.parent)
    with done_ids_path.open("a", encoding="utf-8") as f:
        for cid in chunk_ids:
            f.write(json.dumps({"chunk_id": cid}, ensure_ascii=False) + "\n")


# -----------------------------
# Sharding
# -----------------------------

def shard_paths(run_dir: Path, shard_idx: int) -> Tuple[Path, Path]:
    shards_dir = run_dir / "shards"
    ids_dir = run_dir / "ids"
    ensure_dir(shards_dir)
    ensure_dir(ids_dir)
    vec_path = shards_dir / f"embeddings_{shard_idx:06d}.npy"
    ids_path = ids_dir / f"chunk_ids_{shard_idx:06d}.jsonl"
    return vec_path, ids_path


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    # Atomic write: save to tmp then rename
    tmp = path.with_suffix(path.suffix + ".tmp")
    ensure_dir(path.parent)
    with tmp.open("wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_save_ids_jsonl(path: Path, chunk_ids: Sequence[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    ensure_dir(path.parent)
    with tmp.open("w", encoding="utf-8") as f:
        for cid in chunk_ids:
            f.write(json.dumps({"chunk_id": cid}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


# -----------------------------
# Manifest
# -----------------------------

def init_or_validate_manifest(
    run_dir: Path,
    manifest_path: Path,
    chunks_path: Path,
    cfg: BGEConfig,
    log_path: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    existing = safe_read_json(manifest_path)

    if existing and overwrite:
        # Fully delete the previous run directory
        shutil.rmtree(run_dir, ignore_errors=True)
        ensure_dir(run_dir)
        existing = None

    if existing is None:
        mf: Dict[str, Any] = {
            "created_ts": now_iso(),
            "run_dir": str(run_dir),
            "chunks_source": str(chunks_path),
            "chunks_source_size": chunks_path.stat().st_size if chunks_path.exists() else None,
            "config": config_fingerprint(cfg),
            "dim": None,
            "shards": [],  # list of {"idx": int, "vec_path": str, "ids_path": str, "count": int}
            "total_embedded": 0,
        }
        atomic_write_json(manifest_path, mf)
        append_jsonl(log_path, {"type": "MANIFEST_CREATED", "path": str(manifest_path)})
        return mf

    # Validate semantic config compatibility
    want = config_fingerprint(cfg)
    have = existing.get("config")
    if have != want:
        raise RuntimeError(
            "Manifest exists but embed config does not match.\n"
            f"Run dir: {run_dir}\n"
            f"Have: {have}\n"
            f"Want: {want}\n"
            "Use a different output dir, or pass --overwrite to rebuild."
        )

    return existing


def update_manifest(
    manifest_path: Path,
    mf: Dict[str, Any],
    log_path: Path,
) -> None:
    atomic_write_json(manifest_path, mf)
    append_jsonl(log_path, {"type": "MANIFEST_UPDATED", "path": str(manifest_path)})


# -----------------------------
# Main embedding pipeline
# -----------------------------

def embed_chunks(
    chunks_path: Path,
    out_dir: Path,
    pooling: str,
    normalize: bool,
    batch_size: int,
    max_length: int,
    shard_size: int,
    resume: bool,
    overwrite: bool,
    device: Optional[str],
    seed: Optional[int],
) -> None:
    run_name = compute_run_name(pooling=pooling, normalize=normalize)
    run_dir = out_dir / run_name
    ensure_dir(run_dir)

    log_path = run_dir / "embed_log.jsonl"
    manifest_path = run_dir / "manifest.json"
    done_ids_path = run_dir / "embedded_ids.jsonl"

    cfg = BGEConfig(
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,         # type: ignore[arg-type]
        normalize=normalize,
        seed=seed,
        empty_policy="replace",  # preserve 1:1 alignment for later ID mapping
        empty_replacement=".",
        trust_remote_code=False,
    )

    append_jsonl(
        log_path,
        {
            "type": "RUN_START",
            "chunks_path": str(chunks_path),
            "out_dir": str(out_dir),
            "run_dir": str(run_dir),
            "run_name": run_name,
            "resume": resume,
            "overwrite": overwrite,
            "cfg": config_fingerprint(cfg),
        },
    )

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")

    mf = init_or_validate_manifest(
        run_dir=run_dir,
        manifest_path=manifest_path,
        chunks_path=chunks_path,
        cfg=cfg,
        log_path=log_path,
        overwrite=overwrite,
    )

    done: set[str] = set()
    if resume:
        done = load_done_set(done_ids_path, log_path)

    try:
        import torch
        torch.set_num_threads(4)          # try 4; drop to 2 if laggy
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    embedder = BGEEmbedder(cfg)

    print("Embedder device:", embedder.device)

    if mf.get("dim") is None:
        mf["dim"] = int(embedder.dim) if embedder.dim else None
        update_manifest(manifest_path, mf, log_path)

    # Decide next shard index based on manifest
    existing_shards = mf.get("shards", [])
    shard_idx = 0
    if isinstance(existing_shards, list) and existing_shards:
        shard_idx = int(existing_shards[-1].get("idx", len(existing_shards) - 1)) + 1

    # Buffers for the next shard
    buf_ids: List[str] = []
    buf_texts: List[str] = []

    total_read = 0
    total_new = 0
    total_skipped = 0
    total_dupe_in_file = 0

    # Track duplicates within this run to avoid double embedding if chunks.jsonl has repeats
    seen_in_this_run: set[str] = set()

    def flush_shard() -> None:
        nonlocal shard_idx, buf_ids, buf_texts, mf, done, total_new

        if not buf_ids:
            return

        # Embed this shard
        try:
            vecs = embedder.embed_texts(buf_texts)
        except RuntimeError as e:
            append_jsonl(
                log_path,
                {
                    "type": "ERROR_EMBED_SHARD",
                    "shard_idx": shard_idx,
                    "count": len(buf_ids),
                    "error": str(e),
                },
            )
            raise

        if vecs.shape[0] != len(buf_ids):
            raise RuntimeError(
                "Embedder returned wrong number of vectors.\n"
                f"Expected {len(buf_ids)}, got {vecs.shape[0]}"
            )

        vec_path, ids_path = shard_paths(run_dir, shard_idx)

        # Atomic writes
        atomic_save_npy(vec_path, vecs.astype(np.float32))
        atomic_save_ids_jsonl(ids_path, buf_ids)

        # Append to done log so resume works even if we crash after this point
        append_done_ids(done_ids_path, buf_ids)

        # Update in-memory done set for this run
        done.update(buf_ids)

        # Update manifest
        shard_rec = {
            "idx": shard_idx,
            "vec_path": str(vec_path),
            "ids_path": str(ids_path),
            "count": int(len(buf_ids)),
        }
        mf["shards"].append(shard_rec)
        mf["total_embedded"] = int(mf.get("total_embedded", 0)) + int(len(buf_ids))
        update_manifest(manifest_path, mf, log_path)

        append_jsonl(
            log_path,
            {
                "type": "SHARD_WRITTEN",
                "shard_idx": shard_idx,
                "count": len(buf_ids),
                "vec_path": str(vec_path),
                "ids_path": str(ids_path),
                "total_embedded": mf["total_embedded"],
                "dim": int(vecs.shape[1]) if vecs.ndim == 2 else None,
            },
        )

        total_new += len(buf_ids)

        shard_idx += 1
        buf_ids = []
        buf_texts = []

    # Stream chunks and build shards
    for chunk_id, text in iter_chunks_jsonl(chunks_path):
        total_read += 1

        # Skip placeholder for bad JSON lines
        if chunk_id.startswith("__BAD_JSON_LINE_"):
            append_jsonl(
                log_path,
                {"type": "WARN_BAD_JSON_LINE", "chunk_id": chunk_id},
            )
            continue

        # De-dupe within this run (if chunks.jsonl contains duplicates)
        if chunk_id in seen_in_this_run:
            total_dupe_in_file += 1
            continue
        seen_in_this_run.add(chunk_id)

        # Resume skip
        if resume and chunk_id in done:
            total_skipped += 1
            continue

        buf_ids.append(chunk_id)
        buf_texts.append(text)

        if len(buf_ids) >= shard_size:
            flush_shard()

    # Flush final partial shard
    flush_shard()

    append_jsonl(
        log_path,
        {
            "type": "RUN_END",
            "chunks_path": str(chunks_path),
            "run_dir": str(run_dir),
            "total_read": total_read,
            "total_new_embedded": total_new,
            "total_skipped_resume": total_skipped,
            "total_dupe_in_file": total_dupe_in_file,
            "total_embedded_manifest": int(mf.get("total_embedded", 0)),
        },
    )

    print("Embedding complete.")
    print(f"Run dir: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Log: {log_path}")
    print(f"Embedded IDs: {done_ids_path}")
    print(f"Total read: {total_read}")
    print(f"New embedded: {total_new}")
    print(f"Skipped (resume): {total_skipped}")
    print(f"Duplicates in chunks.jsonl (ignored): {total_dupe_in_file}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed chunks.jsonl with BGE and persist embeddings (sharded + resumable).")
    p.add_argument("--chunks", default="data/chunks/chunks.jsonl", help="Path to chunks.jsonl")
    p.add_argument("--out_dir", default="data/embeddings", help="Output directory root")
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean", help="Pooling mode")
    p.add_argument("--normalize", action="store_true", help="L2 normalize embeddings")
    p.add_argument("--no_normalize", action="store_true", help="Disable normalization")
    p.add_argument("--batch_size", type=int, default=8, help="Embedding batch size")
    p.add_argument("--max_length", type=int, default=256, help="Tokenizer max_length")
    p.add_argument("--shard_size", type=int, default=1000, help="Vectors per shard")
    p.add_argument("--resume", action="store_true", help="Resume: skip chunk_ids already embedded")
    p.add_argument("--no_resume", action="store_true", help="Disable resume behavior")
    p.add_argument("--overwrite", action="store_true", help="Delete existing run directory contents and rebuild")
    p.add_argument("--device", default=None, help='Override device: "mps", "cuda", or "cpu" (default auto)')
    p.add_argument("--seed", type=int, default=None, help="Optional seed (mostly for benchmarking consistency)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    normalize = True
    if args.no_normalize:
        normalize = False
    elif args.normalize:
        normalize = True

    resume = True
    if args.no_resume:
        resume = False
    elif args.resume:
        resume = True

    embed_chunks(
        chunks_path=Path(args.chunks),
        out_dir=Path(args.out_dir),
        pooling=str(args.pooling),
        normalize=bool(normalize),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        shard_size=int(args.shard_size),
        resume=bool(resume),
        overwrite=bool(args.overwrite),
        device=args.device if args.device else None,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
