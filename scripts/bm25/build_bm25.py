# Build a BM25-style inverted index from chunks.jsonl.
# Outputs *streaming-first* artifacts under data/.../indexes/bm25/.
#
# Output contract (minimal, seek-friendly, no doc text duplication):
# - docs.jsonl               : one line per indexed doc (NO text), includes {"doc_id", "len", ...meta...}
# - docs_offsets.sqlite      : doc_id -> byte offset in docs.jsonl
# - inverted_index.jsonl     : one line per term {"term","df","idf","postings":[[doc_id, tf], ...]}
# - offsets.sqlite           : term -> byte offset in inverted_index.jsonl
# - chunks_offsets.sqlite    : chunk_id -> byte offset in *input* chunks.jsonl (immutable source of truth)
# - corpus_stats.json        : small stats + pointers to all artifacts (manifest)
# - build_log.jsonl          : structured events (debugging / reproducibility)
#
# Design constraints:
# - No multi-line terms: tokens containing '\n' or '\t' are skipped (logged).
# - Streaming-first: avoid loading whole corpus or whole vocab into memory.
# - Postings use shard TSVs + external sort for scalability.
# - chunks.jsonl is immutable source of truth; we index chunks by chunk_id and store chunk offsets.

"""
How to run:

python -m scripts.bm25.build_bm25 --config config.json

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sqlite3
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterator, List, Optional, Tuple

from scripts.common.policy_meta import flatten_policy_meta  # type: ignore
from scripts.common.text_processing import init_text_processing_from_config, tokenize  # type: ignore
from scripts.common.config import get_cfg_value, get_int, get_path, load_config  # type: ignore


# -----------------------------
# Configuration knobs
# -----------------------------

# How often to commit sqlite inserts (docs_offsets + chunks_offsets). Larger = faster but more work lost if interrupted.
COMMIT_EVERY_DOCS: int = 250

# How often to log progress (in documents).
PROGRESS_EVERY_DOCS: int = 500

# Number of posting shards (TSV). Higher shards = more files but smaller per-shard sorts.
DEFAULT_N_SHARDS: int = 128


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class ChunkDoc:
    chunk_id: str
    entry_id: str
    attachment_id: str
    page_start: int
    page_end: int
    text: str
    source_path: str
    meta: Dict[str, Any]


# -----------------------------
# Logging + small utils
# -----------------------------

def log_event(log_path: Path, event: Dict[str, Any]) -> None:
    """Append one structured event line to a JSONL log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def ensure_parent_dir(path: Path) -> None:
    """Ensure parent directory exists for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    """Persist a JSON file to disk."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def stable_shard(term: str, n_shards: int) -> int:
    """Stable sharding by term; consistent across runs and machines."""
    h = hashlib.md5(term.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % n_shards


def _resolve_cli_path(base_dir: Path, p: str) -> Path:
    """Resolve path relative to config base dir."""
    pp = Path(p)
    return pp if pp.is_absolute() else (base_dir / pp).resolve()


# -----------------------------
# Metadata sanitation
# -----------------------------

def minimal_meta(meta: Any, min_meta_keys: List[str]) -> Dict[str, Any]:
    """Sanitize metadata and keep only minimal keys (JSON-safe)."""
    if not isinstance(meta, dict):
        return {}

    out: Dict[str, Any] = {}
    for k in min_meta_keys:
        v = meta.get(k)
        if v is None:
            continue

        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            safe_list: List[Any] = []
            for item in v:
                safe_list.append(item if isinstance(item, (str, int, float, bool)) else str(item))
            out[k] = safe_list
        elif isinstance(v, dict):
            safe_dict: Dict[str, Any] = {}
            for dk, dv in v.items():
                safe_dict[str(dk)] = dv if isinstance(dv, (str, int, float, bool)) else str(dv)
            out[k] = safe_dict
        else:
            out[k] = str(v)

    return out


# -----------------------------
# Input streaming
# -----------------------------

def iter_chunks(
    chunks_path: Path,
    log_path: Path,
    min_meta_keys: List[str],
) -> Iterator[Tuple[int, ChunkDoc]]:
    """
    Stream chunks from chunks.jsonl (one JSON object per line),
    yielding (byte_offset, ChunkDoc) so we can build chunk_id -> offset in chunks.jsonl.
    """
    with chunks_path.open("r", encoding="utf-8") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                log_event(log_path, {"type": "SKIP_BAD_JSON", "offset": int(pos), "path": str(chunks_path)})
                continue

            chunk_id = obj.get("chunk_id")
            text = obj.get("text") or ""

            meta_obj = obj.get("meta") or {}
            policy_meta = meta_obj.get("policy") if isinstance(meta_obj, dict) else None
            attach_meta = meta_obj.get("attachment") if isinstance(meta_obj, dict) else None
            extr_meta = meta_obj.get("extracted") if isinstance(meta_obj, dict) else None

            entry_id = None
            if isinstance(policy_meta, dict):
                entry_id = policy_meta.get("page_ptr_id")
            if entry_id is None:
                entry_id = obj.get("page_ptr_id")

            attachment_id = obj.get("policydocument_id")
            if attachment_id is None and isinstance(attach_meta, dict):
                attachment_id = attach_meta.get("policydocument_id")
            if attachment_id is None and isinstance(extr_meta, dict):
                attachment_id = extr_meta.get("stable_id")

            if not chunk_id:
                log_event(
                    log_path,
                    {
                        "type": "SKIP_BAD_RECORD",
                        "offset": int(pos),
                        "path": str(chunks_path),
                        "chunk_id": chunk_id,
                        "entry_id": entry_id,
                        "attachment_id": attachment_id,
                    },
                )
                continue

            try:
                page_start = int(obj.get("page_start"))
                page_end = int(obj.get("page_end"))
            except (TypeError, ValueError):
                log_event(
                    log_path,
                    {
                        "type": "SKIP_BAD_PAGES",
                        "offset": int(pos),
                        "chunk_id": str(chunk_id),
                        "page_start": obj.get("page_start"),
                        "page_end": obj.get("page_end"),
                    },
                )
                continue

            if not str(text).strip():
                log_event(log_path, {"type": "SKIP_EMPTY_TEXT", "offset": int(pos), "chunk_id": str(chunk_id)})
                continue

            source_type = obj.get("source_type") or ""
            is_policy_text = source_type in {"policy_title", "policy_summary"}
            if (entry_id is None or attachment_id is None) and not is_policy_text:
                log_event(
                    log_path,
                    {
                        "type": "WARN_MISSING_IDS",
                        "offset": int(pos),
                        "chunk_id": str(chunk_id),
                        "entry_id": entry_id,
                        "attachment_id": attachment_id,
                    },
                )

            chunk = ChunkDoc(
                chunk_id=str(chunk_id),
                entry_id=str(entry_id) if entry_id is not None else "",
                attachment_id=str(attachment_id) if attachment_id is not None else "",
                page_start=page_start,
                page_end=page_end,
                text=str(text),
                source_path=str(obj.get("source_path") or ""),
                meta=minimal_meta(flatten_policy_meta(obj.get("meta") or {}), min_meta_keys=min_meta_keys),
            )

            yield int(pos), chunk


# -----------------------------
# Token/term helpers
# -----------------------------

def term_frequencies(tokens: List[str]) -> Counter:
    """Convert tokens into term frequencies for one document."""
    return Counter(tokens)


def validate_doc_terms(tf: Counter, log_path: Path, chunk_id: str, max_unique_terms_per_doc: int) -> bool:
    """Decide whether a document is too pathological to index."""
    if len(tf) > max_unique_terms_per_doc:
        log_event(
            log_path,
            {
                "type": "WARN_DOC_TOO_MANY_TERMS",
                "chunk_id": chunk_id,
                "unique_terms": len(tf),
                "cap": max_unique_terms_per_doc,
            },
        )
        return False
    return True


def build_doc_payload_and_tf(chunk: ChunkDoc, doc_id: int) -> Tuple[Dict[str, Any], Counter, int]:
    """
    Convert one chunk into an indexable doc payload and term stats.
    Note: doc payload does NOT include text; text is fetched from chunks.jsonl using chunk offsets.
    """
    tokens = tokenize(chunk.text)
    tf = term_frequencies(tokens)
    doc_len = int(sum(tf.values()))

    doc_payload = {
        "doc_id": doc_id,
        "len": doc_len,  # store dl directly in docs.jsonl (no separate doc_lens file)
        "chunk_id": chunk.chunk_id,
        "entry_id": chunk.entry_id,
        "attachment_id": chunk.attachment_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "source_path": chunk.source_path,
        "meta": chunk.meta,
    }
    return doc_payload, tf, doc_len


def _is_tsv_safe_term(term: str) -> bool:
    """
    Terms written to shard TSVs must not contain TSV delimiters.
    If tokenize() produces '\t' or '\n', we skip that token and log it.
    """
    return ("\t" not in term) and ("\n" not in term)


# -----------------------------
# SQLite helpers
# -----------------------------

def _sqlite_connect(path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with sane defaults for bulk updates."""
    ensure_parent_dir(path)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def _init_offsets_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS offsets (term TEXT PRIMARY KEY, offset INTEGER NOT NULL);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_offsets_term ON offsets(term);")


def _init_docs_offsets_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS docs_offsets (doc_id INTEGER PRIMARY KEY, offset INTEGER NOT NULL);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_offsets_doc_id ON docs_offsets(doc_id);")


def _init_chunks_offsets_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS chunks_offsets (chunk_id TEXT PRIMARY KEY, offset INTEGER NOT NULL);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_offsets_chunk_id ON chunks_offsets(chunk_id);")


# -----------------------------
# External sort + shard merge
# -----------------------------

def _sort_tsv_inplace(src_path: Path, dst_path: Path, log_path: Path) -> None:
    """Use system 'sort' to sort shard TSV by (term, doc_id)."""
    sort_bin = shutil.which("sort")
    if not sort_bin:
        log_event(
            log_path,
            {
                "type": "FATAL_SORT_MISSING",
                "msg": "System 'sort' not found on PATH (required for shard merge).",
                "path": str(src_path),
            },
        )
        raise RuntimeError("System 'sort' not found on PATH (required for shard merge).")

    cmd = [sort_bin, "-t", "\t", "-k1,1", "-k2,2n", str(src_path), "-o", str(dst_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        log_event(
            log_path,
            {
                "type": "FATAL_SORT_FAILED",
                "path": str(src_path),
                "returncode": e.returncode,
                "stdout_tail": (e.stdout or "")[-2000:],
                "stderr_tail": (e.stderr or "")[-2000:],
                "cmd": cmd,
            },
        )
        raise RuntimeError(f"sort failed on shard: {src_path}") from e


def merge_shards_to_inverted_index(
    inv_path: Path,
    shard_paths: List[Path],
    log_path: Path,
    *,
    offsets_db_path: Path,
    n_docs: int,
) -> int:
    """
    Merge sorted shard TSVs into inverted_index.jsonl, while writing term -> byte offset
    into offsets.sqlite. Writes df and idf per term directly into inverted_index.jsonl.
    Strict: each term must appear exactly once across all shards.
    """
    ensure_parent_dir(inv_path)

    offsets_conn = _sqlite_connect(offsets_db_path)
    try:
        _init_offsets_table(offsets_conn)
        offsets_conn.execute("DELETE FROM offsets;")
        offsets_conn.commit()

        n_terms = 0

        with inv_path.open("w", encoding="utf-8", newline="\n") as out:
            for shard_path in shard_paths:
                if not shard_path.exists() or shard_path.stat().st_size == 0:
                    continue

                with NamedTemporaryFile(delete=False) as tmp:
                    tmp_sorted = Path(tmp.name)

                _sort_tsv_inplace(shard_path, tmp_sorted, log_path)

                with tmp_sorted.open("r", encoding="utf-8") as f:
                    cur_term: Optional[str] = None
                    cur_postings: List[Tuple[int, int]] = []

                    def _flush_term(term: str, postings: List[Tuple[int, int]]) -> None:
                        nonlocal n_terms
                        offset = out.tell()
                        try:
                            offsets_conn.execute("INSERT INTO offsets(term, offset) VALUES(?, ?)", (term, offset))
                        except sqlite3.IntegrityError:
                            log_event(log_path, {"type": "FATAL_DUP_TERM_ACROSS_SHARDS", "term": term})
                            raise RuntimeError(f"Term appeared in multiple shards: {term}")

                        df = len(postings)
                        # Standard BM25 IDF (same formula as before, now stored inline).
                        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

                        out.write(
                            json.dumps(
                                {"term": term, "df": int(df), "idf": float(idf), "postings": postings},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        n_terms += 1

                    for line in f:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) != 3:
                            log_event(
                                log_path,
                                {
                                    "type": "FATAL_BAD_SHARD_LINE",
                                    "path": str(tmp_sorted),
                                    "line_preview": repr(line[:200]),
                                    "num_fields": len(parts),
                                },
                            )
                            raise RuntimeError(f"Malformed shard TSV line in {tmp_sorted}")

                        term, doc_id_s, tf_s = parts
                        doc_id = int(doc_id_s)
                        tf = int(tf_s)

                        if cur_term is None:
                            cur_term = term

                        if term != cur_term:
                            _flush_term(cur_term, cur_postings)
                            cur_term = term
                            cur_postings = []

                        cur_postings.append((doc_id, tf))

                    if cur_term is not None:
                        _flush_term(cur_term, cur_postings)

                offsets_conn.commit()
                tmp_sorted.unlink(missing_ok=True)

        return n_terms
    finally:
        offsets_conn.close()


# -----------------------------
# Main build orchestrator
# -----------------------------

def build_bm25_artifacts(
    *,
    chunks_path: Path,
    chunks_offsets_db_path: Path,
    docs_path: Path,
    docs_offsets_db_path: Path,
    inv_path: Path,
    offsets_db_path: Path,
    index_dir: Path,  # used for temp shards + corpus_stats.json by convention
    log_path: Path,
    config_path: Path,
    k1: float,
    b: float,
    max_unique_terms_per_doc: int,
    min_meta_keys: List[str],
    text_cfg: Dict[str, Any],
    n_shards: int = DEFAULT_N_SHARDS,
) -> None:
    """
    Orchestrator:
    1) stream chunks -> docs.jsonl (len embedded), shard postings TSVs,
       and sqlite seek maps for docs/chunks
    2) merge shards -> inverted_index.jsonl (df/idf included) + offsets.sqlite
    3) write corpus_stats.json + build logs
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    # Derived (not in config today): manifest file path under index_dir.
    corpus_stats_path = index_dir / "corpus_stats.json"

    # Ensure parent dirs exist for configured outputs
    for p in [docs_path, inv_path, offsets_db_path, docs_offsets_db_path, chunks_offsets_db_path, corpus_stats_path, log_path]:
        ensure_parent_dir(p)

    # Temporary postings shards (derived)
    tmp_dir = index_dir / "_tmp_postings"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [tmp_dir / f"postings_shard_{i:04d}.tsv" for i in range(n_shards)]

    # Overwrite old shards to avoid mixing builds
    for p in shard_paths:
        if p.exists():
            p.unlink()
    shard_files = [p.open("w", encoding="utf-8", newline="\n") for p in shard_paths]

    # Offsets sqlite: reset and stream inserts
    docs_off_conn = _sqlite_connect(docs_offsets_db_path)
    chunks_off_conn = _sqlite_connect(chunks_offsets_db_path)
    _init_docs_offsets_table(docs_off_conn)
    _init_chunks_offsets_table(chunks_off_conn)

    docs_off_conn.execute("DELETE FROM docs_offsets;")
    chunks_off_conn.execute("DELETE FROM chunks_offsets;")
    docs_off_conn.commit()
    chunks_off_conn.commit()

    # Build meta stamp (helps diagnose stale tmp dirs)
    build_meta_path = tmp_dir / "_build_meta.json"
    write_json(
        build_meta_path,
        {
            "created_at_unix": int(time.time()),
            "config_path": str(config_path),
            "chunks_path": str(chunks_path),
            "index_dir": str(index_dir),
            "n_shards": n_shards,
            "shard_fn": "md5_first4_mod",
            "k1": k1,
            "b": b,
            "max_unique_terms_per_doc": max_unique_terms_per_doc,
            "text_processing": text_cfg,
        },
    )

    # Deduplicate chunk_ids defensively (fine for sample_100; can be replaced later if needed)
    seen_chunk_ids: set[str] = set()

    # Counters + streaming corpus stats
    n_lines_read = 0
    n_skipped_dup = 0
    n_skipped_no_tokens = 0
    n_skipped_bad_terms = 0
    n_skipped_too_many_terms = 0
    n_warn_empty_meta = 0

    n_docs = 0
    sum_dl = 0

    # doc_id increments only when we keep a doc
    next_doc_id = 0
    pending_docs = 0

    try:
        with docs_path.open("w", encoding="utf-8", newline="\n") as docs_f:
            for chunk_offset, chunk in iter_chunks(chunks_path, log_path, min_meta_keys=min_meta_keys):
                n_lines_read += 1

                if chunk.chunk_id in seen_chunk_ids:
                    n_skipped_dup += 1
                    log_event(log_path, {"type": "SKIP_DUP_CHUNK_ID", "chunk_id": chunk.chunk_id})
                    continue
                seen_chunk_ids.add(chunk.chunk_id)

                doc_id = next_doc_id
                doc_payload, tf, doc_len = build_doc_payload_and_tf(chunk=chunk, doc_id=doc_id)

                if not chunk.meta:
                    n_warn_empty_meta += 1
                    if n_warn_empty_meta <= 10:
                        log_event(log_path, {"type": "WARN_EMPTY_FLATTENED_META", "chunk_id": chunk.chunk_id})

                if doc_len == 0:
                    n_skipped_no_tokens += 1
                    log_event(log_path, {"type": "SKIP_NO_TOKENS", "chunk_id": chunk.chunk_id})
                    continue

                if not validate_doc_terms(tf, log_path, chunk.chunk_id, max_unique_terms_per_doc=max_unique_terms_per_doc):
                    n_skipped_too_many_terms += 1
                    continue

                # --- Seek maps (only for kept docs) ---
                # chunk_id -> byte offset in chunks.jsonl (immutable source)
                try:
                    chunks_off_conn.execute(
                        "INSERT INTO chunks_offsets(chunk_id, offset) VALUES(?, ?)",
                        (chunk.chunk_id, int(chunk_offset)),
                    )
                except sqlite3.IntegrityError:
                    # Should be prevented by seen_chunk_ids; log defensively.
                    log_event(log_path, {"type": "WARN_DUP_CHUNK_OFFSET_INSERT", "chunk_id": chunk.chunk_id})
                    continue

                # doc_id -> byte offset in docs.jsonl
                doc_out_offset = docs_f.tell()
                docs_off_conn.execute(
                    "INSERT INTO docs_offsets(doc_id, offset) VALUES(?, ?)",
                    (int(doc_id), int(doc_out_offset)),
                )

                # Write doc payload (streaming). Includes len; no text.
                docs_f.write(json.dumps(doc_payload, ensure_ascii=False) + "\n")

                # Write postings to shards
                for term, freq in tf.items():
                    if not _is_tsv_safe_term(term):
                        n_skipped_bad_terms += 1
                        log_event(
                            log_path,
                            {
                                "type": "SKIP_BAD_TERM_CHARS",
                                "chunk_id": chunk.chunk_id,
                                "term_preview": repr(term[:200]),
                                "term_len": len(term),
                            },
                        )
                        continue
                    shard_idx = stable_shard(term, n_shards)
                    shard_files[shard_idx].write(f"{term}\t{doc_id}\t{int(freq)}\n")

                n_docs += 1
                sum_dl += int(doc_len)
                pending_docs += 1

                # Commit sqlite periodically for performance
                if pending_docs >= COMMIT_EVERY_DOCS:
                    docs_off_conn.commit()
                    chunks_off_conn.commit()
                    pending_docs = 0

                # Progress log
                if n_docs % PROGRESS_EVERY_DOCS == 0:
                    log_event(
                        log_path,
                        {
                            "type": "PROGRESS",
                            "n_docs": n_docs,
                            "n_lines_read": n_lines_read,
                            "sum_dl": sum_dl,
                            "avgdl_so_far": (sum_dl / float(n_docs)) if n_docs else 0.0,
                        },
                    )

                next_doc_id += 1

    finally:
        # Ensure shard file handles are closed
        for sf in shard_files:
            try:
                sf.close()
            except Exception:
                pass

        # Final sqlite commit + close
        try:
            docs_off_conn.commit()
        except Exception:
            pass
        try:
            chunks_off_conn.commit()
        except Exception:
            pass

        try:
            docs_off_conn.close()
        except Exception:
            pass
        try:
            chunks_off_conn.close()
        except Exception:
            pass

    if n_docs == 0:
        raise RuntimeError("No documents indexed. Check chunks.jsonl and logs.")

    avgdl = sum_dl / float(n_docs)
    if avgdl <= 0.0:
        log_event(log_path, {"type": "FATAL_AVGDL_ZERO", "n_docs": n_docs, "avgdl": avgdl})
        raise RuntimeError("avgdl is zero; index build looks broken.")

    # Merge shards -> inverted_index.jsonl (df/idf included) + offsets.sqlite
    n_terms = merge_shards_to_inverted_index(
        inv_path,
        shard_paths,
        log_path,
        offsets_db_path=offsets_db_path,
        n_docs=n_docs,
    )

    # Write corpus stats (small manifest)
    write_json(
        corpus_stats_path,
        {
            "n_docs": n_docs,
            "avgdl": avgdl,
            "n_terms": n_terms,
            "bm25_defaults": {"k1": k1, "b": b, "max_unique_terms_per_doc": max_unique_terms_per_doc},
            "text_processing": text_cfg,
            "min_meta_keys": min_meta_keys,
            "config_path": str(config_path),
            "paths": {
                "chunks_jsonl": str(chunks_path),
                "chunks_offsets_sqlite": str(chunks_offsets_db_path),
                "docs_jsonl": str(docs_path),
                "docs_offsets_sqlite": str(docs_offsets_db_path),
                "inverted_index_jsonl": str(inv_path),
                "offsets_sqlite": str(offsets_db_path),
                "index_dir": str(index_dir),
                "log_path": str(log_path),
                "corpus_stats_json": str(corpus_stats_path),
            },
            "shards": {"n_shards": n_shards, "tmp_dir": str(tmp_dir)},
            "skip_counters": {
                "n_lines_read": n_lines_read,
                "skipped_dup_chunk_id": n_skipped_dup,
                "skipped_no_tokens": n_skipped_no_tokens,
                "skipped_bad_terms": n_skipped_bad_terms,
                "skipped_too_many_terms": n_skipped_too_many_terms,
                "warn_empty_meta": n_warn_empty_meta,
            },
        },
    )

    # Final build log
    log_event(
        log_path,
        {
            "type": "BUILD_COMPLETE",
            "chunks_path": str(chunks_path),
            "index_dir": str(index_dir),
            "n_lines_read": n_lines_read,
            "n_docs": n_docs,
            "n_terms": n_terms,
            "avgdl": avgdl,
            "skipped_dup_chunk_id": n_skipped_dup,
            "skipped_no_tokens": n_skipped_no_tokens,
            "skipped_bad_terms": n_skipped_bad_terms,
            "skipped_too_many_terms": n_skipped_too_many_terms,
            "warn_empty_meta": n_warn_empty_meta,
        },
    )

    print("BM25 index build complete.")
    print(f"Input chunks:  {chunks_path}")
    print(f"Index dir:     {index_dir}")
    print(f"Docs indexed:  {n_docs}")
    print(f"Vocab terms:   {n_terms}")
    print(f"Avgdl:         {avgdl:.2f}")
    print(f"Docs:          {docs_path}")
    print(f"Docs offsets:  {docs_offsets_db_path}")
    print(f"Chunks offsets:{chunks_offsets_db_path}")
    print(f"Index:         {inv_path}")
    print(f"Term offsets:  {offsets_db_path}")
    print(f"Stats:         {corpus_stats_path}")
    print(f"Log:           {log_path}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BM25 artifacts from chunks.jsonl (streaming).")
    p.add_argument("--config", default="config.json", help="Path to config.json")

    p.add_argument("--chunks", default=None, help="Override path to chunks.jsonl")
    p.add_argument("--index_dir", default=None, help="Override BM25 index directory (temp + corpus_stats.json)")
    p.add_argument("--docs", default=None, help="Override output docs.jsonl path")
    p.add_argument("--inv", default=None, help="Override output inverted_index.jsonl path")
    p.add_argument("--log", default=None, help="Override path to build log JSONL")

    p.add_argument("--k1", type=float, default=None)
    p.add_argument("--b", type=float, default=None)
    p.add_argument("--max_unique_terms_per_doc", type=int, default=None)
    p.add_argument("--n_shards", type=int, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    base_dir = config_path.parent.resolve()

    # Initialize text processing from config (tokenize() uses module-level config)
    tp_cfg_obj = init_text_processing_from_config(config_path)
    text_cfg = {
        "use_stopwords": bool(tp_cfg_obj.use_stopwords),
        "min_token_len": int(tp_cfg_obj.min_token_len),
        "stopwords_path": str(tp_cfg_obj.stopwords_path),
    }

    # Canonical paths from config
    chunks_path = get_path(cfg, "paths.chunks_jsonl", base_dir=base_dir)

    index_dir = get_path(cfg, "paths.bm25_index_dir", base_dir=base_dir)
    log_path = get_path(cfg, "paths.bm25_build_log_jsonl", base_dir=base_dir)

    docs_path = get_path(cfg, "paths.bm25_docs_jsonl", base_dir=base_dir)
    inv_path = get_path(cfg, "paths.bm25_inverted_index_jsonl", base_dir=base_dir)

    offsets_db_path = get_path(cfg, "paths.bm25_offsets_sqlite", base_dir=base_dir)
    docs_offsets_db_path = get_path(cfg, "paths.bm25_docs_offsets_sqlite", base_dir=base_dir)
    chunks_offsets_db_path = get_path(cfg, "paths.bm25_chunks_offsets_sqlite", base_dir=base_dir)

    # CLI overrides (optional)
    if args.chunks is not None:
        chunks_path = _resolve_cli_path(base_dir, args.chunks)
    if args.index_dir is not None:
        index_dir = _resolve_cli_path(base_dir, args.index_dir)
    if args.docs is not None:
        docs_path = _resolve_cli_path(base_dir, args.docs)
    if args.inv is not None:
        inv_path = _resolve_cli_path(base_dir, args.inv)
    if args.log is not None:
        log_path = _resolve_cli_path(base_dir, args.log)

    k1_val = get_cfg_value(cfg, "bm25.k1")
    k1 = 1.2 if k1_val is None else float(k1_val)

    b_val = get_cfg_value(cfg, "bm25.b")
    b = 0.75 if b_val is None else float(b_val)

    max_unique_terms_per_doc = int(get_int(cfg, "bm25.max_unique_terms_per_doc", default=50000))
    n_shards_cfg = int(get_int(cfg, "bm25.n_shards", default=DEFAULT_N_SHARDS))
    n_shards = n_shards_cfg if n_shards_cfg else DEFAULT_N_SHARDS

    if args.k1 is not None:
        k1 = float(args.k1)
    if args.b is not None:
        b = float(args.b)
    if args.max_unique_terms_per_doc is not None:
        max_unique_terms_per_doc = int(args.max_unique_terms_per_doc)
    if args.n_shards is not None:
        n_shards = int(args.n_shards)

    ensure_parent_dir(log_path)

    mmk = get_cfg_value(cfg, "bm25.min_meta_keys")
    if not isinstance(mmk, list) or not mmk:
        log_event(
            log_path,
            {"type": "WARN_NO_MIN_META_KEYS_CONFIGURED", "config_path": str(config_path), "index_dir": str(index_dir)},
        )
    min_meta_keys = list(mmk) if isinstance(mmk, list) else []
    # Back-compat: config previously used original_date_announced, but meta now uses announced_date.
    if "original_date_announced" in min_meta_keys:
        min_meta_keys = ["announced_date" if k == "original_date_announced" else k for k in min_meta_keys]
        # de-dupe while preserving order
        seen: set[str] = set()
        min_meta_keys = [k for k in min_meta_keys if not (k in seen or seen.add(k))]

    log_event(
        log_path,
        {
            "type": "BUILD_START",
            "config_path": str(config_path),
            "chunks_path": str(chunks_path),
            "index_dir": str(index_dir),
            "docs_path": str(docs_path),
            "inv_path": str(inv_path),
            "offsets_db_path": str(offsets_db_path),
            "docs_offsets_db_path": str(docs_offsets_db_path),
            "chunks_offsets_db_path": str(chunks_offsets_db_path),
            "k1": k1,
            "b": b,
            "max_unique_terms_per_doc": max_unique_terms_per_doc,
            "n_shards": n_shards,
            "min_meta_keys_count": len(min_meta_keys),
            "min_meta_keys": min_meta_keys,
            "text_processing": text_cfg,
        },
    )

    build_bm25_artifacts(
        chunks_path=chunks_path,
        chunks_offsets_db_path=chunks_offsets_db_path,
        docs_path=docs_path,
        docs_offsets_db_path=docs_offsets_db_path,
        inv_path=inv_path,
        offsets_db_path=offsets_db_path,
        index_dir=index_dir,
        log_path=log_path,
        config_path=config_path,
        k1=k1,
        b=b,
        max_unique_terms_per_doc=max_unique_terms_per_doc,
        min_meta_keys=min_meta_keys,
        text_cfg=text_cfg,
        n_shards=n_shards,
    )


if __name__ == "__main__":
    main()
