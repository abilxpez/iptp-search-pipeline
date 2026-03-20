# Search BM25 artifacts produced by scripts/bm25/build_bm25.py (streaming-first contract).
# Uses seek-based offsets (SQLite) so we don't load postings/docs/chunks fully into memory.
#
# Build contract assumed:
# - docs.jsonl               : one line per doc (NO text), includes {"doc_id","len","chunk_id",...,"meta":{...}}
# - docs_offsets.sqlite      : doc_id -> byte offset in docs.jsonl
# - inverted_index.jsonl     : one line per term {"term","df","idf","postings":[[doc_id, tf], ...]}
# - offsets.sqlite           : term -> byte offset in inverted_index.jsonl
# - chunks_offsets.sqlite    : chunk_id -> byte offset in input chunks.jsonl
# - chunks.jsonl             : immutable source-of-truth for text/snippets
# - corpus_stats.json        : {n_docs, avgdl, ...} + pointers (optional)
#
"""
How to use 

python -m scripts.bm25.search_bm25 \
  --config config.json \
  --q "temporary protected status" \
  --top_k 10 \
  --max_candidates 100

other flag:

 --sort_by_announced_date

"""
from __future__ import annotations
import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.common.config import get_cfg_value, get_path, load_config  # type: ignore
from scripts.common.text_processing import init_text_processing_from_config, tokenize  # type: ignore


# Default BM25 parameters (should match build defaults unless you tune intentionally)
BM25_K1_DEFAULT = 1.2
BM25_B_DEFAULT = 0.75


@dataclass(frozen=True)
class SearchResult:
    doc_id: int
    score: float
    doc: Dict[str, Any]
    snippet: str
    chunk_date: str
    title: str
    summary: str
    announced_date: str
    relevant_chunk: Optional[Dict[str, Any]]
    matched_attachment_ids: List[str]
    matched_doc_ids: List[int]
    matched_chunk_ids: List[str]
    matched_count: int
    best_rank: int
    source: str


# -----------------------------
# Small utils
# -----------------------------

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sqlite_ro(path: Path) -> sqlite3.Connection:
    """
    Open SQLite in read-only mode when possible.
    Falls back to normal open if URI read-only isn't supported in the environment.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing sqlite db: {path}")
    try:
        return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except Exception:
        return sqlite3.connect(str(path))


def seek_jsonl_line_by_offset(path: Path, offset: int) -> Dict[str, Any]:
    with path.open("rb") as f:
        f.seek(int(offset))
        line = f.readline()
        if not line:
            raise RuntimeError(f"Failed to read JSONL line at offset {offset} in {path}")
        return json.loads(line.decode("utf-8"))


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


def source_label(source_type: Any) -> str:
    st = str(source_type or "")
    if st == "policy_title":
        return "title"
    if st == "policy_summary":
        return "summary"
    return "pdf"


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


# -----------------------------
# BM25 scoring
# -----------------------------

def bm25_term_score(*, idf: float, tf: int, dl: int, avgdl: float, k1: float, b: float) -> float:
    denom = tf + k1 * (1.0 - b + b * (float(dl) / float(avgdl)))
    if denom <= 0.0:
        return 0.0
    return float(idf) * (float(tf) * (k1 + 1.0)) / denom


# -----------------------------
# Artifact accessors (seek-based)
# -----------------------------

def get_term_entry(
    *,
    term: str,
    inv_path: Path,
    offsets_conn: sqlite3.Connection,
) -> Optional[Dict[str, Any]]:
    """
    Look up a term in offsets.sqlite and seek to its row in inverted_index.jsonl.
    Returns parsed JSON dict or None if term not present.
    """
    row = offsets_conn.execute("SELECT offset FROM offsets WHERE term=?", (term,)).fetchone()
    if not row:
        return None
    off = int(row[0])
    obj = seek_jsonl_line_by_offset(inv_path, off)
    if obj.get("term") != term:
        raise RuntimeError(f"Term offset mismatch: expected '{term}', got '{obj.get('term')}'")
    return obj


def get_doc_by_id(
    *,
    doc_id: int,
    docs_path: Path,
    docs_offsets_conn: sqlite3.Connection,
) -> Optional[Dict[str, Any]]:
    """
    doc_id -> docs_offsets.sqlite -> docs.jsonl seek.
    Returns None if doc_id not present (e.g., partial/dirty build).
    """
    row = docs_offsets_conn.execute("SELECT offset FROM docs_offsets WHERE doc_id=?", (int(doc_id),)).fetchone()
    if not row:
        return None
    off = int(row[0])
    obj = seek_jsonl_line_by_offset(docs_path, off)
    if int(obj.get("doc_id", -1)) != int(doc_id):
        raise RuntimeError(f"Doc offset mismatch: expected doc_id={doc_id}, got {obj.get('doc_id')}")
    return obj


def get_chunk_text_by_chunk_id(
    *,
    chunk_id: str,
    chunks_path: Path,
    chunks_offsets_conn: sqlite3.Connection,
) -> Optional[str]:
    """
    chunk_id -> chunks_offsets.sqlite -> chunks.jsonl seek -> return text.
    Returns None if missing.
    """
    row = chunks_offsets_conn.execute("SELECT offset FROM chunks_offsets WHERE chunk_id=?", (chunk_id,)).fetchone()
    if not row:
        return None
    off = int(row[0])
    obj = seek_jsonl_line_by_offset(chunks_path, off)
    # Defensive: make sure this offset really points to this chunk
    if str(obj.get("chunk_id", "")) != str(chunk_id):
        raise RuntimeError(f"Chunk offset mismatch: expected chunk_id={chunk_id}, got {obj.get('chunk_id')}")
    return str(obj.get("text") or "")


def get_chunk_by_chunk_id(
    *,
    chunk_id: str,
    chunks_path: Path,
    chunks_offsets_conn: sqlite3.Connection,
) -> Optional[Dict[str, Any]]:
    row = chunks_offsets_conn.execute("SELECT offset FROM chunks_offsets WHERE chunk_id=?", (chunk_id,)).fetchone()
    if not row:
        return None
    off = int(row[0])
    obj = seek_jsonl_line_by_offset(chunks_path, off)
    if str(obj.get("chunk_id", "")) != str(chunk_id):
        raise RuntimeError(f"Chunk offset mismatch: expected chunk_id={chunk_id}, got {obj.get('chunk_id')}")
    return obj


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


# -----------------------------
# Filters
# -----------------------------

def doc_passes_filters(doc: Dict[str, Any], filters: Dict[str, Optional[str]]) -> bool:
    entry_id = filters.get("entry_id")
    administration = filters.get("administration")
    agency = filters.get("agency")
    subject = filters.get("subject")

    if entry_id and str(doc.get("entry_id", "")) != str(entry_id):
        return False

    meta = doc.get("meta") or {}

    if administration and str(meta.get("administration", "")) != str(administration):
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
# Core query scoring
# -----------------------------

def score_query(
    *,
    query: str,
    top_k: int,
    k1: float,
    b: float,
    n_docs: int,
    avgdl: float,
    docs_path: Path,
    inv_path: Path,
    chunks_path: Path,
    offsets_conn: sqlite3.Connection,
    docs_offsets_conn: sqlite3.Connection,
    chunks_offsets_conn: sqlite3.Connection,
    filters: Dict[str, Optional[str]],
    snippet_chars: int = 220,
    max_candidates: Optional[int] = None,
    oversample: int = 10,
    sort_by_announced_date: bool = False,
) -> List[SearchResult]:
    # Tokenize query using the initialized text processing config (stopwords/min_token_len/etc).
    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    # Scores per doc_id (sparse)
    scores: Dict[int, float] = {}

    # Cache dl and doc payloads to avoid repeated seeks
    dl_cache: Dict[int, int] = {}
    doc_cache: Dict[int, Dict[str, Any]] = {}

    # De-duplicate query terms to avoid over-weighting repeats
    for term in sorted(set(q_tokens)):
        term_entry = get_term_entry(term=term, inv_path=inv_path, offsets_conn=offsets_conn)
        if term_entry is None:
            continue

        idf_val = term_entry.get("idf")
        postings = term_entry.get("postings") or []

        if not isinstance(idf_val, (int, float)) or not isinstance(postings, list) or not postings:
            continue

        for pair in postings:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            try:
                doc_id = int(pair[0])
                tf = int(pair[1])
            except (TypeError, ValueError):
                continue

            if doc_id < 0 or doc_id >= int(n_docs):
                continue

            dl = dl_cache.get(doc_id)
            if dl is None:
                doc = doc_cache.get(doc_id)
                if doc is None:
                    doc = get_doc_by_id(doc_id=doc_id, docs_path=docs_path, docs_offsets_conn=docs_offsets_conn) or {}
                    if doc:
                        doc_cache[doc_id] = doc
                if not doc:
                    # If doc missing from docs_offsets.sqlite, skip scoring for it
                    continue
                try:
                    dl = int(doc.get("len", 0))
                except Exception:
                    dl = 0
                dl_cache[doc_id] = dl

            if dl <= 0:
                continue

            contrib = bm25_term_score(
                idf=float(idf_val),
                tf=tf,
                dl=dl,
                avgdl=avgdl,
                k1=k1,
                b=b,
            )
            scores[doc_id] = scores.get(doc_id, 0.0) + float(contrib)

    if not scores:
        return []

    # Candidate pruning: keep a moderately larger pool then filter/snippet
    limit = (
        max_candidates
        if (max_candidates and max_candidates > 0)
        else max(int(top_k * max(1, int(oversample))), top_k)
    )
    candidate_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)[:limit]

    title_by_entry, summary_by_entry = build_title_summary_maps(chunks_path)

    hit_rank = 0
    entry_hits: Dict[Any, Dict[str, Any]] = {}
    for doc_id in candidate_ids:
        doc = doc_cache.get(doc_id)
        if doc is None:
            doc = get_doc_by_id(doc_id=doc_id, docs_path=docs_path, docs_offsets_conn=docs_offsets_conn)
            if doc is None:
                continue
            doc_cache[doc_id] = doc

        if not doc_passes_filters(doc, filters):
            continue

        chunk_id = str(doc.get("chunk_id") or "")
        text = ""
        chunk_date = ""
        chunk_date_int = 0
        chunk_source_type = ""
        chunk_doc_type = ""
        if chunk_id:
            chunk_obj = get_chunk_by_chunk_id(
                chunk_id=chunk_id,
                chunks_path=chunks_path,
                chunks_offsets_conn=chunks_offsets_conn,
            ) or {}
            text = str(chunk_obj.get("text") or "")
            chunk_source_type = str(chunk_obj.get("source_type") or "")
            meta_obj = chunk_obj.get("meta") or {}
            attach_meta = meta_obj.get("attachment") if isinstance(meta_obj, dict) else None
            if isinstance(attach_meta, dict):
                raw_date = attach_meta.get("date")
                if raw_date:
                    chunk_date = str(raw_date)
                    chunk_date_int = _date_to_int(chunk_date)
                dt = attach_meta.get("document_type")
                if dt:
                    chunk_doc_type = str(dt)

        snippet = format_snippet(text, max_chars=snippet_chars)

        hit_rank += 1
        score_value = float(scores[doc_id])
        entry_id = doc.get("entry_id")
        if entry_id is None:
            continue

        attachment_id_raw = doc.get("attachment_id")
        attachment_id = str(attachment_id_raw) if attachment_id_raw is not None else ""
        doc_id_int = int(doc_id)

        entry_info = entry_hits.get(entry_id)
        match_info = {
            "chunk_id": chunk_id,
            "score": score_value,
            "chunk_date": chunk_date,
            "chunk_date_int": chunk_date_int,
            "hit_rank": hit_rank,
            "doc": doc,
            "doc_id": doc_id_int,
            "snippet": snippet,
            "attachment_id": attachment_id,
            "source_type": chunk_source_type,
            "document_type": chunk_doc_type,
            "text": text,
        }
        if entry_info is None:
            entry_key = str(entry_id)
            entry_info = {
                "entry_id": entry_id,
                "doc": doc,
                "doc_id": doc_id_int,
                "snippet": snippet,
                "chunk_id": chunk_id,
                "attachment_id": attachment_id,
                "score": score_value,
                "best_rank": hit_rank,
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
            entry_hits[entry_id] = entry_info
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
                    "doc": doc,
                    "doc_id": doc_id_int,
                    "snippet": snippet,
                    "chunk_id": chunk_id,
                    "attachment_id": attachment_id,
                    "score": score_value,
                    "best_rank": hit_rank,
                    "best_date_int": chunk_date_int,
                    "best_date": chunk_date,
                }
            )
        entry_info["match_count"] += 1
        entry_info["matched_chunks"].append(match_info)

        if attachment_id and attachment_id not in entry_info["matched_attachment_ids"]:
            entry_info["matched_attachment_ids"].append(attachment_id)
        if doc_id_int not in entry_info["matched_doc_ids"]:
            entry_info["matched_doc_ids"].append(doc_id_int)
        if chunk_id and chunk_id not in entry_info["matched_chunk_ids"]:
            entry_info["matched_chunk_ids"].append(chunk_id)

    results: List[SearchResult] = []
    if sort_by_announced_date:
        def _announced_int(info: Dict[str, Any]) -> int:
            doc = info.get("doc") or {}
            meta_flat = doc.get("meta") or {}
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
    for info in sorted_entries[:top_k]:
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
            try:
                info["doc_id"] = int(display_match.get("doc_id"))
            except Exception:
                pass
            info["chunk_id"] = display_match.get("chunk_id") or info.get("chunk_id")
            info["attachment_id"] = display_match.get("attachment_id") or info.get("attachment_id")
            info["best_date_int"] = int(display_match.get("chunk_date_int") or info.get("best_date_int") or 0)
            info["best_date"] = str(display_match.get("chunk_date") or info.get("best_date") or "")

        best_pdf = None
        for m in matched_sorted:
            if m.get("source_type") == "pdf_page":
                best_pdf = m
                break
        if best_pdf is not None:
            best_doc = best_pdf.get("doc") or {}
            info["relevant_chunk"] = {
                "text": format_excerpt(best_pdf.get("text"), max_chars=75),
                "source_type": "pdf_page",
                "attachment_date": best_pdf.get("chunk_date") or "",
                "document_type": best_pdf.get("document_type") or "",
                "document_id": best_doc.get("attachment_id") or "",
                "page_start": best_doc.get("page_start") or "",
                "page_end": best_doc.get("page_end") or "",
                "chunk_id": best_doc.get("chunk_id") or "",
            }
        else:
            info["relevant_chunk"] = None
        selected_source_type = top_source_type
        if not selected_source_type:
            selected_source_type = str((info.get("doc") or {}).get("source_type") or "")
        results.append(
            SearchResult(
                doc_id=info["doc_id"],
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
                source=source_label(selected_source_type),
            )
        )

    return results


def search_bm25(
    *,
    config_path: Path,
    query: str,
    top_k: int,
    filters: Dict[str, Optional[str]],
    k1: Optional[float] = None,
    b: Optional[float] = None,
    snippet_chars: int = 220,
    max_candidates: Optional[int] = None,
    oversample: int = 10,
    sort_by_announced_date: bool = False,
) -> List[SearchResult]:
    """
    Streamlined wrapper that loads config + artifacts, then runs score_query.
    Mirrors the CLI defaults and keeps setup centralized for reuse.
    """
    cfg = load_config(config_path)
    base_dir = config_path.parent.resolve()

    # Initialize tokenization config (stopwords/min_token_len/etc)
    init_text_processing_from_config(config_path)

    chunks_path = get_path(cfg, "paths.chunks_jsonl", base_dir=base_dir)
    docs_path = get_path(cfg, "paths.bm25_docs_jsonl", base_dir=base_dir)
    inv_path = get_path(cfg, "paths.bm25_inverted_index_jsonl", base_dir=base_dir)

    offsets_db_path = get_path(cfg, "paths.bm25_offsets_sqlite", base_dir=base_dir)
    docs_offsets_db_path = get_path(cfg, "paths.bm25_docs_offsets_sqlite", base_dir=base_dir)
    chunks_offsets_db_path = get_path(cfg, "paths.bm25_chunks_offsets_sqlite", base_dir=base_dir)

    corpus_stats_path = get_path(cfg, "paths.bm25_corpus_stats_json", base_dir=base_dir)

    stats = read_json(corpus_stats_path)
    n_docs = int(stats.get("n_docs", 0))
    avgdl = float(stats.get("avgdl", 0.0))
    if n_docs <= 0 or avgdl <= 0.0:
        raise RuntimeError(f"Invalid corpus_stats.json (n_docs={n_docs}, avgdl={avgdl})")

    k1_cfg = get_cfg_value(cfg, "bm25.k1")
    b_cfg = get_cfg_value(cfg, "bm25.b")
    k1_val = float(k1) if k1 is not None else (float(k1_cfg) if k1_cfg is not None else BM25_K1_DEFAULT)
    b_val = float(b) if b is not None else (float(b_cfg) if b_cfg is not None else BM25_B_DEFAULT)

    offsets_conn = _sqlite_ro(offsets_db_path)
    docs_offsets_conn = _sqlite_ro(docs_offsets_db_path)
    chunks_offsets_conn = _sqlite_ro(chunks_offsets_db_path)

    try:
        return score_query(
            query=str(query),
            top_k=int(top_k),
            k1=k1_val,
            b=b_val,
            n_docs=n_docs,
            avgdl=avgdl,
            docs_path=docs_path,
            inv_path=inv_path,
            chunks_path=chunks_path,
            offsets_conn=offsets_conn,
            docs_offsets_conn=docs_offsets_conn,
            chunks_offsets_conn=chunks_offsets_conn,
            filters=filters,
            snippet_chars=int(snippet_chars),
            max_candidates=max_candidates,
            oversample=int(oversample),
            sort_by_announced_date=sort_by_announced_date,
        )
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


# -----------------------------
# Config + CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search BM25 artifacts built from chunks.jsonl (seek-based).")
    p.add_argument("--config", default="config.json", help="Path to config.json")
    p.add_argument("--q", required=True, help="Query string")
    p.add_argument("--top_k", type=int, default=10, help="Number of results to return")
    p.add_argument("--oversample", type=int, default=10, help="FAISS candidates = top_k * oversample for pre-filter")

    # BM25 params
    p.add_argument("--k1", type=float, default=None, help="BM25 k1 parameter (defaults to config/build default)")
    p.add_argument("--b", type=float, default=None, help="BM25 b parameter (defaults to config/build default)")

    # Optional filters
    p.add_argument("--entry_id", default=None, help="Filter to a single entry_id")
    p.add_argument("--administration", default=None, help="Filter by administration (meta)")
    p.add_argument("--agency", default=None, help="Filter by agencies_affected (meta)")
    p.add_argument("--subject", default=None, help="Filter by subject_matter (meta)")
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
        help="Number of raw hits to retrieve before deduping (default=max(top_k*10, top_k))",
    )
    p.add_argument(
        "--sort_by_announced_date",
        action="store_true",
        help="Sort final results by announced_date (desc), then score",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    base_dir = config_path.parent.resolve()

    # Initialize tokenization config (stopwords/min_token_len/etc)
    tp_cfg_obj = init_text_processing_from_config(config_path)

    # Paths from config (centralized)
    chunks_path = get_path(cfg, "paths.chunks_jsonl", base_dir=base_dir)

    docs_path = get_path(cfg, "paths.bm25_docs_jsonl", base_dir=base_dir)
    inv_path = get_path(cfg, "paths.bm25_inverted_index_jsonl", base_dir=base_dir)

    offsets_db_path = get_path(cfg, "paths.bm25_offsets_sqlite", base_dir=base_dir)
    docs_offsets_db_path = get_path(cfg, "paths.bm25_docs_offsets_sqlite", base_dir=base_dir)
    chunks_offsets_db_path = get_path(cfg, "paths.bm25_chunks_offsets_sqlite", base_dir=base_dir)

    corpus_stats_path = get_path(cfg, "paths.bm25_corpus_stats_json", base_dir=base_dir)

    # Load corpus stats
    stats = read_json(corpus_stats_path)
    n_docs = int(stats.get("n_docs", 0))
    avgdl = float(stats.get("avgdl", 0.0))
    if n_docs <= 0 or avgdl <= 0.0:
        raise RuntimeError(f"Invalid corpus_stats.json (n_docs={n_docs}, avgdl={avgdl})")

    # BM25 defaults: prefer CLI override, else config, else hardcoded
    k1_cfg = get_cfg_value(cfg, "bm25.k1")
    b_cfg = get_cfg_value(cfg, "bm25.b")

    k1 = float(args.k1) if args.k1 is not None else (float(k1_cfg) if k1_cfg is not None else BM25_K1_DEFAULT)
    b = float(args.b) if args.b is not None else (float(b_cfg) if b_cfg is not None else BM25_B_DEFAULT)

    filters: Dict[str, Optional[str]] = {
        "entry_id": args.entry_id,
        "administration": args.administration,
        "agency": args.agency,
        "subject": args.subject,
    }

    # Open SQLite connections (seek maps)
    offsets_conn = _sqlite_ro(offsets_db_path)
    docs_offsets_conn = _sqlite_ro(docs_offsets_db_path)
    chunks_offsets_conn = _sqlite_ro(chunks_offsets_db_path)

    try:
        results = score_query(
            query=str(args.q),
            top_k=int(args.top_k),
            k1=k1,
            b=b,
            n_docs=n_docs,
            avgdl=avgdl,
            docs_path=docs_path,
            inv_path=inv_path,
            chunks_path=chunks_path,
            offsets_conn=offsets_conn,
            docs_offsets_conn=docs_offsets_conn,
            chunks_offsets_conn=chunks_offsets_conn,
            filters=filters,
            snippet_chars=int(args.snippet_chars),
            max_candidates=args.max_candidates,
            oversample=int(args.oversample),
            sort_by_announced_date=bool(args.sort_by_announced_date),
        )
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

    # Print summary
    print(f"Query: {args.q}")
    if any(v for v in filters.values()):
        print(f"Filters: { {k: v for k, v in filters.items() if v} }")
    print(f"BM25: k1={k1} b={b} | n_docs={n_docs} avgdl={avgdl:.2f} | stopwords={bool(tp_cfg_obj.use_stopwords)}")
    print(f"Results: {len(results)}\n")

    # Print ranked results (structured, human-readable)
    for rank, r in enumerate(results, start=1):
        doc = r.doc
        meta = doc.get("meta") or {}

        entry_id = doc.get("entry_id")
        title = r.title or meta.get("title") or ""
        meta_flat = doc.get("meta") or {}
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
            print(f"      document_id={ch.get('document_id') or ''}")
            page_start = ch.get("page_start") or ""
            page_end = ch.get("page_end") or ""
            print(f"      pages={page_start}-{page_end}")
            print(f"      id={ch.get('chunk_id') or ''}")
        else:
            print(f"    chunk=None")
        print("")


if __name__ == "__main__":
    main()
