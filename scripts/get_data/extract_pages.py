#!/usr/bin/env python3
"""
Extract page-level text from PDFs referenced by IPTP manifest.jsonl.

Manifest line schema (from assemble_policies.py):
{
  "page_ptr_id": ...,
  "wagtaildoc_id": ...,
  "policydocument_id": ...,
  "document_type": "New Policy" | "Subsequent Action" | "Commentary" | ...,
  "file_title": "...",
  "s3_bucket": "<documents-bucket>",
  "s3_key": "media/documents/....pdf",
  "dest_relpath": "policies/.../documents/...pdf"
}

We assume PDFs are already downloaded locally under:
  <root_dir>/<dest_relpath>

This script is designed to be resumable:
- It records completed PDFs in a state JSONL.
- On rerun, it skips PDFs already marked DONE.

How to run
-----------

python -m scripts.get_data.extract_pages --config config.json

(old way)
python3 scripts/extract_pages.py      

or 

python3 scripts/extract_pages.py \
  --manifest data/sample_100/manifest.jsonl \
  --root-dir data/sample_100 \
  --out data/sample_100/extracted/extracted_pages.jsonl \
  --log data/sample_100/extracted/extract_log.jsonl \
  --state data/sample_100/extracted/extract_state.jsonl

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple
from scripts.common.config import load_config, get_cfg_value

from zoneinfo import ZoneInfo

import fitz  # PyMuPDF


# ----------------------------
# Time helpers
# ----------------------------

def now_pst_iso() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class ManifestItem:
    page_ptr_id: int
    policydocument_id: Optional[int]
    wagtaildoc_id: Optional[int]
    document_type: Optional[str]
    file_title: Optional[str]
    s3_bucket: Optional[str]
    s3_key: Optional[str]
    dest_relpath: str

    @property
    def stable_id(self) -> str:
        """
        A stable identifier for this PDF attachment across runs.
        Prefer policydocument_id if present; else fall back to dest_relpath.
        """
        if self.policydocument_id is not None:
            return f"policydocument:{self.policydocument_id}"
        return f"path:{self.dest_relpath}"


# ----------------------------
# IO helpers
# ----------------------------

def cfg_default(cfg: Dict[str, Any], key: str, fallback: Any) -> Any:
    v = get_cfg_value(cfg, key)
    return fallback if v is None else v


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_manifest(manifest_path: Path) -> List[ManifestItem]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    items: List[ManifestItem] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {manifest_path}: {e}") from e

            if "dest_relpath" not in obj:
                raise ValueError(f"Manifest line {line_num} missing 'dest_relpath'")

            items.append(
                ManifestItem(
                    page_ptr_id=int(obj.get("page_ptr_id")),
                    policydocument_id=(int(obj["policydocument_id"]) if obj.get("policydocument_id") is not None else None),
                    wagtaildoc_id=(int(obj["wagtaildoc_id"]) if obj.get("wagtaildoc_id") is not None else None),
                    document_type=(str(obj["document_type"]) if obj.get("document_type") is not None else None),
                    file_title=(str(obj["file_title"]) if obj.get("file_title") is not None else None),
                    s3_bucket=(str(obj["s3_bucket"]) if obj.get("s3_bucket") is not None else None),
                    s3_key=(str(obj["s3_key"]) if obj.get("s3_key") is not None else None),
                    dest_relpath=str(obj["dest_relpath"]),
                )
            )

    return items


def load_done_set(state_path: Path) -> Set[str]:
    """
    Read state JSONL and return stable_ids marked as DONE.
    """
    done: Set[str] = set()
    if not state_path.exists():
        return done

    with state_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("status") == "DONE" and obj.get("stable_id"):
                done.add(str(obj["stable_id"]))
    return done


def load_stats_done_set(stats_path: Path) -> Set[str]:
    """
    Read stats JSONL and return stable_ids already recorded.
    """
    done: Set[str] = set()
    if not stats_path.exists():
        return done

    with stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sid = obj.get("stable_id")
            if sid:
                done.add(str(sid))
    return done


# ----------------------------
# PDF extraction
# ----------------------------

def open_pdf(pdf_path: Path) -> fitz.Document:
    try:
        return fitz.open(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from e


def guess_mostly_scanned(page_text_lengths: List[int]) -> bool:
    """
    Heuristic: if >=80% of pages have <30 characters of extracted text, treat as scanned/image-based.
    """
    if not page_text_lengths:
        return True
    short_pages = sum(1 for n in page_text_lengths if n < 30)
    return (short_pages / len(page_text_lengths)) >= 0.80


def median_int(xs: List[int]) -> int:
    if not xs:
        return 0
    ys = sorted(xs)
    mid = len(ys) // 2
    if len(ys) % 2 == 1:
        return ys[mid]
    return (ys[mid - 1] + ys[mid]) // 2


def compute_page_text_lengths(pdf_path: Path) -> List[int]:
    doc = open_pdf(pdf_path)
    lens: List[int] = []
    for i in range(len(doc)):
        txt = (doc[i].get_text("text") or "").strip()
        lens.append(len(txt))
    return lens


def iter_page_records(item: ManifestItem, pdf_path: Path) -> Iterator[Dict[str, Any]]:
    doc = open_pdf(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        text = (page.get_text("text") or "").strip()

        yield {
            "page_ptr_id": item.page_ptr_id,
            "policydocument_id": item.policydocument_id,
            "wagtaildoc_id": item.wagtaildoc_id,
            "document_type": item.document_type,
            "file_title": item.file_title,
            "s3_bucket": item.s3_bucket,
            "s3_key": item.s3_key,
            "dest_relpath": item.dest_relpath,

            "stable_id": item.stable_id,
            "page_num": i + 1,  # 1-indexed
            "text": text,

            "source_path": str(pdf_path),
        }


# ----------------------------
# Orchestration
# ----------------------------

def extract_from_manifest(
    manifest_path: Path,
    root_dir: Path,
    out_pages_jsonl: Path,
    out_log_jsonl: Path,
    state_path: Path,
    out_stats_jsonl: Path,
    skip_scanned: bool,
) -> None:
    items = load_manifest(manifest_path)
    done = load_done_set(state_path)
    stats_done = load_stats_done_set(out_stats_jsonl)

    ensure_parent_dir(out_pages_jsonl)
    ensure_parent_dir(out_log_jsonl)
    ensure_parent_dir(state_path)
    ensure_parent_dir(out_stats_jsonl)

    total = len(items)
    processed = 0
    skipped_done = 0
    missing = 0
    failed = 0
    scanned_skipped = 0
    pages_written = 0

    # Append mode so resume doesn't wipe prior output
    with out_pages_jsonl.open("a", encoding="utf-8") as out_f, out_stats_jsonl.open("a", encoding="utf-8") as stats_f:
        for idx, item in enumerate(items, start=1):
            pdf_path = root_dir / item.dest_relpath
            if not pdf_path.exists():
                missing += 1
                append_jsonl(
                    out_log_jsonl,
                    {
                        "ts": now_pst_iso(),
                        "type": "ERROR_MISSING_FILE",
                        "stable_id": item.stable_id,
                        "page_ptr_id": item.page_ptr_id,
                        "dest_relpath": item.dest_relpath,
                        "source_path": str(pdf_path),
                    },
                )
                # Record as DONE? No — keep it retriable.
                continue

            # Always compute/write stats once per stable_id, even if pages are already DONE.
            page_lengths_for_stats: Optional[List[int]] = None
            is_scanned_for_stats: Optional[bool] = None
            if item.stable_id not in stats_done:
                try:
                    page_lengths_for_stats = compute_page_text_lengths(pdf_path)
                    is_scanned_for_stats = guess_mostly_scanned(page_lengths_for_stats)

                    n_pages = len(page_lengths_for_stats)
                    total_chars = sum(page_lengths_for_stats)
                    short_pages = sum(1 for n in page_lengths_for_stats if n < 30)
                    pct_pages_lt_30 = (short_pages / n_pages) if n_pages else 1.0
                    med_chars = median_int(page_lengths_for_stats)

                    stats_rec = {
                        "ts": now_pst_iso(),
                        "stable_id": item.stable_id,
                        "page_ptr_id": item.page_ptr_id,
                        "policydocument_id": item.policydocument_id,
                        "wagtaildoc_id": item.wagtaildoc_id,
                        "document_type": item.document_type,
                        "file_title": item.file_title,
                        "s3_bucket": item.s3_bucket,
                        "s3_key": item.s3_key,
                        "dest_relpath": item.dest_relpath,
                        "source_path": str(pdf_path),

                        "pages": n_pages,
                        "total_chars": total_chars,
                        "median_chars_per_page": med_chars,
                        "pct_pages_lt_30": pct_pages_lt_30,

                        "likely_scanned": bool(is_scanned_for_stats),
                        "scan_heuristic": {"min_chars": 30, "min_frac": 0.80},
                    }
                    stats_f.write(json.dumps(stats_rec, ensure_ascii=False) + "\n")
                    stats_done.add(item.stable_id)

                    if is_scanned_for_stats:
                        append_jsonl(
                            out_log_jsonl,
                            {
                                "ts": now_pst_iso(),
                                "type": "WARN_MOSTLY_EMPTY_TEXT",
                                "stable_id": item.stable_id,
                                "page_ptr_id": item.page_ptr_id,
                                "dest_relpath": item.dest_relpath,
                                "source_path": str(pdf_path),
                                "pages": n_pages,
                            },
                        )
                except Exception as e:
                    append_jsonl(
                        out_log_jsonl,
                        {
                            "ts": now_pst_iso(),
                            "type": "ERROR_STATS_FAILED",
                            "stable_id": item.stable_id,
                            "page_ptr_id": item.page_ptr_id,
                            "dest_relpath": item.dest_relpath,
                            "source_path": str(pdf_path),
                            "error": str(e),
                        },
                    )

            # Now skip page extraction if already DONE.
            if item.stable_id in done:
                skipped_done += 1
                continue

            page_lengths: List[int] = []
            try:
                # Extract all pages
                for rec in iter_page_records(item, pdf_path):
                    page_lengths.append(len(rec["text"]))
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pages_written += 1

                is_scanned = guess_mostly_scanned(page_lengths)
                if is_scanned and skip_scanned:
                    # Here we already extracted text; keep the warning for downstream decisions.
                    scanned_skipped += 1

                # Mark completed (so we can resume safely)
                append_jsonl(
                    state_path,
                    {
                        "ts": now_pst_iso(),
                        "status": "DONE",
                        "stable_id": item.stable_id,
                        "page_ptr_id": item.page_ptr_id,
                        "dest_relpath": item.dest_relpath,
                        "source_path": str(pdf_path),
                        "pages": len(page_lengths),
                        "scanned_guess": bool(is_scanned),
                    },
                )
                done.add(item.stable_id)
                processed += 1

            except Exception as e:
                failed += 1
                append_jsonl(
                    out_log_jsonl,
                    {
                        "ts": now_pst_iso(),
                        "type": "ERROR_EXTRACT_FAILED",
                        "stable_id": item.stable_id,
                        "page_ptr_id": item.page_ptr_id,
                        "dest_relpath": item.dest_relpath,
                        "source_path": str(pdf_path),
                        "error": str(e),
                    },
                )

            # Lightweight progress
            if idx % 10 == 0 or idx == total:
                print(
                    f"Progress: {idx}/{total} | processed={processed} "
                    f"skipped_done={skipped_done} missing={missing} failed={failed} pages={pages_written}"
                )

    print("\nExtraction complete.")
    print(f"Manifest: {manifest_path} ({total} PDFs)")
    print(f"Root dir: {root_dir}")
    print(f"Pages out: {out_pages_jsonl}")
    print(f"Log out : {out_log_jsonl}")
    print(f"State   : {state_path}")
    print(f"Stats out: {out_stats_jsonl}")
    print(f"Processed PDFs : {processed}")
    print(f"Skipped (done) : {skipped_done}")
    print(f"Missing files  : {missing}")
    print(f"Failed PDFs    : {failed}")
    print(f"Pages written  : {pages_written}")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    # -------------------------
    # parse --config
    # -------------------------
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="Path to config.json (optional)")
    pre_args, _ = pre.parse_known_args()

    cfg: Dict[str, Any] = {}
    if pre_args.config:
        cfg = load_config(Path(pre_args.config))

    # -------------------------
    # Config-driven defaults
    # -------------------------
    root_dir_default = str(cfg_default(cfg, "assemble.out_dir", "data/sample_100"))
    extract_dir_default = str(Path(root_dir_default) / "extract_pages")

    manifest_default = str(
        cfg_default(
            cfg,
            "paths.manifest_jsonl",
            str(Path(root_dir_default) / "manifest.jsonl"),
        )
    )

    out_default = str(
        cfg_default(
            cfg,
            "paths.extracted_pages_jsonl",
            str(Path(extract_dir_default) / "extracted_pages.jsonl"),
        )
    )

    log_default = str(
        cfg_default(
            cfg,
            "paths.extract_pages_log_jsonl",
            str(Path(extract_dir_default) / "extract_log.jsonl"),
        )
    )

    state_default = str(
        cfg_default(
            cfg,
            "paths.extract_pages_state_jsonl",
            str(Path(extract_dir_default) / "extract_state.jsonl"),
        )
    )

    stats_default = str(
        cfg_default(
            cfg,
            "paths.extract_pages_stats_jsonl",
            str(Path(extract_dir_default) / "pdf_stats.jsonl"),
        )
    )

    skip_scanned_default = bool(
        cfg_default(cfg, "extract_pages.skip_scanned", False)
    )

    # -------------------------
    # full parser
    # -------------------------
    p = argparse.ArgumentParser(
        description="Extract page-level text from IPTP PDFs via manifest.jsonl."
    )

    p.add_argument("--config", default=pre_args.config, help="Path to config.json (optional)")

    p.add_argument(
        "--manifest",
        default=manifest_default,
        help="Path to manifest.jsonl produced by assemble_policies.py",
    )

    p.add_argument(
        "--root-dir",
        default=root_dir_default,
        help="Root directory that contains dest_relpath files",
    )

    p.add_argument(
        "--out",
        default=out_default,
        help="Output JSONL path for extracted pages (append mode for resume)",
    )

    p.add_argument(
        "--log",
        default=log_default,
        help="Output JSONL path for warnings/errors",
    )

    p.add_argument(
        "--state",
        default=state_default,
        help="State JSONL tracking completed PDFs for safe resume",
    )

    p.add_argument(
        "--stats",
        default=stats_default,
        help="Output JSONL path for per-PDF stats",
    )

    # BooleanOptionalAction allows:
    # --skip-scanned and --no-skip-scanned
    p.add_argument(
        "--skip-scanned",
        action=argparse.BooleanOptionalAction,
        default=skip_scanned_default,
        help="Mark scanned PDFs; extraction still runs but warning is logged.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    extract_from_manifest(
        manifest_path=Path(args.manifest),
        root_dir=Path(args.root_dir),
        out_pages_jsonl=Path(args.out),
        out_log_jsonl=Path(args.log),
        state_path=Path(args.state),
        out_stats_jsonl=Path(args.stats),
        skip_scanned=args.skip_scanned,
    )


if __name__ == "__main__":
    main()
