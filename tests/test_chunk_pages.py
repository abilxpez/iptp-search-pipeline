# Sanity tests for chunk_pages.py
# These tests verify structural correctness, not exact chunk counts.
#
# This file includes:
#   (A) Real-output smoke/sanity checks for data/chunks/chunks.jsonl
#   (B) Focused regression tests for the 9 recent improvements (using tmp_path)
#
# Notes:
#   - The new chunk_id format is: "{entry}_{attachment}_p####-p####_h{16hex}"
#   - The regression tests run chunk_pages_pipeline in a temp directory with tiny input JSONL.

import json
import re
from pathlib import Path
from collections import Counter

import pytest

import scripts.chunk_pages as chunk_pages

# ---------------------------
# (A) Smoke tests on real output file
# ---------------------------

CHUNKS_PATH = Path("data/chunks/chunks.jsonl")


def _read_jsonl(path: Path):
    """Read a JSONL file into a list of dicts (skips blank lines)."""
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# Ensure the chunking pipeline produced an output file with content.
# This is a basic smoke test that the pipeline ran end-to-end.
def test_chunks_file_exists_and_nonempty():
    assert CHUNKS_PATH.exists(), "chunks.jsonl was not created"
    assert CHUNKS_PATH.stat().st_size > 0, "chunks.jsonl is empty"
    print("test_chunks_file_exists_and_nonempty passed")


# Validate that every chunk has the required fields and correct types.
# This protects against schema drift and partial writes.
def test_chunk_schema_and_types():
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            # Required keys
            for key in [
                "chunk_id",
                "entry_id",
                "attachment_id",
                "page_start",
                "page_end",
                "text",
                "source_path",
                "meta",
            ]:
                assert key in obj, f"Missing key: {key}"

            # Type checks
            assert isinstance(obj["chunk_id"], str)
            assert isinstance(obj["entry_id"], str)
            assert isinstance(obj["attachment_id"], str)
            assert isinstance(obj["page_start"], int)
            assert isinstance(obj["page_end"], int)
            assert isinstance(obj["text"], str)
            assert isinstance(obj["meta"], dict)

    print("test_chunk_schema_and_types passed")


# Ensure no chunk has empty or whitespace-only text.
# Empty chunks silently break retrieval quality.
def test_no_empty_chunk_text():
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            assert obj["text"].strip(), f"Empty text in {obj['chunk_id']}"

    print("test_no_empty_chunk_text passed")


# Validate chunk_id encodes page bounds correctly.
# This ensures chunk_id and payload stay in sync.
#
# Updated: chunk_id now ends with "_h{16hex}" instead of "_c###".
def test_chunk_id_page_bounds_match_payload():
    pattern = re.compile(r"_p(\d{4})-p(\d{4})_h([0-9a-f]{16})$")

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            m = pattern.search(obj["chunk_id"])
            assert m, f"Bad chunk_id format: {obj['chunk_id']}"

            start = int(m.group(1))
            end = int(m.group(2))

            assert start == obj["page_start"]
            assert end == obj["page_end"]
            assert start <= end

    print("test_chunk_id_page_bounds_match_payload passed")


# Ensure all chunk_ids are globally unique.
# Duplicate IDs indicate counter or grouping bugs.
def test_no_duplicate_chunk_ids():
    ids = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["chunk_id"])

    counts = Counter(ids)
    duplicates = [cid for cid, c in counts.items() if c > 1]

    assert not duplicates, f"Duplicate chunk_ids found: {duplicates}"
    print("test_no_duplicate_chunk_ids passed")


# Check that chunks within a PDF are in non-decreasing page order.
# This guards against grouping or ordering regressions.
def test_chunks_are_ordered_within_pdf():
    grouped = {}

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            key = (obj["entry_id"], obj["attachment_id"])
            grouped.setdefault(key, []).append(obj)

    for (entry_id, attachment_id), chunks in grouped.items():
        prev = None
        for c in chunks:
            cur = (c["page_start"], c["page_end"])
            if prev is not None:
                assert cur >= prev, (
                    f"Out-of-order chunk in {entry_id}/{attachment_id}: "
                    f"{prev} -> {cur}"
                )
            prev = cur

    print("test_chunks_are_ordered_within_pdf passed")


# Verify metadata is minimal and does not silently bloat.
# This protects index size and future performance.
def test_meta_is_minimal():
    allowed_keys = {
        "iptp_id",
        "title",
        "administration",
        "original_date_announced",
        "effective_date",
        "agencies_affected",
        "subject_matter",
    }

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta_keys = set(obj["meta"].keys())
            assert meta_keys.issubset(
                allowed_keys
            ), f"Unexpected meta keys: {meta_keys - allowed_keys}"

    print("test_meta_is_minimal passed")


# ---------------------------
# (B) Regression tests for the 9 improvements (tmp_path)
# ---------------------------

def _write_jsonl(path: Path, rows):
    """Write rows (dicts) to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_manual_entries(path: Path, entries=None):
    """Create a minimal manual_entries.json needed by chunk_pages_pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if entries is None:
        entries = [{"entry_id": "E1", "title": "t1"}]
    path.write_text(json.dumps(entries), encoding="utf-8")


def _run_pipeline_in_tmp(
    tmp_path: Path,
    *,
    pages_rows,
    target_chars=200,
    min_chars=1,
):
    """
    Run chunk_pages_pipeline in a temp directory.

    Returns a dict of important paths:
      - out: chunks.jsonl
      - log: chunk_log.jsonl
      - bad_keys: bad_keys.jsonl
    """
    entries = tmp_path / "manual_entries.json"
    pages = tmp_path / "extracted_pages.jsonl"
    out = tmp_path / "chunks.jsonl"
    log = tmp_path / "chunk_log.jsonl"

    _write_manual_entries(entries)
    _write_jsonl(pages, pages_rows)

    chunk_pages.chunk_pages_pipeline(
        manual_entries_path=entries,
        extracted_pages_path=pages,
        out_chunks_path=out,
        target_chars=int(target_chars),
        min_chars=int(min_chars),
        log_path=log,
    )

    return {
        "entries": entries,
        "pages": pages,
        "out": out,
        "log": log,
        "bad_keys": out.parent / "bad_keys.jsonl",
    }


def _filter_events(events, type_):
    return [e for e in events if e.get("type") == type_]


# 1) Unsorted input can create duplicate/overlapping chunks + non-unique chunk IDs
# Fix: If a key reappears after being closed:
#   - log ERROR_UNSORTED_INPUT
#   - write key once to bad_keys.jsonl
#   - skip processing that reappearing group (but keep global run going)
def test_unsorted_input_skips_reappearing_key_and_records_bad_key_once(tmp_path):
    rows = [
        {"entry_id": "E1", "attachment_id": "A1", "page_num": 1, "text": "hello world.", "source_path": "p1.pdf"},
        {"entry_id": "E2", "attachment_id": "A2", "page_num": 1, "text": "another doc text.", "source_path": "p2.pdf"},
        # E1/A1 reappears later -> unsorted input scenario
        {"entry_id": "E1", "attachment_id": "A1", "page_num": 2, "text": "more text.", "source_path": "p1.pdf"},
        {"entry_id": "E1", "attachment_id": "A1", "page_num": 3, "text": "even more.", "source_path": "p1.pdf"},
    ]
    paths = _run_pipeline_in_tmp(tmp_path, pages_rows=rows, target_chars=30, min_chars=1)

    logs = _read_jsonl(paths["log"])
    assert any(
        e.get("type") == "ERROR_UNSORTED_INPUT" and e.get("entry_id") == "E1" and e.get("attachment_id") == "A1"
        for e in logs
    ), "Expected ERROR_UNSORTED_INPUT log event"

    bad_keys = _read_jsonl(paths["bad_keys"])
    # Key should be recorded exactly once (no spam)
    assert [b for b in bad_keys if b["entry_id"] == "E1" and b["attachment_id"] == "A1"] == [
        {"entry_id": "E1", "attachment_id": "A1"}
    ]

    chunks = _read_jsonl(paths["out"])
    # Ensure we did NOT write chunks for the reappearing segment (pages 2/3).
    # The initial E1/A1 page 1 may have produced chunks; the later pages must not.
    assert not any(
        c["entry_id"] == "E1" and c["attachment_id"] == "A1" and c["page_end"] >= 2
        for c in chunks
    ), "Reappearing key should be skipped (no chunks covering later pages)"


# 2) iter_pages() accepts page_num <= 0
# Fix: skip and log SKIP_NONPOSITIVE_PAGE_NUM
def test_iter_pages_skips_nonpositive_page_numbers(tmp_path):
    rows = [
        {"entry_id": "E1", "attachment_id": "A1", "page_num": 0, "text": "bad", "source_path": "p.pdf"},
        {"entry_id": "E1", "attachment_id": "A1", "page_num": -2, "text": "bad2", "source_path": "p.pdf"},
        {"entry_id": "E1", "attachment_id": "A1", "page_num": 1, "text": "good text.", "source_path": "p.pdf"},
    ]
    paths = _run_pipeline_in_tmp(tmp_path, pages_rows=rows, target_chars=25, min_chars=1)

    logs = _read_jsonl(paths["log"])
    nonpos = _filter_events(logs, "SKIP_NONPOSITIVE_PAGE_NUM")
    assert {e["page_num"] for e in nonpos} == {0, -2}

    chunks = _read_jsonl(paths["out"])
    assert any(c["page_start"] == 1 for c in chunks), "Valid page should still produce chunks"


# 3) Mixed source_path within the same (entry_id, attachment_id)
# Fix: fail-fast locally -> log ERROR_MIXED_SOURCE_PATH, record key once, skip group
def test_mixed_source_path_group_is_skipped_and_key_recorded(tmp_path):
    rows = [
        {"entry_id": "E1", "attachment_id": "A1", "page_num": 1, "text": "text 1.", "source_path": "p1.pdf"},
        {"entry_id": "E1", "attachment_id": "A1", "page_num": 2, "text": "text 2.", "source_path": "p2.pdf"},
    ]
    paths = _run_pipeline_in_tmp(tmp_path, pages_rows=rows, target_chars=20, min_chars=1)

    chunks = _read_jsonl(paths["out"])
    assert not any(c["entry_id"] == "E1" and c["attachment_id"] == "A1" for c in chunks), (
        "Group with mixed source_path should be skipped"
    )

    logs = _read_jsonl(paths["log"])
    mixed = _filter_events(logs, "ERROR_MIXED_SOURCE_PATH")
    assert mixed, "Expected ERROR_MIXED_SOURCE_PATH log event"
    assert "source_paths" in mixed[0] and len(mixed[0]["source_paths"]) == 2, "Should include both source paths"

    bad_keys = _read_jsonl(paths["bad_keys"])
    assert {"entry_id": "E1", "attachment_id": "A1"} in bad_keys


# 4) Oversized unit produces oversized chunk (embedding risk)
# Fix: split oversized units + log WARN_OVERSIZED_UNIT
#
# We monkeypatch thresholds down so the test runs quickly and deterministically.
def test_oversized_unit_is_split_and_warned(tmp_path, monkeypatch):
    # Make "oversized" easy to trigger in tests
    monkeypatch.setattr(chunk_pages, "MAX_UNIT_CHARS", 100)
    monkeypatch.setattr(chunk_pages, "HARD_SPLIT_CHARS", 50)

    # No sentence punctuation => split_into_units likely produces a single unit
    huge = "A" * 260

    rows = [
        {"entry_id": "E1", "attachment_id": "A1", "page_num": 1, "text": huge, "source_path": "p.pdf"},
    ]
    paths = _run_pipeline_in_tmp(tmp_path, pages_rows=rows, target_chars=60, min_chars=1)

    logs = _read_jsonl(paths["log"])
    assert _filter_events(logs, "WARN_OVERSIZED_UNIT"), "Expected WARN_OVERSIZED_UNIT log event"

    chunks = _read_jsonl(paths["out"])
    assert chunks, "Expected at least one chunk from oversized input"

    # We don't assert exact splitting boundaries; we assert it didn't remain one giant unit-driven chunk.
    # In practice chunk sizes should be bounded by target packing + hard windows.
    assert max(len(c["text"]) for c in chunks) < len(huge), (
        "Oversized unit should not remain as a single massive chunk"
    )


# 5) clean_text() collapses all whitespace (quality hit)
# Fix: preserve paragraph structure via "\n\n" and avoid smashing everything.
def test_clean_text_preserves_paragraph_breaks_and_unwraps_lines():
    raw = "Line one wrapped\nline continues\n\n\nNew para here\nstill para"
    cleaned = chunk_pages.clean_text(raw)

    # Paragraph breaks preserved
    assert "\n\n" in cleaned, "Expected paragraph breaks to be preserved"

    # Single line breaks inside a paragraph should become spaces
    assert "wrapped line continues" in cleaned, "Expected line-wrapped text to be unwrapped into a sentence"


# 7) SKIP_BAD_RECORD logging flags don’t match actual skip condition
# Fix: log booleans for "missing_or_invalid" based on emptiness/missing, not None-only.
def test_skip_bad_record_logs_correct_missing_or_invalid_flags(tmp_path):
    rows = [
        # empty entry_id => should be considered missing/invalid
        {"entry_id": "", "attachment_id": "A1", "page_num": 1, "text": "x", "source_path": "p.pdf"},
        # empty attachment_id => should be considered missing/invalid
        {"entry_id": "E1", "attachment_id": "", "page_num": 1, "text": "x", "source_path": "p.pdf"},
        # missing page_num => should be considered missing/invalid
        {"entry_id": "E1", "attachment_id": "A1", "text": "x", "source_path": "p.pdf"},
        # one valid record so pipeline still produces output
        {"entry_id": "E1", "attachment_id": "A1", "page_num": 1, "text": "good text.", "source_path": "p.pdf"},
    ]
    paths = _run_pipeline_in_tmp(tmp_path, pages_rows=rows, target_chars=20, min_chars=1)

    logs = _read_jsonl(paths["log"])
    bad = _filter_events(logs, "SKIP_BAD_RECORD")
    assert len(bad) == 3, "Expected three SKIP_BAD_RECORD events"

    # Each event includes missing_or_invalid dict; verify at least one event flags each field as missing/invalid.
    assert any(e["missing_or_invalid"]["entry_id"] is True for e in bad)
    assert any(e["missing_or_invalid"]["attachment_id"] is True for e in bad)
    assert any(e["missing_or_invalid"]["page_num"] is True for e in bad)


# 8) load_manual_entries() doesn’t check file existence
# Fix: raise clear FileNotFoundError
def test_load_manual_entries_missing_file_raises(tmp_path):
    missing = tmp_path / "manual_entries.json"
    with pytest.raises(FileNotFoundError) as exc:
        chunk_pages.load_manual_entries(missing)
    assert "manual_entries.json not found" in str(exc.value)


# 9) Duplicate/unused write path removed (maintenance risk)
# Fix: ensure we only rely on write_chunk_jsonl; avoid references to removed write_chunks_jsonl.
def test_writer_api_is_write_chunk_jsonl_and_old_writer_is_absent():
    # New function should exist
    assert hasattr(chunk_pages, "write_chunk_jsonl"), "Expected write_chunk_jsonl to exist"

    # Old/removed function should not be used; this guards against stale tests or imports.
    assert not hasattr(chunk_pages, "write_chunks_jsonl"), (
        "write_chunks_jsonl should be removed to avoid duplicate write paths"
    )


# 10) Chunk ID stability depends on filtering (IDs can shift across runs)
# Fix: chunk_id now includes a stable hash suffix derived from text + identifiers.
#
# This test checks:
#   - running the same input twice yields the same chunk_ids
#   - format includes _h + 16 hex chars
def test_chunk_id_is_stable_across_runs_and_has_hash_suffix(tmp_path):
    rows = [
        {
            "entry_id": "E1",
            "attachment_id": "A1",
            "page_num": 1,
            "text": "Hello world. Second sentence!",
            "source_path": "p.pdf",
        },
    ]

    p1 = _run_pipeline_in_tmp(tmp_path / "run1", pages_rows=rows, target_chars=30, min_chars=1)
    p2 = _run_pipeline_in_tmp(tmp_path / "run2", pages_rows=rows, target_chars=30, min_chars=1)

    c1 = _read_jsonl(p1["out"])
    c2 = _read_jsonl(p2["out"])

    assert [c["chunk_id"] for c in c1] == [c["chunk_id"] for c in c2], "chunk_id should be deterministic across runs"

    # Validate new format: _h + 16 hex chars at end
    pat = re.compile(r"_h([0-9a-f]{16})$")
    for c in c1:
        assert pat.search(c["chunk_id"]), f"chunk_id missing stable hash suffix: {c['chunk_id']}"
