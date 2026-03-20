# test_extract_pages.py
#
# Pytest tests for the IPTP "extract_pages" pipeline.
#
# IMPORTANT:
#
# These tests generate small PDFs on the fly using PyMuPDF (fitz).

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pytest
import fitz  # PyMuPDF

from scripts.extract_pages import (
    extract_all_pages,
)

# -----------------------------
# Helpers
# -----------------------------


def _write_manual_entries(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_pdf(path: Path, page_texts: list[str]) -> None:
    """Create a PDF at `path` with one page per element in `page_texts`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text is not None and text != "":
            # Put text somewhere visible so get_text("text") retrieves it reliably.
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _read_jsonl(path: Path) -> list[dict]:
    lines = []
    if not path.exists():
        return lines
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))
    return lines


def _group_pages_by_attachment(page_recs: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in page_recs:
        grouped[(r["entry_id"], r["attachment_id"])].append(r)
    return grouped


def _count_logs_by_type(log_recs: list[dict]) -> dict[str, int]:
    counts = defaultdict(int)
    for r in log_recs:
        counts[r.get("type")] += 1
    return dict(counts)


def _has_pages_for(page_recs: list[dict], entry_id: str, attachment_id: str) -> bool:
    return any(r["entry_id"] == entry_id and r["attachment_id"] == attachment_id for r in page_recs)


def _long_text(n: int = 31) -> str:
    # Ensure length >= 31 so it is NOT considered "short" (<30).
    return "X" * n


def _short_text() -> str:
    return "hi"


# -----------------------------
# Pytest fixtures
# -----------------------------


@pytest.fixture()
def paths(tmp_path: Path):
    """Standard locations inside a temp dir."""
    entries_path = tmp_path / "data" / "manual_entries.json"
    out_pages = tmp_path / "data" / "extracted_pages.jsonl"
    out_log = tmp_path / "data" / "extract_log.jsonl"
    return entries_path, out_pages, out_log


# -----------------------------
# 12 Most important tests
# -----------------------------


def test_01_output_jsonl_schema_and_types(paths):
    """
    Test: Every output JSONL line is valid JSON and contains the required keys + correct types.
    This catches schema drift and serialization problems early.
    """
    entries_path, out_pages, out_log = paths

    pdf = entries_path.parent / "ok.pdf"
    _make_pdf(pdf, ["Hello world", "Second page"])

    _write_manual_entries(
        entries_path,
        [
            {
                "entry_id": "E1",
                "attachments": [{"attachment_id": "A1", "pdf_path": str(pdf), "is_scanned": False}],
            }
        ],
    )

    extract_all_pages(entries_path, out_pages, out_log)

    page_recs = _read_jsonl(out_pages)
    assert len(page_recs) == 2

    required = {"entry_id", "attachment_id", "page_num", "text", "source_path"}
    for r in page_recs:
        assert required.issubset(r.keys())
        assert isinstance(r["entry_id"], str)
        assert isinstance(r["attachment_id"], str)
        assert isinstance(r["page_num"], int)
        assert isinstance(r["text"], str)
        assert isinstance(r["source_path"], str)
        assert r["page_num"] >= 1
        assert r["text"] is not None


def test_02_page_numbering_is_1_indexed_contiguous_no_duplicates(paths):
    """
    Test: For each attachment, page_num is 1-indexed, contiguous, and has no duplicates.
    This catches off-by-one errors and incomplete extraction loops.
    """
    entries_path, out_pages, out_log = paths

    pdf = entries_path.parent / "three_pages.pdf"
    _make_pdf(pdf, ["p1", "p2", "p3"])

    _write_manual_entries(
        entries_path,
        [{"entry_id": "E1", "attachments": [{"attachment_id": "A1", "pdf_path": str(pdf)}]}],
    )

    extract_all_pages(entries_path, out_pages, out_log)

    page_recs = _read_jsonl(out_pages)
    grouped = _group_pages_by_attachment(page_recs)
    recs = grouped[("E1", "A1")]

    nums = sorted(r["page_num"] for r in recs)
    assert nums == [1, 2, 3]  # 1-indexed and contiguous


def test_03_known_fixture_pdf_page_count(paths):
    """
    Test: A known fixture PDF with N pages produces exactly N JSONL page records.
    This is a direct correctness test for the extraction loop.
    """
    entries_path, out_pages, out_log = paths

    pdf = entries_path.parent / "five_pages.pdf"
    _make_pdf(pdf, ["1", "2", "3", "4", "5"])

    _write_manual_entries(
        entries_path,
        [{"entry_id": "E1", "attachments": [{"attachment_id": "A1", "pdf_path": str(pdf)}]}],
    )

    extract_all_pages(entries_path, out_pages, out_log)

    page_recs = _read_jsonl(out_pages)
    assert sum(1 for r in page_recs if r["entry_id"] == "E1" and r["attachment_id"] == "A1") == 5


def test_04_scanned_attachments_are_skipped_and_logged(paths):
    """
    Test: If is_scanned=true, the attachment is skipped (no pages written) and a SKIP_SCANNED log is produced.
    This validates the early skip path and logging.
    """
    entries_path, out_pages, out_log = paths

    pdf = entries_path.parent / "scanned.pdf"
    _make_pdf(pdf, ["This text should never be extracted"])

    _write_manual_entries(
        entries_path,
        [
            {
                "entry_id": "E1",
                "attachments": [{"attachment_id": "A1", "pdf_path": str(pdf), "is_scanned": True}],
            }
        ],
    )

    extract_all_pages(entries_path, out_pages, out_log)

    page_recs = _read_jsonl(out_pages)
    log_recs = _read_jsonl(out_log)

    assert not _has_pages_for(page_recs, "E1", "A1")
    assert any(r.get("type") == "SKIP_SCANNED" and r.get("entry_id") == "E1" and r.get("attachment_id") == "A1" for r in log_recs)


def test_05_missing_file_is_logged_and_run_continues(paths):
    """
    Test: Missing PDF path is logged (ERROR_MISSING_FILE) and does not crash the run.
    Also checks that subsequent valid attachments still extract.
    """
    entries_path, out_pages, out_log = paths

    missing = entries_path.parent / "does_not_exist.pdf"
    ok_pdf = entries_path.parent / "ok.pdf"
    _make_pdf(ok_pdf, ["ok page 1", "ok page 2"])

    _write_manual_entries(
        entries_path,
        [
            {"entry_id": "E1", "attachments": [{"attachment_id": "M1", "pdf_path": str(missing)}]},
            {"entry_id": "E2", "attachments": [{"attachment_id": "A1", "pdf_path": str(ok_pdf)}]},
        ],
    )

    extract_all_pages(entries_path, out_pages, out_log)

    page_recs = _read_jsonl(out_pages)
    log_recs = _read_jsonl(out_log)

    # Missing file produces no pages and logs an error
    assert not _has_pages_for(page_recs, "E1", "M1")
    assert any(r.get("type") == "ERROR_MISSING_FILE" and r.get("entry_id") == "E1" and r.get("attachment_id") == "M1" for r in log_recs)

    # Subsequent valid PDF still extracted
    assert _has_pages_for(page_recs, "E2", "A1")


def test_06_corrupt_pdf_is_logged_and_run_continues(paths):
    """
    Test: A corrupt/unopenable PDF triggers ERROR_EXTRACT_FAILED and does not stop later attachments.
    This validates exception handling inside the per-attachment extraction block.
    """
    entries_path, out_pages, out_log = paths

    corrupt = entries_path.parent / "corrupt.pdf"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"%PDF-not-a-real-pdf\x00\x01\x02garbage")

    ok_pdf = entries_path.parent / "ok.pdf"
    _make_pdf(ok_pdf, ["ok page 1"])

    _write_manual_entries(
        entries_path,
        [
            {"entry_id": "E1", "attachments": [{"attachment_id": "C1", "pdf_path": str(corrupt)}]},
            {"entry_id": "E2", "attachments": [{"attachment_id": "A1", "pdf_path": str(ok_pdf)}]},
        ],
    )

    extract_all_pages(entries_path, out_pages, out_log)

    page_recs = _read_jsonl(out_pages)
    log_recs = _read_jsonl(out_log)

    assert not _has_pages_for(page_recs, "E1", "C1")
    assert any(
        r.get("type") == "ERROR_EXTRACT_FAILED"
        and r.get("entry_id") == "E1"
        and r.get("attachment_id") == "C1"
        and isinstance(r.get("error"), str)
        and len(r.get("error")) > 0
        for r in log_recs
    )
    assert _has_pages_for(page_recs, "E2", "A1")


def test_07_warn_mostly_empty_text_triggers_at_or_above_80_percent(paths):
    """
    Test: If >=80% of pages have <30 chars of text, WARN_MOSTLY_EMPTY_TEXT is logged.
    This validates the heuristic and the page_lengths tracking.
    """
    entries_path, out_pages, out_log = paths

    # 10 pages: 8 short, 2 long => 80% short => should warn
    pdf = entries_path.parent / "mostly_empty_80.pdf"
    texts = [_short_text()] * 8 + [_long_text()] * 2
    _make_pdf(pdf, texts)

    _write_manual_entries(
        entries_path,
        [{"entry_id": "E1", "attachments": [{"attachment_id": "A1", "pdf_path": str(pdf)}]}],
    )

    extract_all_pages(entries_path, out_pages, out_log)
    log_recs = _read_jsonl(out_log)

    assert any(
        r.get("type") == "WARN_MOSTLY_EMPTY_TEXT"
        and r.get("entry_id") == "E1"
        and r.get("attachment_id") == "A1"
        and r.get("pages") == 10
        for r in log_recs
    )


def test_08_warn_mostly_empty_text_boundary_79_percent_does_not_warn(paths):
    """
    Test: Boundary check below threshold: 79% short pages should NOT trigger warning.
    We use 100 pages so we can hit exactly 79%.
    """
    entries_path, out_pages, out_log = paths

    # 100 pages: 79 short, 21 long => 79% short => should NOT warn
    pdf = entries_path.parent / "mostly_empty_79.pdf"
    texts = [_short_text()] * 79 + [_long_text()] * 21
    _make_pdf(pdf, texts)

    _write_manual_entries(
        entries_path,
        [{"entry_id": "E1", "attachments": [{"attachment_id": "A1", "pdf_path": str(pdf)}]}],
    )

    extract_all_pages(entries_path, out_pages, out_log)
    log_recs = _read_jsonl(out_log)

    assert not any(
        r.get("type") == "WARN_MOSTLY_EMPTY_TEXT" and r.get("entry_id") == "E1" and r.get("attachment_id") == "A1"
        for r in log_recs
    )


def test_09_is_scanned_must_be_boolean_raises_error(paths):
    """
    Test: Enforces strict boolean semantics for is_scanned.
    If is_scanned is provided but is not a boolean (e.g. "false"),
    iter_attachments should raise a ValueError and fail fast.
    """
    entries_path, out_pages, out_log = paths

    pdf = entries_path.parent / "string_false.pdf"
    _make_pdf(pdf, ["this should not silently skip"])

    _write_manual_entries(
        entries_path,
        [
            {
                "entry_id": "E1",
                "attachments": [
                    {
                        "attachment_id": "A1",
                        "pdf_path": str(pdf),
                        "is_scanned": "false",  # invalid: string instead of boolean
                    }
                ],
            }
        ],
    )

    with pytest.raises(ValueError, match="non-boolean 'is_scanned'"):
        extract_all_pages(entries_path, out_pages, out_log)



def test_10_default_attachment_id_is_used_consistently(paths):
    """
    Test: If attachment_id is missing, it defaults to pdf_1, pdf_2, ...
    This prevents downstream index mismatches between pages/logs and later pipeline steps.
    """
    entries_path, out_pages, out_log = paths

    pdf = entries_path.parent / "no_attachment_id.pdf"
    _make_pdf(pdf, ["p1", "p2"])

    _write_manual_entries(
        entries_path,
        [{"entry_id": "E1", "attachments": [{"pdf_path": str(pdf)}]}],  # no attachment_id provided
    )

    extract_all_pages(entries_path, out_pages, out_log)
    page_recs = _read_jsonl(out_pages)

    # Defaults to pdf_1 for the first attachment
    assert _has_pages_for(page_recs, "E1", "pdf_1")
    nums = sorted(r["page_num"] for r in page_recs if r["entry_id"] == "E1" and r["attachment_id"] == "pdf_1")
    assert nums == [1, 2]


def test_11_no_log_mode_does_not_create_log_file(paths):
    """
    Test: When out_log_jsonl=None, the run still succeeds and no log file is created.
    This checks that logging is truly optional.
    """
    entries_path, out_pages, out_log = paths

    pdf = entries_path.parent / "ok.pdf"
    _make_pdf(pdf, ["ok"])

    _write_manual_entries(entries_path, [{"entry_id": "E1", "attachments": [{"attachment_id": "A1", "pdf_path": str(pdf)}]}])

    extract_all_pages(entries_path, out_pages, out_log_jsonl=None)

    page_recs = _read_jsonl(out_pages)
    assert len(page_recs) == 1
    assert not out_log.exists()


def test_12_output_overwrite_semantics_no_duplicate_pages_on_rerun(paths):
    """
    Test: out_pages_jsonl is opened with 'w', so rerunning should overwrite rather than append.
    This ensures repeated runs don't silently duplicate data.
    """
    entries_path, out_pages, out_log = paths

    pdf = entries_path.parent / "ok.pdf"
    _make_pdf(pdf, ["p1", "p2", "p3"])

    _write_manual_entries(entries_path, [{"entry_id": "E1", "attachments": [{"attachment_id": "A1", "pdf_path": str(pdf)}]}])

    # First run
    extract_all_pages(entries_path, out_pages, out_log)
    recs1 = _read_jsonl(out_pages)
    assert len(recs1) == 3

    # Second run (should overwrite, still 3 records)
    extract_all_pages(entries_path, out_pages, out_log)
    recs2 = _read_jsonl(out_pages)
    assert len(recs2) == 3

    # Sanity: still the same 1..3 pages
    nums = sorted(r["page_num"] for r in recs2)
    assert nums == [1, 2, 3]
