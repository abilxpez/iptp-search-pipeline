# Text Preparation Pipeline (PDFs → Chunks)

This pipeline turns IPTP PDFs into clean, chunk-level text records that are ready for indexing and search.

It runs in three stages:

1) `extract_pages.py` → PDF files → `extracted_pages.jsonl` (one JSON object per page)  
2) `chunk_pages.py` → pages → `chunks.jsonl` (one JSON object per chunk)  
3) `text_processing.py` → shared normalization/tokenization/stopword utilities used downstream

Each stage is designed to be runnable on its own. Logs are written as JSONL so failures don’t stop the entire run.

---

## 1) extract_pages.py

### Purpose
`extract_pages.py` reads `data/manual_entries.json`, iterates each PDF attachment, and extracts **page-level text** using PyMuPDF (`fitz`). It writes a streaming JSONL output so we don’t load the corpus into memory.

This stage is intentionally “lossless-ish”: it preserves per-page boundaries and keeps the raw extracted text, since later stages handle cleanup + chunking.

### Inputs
- `data/manual_entries.json`
  - Each entry must contain `entry_id`
  - Each entry may contain `attachments` (list)
  - Each attachment should contain:
    - `pdf_path` (required)
    - `attachment_id` (optional; defaults to `pdf_1`, `pdf_2`, ...)
    - `is_scanned` (optional; defaults to false)

### Outputs
- `data/extracted/extracted_pages.jsonl`
  - One line per page, schema:
    - `entry_id` (str)
    - `attachment_id` (str)
    - `page_num` (int, 1-indexed)
    - `text` (str)
    - `source_path` (str)

- Optional: `data/extracted/extract_log.jsonl`
  - JSONL events with:
    - `ts` (PST/PDT ISO timestamp)
    - `type` (event type)
    - identifiers like `entry_id`, `attachment_id`, `source_path`
    - and sometimes `error`, `pages`, etc.

### Extraction Strategy (High-Level)

1. **Load the PDF manifest**
   - Read `data/manual_entries.json` to get `entry_id` and attachment metadata.

2. **Iterate attachments defensively**
   - Validate required fields and assign default `attachment_id`s when missing.

3. **Skip known problem cases early**
   - Attachments marked `is_scanned=true` are skipped and optionally logged.
   - Missing PDF files are logged and skipped without stopping the run.
   - This makes the extraction step robust during early prototyping when metadata or files may be incomplete. 

4. **Extract page-level text**
   - Open each PDF with PyMuPDF.
   - Extract plain text per page and write **one JSON object per page** to `extracted_pages.jsonl` (streaming, no full-corpus loads).

5. **Warn on likely scanned PDFs**
   - If most pages are near-empty, log a warning to flag PDFs that may need OCR later. This does not stop the pipeline. 
   - logs: `type=WARN_MOSTLY_EMPTY_TEXT`

6. **Fail safely**
   - Any extraction error is logged, and the pipeline continues with the next PDF.
   - logged as: `type=ERROR_EXTRACT_FAILED`
   - The script continues to the next PDF so a single bad file does not derail the run.

7. **Emit quick stats**
   - Print counts of pages written, PDFs skipped, and PDFs failed for sanity checking.


### How to run
You can run the script with defaults (works out of the box):
```bash
python scripts/extract_pages.py
```
Or override paths via optional CLI flags:
```bash 
python scripts/extract_pages.py \
  --entries data/manual_entries.json \
  --out data/extracted/extracted_pages.jsonl \
  --log data/extracted/extract_log.jsonl
```

### Notes/ future improvements
- OCR support for scanned/image PDFs is out of scope for this stage but likely needed.
- If extraction quality varies by PDF type, we can add additional logging fields later (e.g., average page text length, percent empty pages).

### Testing

- Existing tests: (none yet)
- To be added soon:
    - verify JSONL schema per page (required keys + types)
    - verify page_num is 1-indexed and contiguous per PDF
    - verify is_scanned=true attachments are skipped + logged
    - verify missing files are logged but do not crash the run


## Chunk PDF pages into search-friendly chunks (`chunk_pages.py`)

After `extract_pages.py` produces page-level text (`extracted_pages.jsonl`), the next step is to convert those pages into **chunk-level records** that are better suited for search. Chunking balances search quality and traceability by grouping text into moderately sized units while preserving page ranges.

### Purpose

`chunk_pages.py` transforms page-level text into clean, search-friendly chunks. It normalizes PDF artifacts, groups related text, and emits structured chunk records that are ready for indexing.

This stage is intentionally opinionated: it removes boilerplate, drops low-value text, and keeps metadata minimal to avoid bloating downstream indexes.

---

### Inputs

- `data/extracted/extracted_pages.jsonl` (from `extract_pages.py`)
  - JSONL, one record per page:
    - `entry_id`
    - `attachment_id`
    - `page_num` (1-indexed)
    - `text`
    - `source_path`

- `data/manual_entries.json`
  - Used to attach **minimal metadata** to each chunk for filtering and display

---

### Outputs

- `data/chunks/chunks.jsonl`
  - JSONL, one record per chunk:
    - `chunk_id`
    - `entry_id`
    - `attachment_id`
    - `page_start`, `page_end`
    - `text`
    - `source_path`
    - `meta` (minimal metadata only)

- `data/chunks/chunk_log.jsonl`
  - JSONL log of warnings and edge cases:
    - malformed page records
    - unsorted input
    - unusually large PDFs
    - PDFs that produce zero chunks

---

### Chunking Strategy (High-Level)

1. **Stream pages**
   - Read `extracted_pages.jsonl` line-by-line (no full-file loads).

2. **Group by PDF**
   - Pages are grouped by `(entry_id, attachment_id)` so each PDF is processed independently.

3. **Clean page text**
   - Fix hyphenation across line breaks.
   - Normalize whitespace.
   - Remove recurring Federal Register headers and footer tokens.

4. **Split into units**
   - Cleaned text is split into sentence-like units using a lightweight regex.

5. **Pack units into chunks**
   - Units are accumulated until a target character size is reached (default ~1500 chars).

6. **Emit valid chunks**
   - Chunks are dropped if they are too small or mostly boilerplate.
   - Valid chunks are written immediately (streaming).

7. **Fallback for short PDFs**
   - If a PDF produces zero chunks, the script retries once with a relaxed minimum size.

---

### Key Design Decisions
- **Character-based chunk sizing (~1,500 chars)**
  - Chunks are sized to a consistent target length of roughly 1,500 characters.
  - For keyword search (BM25), consistent chunk sizes produce more predictable signal density and length normalization, which leads to more stable and interpretable ranking behavior across diverse PDFs.

- **Streaming I/O**
  - Pages and chunks are processed incrementally to keep memory usage stable for large corpora.

- **Grouping assumption with safeguards**
  - Input is expected to be ordered by `(entry_id, attachment_id)`.
  - If a group reappears later, a warning is logged (`WARN_UNSORTED_INPUT`).

- **Boilerplate removal**
  - Common non-content sections (tables of contents, Federal Register boilerplate, contact blocks) are filtered to improve search quality.

- **Stable, descriptive chunk IDs**
  - Each chunk ID encodes provenance:
    ```
    {entry_id}_{attachment_id}_p{page_start}-p{page_end}_c{chunk_index}
    ```
  - This makes chunks easy to trace back to source PDFs and page ranges.

- **Minimal metadata**
  - Only a small subset of metadata is copied into each chunk:
    - `iptp_id`, `title`, `administration`, `original_date_announced`,
      `effective_date`, `agencies_affected`, `subject_matter`
  - Chunks are the unit indexed and scanned most often, so keeping metadata minimal
    avoids inflating `chunks.jsonl`, reduces I/O and memory overhead, and keeps
    downstream search fast.

- **Fallback logic**
  - Short but important documents are given a second chance by relaxing
    `min_chars` once before being dropped entirely.

---

### How to Run

Run with defaults:
```bash
python scripts/chunk_pages.py
```
Or override paths and chunk sizing:
```bash 
python scripts/chunk_pages.py \
  --entries data/manual_entries.json \
  --pages data/extracted/extracted_pages.jsonl \
  --out data/chunks/chunks.jsonl \
  --target_chars 1500 \
  --min_chars 300 \
  --log data/chunks/chunk_log.jsonl
```
#### CLI Flags (All Optional)

- --entries Default: data/manual_entries.json 
- --pages Default: data/extracted/extracted_pages.jsonl 
- --out Default: data/chunks/chunks.jsonl 
- --target_chars Default: 1500 Target size (in characters) for each chunk. 
- --min_chars Default: 300 Minimum size required for a chunk to be kept. 
- --log Default: data/chunks/chunk_log.jsonl


### Testing

Tests for this stage focus on structural correctness and invariants, not exact chunk counts.

Current tests live in:
- `tests/test_chunk_pages.py`

They verify that:
- `chunks.jsonl` is created and non-empty
- all chunks follow the expected schema and types
- no chunk has empty or whitespace-only text
- `chunk_id` page ranges match `page_start` / `page_end`
- all `chunk_id`s are globally unique
- chunks within a PDF appear in non-decreasing page order
- metadata remains minimal and does not silently bloat

These tests are intentionally high-level so chunking heuristics can evolve
without breaking the test suite.


## Shared normalization + tokenization utilities (`text_processing.py`)

After `chunk_pages.py` produces clean, search-friendly chunks, the next step is to ensure that **both indexing and search interpret text the same way**. This is where `text_processing.py` fits in.

This module provides a shared foundation for text handling across the system. Any component that needs to reason about text (BM25 today, semantic search and analytics later) relies on these utilities to stay consistent.

---

### Purpose

`text_processing.py` centralizes text normalization and tokenization so downstream components behave predictably and remain easy to evolve. It ensures that the text used during indexing is processed in the same way as text used during querying.

The focus here is on:
- consistency across the pipeline
- transparency and debuggability
- easy iteration without introducing heavy dependencies

---

### Inputs

- Raw text strings (typically chunk text from `chunks.jsonl`)
- Optional stopword list file:
  - `data/text/stopwords_en.txt` (newline-delimited)
  - blank lines and `#` comments are ignored
  - if the file is missing, stopwords default to an empty set (MVP-friendly)

---

### Outputs

- Normalized text (string)
- Token lists (list of strings), filtered for:
  - alphanumeric terms
  - minimum token length
  - optional stopword removal

---

### Processing Strategy (High-Level)

1. **Load stopwords once**
   - `load_stopwords()` reads `data/text/stopwords_en.txt` (if present).
   - A module-level `STOPWORDS` set is initialized at import time and reused across calls.

2. **Normalize text**
   - `normalize_text()` applies:
     - de-hyphenation across line breaks (`immi-\ngration → immigration`)
     - consistent handling of tabs and newlines
     - whitespace collapsing and trimming

3. **Tokenize**
   - `tokenize()` lowercases normalized text and extracts tokens using a compiled regex:
     - `[a-z0-9]+`
   - Tokens are then filtered to remove:
     - very short tokens (`len < 2`)
     - stopwords (configurable via file)

4. **Explicit token filtering**
   - `filter_tokens()` allows downstream code to apply stopword and length filtering explicitly.
   - Useful when experimenting with different stopword sets or search strategies.

5. **Convenience helper**
   - `tokenize_for_bm25()` provides a single entry point for:
     - normalize → tokenize → filter
   - Keeps BM25 code concise and consistent.

---

### Key Design Decisions

- **Single source of truth**
  - All normalization and tokenization logic lives in one place, preventing subtle mismatches between indexing and querying.

- **File-driven stopwords**
  - Stopwords can be updated or tuned without code changes, making experimentation easier.

- **Configurable stopword usage**
  - Callers can toggle stopword removal (`use_stopwords=True/False`) depending on the search strategy.

- **Predictable token boundaries**
  - Regex-based tokenization produces stable, interpretable tokens that are easy to debug.

- **Forward-compatible**
  - The same utilities can be reused for semantic search, hybrid ranking, or text analytics without rewriting preprocessing logic.

---

### How It’s Used

- **BM25 indexing (`build_bm25.py`)**
  - Uses `tokenize(text, use_stopwords=...)` to generate term frequencies per chunk.

- **BM25 search (`search_bm25.py`)**
  - Uses the same tokenization logic so query parsing matches indexing behavior exactly.

This symmetry is critical for meaningful BM25 scoring.

---

### How to Run

This module is not typically executed directly. It is imported by other scripts.

For a quick sanity check:

```bash
python -c "from scripts.text_processing import tokenize; print(tokenize('Hello, world! Immigration immi-\\ngration.'))"
```

### Testing (Future)

No tests exist yet for this module.

Suggested future tests:
- Stopword loading
    - missing stopword file returns an empty set
    - blank lines and # comments are ignored
    - stopwords are lowercased consistently
- Normalization correctness
    - hyphenation repair across line breaks works
    - whitespace normalization behaves as expected
- Tokenization invariants
    - only alphanumeric tokens are returned
    - output tokens are lowercased
    - tokens shorter than two characters are removed
    - stopword removal toggles correctly
- Helper consistency
    - tokenize_for_bm25() produces the same result as explicitly chaining normalization, tokenization, and filtering