# BM25 Index Pipeline

This document describes how IPTP chunk-level text is converted into a searchable BM25 index.  
This is the **primary retrieval architecture** in the current system and serves as the foundation for keyword search, filtering, and future hybrid retrieval.

The pipeline operates in two stages:

1) `build_bm25.py`   
2) `search_bm25.py` 

This document focuses on **index construction**.

BM25 Pipeline Visualization:

chunks.jsonl → build_bm25.py → BM25 index (disk) → search_bm25.py → ranked chunk results

---

## BM25 Index 

BM25 is a **keyword-based ranking model**, not a semantic model.

At index-build time, we are not *learning* anything. Instead, we are **precomputing corpus statistics** that let us efficiently answer the question:

> *“Given these query words, which text chunks mention them most strongly and informatively?”*

BM25 scores text using three core ideas:

1. **Term frequency** – chunks that mention a query term more often are likely more relevant
2. **Inverse document frequency** – rarer terms carry more signal than very common ones
3. **Length normalization** – longer chunks are slightly penalized so they don’t dominate purely by size

Because these statistics are computed once during indexing, search-time scoring is fast, transparent, and easy to reason about.

---

## 1) build_bm25.py

### Purpose

`build_bm25.py` builds a compact, disk-backed BM25-style inverted index from
`chunks.jsonl`. It transforms each chunk into an indexable document, computes
term statistics, and writes a set of artifacts optimized for fast lookup and
iteration during search.

---

### Inputs

- `data/chunks/chunks.jsonl`
  - One JSON object per chunk:
    - `chunk_id`
    - `entry_id`
    - `attachment_id`
    - `page_start`, `page_end`
    - `text`
    - `source_path`
    - `meta`

---

### Outputs

All artifacts are written under a single directory (default: `data/indexes/bm25/`):

- `docs.jsonl`  
  Minimal per-document records used for lookup, snippets, and citation.

- `inverted_index.jsonl`  
  One JSON object per term:
  ```json
  { "term": "immigration", "postings": [[doc_id, tf], ...] }
  ```
- `term_offsets.json`
    Byte offsets into inverted_index.jsonl for seek-based term access.

- `doc_lens.json`
    List of document lengths indexed by doc_id.

- `idf.json`
    BM25 IDF values computed over the indexed corpus.

- `corpus_stats.json`
    Corpus-level statistics:
    - number of indexed documents
    - average document length
    - stopword configuration used

- `build_log.jsonl`
    Structured JSONL log of skips, warnings, and final build stats.

## Indexing Strategy (High-Level)

1. **Stream chunk records**  
   Read `chunks.jsonl` line-by-line so the full corpus is never loaded into memory.

2. **Validate and sanitize input**  
   Skip malformed records, empty text, or duplicate `chunk_id`s.  
   Normalize metadata and retain only approved, JSON-safe fields.

3. **Tokenize document text**  
   Delegate normalization and tokenization to `text_processing.tokenize`, with
   optional stopword removal controlled by a build-time flag.

4. **Compute term statistics**  
   For each document, compute term frequencies and update:
   - document frequency (DF) per term  
   - postings lists mapping terms to `(doc_id, tf)`

5. **Apply scalability guardrails**  
   Detect and skip pathological documents with extremely large vocabularies to
   keep the index bounded and performant.

6. **Assign compact document IDs**  
   Documents that pass validation are assigned dense `doc_id`s (`0..N-1`), ensuring
   alignment across all persisted artifacts.

7. **Persist index artifacts**  
   Write document records, postings, corpus statistics, and byte offsets
   deterministically for reliable lookup and debugging.

---

## Key Design Decisions

- **Streaming-first architecture**  
  All inputs and outputs are processed incrementally to support large corpora without high memory overhead.

- **Compact, disk-backed index**  
  The inverted index is stored as JSONL with explicit byte offsets, enabling seek-based term access instead of loading the full vocabulary into memory.

- **Minimal metadata propagation**  
  Only a small, approved set of metadata fields is carried into the index to avoid unnecessary I/O and keep retrieval fast.

- **Deterministic output**  
  Terms and postings are written in sorted order, making the index reproducible and easier to inspect, diff, and test.

- **Defensive robustness**  
  The build process logs and skips bad inputs rather than failing the entire run, allowing indexing to proceed even with imperfect upstream data.

### How to Run

Run with defaults:
``` bash
python scripts/build_bm25.py
```

Or override paths and options:
```bash
python scripts/build_bm25.py \
  --chunks data/chunks/chunks.jsonl \
  --out_dir data/indexes/bm25 \
  --log data/indexes/bm25/build_log.jsonl \
  --use_stopwords
```

### CLI Flags (All Optional)

- `--chunks`  
  Default: `data/chunks/chunks.jsonl`  
  Path to the chunk-level input file.

- `--out_dir`  
  Default: `data/indexes/bm25`  
  Output directory for all BM25 artifacts.

- `--log`  
  Default: `data/indexes/bm25/build_log.jsonl`  
  JSONL log file for warnings, skips, and build summary events.

- `--use_stopwords`  
  Default: `False`  
  Enable stopword removal during tokenization (keeps negation words intact).

---

### Testing

Tests for the BM25 build stage focus on **invariants and consistency**, not exact scores or rankings, so the search layer can evolve without silently  breaking the index.

Current tests live in:

- `tests/test_build_bm25.py`

They verify that:

- **Artifacts exist and align**
  - All expected output files are created.
  - `docs.jsonl`, `doc_lens.json`, and corpus statistics agree on document counts.

- **Document length correctness**
  - `doc_lens[doc_id]` matches the number of tokens produced by
    `text_processing.tokenize(...)` (spot-checked).

- **Chunk provenance integrity**
  - `chunk_id` page ranges match `page_start` / `page_end`.

- **Postings list sanity**
  - Postings contain unique, non-negative `doc_id`s with positive term frequencies.

- **Offset correctness**
  - `term_offsets.json` covers all indexed terms and allows correct seek-based lookup
    into `inverted_index.jsonl`.

- **Metadata constraints**
  - `meta` contains only approved keys and JSON-safe values, preventing silent index bloat.

- **Encoding safety**
  - Indexed document text is valid UTF-8 and contains no null bytes.

These tests are intentionally **medium-granularity**:
- They catch structural and alignment bugs early.
- They avoid locking in exact corpus statistics or term counts, which may change as tokenization and chunking heuristics improve.

---

## BM25 Search (`search_bm25.py`)

This module provides **efficient, seek-based BM25 search** over the artifacts produced by `build_bm25.py`. It is designed to support fast querying without loading large index structures fully into memory, making it suitable for iterative development and scaling to larger corpora.

### Purpose

`search_bm25.py` performs keyword search over **chunk-level documents** indexed by `build_bm25.py`

Given a query string, it:

* scores indexed text chunks using BM25,
* applies optional metadata filters,
* and returns the top-ranked chunk-level results with provenance and snippets.

The search layer is intentionally **stateless**: it does not modify the index or compute new corpus statistics. It only reads and uses artifacts produced at build time.

---

### Inputs

**Required**

* A query string (e.g. `"ICE raids"`)

**Optional**

* Path to a BM25 index directory
* BM25 parameters (`k1`, `b`)
* Metadata filters:

  * `entry_id`
  * `administration`
  * `agency`
  * `subject`

All search behavior depends entirely on the artifacts already present in the index directory.

---

### Outputs

The search returns a **ranked list of chunk-level results**.

Each result includes:

* a BM25 relevance score,
* chunk provenance:

  * `chunk_id`
  * `entry_id`
  * `attachment_id`
  * `page_start`, `page_end`
* the chunk text,
* minimal metadata (for display and filtering),
* a short text snippet (for quick inspection in the CLI).

The CLI prints a compact, human-readable summary suitable for debugging, evaluation, or downstream reranking.

---

### Search Strategy (High-Level)

1. **Load compact index data**

   * Load small, fixed-size artifacts into memory:

     * IDF values
     * document lengths
     * corpus statistics
     * byte-offset maps
   * Build `docs_offsets.json` lazily if it does not exist.

2. **Tokenize the query**

   * Use the *same tokenization and stopword configuration* as indexing to ensure consistency.

3. **Sparse scoring via postings**

   * For each unique query term:

     * Seek directly to its postings list using byte offsets.
     * Accumulate BM25 scores only for documents that contain the term.

4. **Candidate pruning**

   * Rank candidate documents by score.
   * Keep a bounded candidate set to avoid loading unnecessary documents.

5. **On-demand document loading**

   * Load full document records only for candidate `doc_id`s.
   * Apply metadata filters at this stage.

6. **Final ranking**

   * Sort filtered results by score.
   * Return the top-k results.

This approach keeps search fast while avoiding full index scans or large in-memory structures.

---

### Metadata Filtering

Search supports lightweight, optional filters applied **after scoring**:

* `entry_id` (top-level document field)
* `administration` (metadata)
* `agency` → `agencies_affected` (metadata)
* `subject` → `subject_matter` (metadata)

Filtering after scoring keeps the scoring logic simple and reusable, while still supporting targeted queries and debugging workflows.

---

### Key Design Decisions

* **Seek-based file access**
  Large JSONL files (`inverted_index.jsonl`, `docs.jsonl`) are accessed via byte offsets, avoiding full loads.

* **Strict consistency with indexing**
  Tokenization, stopword usage, and BM25 defaults are inherited from the build step to prevent silent mismatches.

* **Defensive correctness**
  Offset mismatches, invalid doc IDs, and malformed records raise clear errors rather than failing silently.

* **Composable search layer**
  Document loading, filtering, and scoring are separated so this layer can later support hybrid retrieval or reranking.

Here’s a **clean, consistent rewrite** of the **How to Run** section for `search_bm25.py`, matching the style of the other pipeline docs and avoiding unnecessary repetition.

---

### How to Run

Run a basic query with defaults:

```bash
python -m scripts.search_bm25 --q "ICE raids"
```

This uses:

* the default BM25 index directory (`data/indexes/bm25`)
* standard BM25 parameters (`k1=1.2`, `b=0.75`)
* the same stopword configuration used during index build

---

Run with optional filters and parameters:

```bash
python -m scripts.search_bm25 \
  --q "asylum processing" \
  --administration Biden \
  --top_k 20
```

---

#### CLI Flags (All Optional)

* `--q` *(required)*
  Query string to search for.

* `--index_dir`
  Default: `data/indexes/bm25`
  Path to the BM25 index directory.

* `--top_k`
  Default: `10`
  Number of ranked results to return.

* `--k1`
  Default: `1.2`
  BM25 term-frequency saturation parameter.

* `--b`
  Default: `0.75`
  BM25 length-normalization parameter.

* `--entry_id`
  Filter results to a single `entry_id`.

* `--administration`
  Filter by `administration` metadata.

* `--agency`
  Filter by `agencies_affected` metadata.

* `--subject`
  Filter by `subject_matter` metadata.

---


### Testing

Search testing will focus on **expected retrieval behavior** and correctness, rather than low-level implementation details. Tests will be added incrementally and compared against the current search baseline to ensure expected retrieval behavior does not regress.

Planned search-specific tests include:

* **Golden query behavior**

  * Maintain a small set of representative queries with expected results (e.g., specific `chunk_id`s appearing in the top-*k*).
  * Use these to catch regressions when search logic or tokenization changes.

* **Baseline comparison**

  * Compare BM25 results against the current search system for a fixed query set.
  * Check overlap or recall@k to ensure BM25 surfaces comparable or better results.

* **Filter correctness**

  * Verify that metadata filters (`entry_id`, `administration`, `agency`, `subject`) correctly restrict results to matching documents.

* **Scoring sanity**

  * On small, controlled corpora:

    * chunks with more query-term matches score higher
    * rarer terms contribute more than common terms
    * extremely long chunks do not dominate purely by size

* **Offset integrity**

  * Spot-check that term and document offsets seek to the correct records during search.

These tests aim to validate that search behaves **as expected from a user and retrieval perspective**, while allowing the underlying index and heuristics to evolve.
