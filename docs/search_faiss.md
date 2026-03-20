# `search_faiss`: Design & Architecture

This document describes the design decisions behind `scripts/search_faiss.py`, the FAISS-based semantic search component of the IPTP pipeline.

The primary goals of `search_faiss` are:

* **Scalability**: support millions of chunks without loading large files into memory
* **Consistency**: mirror `search_bm25.py` in interface, filters, and output format
* **Correctness**: enforce embedding/index semantic compatibility
* **Performance**: make later queries fast via one-time, streaming-built helper artifacts

---

## High-level overview

`search_faiss` performs semantic search over chunk embeddings produced by:

1. `embed_chunks.py` → sharded `.npy` embeddings + `manifest.json`
2. `build_faiss.py` → `index.faiss` + `row_to_chunk_id.jsonl`

At query time, `search_faiss`:

1. Embeds the query using **the exact same embedding configuration** as indexing
2. Runs a FAISS nearest-neighbor search
3. Maps FAISS result rows → `chunk_id` (seek-based, streaming-safe)
4. Loads chunk records from `chunks.jsonl` (seek-based, streaming-safe)
5. Applies the same metadata filters as BM25
6. Returns ranked results in the same printed format as `search_bm25.py`

---

## Core design decisions

### Decision A — Never load `row_to_chunk_id.jsonl` into memory

**Problem**
`row_to_chunk_id.jsonl` can be extremely large (one row per embedding). Loading it fully does not scale.

**Solution**
We build a one-time **offset index**:

```
row_to_chunk_id_offsets.npy
offsets[row] = byte offset of that row’s JSONL line
```

At query time:

* seek to `offsets[row]`
* read one line
* parse JSON
* extract `chunk_id`

**Why**
This is the exact same scaling pattern used by BM25’s `docs_offsets.json`.

**Cost**

* One-time, streaming O(N) scan
* Stored as `int64` NumPy array (compact, mmap-friendly)

**Guarantees**

* Rows must be contiguous `0..N-1`
* Defensive verification that the `"row"` field matches the requested row

---

### Decision B — Never load `chunks.jsonl` into memory

**Problem**
Chunks are identified by string `chunk_id`. There is no contiguous integer ID, so a simple offset array is not feasible.

**Solution**
Build a **disk-backed SQLite index**:

```
chunk_id TEXT PRIMARY KEY → byte offset in chunks.jsonl
```

At query time:

* look up offset via SQLite
* seek into `chunks.jsonl`
* parse exactly one JSON record

**Why SQLite**

* Handles millions of rows reliably
* On-disk (constant memory usage)
* Simple, dependency-free, battle-tested
* Much cleaner than custom hash-bucket files

**Cost**

* One-time, streaming scan of `chunks.jsonl`
* Reused across all future searches

**Staleness protection**

* SQLite stores a signature of `chunks.jsonl` (size + mtime)
* Automatically rebuilds if the source file changes

---

### Decision C — Oversample then filter (post-retrieval filtering)

**Problem**
Metadata filters (`administration`, `agency`, `subject`, etc.) cannot be applied inside FAISS.

**Solution**

* Retrieve `top_k * oversample` candidates from FAISS
* For each candidate (in rank order):

  * load chunk
  * apply filters
* Stop once `top_k` passing results are found

**Defaults**

* `oversample = 10`
* One bounded expansion pass if filters are very strict

**Why**

* Same strategy as `search_bm25.py`
* Keeps FAISS search fast while preserving filter correctness

---

### Decision D — Cold start vs warm start behavior

`search_faiss` supports two operating modes transparently:

#### Cold start

* Offset files and SQLite lookup DB do not exist
* They are built automatically via streaming scans

#### Warm start

* All helper artifacts already exist
* Query-time work is minimal and fast

**Artifacts created lazily**

* `run_dir/faiss/row_to_chunk_id_offsets.npy`
* `data/chunks/chunks_lookup.sqlite`

**Key property**

> All helper artifacts are *derived* from existing authoritative files.
> Nothing is duplicated semantically, only indexed for fast lookup.

---

### Decision E — Never rebuild the FAISS index at query time

**Problem**
Rebuilding FAISS from shard `.npy` files would be expensive and unnecessary.

**Solution**

* `search_faiss` **requires** a prebuilt `index.faiss`
* Index construction is the responsibility of `build_faiss.py`

**Benefit**

* Query latency is predictable
* Index construction and search are cleanly separated stages

---

### Decision F — Metric handling (Inner Product vs L2)

FAISS supports multiple similarity metrics. `search_faiss` enforces consistency with how embeddings were built.

**Rules**

* If embeddings were **L2-normalized** (`normalize=True`):

  * Use **inner product (IP)**
  * Scores ≈ cosine similarity
* If embeddings were **not normalized**:

  * Use **L2 distance**
  * Scores are reported as `-distance` so “higher is better”

**Source of truth**

* `manifest.json["config"]["normalize"]`
* `manifest.json["faiss"]["metric"]` (if present)

**Guarantee**

> Query-time scoring semantics always match index-time semantics.

---

### Decision G — Embedding configuration is derived from the manifest

**Problem**
Using a different embedder config at query time silently breaks retrieval quality.

**Solution**

* `search_faiss` reconstructs `BGEConfig` from `manifest.json["config"]`
* Only runtime knobs (device, batch size) are overridable
* Semantic knobs (model, pooling, normalization, max_length) are locked

**Guarantee**

> Query embeddings live in the same vector space as indexed embeddings.

---

### Decision H — Streaming everywhere, bounded memory always

At no point does `search_faiss` load:

* `row_to_chunk_id.jsonl`
* `chunks.jsonl`
* all chunk metadata
* all chunk IDs

Memory usage is bounded by:

* FAISS index (expected)
* one query embedding
* `top_k * oversample` chunk records (small)

This allows the system to scale to **millions of chunks** without architectural changes.

---

## Interface consistency with BM25

`search_faiss.py` intentionally mirrors `search_bm25.py`:

* Same CLI flags:

  * `--q`, `--top_k`
  * `--entry_id`, `--administration`, `--agency`, `--subject`
* Same printed output:

  * rank, score, `chunk_id`
  * entry / attachment / page range
  * title (if present)
  * snippet
* Same filter semantics

This allows FAISS and BM25 to be swapped or compared easily.

---

## Summary of guarantees

`search_faiss` guarantees:

* No large JSONL file is loaded fully into memory
* Query-time semantics match index-time semantics
* Metadata filtering is correct and deterministic
* First run may build helpers; later runs are fast
* Design scales to very large corpora
