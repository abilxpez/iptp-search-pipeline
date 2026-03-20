# Hybrid Legal Policy Retrieval: Design Choices, Tradeoffs, and Empirical Behavior (Scaffold v2)

## Abstract
This document presents a production-oriented retrieval system for legal and policy search. The pipeline combines lexical retrieval (BM25), dense semantic retrieval (FAISS-based), reciprocal rank fusion (RRF), and cross-encoder reranking. Rather than treating architecture as fixed, this scaffold makes implementation knobs explicit: chunking policy, pooling strategy, embedding normalization, ANN index/search parameters, fusion constant, rerank depth, and text truncation. We evaluate each stage under shared relevance labels and report `nDCG@5`, `MRR@5`, `Success@5`, and `P@5` to show where additional complexity helps, where it does not, and why. The goal is to provide both a clear systems blueprint and a defensible decision record that can guide follow-up ablations.

## 1. Problem Statement and Audience

This retrieval system is built for legal and policy search, where users care about both exact terminology and broader meaning. In this domain, query intent often mixes citation-like specificity (agency names, document types, policy labels) with conceptual language (policy themes, practical impact, procedural effects). Systems that rely only on keyword matching or only on semantic similarity tend to struggle with one side of this.

The goal of this system is to handle both types of queries while keeping the retrieval process understandable. In legal and policy settings, it’s not enough to return results—the user also needs to understand why a result was returned and which part of the document is actually relevant.

This document focuses on two things:

1. Making retrieval behavior easy to inspect at each stage of the pipeline.
2. Making the system reproducible, so results can be traced back to specific design and configuration choices.

In addition to comparing overall performance, this work looks at how different design decisions—like chunking, indexing, fusion, and reranking—affect system behavior, and what tradeoffs they introduce in terms of accuracy, latency, and complexity.

## 2. Research Questions and Contributions
We ask three primary questions:
1. How much quality does each retrieval stage add relative to simpler alternatives?
2. Which configuration decisions are most consequential for quality, robustness, and latency?
3. When do more advanced stages fail to improve outcomes despite higher complexity?

Contributions:
1. A modular four-stage architecture with explicit stage boundaries.
2. A decision-led evaluation framing tied to measurable outcomes.
3. A concrete implementation ledger for retrieval output semantics (top-hit source and display rules).
4. A prioritized ablation roadmap grounded in current artifacts.

## 3. System Overview
The pipeline is a staged retrieval cascade:
1. `Step 0`: data acquisition and corpus construction from AWS artifacts.
2. `Stage 1`: BM25 lexical retrieval.
3. `Stage 2`: semantic retrieval using dense embeddings and FAISS ANN search.
4. `Stage 3`: reciprocal rank fusion (RRF) of lexical + semantic results.
5. `Stage 4`: cross-encoder reranking of fused candidates.

This structure separates recall-oriented stages (BM25, semantic, fusion) from precision-oriented reranking (cross-encoder), so each stage can be measured independently and tuned without redesigning the full stack.

## 4. Step 0: Data Acquisition and Corpus Construction (`scripts/get_data`)
This step establishes data provenance and directly shapes retrieval quality by controlling what text enters the index. The ingestion workflow downloads source files from AWS/S3, assembles policy manifests, extracts page text, and creates chunks for indexing.

### 4.1 What this step does
1. Builds a local reproducible corpus from cloud-hosted policy artifacts.
2. Converts document-level assets into page-level and chunk-level retrieval units.
3. Produces intermediate logs/state files for resume-safe processing and anomaly auditing.

### 4.2 Key design decisions
1. AWS download behavior (`skip_existing`, `verify_size`, jobs/parallelism, failure logging).
2. Extraction/chunking policy (chunk boundaries and overlap), which impacts both lexical and dense retrieval.
3. Manifest-first orchestration to keep IDs and metadata aligned across all stages.
4. Policy metadata flattening into chunk records to support filtering and downstream display.

### 4.3 Current configuration context
Representative active settings (from `config.json`):
1. Sample size for assembled corpus: `assemble.n = 100`.
2. Chunking defaults: `target_chars=1500`, `min_chars=300`.
3. Overlap enabled with bounded overlap units.

### 4.4 Representative code
1. `scripts/get_data/assemble_policies.py`
2. `scripts/get_data/download_documents.py`
3. `scripts/get_data/extract_pages.py`
4. `scripts/get_data/chunk_pages.py`

### 4.5 Script-by-script walkthrough (with sub-function intent)
This subsection expands the beginning of the paper so readers can follow exactly how raw source data becomes retrieval-ready chunks.

#### A) `scripts/get_data/assemble_policies.py`
Primary role:
1. Read IPTP parquet exports from AWS backup prefixes.
2. Join policy metadata, related entities, and document references.
3. Write per-policy `policy.json`, `policies_index.jsonl`, and `manifest.jsonl`.

Key sub-functions and what they do:
1. `list_parquet_keys(...)`: lists relevant parquet shards from S3 prefixes.
2. `cache_table_parts(...)`: downloads/caches table shards locally for reproducible reads.
3. `read_table(...)`: loads cached parquet shards into a unified dataframe.
4. `choose_diverse_sample(...)`: builds bucket keys from subject category, department, and action type; then performs round-robin bucket sampling.
5. `html_to_text(...)`: normalizes policy description HTML into plain text for downstream summary chunks.
6. `build_policy_folder_name(...)`: creates stable policy folder naming.

Why these matter:
1. `list_parquet_keys(...)` + `cache_table_parts(...)` turn cloud snapshots into a stable local input set, which is required for reproducible experiments.
2. `read_table(...)` centralizes multi-shard loading so downstream joins are done on complete, not partial, table views.
3. `choose_diverse_sample(...)` directly controls corpus composition and therefore retrieval/evaluation coverage.
4. `html_to_text(...)` defines summary text quality before chunking/indexing.
5. `build_policy_folder_name(...)` makes reruns deterministic and traceable on disk.

Key design decision to highlight:
1. The dataset sample is intentionally diversified (stratified-like bucketed sampling), not naive random-only sampling. The objective is to reduce over-representation of a few policy types and increase retrieval-evaluation coverage across topic/agency/action combinations.

#### B) `scripts/get_data/download_documents.py`
Primary role:
1. Read `manifest.jsonl`.
2. Download referenced source files (primarily PDFs) from AWS S3 into local `out_dir`.
3. Record failures for resumable reruns.

Key sub-functions and what they do:
1. `parse_manifest_line(...)`: normalizes manifest schema variants into `(bucket, key, dest, expected_size)`.
2. `head_object_size(...)`: checks object size from AWS for validation/resume logic.
3. `should_skip(...)`: skip-on-exists logic with optional size checking.
4. `download_one(...)`: performs one `aws s3 cp` transfer, with optional size verification and retry-safe behavior.

Why these matter:
1. `parse_manifest_line(...)` prevents failures from small schema differences across manifest generations.
2. `head_object_size(...)` provides expected byte size so resume logic can validate file integrity.
3. `should_skip(...)` avoids unnecessary redownloads while still protecting correctness:
   - if `skip_existing` is off, it never skips;
   - if expected size is known, it skips only on exact size match;
   - if expected size is unknown, it skips only when the existing file is non-empty.
   This makes resume behavior efficient without silently accepting obvious truncation.
4. `download_one(...)` is the execution point for transfer + verification + structured failure metadata.

Key design decision to highlight:
1. Downloads are explicitly AWS-based (`aws s3 cp` / `s3api head-object`) with parallel execution plus optional size verification, enabling robust resume behavior and reducing silent corruption/truncation risk.

#### C) `scripts/get_data/extract_pages.py`
Primary role:
1. Read local downloaded files referenced in manifest.
2. Open PDFs and extract page-level text into `extracted_pages.jsonl`.
3. Log extraction stats and track completion state for resume.

Key sub-functions and what they do:
1. `open_pdf(...)`: opens documents with `fitz` (PyMuPDF).
2. `iter_page_records(...)`: iterates page-by-page and extracts text via `page.get_text("text")`.
3. `compute_page_text_lengths(...)`: measures extracted text length per page.
4. `guess_mostly_scanned(...)`: heuristic to flag mostly image/scanned PDFs (low text yield pages).
5. `extract_from_manifest(...)`: orchestrates full extraction loop with logging/state updates.

Why these matter:
1. `open_pdf(...)` defines the extraction backend boundary (PyMuPDF) and error handling point.
2. `iter_page_records(...)` standardizes page-level text output schema used by chunking.
3. `compute_page_text_lengths(...)` + `guess_mostly_scanned(...)` provide quality diagnostics for OCR-like failure modes.
4. `extract_from_manifest(...)` enforces resumability and logging, which is essential for long-running corpus builds.

Key design decision to highlight:
1. Text extraction uses PyMuPDF for deterministic page-level text retrieval from downloaded PDFs. The pipeline records scanned/low-text heuristics so downstream stages can audit extraction quality instead of assuming all PDFs are text-rich.

#### D) `scripts/get_data/chunk_pages.py`
Primary role:
1. Convert extracted page text into chunk-level records for indexing.
2. Add policy/attachment metadata into each chunk.
3. Write `chunks.jsonl` with title, summary, and pdf-page chunks.

Key sub-functions and what they do:
1. `iter_pages(...)`: streams extracted page records.
2. `clean_text(...)`: normalizes spacing and removes extraction artifacts/control chars.
3. `split_into_units(...)`: sentence/paragraph-like segmentation for chunk assembly.
4. `split_oversized_unit(...)`: hard-splits long units when they exceed max thresholds.
5. `chunk_pages_for_pdf(...)`: core buffer-based chunk assembly over page units.
6. `make_chunk_from_buffer(...)`: emits a chunk record once buffer reaches boundaries.
7. `build_summary_chunks(...)`: constructs policy-level summary chunks.
8. `build_title_chunks(...)`: constructs policy-title chunks.
9. `chunk_pages_pipeline(...)`: full orchestration and output writing.

Why these matter:
1. `iter_pages(...)` and `chunk_pages_pipeline(...)` define ordering and grouping guarantees for deterministic chunk IDs.
2. `clean_text(...)` influences tokenization quality for both BM25 and embeddings.
3. `split_into_units(...)` and `split_oversized_unit(...)` control semantic coherence vs chunk size constraints.
4. `chunk_pages_for_pdf(...)` and `make_chunk_from_buffer(...)` implement the actual retrieval-unit contract.
5. `build_summary_chunks(...)` and `build_title_chunks(...)` inject non-PDF policy signals that affect top-hit behavior in retrieval.

Key design decisions to highlight:
1. Character-budgeted chunking: buffers accumulate units until approximately `target_chars` (currently 1500), then emit.
2. Minimum chunk threshold: short chunks below `min_chars` (currently 300) are suppressed unless fallback logic applies.
3. Oversized text handling: very long units are split with punctuation-aware and hard-window fallbacks.
4. Optional overlap: retained trailing units from prior chunk preserve local context continuity.
5. Multi-source chunk types: policy title, policy summary, and PDF-page chunks are all indexed, enabling mixed retrieval behavior later.

### 4.6 Step-0 importance
The retrieval system is only as good as its corpus construction contract. These implementation choices in Step 0 create the exact objects ranked in BM25 and FAISS later, so this section should be treated as method-critical rather than preprocessing boilerplate. In the final essay, this is where to explicitly connect engineering choices (sampling diversity, AWS robustness, text extraction method, chunk policy) to downstream retrieval behavior and metric differences.

### 4.7 Chunking rules (chunk_pages.py)
To avoid ambiguity in the essay, the chunking policy below is verified directly from `scripts/get_data/chunk_pages.py` and current `config.json`.

#### A) PDF chunk assembly trigger
1. Page text is cleaned and split into sentence-like units.
2. Units are appended to a running buffer with character count tracking.
3. A chunk *attempt* occurs when `buffer_text_len >= target_chars`.
4. Current default: `target_chars = 1500`.

#### B) Chunk acceptance criteria
A candidate chunk is emitted only if:
1. Buffer is non-empty.
2. Joined chunk text length is at least `min_chars` (default `300`).
3. Text is not flagged by boilerplate heuristics.

Important behavior:
1. If chunk attempt fails (for example text still below `min_chars` or considered boilerplate), the buffer is **not** cleared; accumulation continues until a valid chunk can be emitted.

#### C) Oversized unit handling
1. If a single unit exceeds `max_unit_chars` (default `2000`), it is split.
2. Split strategy is punctuation-aware first, then hard windows.
3. Hard split window default: `hard_split_chars = 800`.

#### D) Overlap behavior after emission
If overlap is enabled (current default: enabled):
1. After a chunk is emitted, trailing units from that chunk are kept for context carryover.
2. Target retained units is `max(overlap.units, overlap.min_units)` (defaults: `2` and `1`).
3. Additional retained units are constrained by character budget `target_chars * overlap.max_frac_of_target` (default `0.2`, so ~300 chars with target 1500).
4. At least one trailing unit is always retained when overlap is enabled.

#### E) End-of-document flush and fallback
1. After all pages are processed, remaining buffer is attempted once more with normal `min_chars`.
2. If a PDF still yields zero chunks, pipeline retries that PDF with relaxed minimum size:
   `min_chars_relaxed = min(original_min_chars, 150)`.
3. If still zero, a warning is logged for audit.

#### F) Summary/title chunk rules (separate from PDF chunks)
1. Title chunks are always one chunk per policy when title text exists.
2. Summary chunk behavior:
   - If any summary paragraph exceeds 1500 chars, summary is split by paragraph.
   - Otherwise summary paragraphs are merged into a single summary chunk.

These details should be stated in the methods section because they directly shape retrieval granularity, duplicate evidence behavior, and downstream score/rank dynamics.

## 5. Stage 1: BM25 Lexical Retrieval (`scripts/bm25`)
BM25 provides exact-term retrieval and serves as a strong baseline for legal terminology, named entities, and statutory phrasing. It is generally robust when query language overlaps corpus language and can fail on paraphrases or concept-level queries with weak lexical overlap.

### 5.1 What this stage does
1. Retrieves chunk-level candidates by weighted term matching.
2. Aggregates chunk-level evidence into entry-level results.
3. Exposes interpretable lexical behavior and stable baseline ranking quality.

### 5.2 Key decisions
1. Candidate depth and oversampling (`top_k`, `oversample`, `max_candidates`).
2. BM25 hyperparameters (`k1`, `b`) and tokenization assumptions.
3. Aggregation policy from chunk-level hits to entry-level display objects.
4. Filtering policy (entry/administration/agency/subject filters applied post retrieval).

### 5.3 Implementation notes to preserve
1. Search/export are now wrapper-driven (`export_bm25_results_csv.py` calls `search_bm25`).
2. Snippet length is controlled by `snippet_chars` passed into search.
3. Export includes explicit `source` field (`title`, `summary`, `pdf`) based on top-scoring matched chunk type.

### 5.4 Representative code
1. `scripts/bm25/build_bm25.py`
2. `scripts/bm25/search_bm25.py`
3. `scripts/bm25/export_bm25_results_csv.py`

### 5.5 BM25 Build Pipeline (expanded, current-state)
This subsection captures the build stage specifically (`build_bm25.py`) and updates older documentation details where needed.

#### A) Build purpose
`build_bm25.py` converts `chunks.jsonl` into seek-friendly BM25 artifacts for fast query-time scoring without loading the whole corpus into memory.

#### B) Current artifact contract (as implemented)
1. `docs.jsonl`
   - one record per indexed doc, no full text duplication
   - includes `doc_id`, `len`, `chunk_id`, entry/attachment IDs, page ranges, and minimal metadata
2. `docs_offsets.sqlite`
   - `doc_id -> byte offset` into `docs.jsonl`
3. `inverted_index.jsonl`
   - one term row with inline `df`, `idf`, and postings list `[[doc_id, tf], ...]`
4. `offsets.sqlite`
   - `term -> byte offset` into `inverted_index.jsonl`
5. `chunks_offsets.sqlite`
   - `chunk_id -> byte offset` into original `chunks.jsonl` (source-of-truth text lookup)
6. `corpus_stats.json`
   - corpus-level stats and pointers to artifact paths
7. `build_log.jsonl`
   - structured skip/warn/progress/build summary events

Note for essay clarity:
1. Some older text may refer to JSON offset artifacts; current implementation uses SQLite offset maps for term/doc/chunk lookups.

#### C) Build flow (high-level, but implementation-aligned)
1. Stream `chunks.jsonl` line-by-line (`iter_chunks(...)`) and sanitize metadata.
2. Tokenize each chunk with shared text-processing config.
3. Build per-doc term frequency map and document length.
4. Write compact doc payload to `docs.jsonl`; write doc/chunk offsets to SQLite.
5. Emit raw postings into stable hash shards (`_tmp_postings/postings_shard_*.tsv`) for scalability.
6. External-sort each shard by `(term, doc_id)` using system `sort`.
7. Merge sorted shards into `inverted_index.jsonl`; compute/store `df` and BM25 `idf`; write term offsets to `offsets.sqlite`.
8. Write `corpus_stats.json` and final build summary logs.

#### D) Build-stage design decisions worth highlighting
1. Streaming-first indexing:
   keeps memory bounded and makes indexing robust for larger corpora.
2. No text duplication in BM25 docs:
   chunk text remains in `chunks.jsonl`; index stores pointers/metadata only.
3. SQLite seek maps:
   efficient point lookup for terms, docs, and chunks at query time.
4. Sharded postings + external sort:
   handles large vocab/postings growth without large in-memory global sorts.
5. Defensive skip/log behavior:
   malformed input is logged and skipped, so one bad record does not kill the full build.

#### E) Parameters and defaults to surface in writeup
1. BM25 scoring defaults used downstream: `k1=1.2`, `b=0.75`.
2. Build scalability knobs:
   - posting shards (`n_shards`, default 128)
   - commit cadence for SQLite writes
   - max unique terms per doc guardrail
3. Tokenization consistency:
   build and search share text-processing initialization from config, which is important for score reproducibility.

#### F) Why this matters before the search section
The build contract determines what search can do efficiently. Because search is seek-based over precomputed offsets and compact payloads, any essay discussion of BM25 ranking behavior should first anchor how doc/posting/offset artifacts are produced and constrained at build time.

## 6. Stage 2: Semantic Retrieval with BGE + FAISS (`scripts/semantic`)
Semantic retrieval addresses lexical mismatch by embedding queries and chunks into a dense vector space. This stage improves conceptual recall and is sensitive to embedding and ANN design choices.

### 6.1 What this stage does
1. Encodes text using a BGE encoder.
2. Retrieves nearest neighbors with FAISS ANN search.
3. Adds semantically related candidates BM25 may miss.

### 6.2 Key decisions (current artifact values)
1. Embedding model: `BAAI/bge-base-en-v1.5`.
2. Pooling strategy: `mean`.
3. Normalization: `true`.
4. Max input length: `256`.
5. FAISS index: HNSW (`ip`) with `m=32`, `ef_construction=200`, `ef_search=64`.

### 6.3 Why these matter
1. Pooling choice changes representation quality and downstream rank behavior.
2. Normalization and metric choice affect score geometry and neighbor stability.
3. HNSW parameters control recall-latency tradeoffs and can influence fusion quality.

### 6.4 Representative code/artifacts
1. `scripts/semantic/bge_embedder.py`
2. `scripts/semantic/embed_chunks.py`
3. `scripts/semantic/build_faiss.py`
4. `scripts/semantic/search_faiss.py`
5. `scripts/semantic/export_faiss_results_csv.py`
6. `data/embeddings/bge_mean_norm/manifest.json`

### 6.5 Semantic Build Pipeline (expanded, current-state)
This subsection expands the semantic side in the same style as BM25 build documentation. It covers embedding generation and FAISS index construction before search-time retrieval.

#### A) `scripts/semantic/bge_embedder.py` (embedding core)
Primary role:
1. Provide a reusable embedding engine (`BGEEmbedder`) configured by `BGEConfig`.
2. Convert raw text into dense vectors with deterministic shape/type for downstream FAISS indexing and search.
3. Keep semantic knobs (pooling/normalize/model/max length) explicit and reproducible.

Function-specific logic and why it exists:
1. `BGEConfig`
   - what: defines model/runtime/embedding behavior in one object
   - why: makes experiment settings explicit and portable
2. `BGEEmbedder.__init__`
   - what: resolves device, optionally seeds torch, loads tokenizer/model, sets AMP mode and output dim
   - why: centralizes runtime initialization to avoid hidden behavior drift across calls
3. `_sanitize_texts(...)`
   - what: strips input texts and handles empty strings (`replace` or `error`)
   - why: preserves 1:1 alignment with chunk IDs and avoids silent embedding-count mismatches
4. `_pool(...)`
   - what: computes either `cls` pooling or attention-mask-aware `mean` pooling
   - why: pooling strategy materially affects retrieval quality and must be easily ablated
5. `_l2_normalize(...)`
   - what: applies stable L2 normalization
   - why: enables cosine-like behavior under FAISS inner-product search when enabled
6. `_encode_batch(...)`
   - what: tokenizes one batch, runs model inference, pools, optionally normalizes, returns float32 NumPy
   - why: defines the canonical embedding contract consumed by all later stages
7. `embed_texts(...)`
   - what: batches full corpus input and retries OOM cases by halving batch size
   - why: robustness on constrained hardware without manual restarts

Design decisions to surface:
1. Pooling is a first-class experimental knob (`mean` default, `cls` benchmark path).
2. Normalization is explicit, allowing metric-aware index behavior (`ip` vs `l2` implications).
3. Runtime knobs (device/AMP/batching) are separated from semantic identity knobs.
4. Output is always CPU `float32`, simplifying FAISS compatibility and reducing dtype surprises.
5. Empty-text replacement is defaulted for alignment safety in production-style pipelines.

#### B) `scripts/semantic/embed_chunks.py` (embedding run orchestration)
Primary role:
1. Stream `chunks.jsonl` and generate embeddings shard-by-shard.
2. Persist embeddings and chunk-id mappings atomically.
3. Support resumable runs and manifest-based reproducibility checks.

Function-specific logic and why it exists:
1. `compute_run_name(...)`
   - what: creates deterministic run folder name (`bge_{pooling}_{norm/raw}`)
   - why: prevents accidental mixing of semantically different embedding runs
2. `config_fingerprint(...)`
   - what: extracts embedding-identity fields (model/pooling/normalize/max_length/etc.)
   - why: resume validation should key on semantic compatibility, not runtime-only knobs
3. `iter_chunks_jsonl(...)`
   - what: streams `(chunk_id, text)` records from `chunks.jsonl`
   - why: memory-safe ingestion for large corpora
4. `load_done_set(...)` and `append_done_ids(...)`
   - what: maintain durable record of already embedded chunk IDs
   - why: resumability and crash recovery without re-embedding completed items
5. `shard_paths(...)`, `atomic_save_npy(...)`, `atomic_save_ids_jsonl(...)`
   - what: deterministic shard naming and atomic persistence of vectors/ID maps
   - why: prevents partial shard writes and preserves row-to-id integrity
6. `init_or_validate_manifest(...)`
   - what: create a new run manifest or validate compatibility of an existing one
   - why: ensures reproducibility and blocks incompatible config reuse in the same run dir
7. `embed_chunks(...)` with internal `flush_shard()`
   - what: orchestrates buffering, embedding, shard flush, done-id append, and manifest update
   - why: converts streaming input into resumable, append-safe semantic artifacts

Artifact outputs from this stage:
1. `run_dir/manifest.json`
2. `run_dir/shards/embeddings_*.npy`
3. `run_dir/ids/chunk_ids_*.jsonl`
4. `run_dir/embedded_ids.jsonl`
5. `run_dir/embed_log.jsonl`

Design decisions to surface:
1. Stage separation: embedding artifacts are persisted independently from FAISS index artifacts.
2. Atomic writes reduce risk of partial-file corruption on interruption.
3. Resume-first design makes long runs practical and reproducible.
4. Manifest fingerprinting blocks accidental mixing of incompatible embedding configs in one run directory.
5. Sharded output format supports future cloud upload/parallel rebuild patterns.

#### C) `scripts/semantic/build_faiss.py` (index construction from embedding shards)
Primary role:
1. Read embedding manifest + shard files.
2. Validate shard integrity and add vectors into a FAISS index.
3. Write index, row-to-chunk mapping, and build metadata atomically.

Function-specific logic and why it exists:
1. `validate_manifest(...)`
   - what: checks required manifest structure (`dim`, normalize flag, shard list)
   - why: fail fast on incompatible/malformed embedding runs before index construction
2. `load_embeddings_npy(...)` and `load_ids_jsonl(...)`
   - what: strict shard readers for vectors and chunk-ID mappings
   - why: classify corruption clearly and keep shard errors debuggable
3. `make_index(...)`
   - what: creates FAISS index family (`flat`/`hnsw`) with chosen metric/params
   - why: ANN topology and metric are explicit experiment knobs, not hidden defaults
4. `AtomicMappingWriter`
   - what: writes `row_to_chunk_id.jsonl` via temp file + commit/abort lifecycle
   - why: preserves row-mapping integrity alongside index integrity
5. `build_faiss(...)`
   - what: end-to-end orchestration of shard validation, add-to-index, mapping/meta/index writes, manifest update
   - why: provides one controlled boundary from embedding shards to query-ready ANN artifacts
6. `on_shard_error` policy inside `build_faiss(...)`
   - what: chooses `skip` vs `fail` when shard-level errors occur
   - why: explicit reliability tradeoff between maximal coverage and strict all-shards guarantees

Core flow in code:
1. Validate manifest and derive default metric from normalization:
   - normalized vectors -> default `ip`
   - non-normalized vectors -> default `l2`
2. Initialize index (`flat` or `hnsw`) with provided params.
3. Iterate shards and validate each before add:
   - path resolution, file existence, shape/dim/count checks, finite-value checks, duplicate chunk-ID checks
4. Add vectors, append row mapping, and track indexed/skipped shards.
5. Finalize atomically:
   - `faiss/index.faiss`
   - `faiss/row_to_chunk_id.jsonl`
   - `faiss/build_faiss_meta.json`
   - `faiss/build_faiss_errors.jsonl`
6. Update embedding manifest with `faiss` summary block.

Design decisions to surface:
1. Fault-tolerant default (`on_shard_error=skip`) prioritizes pipeline continuity.
2. Strict shard validation prevents silent index corruption (dim/count/nonfinite/duplicate checks).
3. Atomic commit discipline ensures either valid artifacts or clean failure, not partial mixed state.
4. Explicit row mapping keeps ANN row IDs auditable and reversible to chunk IDs.
5. Metric/index choices are explicit knobs (`flat` vs `hnsw`, `ip` vs `l2`, HNSW params).

#### D) Semantic build-stage design

The semantic retrieval pipeline consists of three distinct components:

1. **Representation (`bge_embedder.py`)**  
   Defines how text is encoded into dense vectors. This stage determines what semantic relationships are captured in the embedding space.
2. **Artifact generation and storage (`embed_chunks.py`)**  
   Handles how embeddings are computed, stored, and versioned. This stage ensures consistency across runs and enables resumable processing for large datasets.
3. **ANN index construction (`build_faiss.py`)**  
   Builds the vector index used at query time. This stage determines how embeddings are organized and searched under latency and recall constraints.

Changes in retrieval behavior can originate from any of these components. Differences in results are not only tied to the embedding model itself, but also to how embeddings are generated, stored, and indexed.

## 6.6 Retrieval Output Semantics 
This section documents the active logic used to surface entry-level results from chunk-level hits. It is critical for interpretability and for consistent behavior across BM25 and FAISS retrieval outputs.

### 6.6.1 Per-entry hit ordering and tie-breaker
Within an entry, matched chunks are ordered by:
1. Highest score first.
2. If score ties, most recent chunk date first.
3. If still tied, lower hit rank first.

This is the current tie-break policy: highest score + most recent chunk.

### 6.6.2 Source labeling rule
`source` always reflects the top-scoring matched chunk type:
1. `title`
2. `summary`
3. `pdf`

### 6.6.3 Display rules for `doc` and `snippet`
Given top matched type:
1. `title` top hit:
   Display highest-scoring `pdf` chunk under that entry as `doc` + `snippet`.
   If no `pdf` chunk exists, fallback to `summary`.
   If no summary exists, keep title.
2. `summary` top hit:
   Display `summary` as `doc` + `snippet`.
3. `pdf` top hit:
   Display that `pdf` chunk as `doc` + `snippet`.

Short-form decision statement:
1. title + chunk
2. summary + summary
3. chunk + chunk

### 6.6.4 Matched lists behavior
For all scenarios above, matched evidence is still fully gathered:
1. `matched_attachment_ids`
2. `matched_doc_ids`
3. `matched_chunk_ids`
4. `matched_count`
5. `best_rank`

### 6.6.5 Shared search logic across BM25 and FAISS
The downstream entry-level output semantics are intentionally shared between `search_bm25` and `search_faiss`.

What differs:
1. Candidate generation step only:
   - `search_bm25`: lexical BM25 postings/scoring.
   - `search_faiss`: FAISS nearest-neighbor retrieval over dense embeddings.

What is shared after candidate generation:
1. metadata filters
2. entry-level aggregation
3. matched-list collection
4. source labeling
5. display selection rules
6. ranking/tie-break behavior

### 6.6.6 Common retrieval-to-display pipeline
1. Generate candidate chunk/doc hits with raw scores.
2. Apply metadata filters (`entry_id`, `administration`, `agency`, `subject`).
3. Group hits by policy entry (`entry_id` / `page_ptr_id`).
4. Track matched evidence lists and match count per entry.
5. Sort per-entry matched chunks using shared ordering:
   - score descending
   - chunk date descending
   - hit rank ascending
6. Assign `source` from top-scoring matched chunk type.
7. Apply display mapping for `doc` + `snippet` based on top source type.
8. Build optional `relevant_chunk` from best `pdf_page` evidence.
9. Rank entries globally and return top-k.

### 6.6.7 Practical implication 
Because BM25 and FAISS share the same output semantics, differences in final rankings are primarily attributable to candidate generation quality (lexical vs semantic recall), not inconsistent display/aggregation policy.

## 7. Stage 3: Fusion with Reciprocal Rank Fusion (RRF) (`scripts/hybrid/rrf.py`)
RRF combines BM25 and semantic rankings by summing reciprocal rank contributions. The core motivation is robustness across mixed query types.

### 7.1 What this stage does
1. Merges complementary lexical and semantic evidence.
2. Reduces dependence on a single retrieval mechanism.

### 7.2 Key decisions
1. Fusion constant `rrf_k` (current default in scripts: `60`).
2. Input depth from each source before fusion.
3. Entry-level merge behavior for document/chunk-level matches.

### 7.3 Expected effects and risk
1. Often increases consistency across heterogeneous query patterns.
2. Can flatten strong single-source winners if source depth or `rrf_k` is poorly tuned.

### 7.4 Fusion scoring policy (explicit)
RRF in this pipeline is unweighted across sources:
1. BM25 and semantic contributions are treated equally.
2. No per-source multiplier is applied.
3. Fusion uses rank positions, not raw score calibration across models.

Implication:
1. A result gains from appearing in both lists, regardless of score-scale differences between lexical and semantic systems.
2. This avoids fragile cross-model score normalization assumptions.
3. It can also under-emphasize an exceptionally strong single-source hit if the other source disagrees.

### 7.5 Entry-level merge behavior in practice
At fusion time, results are merged by entry identity (policy-level), then aggregated:
1. Source membership is tracked (`bm25`, `semantic`, or both).
2. Rank-origin metadata is retained for diagnostics (source ranks/scores where available).
3. Matched IDs/snippet/context fields are propagated from upstream search outputs.
4. Final fused ordering is by descending RRF score.

Why this matters:
1. The system fuses at a policy-entry level, not as independent chunk-only lists.
2. This makes final ranking easier to read in policy workflows where the unit of decision is usually the policy entry.

### 7.6 Source weighting 
Current decision: keep RRF source weights equal.

Rationale:
1. Simplicity and interpretability: one parameter (`rrf_k`) instead of source weight grid + calibration.
2. Robustness: avoids overfitting source weights to a small judged query set.
3. Fair baseline for ablations: first establish whether fusion helps before introducing weighted fusion complexity.

Future extension:
1. Weighted RRF can be added as an ablation once baseline behavior is stable.
2. Candidate design: `w_bm25` and `w_semantic` with constraints (`w_bm25 + w_semantic = 1`) to preserve interpretability.

## 8. Stage 4: Cross-Encoder Reranking (`scripts/hybrid/cross_encoder_rerank.py`)
The cross-encoder is applied after recall-focused retrieval to improve top-rank precision.

### 8.1 What this stage does
1. Re-scores RRF candidates with pairwise relevance modeling.
2. Reorders top candidates for better top-k utility.

### 8.2 Key decisions
1. Cross-encoder checkpoint: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
2. Rerank depth (`ce_top_k`) and final depth (`final_k`).
3. Input truncation (`ce_max_chars`) and runtime execution settings.

### 8.3 Tradeoff to surface
1. Higher compute/latency cost for potential precision gains.
2. Sensitive to truncation and upstream candidate pool quality.

### 8.4 Cross-encoder rerank flow (implementation-aligned)
Current pipeline behavior in `scripts/hybrid/cross_encoder_rerank.py`:
1. Input candidates:
   - either load existing RRF JSONL (`--rrf_results`)
   - or generate candidates inline via `--auto_rrf`
2. Candidate text assembly:
   - prefer full chunk text from `chunks.jsonl` (via chunk offset lookup)
   - fallback to existing snippet when chunk text cannot be fetched
3. Pair construction:
   - create `(query, candidate_text)` pairs for each rerank candidate
4. CE scoring:
   - tokenize query-text pairs with truncation (`max_length=512`)
   - run `AutoModelForSequenceClassification`
   - use logits as relevance scores (`ce_score`)
5. CE sort:
   - primary key: `ce_score` desc
   - tie-break: `rrf_score` desc, then entry ID
6. Merge policy:
   - only top `ce_top_k` fused candidates are reranked
   - reranked subset is concatenated with untouched remainder
7. Optional date-first output:
   - if `sort_by_announced_date`, final ordering is announced-date-first over reranked results
8. Final truncation:
   - output first `final_k` hits

### 8.5 Function-specific notes
1. `_get_text_for_hit(...)`:
   - what: fetches text for each fused hit by chunk ID from chunk-offset sqlite map
   - why: CE quality is better when using raw chunk text vs short snippet-only context
2. `_cross_encoder_rerank(...)`:
   - what: loads CE model/tokenizer, builds pair batches, computes logits, writes `ce_score`
   - why: provides precise pairwise query-document scoring after recall is already handled upstream
3. `run_cross_encoder(...)`:
   - what: orchestrates candidate acquisition, rerank depth, optional date sort, and final output size
   - why: keeps CE stage pluggable on top of RRF while preserving reproducible CLI behavior

### 8.6 Key design decisions to call out in the essay
1. Rerank depth control (`ce_top_k`):
   - reranking all candidates is expensive; reranking only top fused candidates reduces cost.
2. Text truncation control (`ce_max_chars`, tokenizer max length):
   - long legal text must be bounded, but truncation can remove decisive evidence.
3. Two-tier ranking strategy:
   - CE score dominates reranked subset, while original fused order still influences tie behavior and untouched tail.
4. Candidate-source dependence:
   - CE can only improve what upstream recall provides; if relevant policy is absent from fused pool, CE cannot recover it.
5. Auto-RRF coupling:
   - CE stage can be run end-to-end from query in one command, but this also couples CE outcomes to current BM25/FAISS/RRF settings.

### 8.7 Observed behavior pattern to discuss
From current results, CE is not uniformly monotonic at aggregate level. This should be framed as expected for a precision stage with finite rerank depth and truncated inputs:
1. CE can fix ordering among semantically similar candidates.
2. CE can regress when truncated text hides critical legal qualifiers.
3. CE effectiveness depends strongly on upstream candidate quality and diversity.

### 8.8 Suggested ablations for Stage 4 specifically
1. `ce_top_k` sweep (for example 10/20/50/100).
2. `ce_max_chars` sweep (short vs long context windows).
3. Candidate text source ablation:
   - full chunk text vs snippet-only text
4. Date-sort interaction ablation:
   - evaluate with and without `sort_by_announced_date` post-CE sorting.
5. Model checkpoint ablation:
   - compare current CE checkpoint to stronger/domain-adapted alternatives.

## 9. Evaluation Setup and Metric Definitions

All retrieval stages are evaluated on the same judged query set using a shared evaluation script. Metrics are computed per query and macro-averaged to allow direct comparison across methods.

### 9.1 Metrics
1. `nDCG@5` — graded ranking quality within the top 5 results.
2. `Success@5 (>=2)` — whether at least one sufficiently relevant result appears in the top 5.
3. `MRR@5 (>=2)` — reciprocal rank of the first sufficiently relevant result within the top 5.
4. `P@5 (>=2)` — fraction of sufficiently relevant results in the top 5.

### 9.2 Summary values (15-query judged set)
1. BM25: `nDCG@5 = 0.9773`, `MRR@5 = 0.9667`, `Success@5 = 1.0`, `P@5 = 0.3733`
2. Semantic: `nDCG@5 = 0.9679`, `MRR@5 = 0.9467`, `Success@5 = 1.0`, `P@5 = 0.4400`
3. RRF: `nDCG@5 = 0.9740`, `MRR@5 = 0.9333`, `Success@5 = 0.9333`, `P@5 = 0.3867`
4. Cross-encoder (on RRF): `nDCG@5 = 0.9715`, `MRR@5 = 0.9000`, `Success@5 = 0.9333`, `P@5 = 0.3867`

### 9.3 Observed behavior
1. BM25 performs strongest for early-rank placement (`MRR@5`).
2. Semantic retrieval achieves the highest density of relevant results in the top 5 (`P@5`).
3. Additional stages (fusion and reranking) do not consistently improve all metrics under the current configuration.

### 9.4 Evaluation code/artifacts
1. `tests/eval_ir_metrics.py`
2. `data/evals/eval_bm25/summary_metrics.csv`
3. `data/evals/eval_semantic/summary_metrics.csv`
4. `data/evals/eval_rrf/summary_metrics.csv`
5. `data/evals/eval_ce/summary_metrics.csv`

## 10. Per-Query Behavior and Case Studies

This section examines query-level differences across retrieval stages to highlight where each method improves or degrades performance.

### 10.1 Observed counts
1. Semantic retrieval achieves higher `nDCG@5` than BM25 on 4 queries.
2. BM25 achieves higher `nDCG@5` than semantic retrieval on 6 queries.
3. BM25 and semantic retrieval are tied on 5 queries.
4. RRF outperforms the better baseline (`max(BM25, semantic)`) on 3 queries.
5. Cross-encoder reranking improves over RRF on 7 queries.

### 10.2 Representative examples
1. **BM25-over-semantic case (largest gap in current set)**  
   `ICE detention standards`: BM25 `0.896` vs semantic `0.768` (`Δ=-0.128` for semantic).

2. **BM25-over-semantic case (policy wording specificity)**  
   `categorical parole programs CHNV`: BM25 `0.945` vs semantic `0.924` (`Δ=-0.021` for semantic).

3. **Semantic-over-BM25 case (representative)**  
   `OMB pauses agency grant loan financial assistance programs`: semantic `1.000` vs BM25 `0.973` (`Δ=+0.027` for semantic).

4. **Semantic-over-BM25 case**  
   `asylum cooperative agreements interim final rule`: semantic `1.000` vs BM25 `0.973` (`Δ=+0.027` for semantic).

5. **RRF win over best baseline (`max(BM25, semantic)`)**  
   `ICE detention standards`: RRF `0.983` vs `max(BM25=0.896, semantic=0.768) = 0.896` (`Δ=+0.087`).

6. **Cross-encoder improvement**  
   `public charge rule` improves from RRF `0.933` to CE `1.000`.

7. **Cross-encoder regression**  
   `asylum policy` drops from RRF `0.928` to CE `0.794`.

## 11. Decision Ledger (Design Knobs and Alternatives)
This section should function as the technical core of the essay: a transparent map from configuration to behavior.

### 11.1 Embedding-side knobs
1. Pooling (`mean` vs `cls`)
2. Normalization (on/off)
3. Encoder checkpoint choice
4. Max length and chunking interactions

### 11.2 ANN/retrieval knobs
1. FAISS index type and HNSW settings
2. Candidate depth/oversampling in BM25 and FAISS
3. Fusion constant `rrf_k`
4. Output surfacing rules (source and display policy)

### 11.3 Reranking knobs
1. CE model checkpoint
2. `ce_top_k`, `final_k`, `ce_max_chars`

For each knob, document:
1. Current value
2. Candidate alternatives
3. Expected effect
4. Observed effect in metrics/per-query behavior
5. Operational cost (latency, memory, complexity)

## 12. Threats to Validity and Limitations
Discuss constraints that affect confidence and generalization:
1. Limited judged query set size.
2. Domain/time sensitivity in policy corpora.
3. Potential mismatch between offline ranking metrics and user outcomes.
4. Partial ablation coverage across all major knobs.
5. Limited formal latency benchmarking in current reporting.

## 13. Future Work and Experiment Plan
Convert current findings into a concrete ablation roadmap:
1. Pooling ablation (`mean` vs `cls`) with fixed index/search settings.
2. RRF sweep across `rrf_k` and source depths.
3. CE depth/truncation sweep (`ce_top_k`, `ce_max_chars`).
4. FAISS search sweep (`ef_search`) for recall/latency frontier.
5. Optional larger/domain-adapted reranker benchmark.
6. Statistical confidence reporting on paired per-query deltas.

## 14. Reproducibility Appendix
Current config anchor:
1. `config.json`

Current pipeline commands:
```bash
# Environment
conda activate iptp

# Step 0: data assembly and preprocessing
python -m scripts.get_data.assemble_policies --config config.json
python -m scripts.get_data.download_documents --config config.json
python -m scripts.get_data.extract_pages --config config.json
python -m scripts.get_data.chunk_pages --config config.json

# Stage 1: BM25
python -m scripts.bm25.build_bm25 --config config.json
python -m scripts.bm25.search_bm25 --config config.json --q "temporary protected status" --top_k 10 --max_candidates 100
python -m scripts.bm25.export_bm25_results_csv --config config.json --queries data/text/queries.txt --out data/text/bm25_results.csv

# Stage 2: Semantic
python -m scripts.semantic.embed_chunks --chunks data/sample_100/chunks/chunks.jsonl --normalize --overwrite
python -m scripts.semantic.build_faiss --run_dir data/embeddings/bge_mean_norm --overwrite
python -m scripts.semantic.search_faiss --q "temporary protected status" --run_dir data/embeddings/bge_mean_norm --chunks data/sample_100/chunks/chunks.jsonl --top_k 10
python -m scripts.semantic.export_faiss_results_csv --queries data/text/queries.txt --out data/text/faiss_results.csv

# Stage 3: RRF
python -m scripts.hybrid.rrf --q "temporary protected status" --config config.json --run_dir data/embeddings/bge_mean_norm --chunks data/sample_100/chunks/chunks.jsonl
python -m scripts.hybrid.export_rrf_results_csv --queries data/text/queries.txt --config config.json --run_dir data/embeddings/bge_mean_norm --chunks data/sample_100/chunks/chunks.jsonl --out data/text/results/rrf_results.csv

# Stage 4: Cross-encoder rerank
python -m scripts.hybrid.cross_encoder_rerank --q "temporary protected status" --auto_rrf --cross_encoder_model "cross-encoder/ms-marco-MiniLM-L-6-v2"
python -m scripts.hybrid.export_cross_encoder_results_csv --queries data/text/queries.txt --config config.json --run_dir data/embeddings/bge_mean_norm --chunks data/sample_100/chunks/chunks.jsonl --cross_encoder_model "cross-encoder/ms-marco-MiniLM-L-6-v2" --out data/text/results/cross_encoder_results.csv
```

Current evaluation command style:
```bash
python tests/eval_ir_metrics.py -i data/text/results/bm25_results_scored.csv -o data/evals/eval_bm25
python tests/eval_ir_metrics.py -i data/text/results/semantic_results_scored.csv -o data/evals/eval_semantic
python tests/eval_ir_metrics.py -i data/text/results/rrf_results_scored.csv -o data/evals/eval_rrf
python tests/eval_ir_metrics.py -i data/text/results/cross_enc_results_scored.csv -o data/evals/eval_ce
```

Current runtime context:
1. Development hardware: Apple M2 (Apple Silicon, macOS).
2. Reported numbers in this scaffold are retrieval quality metrics.
3. Formal latency benchmarks should include backend (`mps`/`cpu`), corpus size, batch size, and per-stage `p50`/`p95`.
