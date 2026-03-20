# Hybrid Retrieval for Legal Policy Search: Design-Oriented Technical Report (v2)

## Abstract
This report documents a four-stage legal-policy retrieval system with emphasis on system design and implementation choices. The pipeline consists of (1) corpus acquisition and chunking, (2) BM25 lexical indexing and retrieval, (3) dense embedding and FAISS ANN retrieval, (4) reciprocal rank fusion (RRF), and (5) cross-encoder reranking. Rather than presenting stages as black boxes, this version explains how each stage is constructed, what artifacts it produces, how those artifacts are consumed by the next stage, and which knobs are most likely to change quality and runtime behavior.

## 1. Design Goals and Constraints
The system was designed under the following constraints:
- Retrieval quality must remain interpretable for legal/policy workflows.
- Pipeline stages should be independently runnable and inspectable.
- Artifacts should be persisted between stages to support reproducibility and ablation.
- Candidate generation (recall) and candidate ordering (precision) should be separated.

These constraints directly motivate a staged architecture where each component has a clear contract.

## 2. End-to-End Architecture
The system is organized as a dataflow with persistent artifacts:

1. `scripts/get_data/*` -> raw documents, extracted pages, chunked corpus.
2. `scripts/bm25/build_bm25.py` -> lexical index artifacts.
3. `scripts/semantic/embed_chunks.py` -> embedding shards + embedding manifest.
4. `scripts/semantic/build_faiss.py` -> ANN index over embeddings.
5. `scripts/bm25/search_bm25.py` + `scripts/semantic/search_faiss.py` -> ranked candidate lists.
6. `scripts/hybrid/rrf.py` -> fused ranked list.
7. `scripts/hybrid/cross_encoder_rerank.py` -> reranked final list.
8. `tests/eval_ir_metrics.py` -> per-query and summary metrics.

This design makes every stage auditable: each stage consumes explicit files and writes explicit files.

## 3. Stage 0 Design: Data Acquisition and Text Preparation
### 3.1 What is designed here
The data stage is not only ingestion; it defines the unit of retrieval. The pipeline first assembles policy metadata and file manifests, then downloads files, extracts page text, and finally chunks text into retrieval units.

### 3.2 Artifact chain
- `assemble_policies.py` -> `manifest.jsonl`, policy folders, policy index.
- `download_documents.py` -> local document files.
- `extract_pages.py` -> page-level JSONL (`extracted_pages.jsonl`).
- `chunk_pages.py` -> chunk-level JSONL (`chunks.jsonl`).

### 3.3 Design choices
- Chunk sizing (`target_chars`, `min_chars`, `hard_split_chars`) determines context granularity.
- Overlap strategy improves recall across chunk boundaries.
- Resume-safe logs/state files make long ingestion runs fault-tolerant.

## 4. Stage 1 Design: BM25 (Lexical Retrieval)
### 4.1 How BM25 is designed in this project
BM25 is implemented as a build phase and a search phase.

Build phase (`scripts/bm25/build_bm25.py`):
- Reads `chunks.jsonl` as source-of-truth text.
- Tokenizes text and computes per-document term frequencies.
- Builds an inverted index and supporting offsets for fast seek-based lookup.
- Persists corpus statistics for scoring (e.g., avg document length).

Search phase (`scripts/bm25/search_bm25.py`):
- Tokenizes query using same text processing path.
- Fetches postings for query terms from the inverted index.
- Scores candidates with BM25 formula (`k1`, `b`).
- Aggregates chunk-level matches to entry-level results and returns top candidates.

### 4.2 Why this design
- Inverted index + offset lookup keeps query-time IO bounded.
- Separation of build/search supports stable indexing with repeated query experiments.
- Entry-level aggregation aligns retrieval outputs with policy-level evaluation.

### 4.3 Core artifacts
Under `data/sample_100/indexes/bm25`:
- `inverted_index.jsonl`
- `offsets.sqlite`
- `docs.jsonl`
- `docs_offsets.sqlite`
- `chunks_offsets.sqlite`
- `corpus_stats.json`

### 4.4 Current knobs and implications
- `k1`, `b`: term saturation and length normalization.
- `top_k`, `oversample`, `max_candidates`: recall-depth vs noise.
- Tokenization/stopword choices: can significantly shift legal phrase matching behavior.

## 5. Stage 2 Design: Semantic Retrieval (Embed First, Then ANN Index)
### 5.1 How semantic retrieval is designed
This stage is intentionally two-step:
1. Embed all chunks into dense vectors.
2. Build an ANN index over those vectors.

Embedding step (`scripts/semantic/embed_chunks.py`, encoder in `scripts/semantic/bge_embedder.py`):
- Streams `chunks.jsonl`.
- Encodes chunk text using `BAAI/bge-base-en-v1.5`.
- Applies `mean` pooling over token embeddings (mask-aware) and normalization (`true`).
- Writes vectors in shards plus ID maps and a manifest.

ANN indexing step (`scripts/semantic/build_faiss.py`):
- Loads embedding shards.
- Builds a FAISS HNSW index.
- Persists index binary + row-to-chunk mapping + index metadata.

Query step (`scripts/semantic/search_faiss.py`):
- Embeds query with manifest-consistent config.
- Runs ANN search in FAISS.
- Maps vector hits back to chunk/document metadata.
- Aggregates to entry-level results.

### 5.2 Why this design
- Embedding once and reusing vectors decouples expensive encoding from query-time retrieval.
- Sharded vector artifacts support resume and incremental debugging.
- Manifest-driven query embedding prevents config drift between corpus and query embeddings.

### 5.3 Current semantic configuration
From `data/embeddings/bge_mean_norm/manifest.json`:
- Model: `BAAI/bge-base-en-v1.5`
- Pooling: `mean` (configured in `scripts/semantic/bge_embedder.py`; `cls` is also supported for ablation)
- Normalize: `true`
- Max length: `256`
- FAISS: HNSW Flat (`ip`, `m=32`, `ef_construction=200`, `ef_search=64`)

### 5.4 Primary knobs
- Pooling (`mean` vs `cls`)
- Normalization (on/off)
- HNSW search depth (`ef_search`)
- Semantic candidate depth (`faiss_top_k`, oversampling)

## 6. Stage 3 Design: Reciprocal Rank Fusion (RRF)
### 6.1 How fusion is implemented
RRF consumes two ranked lists (BM25 and semantic) and computes fused scores by reciprocal-rank accumulation:
- For each source rank `r`, contribution is `1 / (rrf_k + r)`.
- Contributions are summed per entry ID.
- Result is a unified ranking with source provenance fields.

### 6.2 Why this design
- Avoids direct score calibration across heterogeneous scoring spaces.
- Improves robustness when lexical and semantic systems fail on different queries.
- Keeps fusion logic simple and auditable.

### 6.3 Key knobs
- `rrf_k` (current default `60`): controls emphasis on top ranks.
- Source list depth: determines how much tail evidence enters fusion.
- Dedup/merge strategy at entry level: affects final rank stability.

## 7. Stage 4 Design: Cross-Encoder Reranking
### 7.1 How reranking is implemented
The reranker receives fused candidates and re-scores query-document pairs with a cross-encoder:
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Input text is truncated to `ce_max_chars`.
- Top `ce_top_k` fused candidates are reranked.
- Final output list is cut to `final_k`.

### 7.2 Why this design
- First stages prioritize recall; this stage prioritizes precision in the final ordering.
- Cross-encoder pair modeling can capture relevance signals missed by independent encoders.

### 7.3 Design risks
- If upstream candidate pool misses relevant entries, reranker cannot recover them.
- Truncation may remove legally important context.
- Rerank depth too shallow can cap potential gains; too deep can add latency with little benefit.

## 8. Evaluation and Current Results
### 8.1 Metrics and protocol
Evaluation is run with `tests/eval_ir_metrics.py` using graded relevance and macro-averaged per-query metrics:
- `nDCG@5`
- `Success@5(>=2)`
- `MRR@5(>=2)`
- `P@5(>=2)`

### 8.2 Aggregate scores (15 queries)
| System | nDCG@5 | Success@5 (>=2) | MRR@5 (>=2) | P@5 (>=2) |
|---|---:|---:|---:|---:|
| BM25 | 0.9773 | 1.0000 | 0.9667 | 0.3733 |
| Semantic (BGE+FAISS) | 0.9679 | 1.0000 | 0.9467 | 0.4400 |
| RRF | 0.9740 | 0.9333 | 0.9333 | 0.3867 |
| CE on RRF | 0.9715 | 0.9333 | 0.9000 | 0.3867 |

### 8.3 Design-oriented interpretation
- BM25 currently gives the strongest early-rank behavior (`MRR@5`).
- Semantic retrieval contributes denser relevant sets (`P@5`).
- RRF and CE provide targeted per-query gains but do not yet outperform BM25 on all headline metrics.
- The result supports the architecture, but indicates tuning work is concentrated in fusion/rerank controls.

### 8.4 Error Analysis and Case Studies
Per-query behavior confirms that stage gains are query-dependent rather than uniform.

- Hybrid recovery example:
  - `ICE detention standards` improves from semantic `0.768` to RRF `0.983` (`+0.215`), indicating lexical+semantic complementarity.
- Cross-encoder gain example:
  - `public charge rule` improves from RRF `0.933` to CE `1.000` (`+0.067`), showing rerank precision gains on some high-salience policy queries.
- Cross-encoder regression example:
  - `asylum policy` drops from RRF `0.928` to CE `0.794` (`-0.134`), showing that reranking can over-correct when candidate context or truncation is suboptimal.

## 9. Stage Contracts and Decision Ledger (Design Doc View)
Each stage has a contract that enables modular experimentation.

- Stage 0 contract:
  - Input: AWS-backed policy/document sources.
  - Output: `chunks.jsonl` + metadata and logs.
- Stage 1 contract:
  - Input: `chunks.jsonl`.
  - Output: BM25 index artifacts + ranked lexical candidates.
- Stage 2 contract:
  - Input: `chunks.jsonl`.
  - Output: embedding run directory + FAISS index + ranked semantic candidates.
- Stage 3 contract:
  - Input: BM25 and semantic ranked lists.
  - Output: fused ranked list with source traces.
- Stage 4 contract:
  - Input: fused ranked list + chunk text.
  - Output: reranked final results.

This contract structure is the main reason the system can be analyzed both as a research pipeline and as production-oriented software.

Decision ledger summary (high-impact knobs):
- Embedding representation: pooling (`mean` vs `cls`), normalization, max length.
- ANN retrieval: HNSW parameters (`m`, `ef_construction`, `ef_search`) and candidate depth.
- Lexical retrieval: BM25 (`k1`, `b`), lexical candidate depth, oversampling.
- Fusion: `rrf_k` and source list depth.
- Reranking: CE model choice, `ce_top_k`, `ce_max_chars`, `final_k`.

## 10. Reproducibility (Current State)
Configuration anchor:
- `config.json`

Representative commands already documented in source headers:
```bash
python -m scripts.get_data.download_documents --config config.json
python -m scripts.get_data.extract_pages --config config.json
python -m scripts.get_data.chunk_pages --config config.json

python -m scripts.bm25.build_bm25 --config config.json
python -m scripts.bm25.search_bm25 --config config.json --q "temporary protected status" --top_k 10 --max_candidates 100

python -m scripts.semantic.embed_chunks --chunks data/sample_100/chunks/chunks.jsonl --normalize --overwrite
python -m scripts.semantic.build_faiss --run_dir data/embeddings/bge_mean_norm --overwrite

python -m scripts.hybrid.search_rrf --q "temporary protected status" --config config.json --run_dir data/embeddings/bge_mean_norm --chunks data/sample_100/chunks/chunks.jsonl
python -m scripts.hybrid.search_cross_encoder_rerank --q "temporary protected status" --auto_rrf --cross_encoder_model "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

Environment note:
- Development machine: Apple M2 (Apple Silicon, macOS).
- Current report is quality-focused; full latency benchmarking is pending.

## 11. User-Facing Limitations, Threats to Validity, and Next Work
### 11.1 User-facing limitations
- Some queries still show non-monotonic behavior across stages, so the “most complex” stage is not always the most reliable.
- Offline retrieval metrics do not fully capture user workflow outcomes such as trust, interpretability, and task completion speed.
- Reranking behavior can be sensitive to truncation and candidate pool composition, which may affect user-visible relevance consistency.

### 11.2 Threats to validity
- Query set is small (`n=15`), so conclusions are directional rather than definitive.
- Current comparison is primarily between fixed configurations, not full ablation sweeps with statistical significance testing.
- Domain/time effects in policy corpora may shift results as content and language evolve.

### 11.3 Next work
1. Pooling ablation (`mean` vs `cls`) with fixed evaluation set.
2. RRF sweep (`rrf_k` and source depths).
3. CE depth/truncation sweep (`ce_top_k`, `ce_max_chars`).
4. FAISS recall/latency sweep over `ef_search`.
5. Stage-wise latency profiling on Apple M2 with `mps` vs `cpu`.

## 12. Conclusion
This design-oriented version shows the system as a sequence of explicit contracts: build artifacts, query artifacts, fusion artifacts, and reranked outputs. The architecture is operational and analyzable today. The empirical signal is that BM25 remains a strong baseline, semantic retrieval contributes complementary relevance density, and fusion/reranking behavior depends heavily on candidate and ranking controls. The next gains are likely to come from targeted ablations on already-exposed knobs rather than introducing additional architectural complexity.
