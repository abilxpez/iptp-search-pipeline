# Hybrid Retrieval for Legal Policy Search: A Design-Driven Evaluation

## Abstract
This report presents a four-stage retrieval system for legal and policy search and evaluates how key design choices affect retrieval quality. The system combines lexical retrieval (BM25), dense semantic retrieval with ANN search, reciprocal rank fusion (RRF), and cross-encoder reranking. The central objective is not only to compare models, but to make implementation decisions explicit and testable, including pooling strategy, normalization, ANN index settings, fusion constants, and rerank depth. On a 15-query judged set, BM25 is the strongest single method on `nDCG@5` and `MRR@5`, semantic retrieval provides the highest `P@5`, and downstream fusion/reranking improve some hard queries while degrading others. These outcomes motivate a design-led interpretation: additional pipeline complexity is useful when tuned to specific failure modes, but does not guarantee monotonic gains.

## 1. Introduction
Legal-policy retrieval requires balancing exact terminology with semantic flexibility. Lexical methods are reliable for statute-like phrasing and named entities, while dense methods help on paraphrased or concept-level queries. A hybrid architecture is therefore attractive, but introduces many design choices that can materially shift behavior.

This report documents the current pipeline as both an experimental system and an engineering artifact. The emphasis is on decision transparency: which choices were made, where they appear in code, and how observed metrics changed under the current configuration.

## 2. System and Data Pipeline
### 2.1 Step 0: Data Acquisition and Corpus Construction
The corpus is built from AWS-backed source artifacts via scripts under `scripts/get_data`:
- `assemble_policies.py` builds local policy folders and manifests.
- `download_documents.py` downloads referenced files from S3.
- `extract_pages.py` extracts page-level PDF text.
- `chunk_pages.py` produces chunk-level retrieval units.

Configuration is anchored in root `config.json`. Relevant active settings include:
- Sample size: `assemble.n = 100`.
- Chunking: `target_chars=1500`, `min_chars=300`, overlap enabled with `units=2`.
- Download behavior: `skip_existing=true`, `verify_size=true`, `jobs=12`.

These preprocessing decisions define retrieval units and metadata quality and therefore affect all downstream stages.

### 2.2 Stage 1: BM25 Lexical Retrieval
BM25 index construction and retrieval are implemented in `scripts/bm25/build_bm25.py` and `scripts/bm25/search_bm25.py`.

Current defaults (from `config.json` and script interfaces):
- `k1=1.2`, `b=0.75`.
- Top-k and oversampling controls at query time.
- Entry-level aggregation from chunk-level hits.

Design intent: preserve strong exact-match performance on legal terminology and maintain interpretability via term-based scoring.

### 2.3 Stage 2: Semantic Retrieval (Dense + FAISS)
Dense embeddings are produced and indexed using `scripts/semantic/embed_chunks.py` and `scripts/semantic/build_faiss.py`, queried via `scripts/semantic/search_faiss.py`.

Current embedding/index artifact (`data/embeddings/bge_mean_norm/manifest.json`):
- Model: `BAAI/bge-base-en-v1.5`
- Pooling: `mean`
- Normalization: `true`
- Max length: `256`
- Dimension: `768`
- FAISS: HNSW Flat, metric `ip`
- HNSW params: `m=32`, `ef_construction=200`, `ef_search=64`

Design intent: increase semantic recall while controlling ANN latency/recall tradeoff.

### 2.4 Stage 3: Reciprocal Rank Fusion (RRF)
RRF is implemented in `scripts/hybrid/rrf.py`, combining BM25 and semantic ranks with reciprocal-rank accumulation.

Current default:
- `rrf_k=60`

Design intent: reduce single-model brittleness by combining lexical and semantic evidence at rank level rather than score calibration.

### 2.5 Stage 4: Cross-Encoder Reranking
Reranking is implemented in `scripts/hybrid/cross_encoder_rerank.py` on top of fused candidates.

Current model and key controls:
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Candidate rerank depth: `ce_top_k` (default `20`)
- Input truncation: `ce_max_chars` (default `4000`)
- Final output depth: `final_k`

Design intent: improve top-rank precision after recall-focused retrieval stages.

## 3. Evaluation Setup
Evaluation uses `tests/eval_ir_metrics.py` over scored CSV outputs and reports macro-averaged per-query metrics:
- `nDCG@5`
- `Success@5(>=2)`
- `MRR@5(>=2)`
- `P@5(>=2)`

Current evaluation directories:
- `data/evals/eval_bm25`
- `data/evals/eval_semantic`
- `data/evals/eval_rrf`
- `data/evals/eval_ce`

All summary scores below are computed on 15 queries.

## 4. Results
### 4.1 Aggregate Comparison
| System | nDCG@5 | Success@5 (>=2) | MRR@5 (>=2) | P@5 (>=2) |
|---|---:|---:|---:|---:|
| BM25 | 0.9773 | 1.0000 | 0.9667 | 0.3733 |
| Semantic (BGE+FAISS) | 0.9679 | 1.0000 | 0.9467 | 0.4400 |
| RRF (BM25+Semantic) | 0.9740 | 0.9333 | 0.9333 | 0.3867 |
| CE on RRF | 0.9715 | 0.9333 | 0.9000 | 0.3867 |

### 4.2 Interpretation
Three patterns stand out.

1. No single stage dominates all metrics.
BM25 leads on `nDCG@5` and `MRR@5`; semantic retrieval leads on `P@5`. This indicates complementarity rather than strict replacement.

2. Added complexity is non-monotonic.
RRF and CE improve selected queries but do not improve aggregate headline metrics in this run configuration.

3. Precision and early-rank behavior can diverge.
Semantic retrieval increases relevant-item density (`P@5`) while BM25 still places first highly relevant hits earlier on average (`MRR@5`).

## 5. Per-Query Behavior and Failure Modes
A per-query comparison across all four systems shows heterogeneous gains.

Observed counts on the 15-query set:
- Semantic `nDCG@5` > BM25 on 4 queries.
- RRF > both BM25 and semantic on 3 queries.
- CE > RRF on 7 queries.

Representative examples:
- Strong hybrid recovery: `ICE detention standards` improves from semantic `0.768` to RRF `0.983` (`+0.215`).
- CE gain case: `public charge rule` improves from RRF `0.933` to CE `1.000` (`+0.067`).
- CE regression case: `asylum policy` drops from RRF `0.928` to CE `0.794` (`-0.134`).

Implication: CE reranking is not uniformly beneficial under current candidate depth and truncation settings; targeted tuning is required.

## 6. Design Decisions and Tradeoffs
This section captures design choices that are most likely to shift outcomes if changed.

### 6.1 Retrieval Unit and Chunking
- Decision: chunk-level indexing with overlap.
- Benefit: better coverage of long policy documents.
- Risk: duplicate/near-duplicate evidence can distort top ranks and fusion dynamics.

### 6.2 Lexical Parameters (`k1`, `b`, candidate depth)
- Decision: BM25 defaults near standard IR settings.
- Benefit: stable exact-match baseline.
- Risk: can underperform on paraphrased queries if lexical overlap is weak.

### 6.3 Embedding Representation (Pooling and Normalization)
- Decision: `mean` pooling with normalized vectors.
- Benefit: robust default for sentence/chunk retrieval with inner-product ANN.
- Risk: not guaranteed optimal for this legal-policy corpus versus `cls` pooling.

### 6.4 ANN Design (HNSW settings)
- Decision: HNSW with moderate search depth (`ef_search=64`).
- Benefit: practical retrieval speed with strong approximate recall.
- Risk: ANN miss behavior can cascade into weaker fusion and rerank candidate pools.

### 6.5 Fusion Constant and Source Depth
- Decision: `rrf_k=60` with high candidate caps.
- Benefit: robust score-free rank merging.
- Risk: can flatten high-confidence winners from one source if not tuned for corpus/query mix.

### 6.6 Cross-Encoder Rerank Depth and Truncation
- Decision: rerank top fused candidates with fixed max chars.
- Benefit: deeper semantic relevance modeling than bi-encoder stages.
- Risk: truncation and candidate quality bottlenecks can cause regressions on some queries.

## 7. Reproducibility (Current State)
This project is reproducible with the current codebase and documented command interfaces.

Configuration anchor:
- `config.json`

Representative run commands are embedded in script headers and include:
- `python -m scripts.get_data.download_documents --config config.json`
- `python -m scripts.get_data.extract_pages --config config.json`
- `python -m scripts.get_data.chunk_pages --config config.json`
- `python -m scripts.bm25.build_bm25 --config config.json`
- `python -m scripts.bm25.search_bm25 --config config.json --q "..."`
- `python -m scripts.semantic.embed_chunks --chunks ... --normalize --overwrite`
- `python -m scripts.semantic.build_faiss --run_dir data/embeddings/bge_mean_norm --overwrite`
- `python -m scripts.hybrid.search_rrf --q "..." --config config.json --run_dir ... --chunks ...`
- `python -m scripts.hybrid.search_cross_encoder_rerank --q "..." --auto_rrf --cross_encoder_model "cross-encoder/ms-marco-MiniLM-L-6-v2"`
- `python tests/eval_ir_metrics.py -i <scored_csv> -o <eval_dir>`

Runtime context:
- Development machine: Apple M2 (Apple Silicon, macOS).
- Reported results are quality metrics; no formal latency benchmark is claimed in this draft.

## 8. Limitations
- Small judged query set (`n=15`) limits statistical strength.
- Current comparisons are primarily configuration snapshots, not full ablation sweeps.
- Offline ranking metrics may not fully reflect user task outcomes in legal research workflows.
- Latency/throughput was not benchmarked yet, despite being an important downstream criterion.

## 9. Next Experimental Plan (Master’s-Level Extension)
To transition this report from strong system documentation to stronger empirical evidence, the following experiments are priority:

1. Pooling ablation: `mean` vs `cls` with all other settings fixed.
2. RRF sweep: vary `rrf_k` and per-source candidate depth.
3. CE sweep: vary `ce_top_k` and `ce_max_chars` jointly.
4. FAISS recall/latency sweep: vary `ef_search` and capture quality plus runtime.
5. Statistical reporting: paired per-query deltas with confidence intervals where feasible.

## 10. Conclusion
The current pipeline demonstrates that hybrid legal-policy retrieval is best understood as a sequence of design decisions, not a single-model contest. BM25, dense retrieval, fusion, and reranking each contribute different strengths, and the present results show both gains and regressions depending on query type and stage configuration. The practical outcome is clear: the architecture is viable, but its strongest form will come from targeted ablations on the decision knobs already identified and implemented.
