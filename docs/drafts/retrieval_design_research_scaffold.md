# Hybrid Legal Policy Retrieval: Design Choices, Tradeoffs, and Empirical Behavior

## Abstract
This document presents a production-oriented retrieval system for legal/policy search. The pipeline combines lexical retrieval (BM25), dense semantic retrieval (FAISS-based), reciprocal rank fusion (RRF), and cross-encoder reranking. Rather than treating architecture as fixed, we make design knobs explicit: pooling strategy, embedding normalization, ANN index/search parameters, fusion constant, rerank depth, and text truncation. We evaluate each stage with shared relevance labels and report `nDCG@5`, `MRR@5`, `Success@5`, and `P@5` to show where additional complexity helps, where it does not, and why. The goal is to provide a clear systems blueprint and a defensible decision record that can guide follow-up ablations.

## 1. Problem Statement and Audience
This system is designed for retrieval of policy/legal content where users care about both exact legal terminology and broader semantic relevance. This document emphasizes interpretable retrieval behavior and technical reproducibility. The core question is not only “which model scores highest,” but “which design decisions changed behavior and at what operational cost.”

## 2. Research Questions and Contributions
We ask three primary questions. First, how much quality does each retrieval stage add compared with simpler alternatives. Second, which configuration decisions are most consequential for quality, robustness, and latency. Third, when do more advanced stages fail to improve outcomes despite higher complexity. Contributions are a modular four-stage architecture, a decision-led evaluation framing, and a concrete ablation roadmap tied to current artifacts.

## 3. System Overview
The pipeline is a staged retrieval cascade:
1. `Step 0`: data acquisition and corpus construction from AWS artifacts.
2. `Stage 1`: BM25 lexical retrieval.
3. `Stage 2`: semantic retrieval using dense embeddings and FAISS ANN search.
4. `Stage 3`: reciprocal rank fusion (RRF) of lexical + semantic results.
5. `Stage 4`: cross-encoder reranking of fused candidates.

This design separates recall-oriented stages (BM25, semantic, fusion) from precision-oriented reranking (cross-encoder), so each stage can be measured independently and swapped without redesigning the full system.

## 4. Step 0: Data Acquisition and Corpus Construction (`scripts/get_data`)
This step establishes data provenance and directly shapes retrieval quality by controlling what text enters the index. The ingestion workflow downloads source files from AWS/S3, assembles policy manifests, extracts page text, and creates chunks for indexing.

- What it does:
  - Builds a local reproducible corpus from cloud-hosted policy artifacts.
  - Converts document-level assets into page-level and chunk-level retrieval units.
  - Produces intermediate logs/state files for resume-safe processing.
- Key decisions made:
  - AWS download behavior: `skip_existing`, `verify_size`, parallel jobs, and failure logging.
  - Extraction/chunking policy: chunk boundaries and overlap, which affect both lexical and dense retrieval.
  - Manifest-first orchestration to keep IDs and metadata aligned across stages.
- Representative code:
  - `scripts/get_data/download_documents.py`
  - `scripts/get_data/assemble_policies.py`
  - `scripts/get_data/extract_pages.py`
  - `scripts/get_data/chunk_pages.py`

## 5. Stage 1: BM25 Lexical Retrieval (`scripts/bm25`)
BM25 provides exact-term retrieval and serves as a strong baseline for legal terminology, named entities, and statutory phrasing. It is usually robust when query language closely matches corpus language and can fail on paraphrases or conceptual similarity without lexical overlap.

- What it does:
  - Retrieves documents/chunks by weighted term matching.
  - Prioritizes interpretable lexical evidence for top candidates.
- Key decisions made:
  - Candidate depth and oversampling (`top_k`, `oversample`, max candidates before dedupe).
  - BM25 hyperparameters (`k1`, `b`) and indexing/tokenization assumptions.
  - Aggregation from chunk-level matches to entry-level ranking.
- Representative code:
  - `scripts/bm25/build_bm25.py`
  - `scripts/bm25/search_bm25.py`

## 6. Stage 2: Semantic Retrieval with BGE + FAISS (`scripts/semantic`)
Semantic retrieval addresses lexical mismatch by embedding queries and chunks into a dense vector space. This stage is sensitive to embedding and ANN design choices and typically trades some interpretability for broader recall.

- What it does:
  - Encodes text using a BGE encoder and retrieves nearest neighbors with FAISS.
  - Adds semantically related candidates that BM25 may miss.
- Key decisions made (current artifact values):
  - Embedding model: `BAAI/bge-base-en-v1.5`.
  - Pooling: `mean`.
  - Normalization: `true`.
  - Max input length: `256`.
  - FAISS index: HNSW (metric `ip`), with `m=32`, `ef_construction=200`, `ef_search=64`.
- Why these matter:
  - `mean` vs `cls` changes representation quality and downstream ranking behavior.
  - Normalization interacts with inner-product similarity and score stability.
  - HNSW parameters tune recall-latency tradeoff.
- Representative code/artifacts:
  - `scripts/semantic/bge_embedder.py`
  - `scripts/semantic/embed_chunks.py`
  - `scripts/semantic/build_faiss.py`
  - `scripts/semantic/search_faiss.py`
  - `data/embeddings/bge_mean_norm/manifest.json`

## 7. Stage 3: Fusion with Reciprocal Rank Fusion (RRF) (`scripts/hybrid/rrf.py`)
RRF combines BM25 and semantic rankings into a single list by summing reciprocal rank contributions. The motivation is robustness: one method can recover when the other misses a relevant item.

- What it does:
  - Merges complementary signals from lexical and semantic retrieval.
  - Reduces dependence on any single retrieval mechanism.
- Key decisions made:
  - Fusion constant `rrf_k` (current default in scripts: `60`).
  - Input candidate depth from each source before fusion.
  - Entry-level merge behavior for chunk/document matches and deduplication.
- Expected effects:
  - Often improves consistency across query types.
  - Can suppress very strong single-source rankings if fusion depth is poorly tuned.
- Representative code:
  - `scripts/hybrid/rrf.py`

## 8. Stage 4: Cross-Encoder Reranking on Fused Candidates (`scripts/hybrid/cross_encoder_rerank.py`)
The cross-encoder is applied after recall-focused retrieval to improve precision in the top ranks. It jointly encodes query-document text pairs and can better capture nuanced relevance than bi-encoder retrieval scores.

- What it does:
  - Re-scores RRF candidates with pairwise relevance modeling.
  - Reorders top results for better top-`k` usefulness.
- Key decisions made:
  - Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - Rerank depth (`ce_top_k`) and final output depth (`final_k`).
  - Text truncation (`ce_max_chars`) and batch/device strategy.
- Tradeoff to surface:
  - Higher compute cost and latency for potential gains in rank precision.
  - Sensitive to truncation and candidate pool quality from previous stages.
- Representative code:
  - `scripts/hybrid/cross_encoder_rerank.py`
  - `scripts/hybrid/export_cross_encoder_results_csv.py`

## 9. Evaluation Setup and Metric Definitions
All stages are compared under the same judged query set and evaluation script to make stage-wise differences interpretable. We report macro-averaged per-query metrics to avoid domination by any single query.

- Metrics:
  - `nDCG@5` (graded ranking quality)
  - `Success@5(>=2)` (at least one sufficiently relevant result)
  - `MRR@5(>=2)` (early placement of first sufficiently relevant result)
  - `P@5(>=2)` (fraction of sufficiently relevant results in top 5)
- Current summary values to report:
  - BM25: `nDCG@5=0.9773`, `MRR@5=0.9667`, `Success@5=1.0`, `P@5=0.3733`
  - Semantic: `nDCG@5=0.9679`, `MRR@5=0.9467`, `Success@5=1.0`, `P@5=0.4400`
  - RRF: `nDCG@5=0.9740`, `MRR@5=0.9333`, `Success@5=0.9333`, `P@5=0.3867`
  - CE on RRF: `nDCG@5=0.9715`, `MRR@5=0.9000`, `Success@5=0.9333`, `P@5=0.3867`
- Evaluation code/artifacts:
  - `tests/eval_ir_metrics.py`
  - `data/evals/eval_bm25/summary_metrics.csv`
  - `data/evals/eval_semantic/summary_metrics.csv`
  - `data/evals/eval_rrf/summary_metrics.csv`
  - `data/evals/eval_ce/summary_metrics.csv`

## 10. Decision Ledger (Design Knobs and Alternatives)
This section should act as the paper’s core engineering contribution: a transparent map from configuration choices to observed behavior.

- Embedding-side knobs:
  - Pooling (`mean` vs `cls`)
  - Normalization (on/off)
  - Encoder checkpoint (base vs larger/domain-adapted)
  - Max length and chunk policy interactions
- ANN/retrieval knobs:
  - FAISS index type and HNSW parameters
  - Candidate depth/oversampling in BM25 and FAISS
  - Fusion constant `rrf_k`
- Reranking knobs:
  - CE model checkpoint
  - `ce_top_k`, `final_k`, `ce_max_chars`
- For each knob, document:
  - Current value
  - Candidate alternatives
  - Expected effect
  - Observed effect in metrics/per-query behavior
  - Operational cost (latency, memory, complexity)

## 11. Error Analysis and Case Studies
Include short qualitative examples showing where each stage helps or hurts. This is especially important for legal/policy users who require interpretable justification for ranking outcomes.

- Suggested structure per case:
  - Query type (exact legal term, paraphrase, multi-concept, temporal/policy-specific)
  - Stage-by-stage top result changes
  - Why the ranking changed (lexical match, semantic similarity, fusion effect, CE refinement)

## 12. Threats to Validity and Limitations
Discuss constraints that affect confidence in conclusions and generalization.

- Limited query-set size and potential judgment bias.
- Domain and time sensitivity of policy corpora.
- Potential mismatch between offline metrics and real user utility.
- Incomplete ablations for all major knobs.

## 13. Future Work and Experiment Plan
Convert current findings into a concrete ablation roadmap.

1. Pooling ablation: `mean` vs `cls` with fixed FAISS and evaluation set.
2. RRF sweep: vary `rrf_k` and source depths.
3. CE depth sweep: vary `ce_top_k` and measure latency-quality frontier.
4. FAISS search sweep: vary `ef_search` for recall/latency tradeoff.
5. Optional: larger or domain-adapted reranker for legal/policy language.

## 14. Reproducibility Appendix
This section documents the current rerun path using the commands already embedded in the codebase.

Current config anchor:
- Root config: `config.json`

Current pipeline commands:
```bash
# Step 0: data assembly and preprocessing
python -m scripts.get_data.assemble_policies --config config.json --backup-prefix "$BACKUP_PREFIX" --documents-bucket "$DOCUMENTS_BUCKET"
python -m scripts.get_data.download_documents --config config.json
python -m scripts.get_data.extract_pages --config config.json
python -m scripts.get_data.chunk_pages --config config.json

# Stage 1: BM25
python -m scripts.bm25.build_bm25 --config config.json
python -m scripts.bm25.search_bm25 --config config.json --q "temporary protected status" --top_k 10 --max_candidates 100

# Stage 2: Semantic
python -m scripts.semantic.embed_chunks --chunks data/sample_100/chunks/chunks.jsonl --normalize --overwrite
python -m scripts.semantic.build_faiss --run_dir data/embeddings/bge_mean_norm --overwrite

# Stage 3: RRF
python -m scripts.hybrid.rrf --q "temporary protected status" --config config.json --run_dir data/embeddings/bge_mean_norm --chunks data/sample_100/chunks/chunks.jsonl

# Stage 4: Cross-encoder rerank
python -m scripts.hybrid.cross_encoder_rerank --q "temporary protected status" --auto_rrf --cross_encoder_model "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

Current evaluation command style:
```bash
python tests/eval_ir_metrics.py -i "data/text/bm25_results_scored.csv" -o eval_out_bm25
python tests/eval_ir_metrics.py -i "data/text/faiss_results_scored.csv" -o eval_out_faiss
```
Equivalent runs were used to produce the stage summaries under `data/evals/eval_bm25`, `data/evals/eval_semantic`, `data/evals/eval_rrf`, and `data/evals/eval_ce`.

Current runtime context:
- Development hardware: Apple M2 (Apple Silicon, macOS).
- Reported numbers in this document are retrieval quality metrics; no formal latency benchmark is claimed yet.
- When speed benchmarking is added, report backend (`mps` or `cpu`), corpus size, batch sizes, and per-stage latency (`p50`/`p95`).
