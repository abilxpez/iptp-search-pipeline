#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

"""
How to run 

python tests/eval_ir_metrics.py -i "data/text/bm25_results_scored.csv" -o eval_out_bm25

python tests/eval_ir_metrics.py -i "data/text/faiss_results_scored.csv" -o eval_out_faiss


"""


def dcg_at_k(rels, k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        dcg += float(rel) / math.log2(i + 1)
    return dcg


def ndcg_at_k(rels, k: int) -> float:
    actual = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return (actual / ideal) if ideal > 0 else 0.0


def precision_at_k(rels, k: int, threshold: int) -> float:
    topk = rels[:k]
    return (sum(1 for r in topk if r >= threshold) / len(topk)) if topk else 0.0


def success_at_k(rels, k: int, threshold: int) -> float:
    topk = rels[:k]
    return 1.0 if any(r >= threshold for r in topk) else 0.0


def mrr_at_k(rels, k: int, threshold: int) -> float:
    topk = rels[:k]
    for idx, r in enumerate(topk, start=1):
        if r >= threshold:
            return 1.0 / idx
    return 0.0


def evaluate(df: pd.DataFrame, k: int, threshold: int, query_col: str, rank_col: str, rel_col: str):
    df = df.copy()

    # Coerce types
    df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
    df[rel_col] = pd.to_numeric(df[rel_col], errors="coerce")
    df = df.dropna(subset=[query_col, rank_col, rel_col])
    df[rank_col] = df[rank_col].astype(int)
    df[rel_col] = df[rel_col].astype(int)

    # Preserve query order as first appearance in the CSV
    df["__row_idx"] = range(len(df))
    first_idx = df.groupby(query_col)["__row_idx"].min()
    query_order_map = {q: i for i, q in enumerate(first_idx.sort_values().index.tolist())}
    df["__query_order"] = df[query_col].map(query_order_map).astype(int)

    # Sort by *original query order*, then by rank within query
    df = df.sort_values(["__query_order", rank_col, "__row_idx"]).copy()

    per_query_rows = []
    for q, g in df.groupby(query_col, sort=False):
        # Strictly take rank<=k if present; otherwise take first k rows
        rels_rank_filtered = g.loc[g[rank_col] <= k, rel_col].tolist()
        rels_topk = rels_rank_filtered if len(rels_rank_filtered) > 0 else g[rel_col].tolist()[:k]

        per_query_rows.append(
            {
                "query": q,
                f"nDCG@{k}": ndcg_at_k(rels_topk, k),
                f"Success@{k}(>={threshold})": success_at_k(rels_topk, k, threshold),
                f"MRR@{k}(>={threshold})": mrr_at_k(rels_topk, k, threshold),
                f"P@{k}(>={threshold})": precision_at_k(rels_topk, k, threshold),
                "num_results_considered": len(rels_topk),
            }
        )

    per_query = pd.DataFrame(per_query_rows)

    summary = {
        "num_queries": int(per_query.shape[0]),
        f"nDCG@{k}": float(per_query[f"nDCG@{k}"].mean()) if not per_query.empty else 0.0,
        f"Success@{k}(>={threshold})": float(per_query[f"Success@{k}(>={threshold})"].mean())
        if not per_query.empty
        else 0.0,
        f"MRR@{k}(>={threshold})": float(per_query[f"MRR@{k}(>={threshold})"].mean())
        if not per_query.empty
        else 0.0,
        f"P@{k}(>={threshold})": float(per_query[f"P@{k}(>={threshold})"].mean())
        if not per_query.empty
        else 0.0,
    }

    return per_query, summary


def save_plot_sorted(per_query: pd.DataFrame, out_png: Path, k: int):
    col = f"nDCG@{k}"
    if per_query.empty:
        return

    plot_df = per_query.sort_values(col, ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(plot_df)), plot_df[col].values)
    plt.xlabel("Queries (sorted by nDCG)")
    plt.ylabel(f"nDCG@{k}")
    plt.title(f"Per-query nDCG@{k} (sorted)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input CSV")
    ap.add_argument("-o", "--out_dir", default="eval_out", help="Output directory")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=int, default=2)
    ap.add_argument("--query_col", default="query")
    ap.add_argument("--rank_col", default="rank")
    ap.add_argument("--rel_col", default="relevance")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    per_query, summary = evaluate(
        df, k=args.k, threshold=args.threshold,
        query_col=args.query_col, rank_col=args.rank_col, rel_col=args.rel_col
    )

    # Save outputs
    per_query_csv = out_dir / "per_query_metrics.csv"
    per_query.to_csv(per_query_csv, index=False)

    summary_csv = out_dir / "summary_metrics.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    plot_png = out_dir / f"ndcg_at_{args.k}_per_query.png"
    save_plot_sorted(per_query, plot_png, k=args.k)

    # Print summary
    print("\n=== Macro-averaged metrics (per-query average) ===")
    print(f"Queries: {summary['num_queries']}")
    for k, v in summary.items():
        if k != "num_queries":
            print(f"{k}: {v:.4f}")

    print(f"\nSaved per-query table: {per_query_csv}")
    print(f"Saved summary table:   {summary_csv}")
    print(f"Saved plot:            {plot_png}")


if __name__ == "__main__":
    main()