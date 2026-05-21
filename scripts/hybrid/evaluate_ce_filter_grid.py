from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _norm_title(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _read_rows(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    by_query: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["rank"] = int(row["rank"])
            row["score"] = float(row["score"])
            row["relevance"] = int(row["relevance"])
            by_query[str(row["query"])].append(row)
    for rows in by_query.values():
        rows.sort(key=lambda r: int(r["rank"]))
    return dict(by_query)


def _attach_ce_scores(
    rrf_by_query: Dict[str, List[Dict[str, Any]]],
    ce_by_query: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    merged: Dict[str, List[Dict[str, Any]]] = {}
    for query, rrf_rows in rrf_by_query.items():
        ce_by_title = {_norm_title(row["title"]): float(row["score"]) for row in ce_by_query.get(query, [])}
        out: List[Dict[str, Any]] = []
        for row in rrf_rows:
            item = dict(row)
            item["ce_score"] = ce_by_title.get(_norm_title(row["title"]))
            out.append(item)
        merged[query] = out
    return merged


def _filter_rows(
    rows: List[Dict[str, Any]],
    *,
    threshold: float,
    ce_top_k: int,
    min_return: int,
) -> Tuple[List[Dict[str, Any]], int]:
    window = rows[: int(ce_top_k)]
    kept = [row for row in window if row.get("ce_score") is not None and float(row["ce_score"]) >= float(threshold)]
    forced = 0
    if len(kept) < int(min_return):
        for row in window:
            if row not in kept:
                kept.append(row)
                forced += 1
            if len(kept) >= int(min_return):
                break
    return kept, forced


def _precision(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(int(row["relevance"]) >= 2 for row in rows) / float(len(rows))


def _recall_within_window(kept: List[Dict[str, Any]], window: List[Dict[str, Any]]) -> float:
    relevant_total = sum(int(row["relevance"]) >= 2 for row in window)
    if relevant_total <= 0:
        return 0.0
    relevant_kept = sum(int(row["relevance"]) >= 2 for row in kept)
    return relevant_kept / float(relevant_total)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / float(len(vals)) if vals else 0.0


def evaluate_grid(
    merged: Dict[str, List[Dict[str, Any]]],
    *,
    thresholds: List[float],
    ce_top_ks: List[int],
    min_returns: List[int],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for ce_top_k in ce_top_ks:
        for threshold in thresholds:
            for min_return in min_returns:
                per_query = []
                for query, rows in merged.items():
                    window = rows[: int(ce_top_k)]
                    kept, forced = _filter_rows(
                        rows,
                        threshold=threshold,
                        ce_top_k=ce_top_k,
                        min_return=min_return,
                    )
                    per_query.append(
                        {
                            "query": query,
                            "kept": kept,
                            "forced": forced,
                            "precision": _precision(kept),
                            "recall": _recall_within_window(kept, window),
                            "count": len(kept),
                            "relevant_kept": sum(int(row["relevance"]) >= 2 for row in kept),
                            "relevant_available": sum(int(row["relevance"]) >= 2 for row in window),
                        }
                    )
                results.append(
                    {
                        "ce_top_k": ce_top_k,
                        "threshold": threshold,
                        "min_return": min_return,
                        "avg_count": _mean(item["count"] for item in per_query),
                        "avg_precision": _mean(item["precision"] for item in per_query),
                        "avg_recall_within_window": _mean(item["recall"] for item in per_query),
                        "queries_with_forced": sum(item["forced"] > 0 for item in per_query),
                        "forced_results": sum(item["forced"] for item in per_query),
                        "queries_empty": sum(item["count"] == 0 for item in per_query),
                        "per_query": per_query,
                    }
                )
    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("summary")
    print("ce_top_k,threshold,min_return,avg_count,avg_precision,avg_recall_within_window,queries_with_forced,forced_results,queries_empty")
    for row in sorted(
        results,
        key=lambda r: (
            int(r["ce_top_k"]),
            float(r["threshold"]),
            int(r["min_return"]),
        ),
    ):
        print(
            f"{row['ce_top_k']},{row['threshold']},{row['min_return']},"
            f"{row['avg_count']:.3f},{row['avg_precision']:.3f},{row['avg_recall_within_window']:.3f},"
            f"{row['queries_with_forced']},{row['forced_results']},{row['queries_empty']}"
        )


def print_details(results: List[Dict[str, Any]], *, ce_top_k: int, threshold: float, min_return: int) -> None:
    match = next(
        (
            row
            for row in results
            if int(row["ce_top_k"]) == int(ce_top_k)
            and float(row["threshold"]) == float(threshold)
            and int(row["min_return"]) == int(min_return)
        ),
        None,
    )
    if match is None:
        return
    print("")
    print(f"details ce_top_k={ce_top_k} threshold={threshold} min_return={min_return}")
    for item in match["per_query"]:
        kept = [
            (
                int(row["rank"]),
                round(float(row["ce_score"]), 3) if row.get("ce_score") is not None else None,
                int(row["relevance"]),
                row["title"],
            )
            for row in item["kept"]
        ]
        print(
            f"{item['query']}: count={item['count']} precision={item['precision']:.2f} "
            f"recall_window={item['recall']:.2f} forced={item['forced']} kept={kept}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RRF-order + CE-score filtering over scored CSVs.")
    parser.add_argument("--rrf", default="data/text/results/rrf_results_scored.csv")
    parser.add_argument("--ce", default="data/text/results/cross_enc_results_scored.csv")
    parser.add_argument("--thresholds", default="1.5,2.0")
    parser.add_argument("--ce_top_ks", default="5")
    parser.add_argument("--min_returns", default="0,1,2")
    parser.add_argument("--detail_threshold", type=float, default=2.0)
    parser.add_argument("--detail_ce_top_k", type=int, default=5)
    parser.add_argument("--detail_min_return", type=int, default=2)
    return parser.parse_args()


def _parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    rrf_by_query = _read_rows(Path(args.rrf))
    ce_by_query = _read_rows(Path(args.ce))
    merged = _attach_ce_scores(rrf_by_query, ce_by_query)

    results = evaluate_grid(
        merged,
        thresholds=_parse_float_list(args.thresholds),
        ce_top_ks=_parse_int_list(args.ce_top_ks),
        min_returns=_parse_int_list(args.min_returns),
    )
    print_summary(results)
    print_details(
        results,
        ce_top_k=args.detail_ce_top_k,
        threshold=args.detail_threshold,
        min_return=args.detail_min_return,
    )


if __name__ == "__main__":
    main()
