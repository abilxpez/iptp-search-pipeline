from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _norm_title(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _load_labels(paths: List[Path]) -> Tuple[List[str], Dict[Tuple[str, str], int]]:
    queries: List[str] = []
    labels: Dict[Tuple[str, str], int] = {}
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                query = str(row["query"])
                if query not in queries:
                    queries.append(query)
                key = (query, _norm_title(row["title"]))
                rel = int(row["relevance"])
                labels[key] = max(rel, labels.get(key, rel))
    return queries, labels


def _post_search(api_url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url.rstrip("/") + "/search",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _precision(rels: List[Optional[int]]) -> float:
    known = [r for r in rels if r is not None]
    if not known:
        return 0.0
    return sum(r >= 2 for r in known) / float(len(known))


def _mean(values: List[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def evaluate(
    *,
    api_url: str,
    queries: List[str],
    labels: Dict[Tuple[str, str], int],
    threshold: float,
    ce_top_k: int,
    ce_min_return: int,
    final_k: int,
    timeout: float,
) -> Dict[str, Any]:
    per_query: List[Dict[str, Any]] = []
    for query in queries:
        payload = {
            "q": query,
            "final_k": final_k,
            "ce_filter": True,
            "ce_threshold": threshold,
            "ce_top_k": ce_top_k,
            "ce_min_return": ce_min_return,
        }
        started = time.perf_counter()
        try:
            response = _post_search(api_url, payload, timeout)
            error = None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            response = {"results": [], "elapsed_ms": None, "timings_ms": {}}
            error = str(exc)
        elapsed_wall_ms = round((time.perf_counter() - started) * 1000.0, 2)

        rows = []
        rels: List[Optional[int]] = []
        for result in response.get("results", []):
            title = str(result.get("title") or "")
            rel = labels.get((query, _norm_title(title)))
            rels.append(rel)
            rows.append(
                {
                    "title": title,
                    "ce_score": result.get("ce_score"),
                    "relevance": rel,
                }
            )

        per_query.append(
            {
                "query": query,
                "count": len(rows),
                "known_count": sum(r is not None for r in rels),
                "precision_known": _precision(rels),
                "relevant_known": sum((r or 0) >= 2 for r in rels),
                "elapsed_ms": response.get("elapsed_ms"),
                "wall_ms": elapsed_wall_ms,
                "ce_inference_ms": (response.get("timings_ms") or {}).get("service.ce.inference"),
                "faiss_embed_ms": (response.get("timings_ms") or {}).get("service.faiss.embed_query"),
                "error": error,
                "results": rows,
            }
        )

    known_precisions = [row["precision_known"] for row in per_query if row["known_count"] > 0]
    return {
        "threshold": threshold,
        "ce_top_k": ce_top_k,
        "ce_min_return": ce_min_return,
        "avg_count": _mean([float(row["count"]) for row in per_query]),
        "avg_known_count": _mean([float(row["known_count"]) for row in per_query]),
        "avg_precision_known": _mean(known_precisions),
        "avg_elapsed_ms": _mean([float(row["elapsed_ms"] or 0) for row in per_query]),
        "median_elapsed_ms": statistics.median([float(row["elapsed_ms"] or 0) for row in per_query]),
        "avg_ce_inference_ms": _mean([float(row["ce_inference_ms"] or 0) for row in per_query]),
        "errors": sum(1 for row in per_query if row["error"]),
        "per_query": per_query,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate live API CE filtering over labeled query set.")
    parser.add_argument("--api_url", default="http://127.0.0.1:8010")
    parser.add_argument(
        "--labels",
        default=(
            "data/text/results/rrf_results_scored.csv,"
            "data/text/results/bm25_results_scored.csv,"
            "data/text/results/cross_enc_results_scored.csv"
        ),
    )
    parser.add_argument("--thresholds", default="1.5,2.0")
    parser.add_argument("--ce_top_ks", default="5,10")
    parser.add_argument("--ce_min_returns", default="0,1,2")
    parser.add_argument("--final_k", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--detail_threshold", type=float, default=2.0)
    parser.add_argument("--detail_ce_top_k", type=int, default=5)
    parser.add_argument("--detail_ce_min_return", type=int, default=0)
    return parser.parse_args()


def _parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    label_paths = [Path(part.strip()) for part in args.labels.split(",") if part.strip()]
    queries, labels = _load_labels(label_paths)

    summaries = []
    detailed = None
    for ce_top_k in _parse_int_list(args.ce_top_ks):
        for threshold in _parse_float_list(args.thresholds):
            for ce_min_return in _parse_int_list(args.ce_min_returns):
                result = evaluate(
                    api_url=args.api_url,
                    queries=queries,
                    labels=labels,
                    threshold=threshold,
                    ce_top_k=ce_top_k,
                    ce_min_return=ce_min_return,
                    final_k=args.final_k,
                    timeout=args.timeout,
                )
                summaries.append(result)
                if (
                    float(threshold) == float(args.detail_threshold)
                    and int(ce_top_k) == int(args.detail_ce_top_k)
                    and int(ce_min_return) == int(args.detail_ce_min_return)
                ):
                    detailed = result

    print("summary")
    print("ce_top_k,threshold,ce_min_return,avg_count,avg_known_count,avg_precision_known,avg_elapsed_ms,median_elapsed_ms,avg_ce_inference_ms,errors")
    for row in summaries:
        print(
            f"{row['ce_top_k']},{row['threshold']},{row['ce_min_return']},"
            f"{row['avg_count']:.3f},{row['avg_known_count']:.3f},{row['avg_precision_known']:.3f},"
            f"{row['avg_elapsed_ms']:.2f},{row['median_elapsed_ms']:.2f},{row['avg_ce_inference_ms']:.2f},{row['errors']}"
        )

    if detailed is not None:
        print("")
        print(
            f"details ce_top_k={detailed['ce_top_k']} threshold={detailed['threshold']} "
            f"ce_min_return={detailed['ce_min_return']}"
        )
        for row in detailed["per_query"]:
            compact = [
                (item["relevance"], None if item["ce_score"] is None else round(float(item["ce_score"]), 3), item["title"])
                for item in row["results"]
            ]
            print(
                f"{row['query']}: count={row['count']} known={row['known_count']} "
                f"precision_known={row['precision_known']:.2f} elapsed={row['elapsed_ms']} results={compact}"
            )


if __name__ == "__main__":
    main()
