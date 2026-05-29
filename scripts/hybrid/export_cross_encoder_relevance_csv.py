"""
Export cross-encoder scores and ranks for the labeled relevance dataset.

This script is modeled after the existing cross-encoder export flow, but it
operates on the full relevance training file instead of first running retrieval.

It reads the labeled rows, scores each query/result pair with a Hugging Face
cross-encoder, ranks rows within each query by CE score, and writes an augmented
CSV containing the original columns plus:

- ce_score
- ce_rank
- ce_text_used

Usage:

python -m scripts.hybrid.export_cross_encoder_relevance_csv \
  --input data/text/results/relevance_training_data_all.csv \
  --output data/text/results/relevance_training_data_all_ce.csv \
  --cross_encoder_model "cross-encoder/ms-marco-MiniLM-L-6-v2"
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_INPUT = Path("data/text/results/relevance_training_data_all.csv")
DEFAULT_OUTPUT = Path("data/text/results/relevance_training_data_all_ce.csv")
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_CHARS = 4000


def _resolve_device(override: Optional[str]) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_by_query(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_normalize_text(row.get("query"))].append(row)
    return dict(grouped)


def _shorten_text(text: str, max_chars: int) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return ""
    if max_chars and len(cleaned) > max_chars:
        return cleaned[: max_chars - 1] + "…"
    return cleaned


def _pick_ce_text(row: Dict[str, str], max_chars: int) -> str:
    full_chunk_text = _normalize_text(row.get("full_chunk_text"))
    if full_chunk_text:
        return _shorten_text(full_chunk_text, max_chars)

    title = _normalize_text(row.get("title"))
    if title:
        return _shorten_text(title, max_chars)

    reason = _normalize_text(row.get("reason"))
    return _shorten_text(reason, max_chars)


def _load_ce_model(model_id: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return tokenizer, model


def _score_pairs(
    *,
    query: str,
    texts: Sequence[str],
    tokenizer,
    model,
    batch_size: int,
    device: torch.device,
) -> List[float]:
    scores: List[float] = []
    with torch.no_grad():
        step = max(1, int(batch_size))
        for start in range(0, len(texts), step):
            batch_texts = list(texts[start : start + step])
            batch_queries = [query] * len(batch_texts)
            enc = tokenizer(
                batch_queries,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits
            if logits.ndim == 2 and logits.shape[-1] > 1:
                logits = logits[:, -1]
            else:
                logits = logits.squeeze(-1)
            scores.extend(logits.detach().float().cpu().tolist())

    return scores


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cross-encoder scores and ranks for the relevance training dataset."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input relevance CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument(
        "--cross_encoder_model",
        default=DEFAULT_MODEL,
        help='HF cross-encoder model id, e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"',
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Cross-encoder batch size")
    parser.add_argument("--ce_max_chars", type=int, default=DEFAULT_MAX_CHARS, help="Max chars per candidate text")
    parser.add_argument("--device", default=None, help='Override CE device: "cpu", "mps", or "cuda"')
    return parser.parse_args()


def _rank_group(rows: Sequence[Dict[str, str]]) -> List[Tuple[int, Dict[str, str]]]:
    def sort_key(item: Tuple[int, Dict[str, str]]) -> Tuple[float, int, str]:
        idx, row = item
        ce_score = float(row.get("_ce_score", "-inf"))
        try:
            rrf_rank = int(float(_normalize_text(row.get("rrf_rank")) or "999999"))
        except Exception:
            rrf_rank = 999999
        title = _normalize_text(row.get("title"))
        return (-ce_score, rrf_rank, title)

    ranked = sorted(enumerate(rows), key=sort_key)
    return [(rank + 1, row) for rank, (_, row) in enumerate(ranked)]


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(input_path)
    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")

    required = {"query", "title", "full_chunk_text"}
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise RuntimeError(f"Input file missing required columns: {missing}")

    grouped = _group_by_query(rows)
    device = _resolve_device(args.device)
    tokenizer, model = _load_ce_model(str(args.cross_encoder_model), device)

    scored_total = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) + ["ce_score", "ce_rank", "ce_text_used"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for query in sorted(grouped.keys()):
            group_rows = grouped[query]
            ce_texts = [_pick_ce_text(row, int(args.ce_max_chars)) for row in group_rows]
            scores = _score_pairs(
                query=query,
                texts=ce_texts,
                tokenizer=tokenizer,
                model=model,
                batch_size=int(args.batch_size),
                device=device,
            )

            for row, score, ce_text in zip(group_rows, scores, ce_texts):
                row["_ce_score"] = f"{float(score):.6f}"
                row["ce_text_used"] = ce_text

            ranked = _rank_group(group_rows)
            for ce_rank, row in ranked:
                out_row = dict(row)
                out_row["ce_score"] = out_row.pop("_ce_score")
                out_row["ce_rank"] = ce_rank
                writer.writerow(out_row)
                scored_total += 1

    print(f"Wrote {scored_total} scored rows to {output_path}")
    print(f"Queries: {len(grouped)}")
    print(f"Model: {args.cross_encoder_model}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
