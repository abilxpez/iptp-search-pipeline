#!/usr/bin/env python3
"""
How to use:

python scripts/get_data/inspect_chunk_overlap.py \
  --chunks data/sample_100/chunks/chunks.jsonl


"""


import argparse
import json
import re
from collections import defaultdict
from statistics import mean, median

RE_WORD = re.compile(r"[A-Za-z0-9]+")

def word_set(text: str):
    # lower, basic tokenization; ignore very short tokens
    return {w.lower() for w in RE_WORD.findall(text) if len(w) >= 2}

def containment(a_set, b_set) -> float:
    # fraction of A contained in B
    if not a_set:
        return 0.0
    return len(a_set & b_set) / len(a_set)

def percentile(sorted_vals, p: float) -> float:
    # p in [0, 1]
    if not sorted_vals:
        return 0.0
    idx = int(round(p * (len(sorted_vals) - 1)))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]

def parse_args():
    ap = argparse.ArgumentParser(description="Compute overlap stats for consecutive chunks.")
    ap.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    ap.add_argument("--max-groups", type=int, default=None, help="Limit number of groups processed (debug)")
    ap.add_argument("--min-containment", type=float, default=0.0, help="Only include pairs with containment >= this")
    ap.add_argument("--show-extremes", type=int, default=0, help="Print N worst/best pairs (0 = off)")
    ap.add_argument("--context-chars", type=int, default=240, help="Chars of prev tail / next head when showing extremes")
    return ap.parse_args()

def main():
    args = parse_args()

    groups = defaultdict(list)
    with open(args.chunks, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            key = (str(o.get("page_ptr_id")), str(o.get("policydocument_id")))
            groups[key].append(o)

    keys = list(groups.keys())
    if args.max_groups is not None:
        keys = keys[: args.max_groups]

    pair_scores = []
    pair_records = []  # (score, key, prev_id, next_id, prev_text, next_text)

    for i, key in enumerate(keys, start=1):
        chunks = groups[key]
        # deterministic order: (page_start, page_end, chunk_id)
        chunks.sort(key=lambda c: (int(c.get("page_start", 0)), int(c.get("page_end", 0)), c.get("chunk_id", "")))

        for j in range(len(chunks) - 1):
            prev = chunks[j]
            nxt = chunks[j + 1]

            prev_ws = word_set(prev.get("text", ""))
            next_ws = word_set(nxt.get("text", ""))

            c = containment(prev_ws, next_ws)  # "prev contained in next"
            if c < args.min_containment:
                continue

            pair_scores.append(c)
            pair_records.append((
                c,
                key,
                prev.get("chunk_id", ""),
                nxt.get("chunk_id", ""),
                prev.get("text", ""),
                nxt.get("text", ""),
                (prev.get("page_start"), prev.get("page_end")),
                (nxt.get("page_start"), nxt.get("page_end")),
            ))

    print(f"groups_loaded: {len(groups)}")
    print(f"groups_processed: {len(keys)}")
    print(f"pairs_used: {len(pair_scores)} (min_containment={args.min_containment})")

    if not pair_scores:
        print("No pairs to summarize.")
        return

    s = sorted(pair_scores)
    print()
    print("containment_prev_in_next stats:")
    print(f"  mean   {mean(s):.3f}")
    print(f"  median {median(s):.3f}")
    print(f"  p10    {percentile(s, 0.10):.3f}")
    print(f"  p25    {percentile(s, 0.25):.3f}")
    print(f"  p75    {percentile(s, 0.75):.3f}")
    print(f"  p90    {percentile(s, 0.90):.3f}")

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    print()
    for t in thresholds:
        frac = sum(1 for x in s if x >= t) / len(s)
        print(f"  frac >= {t:.2f}: {frac:.3f}")

    if args.show_extremes and pair_records:
        k = args.show_extremes
        pair_records.sort(key=lambda r: r[0])  # by containment
        worst = pair_records[:k]
        best = pair_records[-k:]

        def preview(txt, n, which):
            if not txt:
                return ""
            if which == "tail":
                return txt[-n:].replace("\n", " ")
            return txt[:n].replace("\n", " ")

        print("\n" + "=" * 90)
        print(f"WORST {k} pairs")
        print("=" * 90)
        for c, key, pid, nid, ptxt, ntxt, ppages, npages in worst:
            print(f"\nkey={key}  containment={c:.3f}")
            print(f"  PREV {pid} pages={ppages}")
            print(f"  NEXT {nid} pages={npages}")
            print(f"  prev_tail: {preview(ptxt, args.context_chars, 'tail')}")
            print(f"  next_head: {preview(ntxt, args.context_chars, 'head')}")

        print("\n" + "=" * 90)
        print(f"BEST {k} pairs")
        print("=" * 90)
        for c, key, pid, nid, ptxt, ntxt, ppages, npages in best:
            print(f"\nkey={key}  containment={c:.3f}")
            print(f"  PREV {pid} pages={ppages}")
            print(f"  NEXT {nid} pages={npages}")
            print(f"  prev_tail: {preview(ptxt, args.context_chars, 'tail')}")
            print(f"  next_head: {preview(ntxt, args.context_chars, 'head')}")

if __name__ == "__main__":
    main()
