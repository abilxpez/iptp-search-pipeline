#!/usr/bin/env python3
"""
How to run
------------
python3 scripts/probe_associated_and_commentary.py --backup-prefix "$BACKUP_PREFIX" --sample-n 15
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import pandas as pd


# -----------------------------
# Shell + S3 helpers
# -----------------------------
def sh(cmd: list[str], check: bool = True) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDERR:\n{p.stderr.strip()}\nSTDOUT:\n{p.stdout.strip()}"
        )
    return p.stdout


def split_s3_uri(s3_uri: str) -> tuple[str, str]:
    assert s3_uri.startswith("s3://"), f"Expected s3://..., got {s3_uri}"
    rest = s3_uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def list_objects_recursive(backup_prefix: str, prefix: str) -> list[str]:
    """Return bucket-relative keys under backup_prefix/prefix (recursive)."""
    full = backup_prefix.rstrip("/") + "/" + prefix.strip("/") + "/"
    out = sh(["aws", "s3", "ls", full, "--recursive"], check=False)
    keys = []
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        # expected: date time size key
        if len(parts) >= 4:
            keys.append(parts[-1])
    return keys


def choose_one_parquet_key(keys: list[str]) -> str | None:
    """Pick the first parquet key in the listing."""
    for k in keys:
        if k.endswith(".parquet"):
            return k
    return None


def download_s3_key(backup_prefix: str, key: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    bucket, _ = split_s3_uri(backup_prefix.rstrip("/"))
    s3_uri = f"s3://{bucket}/{key}"
    sh(["aws", "s3", "cp", s3_uri, str(out_path)])
    return out_path


def load_table_shard(backup_prefix: str, table_prefix: str, out_dir: Path) -> pd.DataFrame:
    """
    Load a single parquet shard for a given table prefix.
    NOTE: This mirrors your existing approach (one shard is enough to map schema & values).
    """
    # keys under: <BACKUP_PREFIX>/<table_prefix>/1/part-....parquet
    keys = list_objects_recursive(backup_prefix, table_prefix)
    parquet_key = choose_one_parquet_key(keys)
    if parquet_key is None:
        raise RuntimeError(f"No parquet found for table prefix: {table_prefix}")

    table_name = table_prefix.strip("/")

    # store as: out_dir/<table_name>/<filename>
    fname = parquet_key.split("/")[-1]
    local_path = out_dir / table_name / fname
    download_s3_key(backup_prefix, parquet_key, local_path)

    return pd.read_parquet(local_path)


# -----------------------------
# Probes
# -----------------------------
def probe_commentary(backup_prefix: str, out_dir: Path, sample_n: int = 10) -> None:
    """
    Check whether commentary is represented as iptp_policydocument rows with a document_type_id
    that maps to iptp_documenttype.title == 'Commentary'.
    """
    print("\n==============================")
    print("PROBE 1: COMMENTARY")
    print("==============================\n")

    doc_types = load_table_shard(backup_prefix, "public.iptp_documenttype", out_dir)
    policy_docs = load_table_shard(backup_prefix, "public.iptp_policydocument", out_dir)

    # Find commentary documenttype IDs
    dt = doc_types.copy()
    dt["title_str"] = dt["title"].astype(str)
    commentary = dt[dt["title_str"].str.strip().str.lower() == "commentary"]

    if commentary.empty:
        print("No documenttype row with title == 'Commentary' found in this shard.")
        print("Document types seen:", dt["title_str"].dropna().unique().tolist())
        return

    commentary_ids = commentary["id"].tolist()
    print(f"Found Commentary document_type_id(s): {commentary_ids}")

    # Filter policydocument
    pdx = policy_docs.copy()

    # Ensure numeric compare works even if floats
    def to_int_safe(x):
        try:
            if pd.isna(x):
                return None
            return int(x)
        except Exception:
            return None

    pdx["document_type_id_int"] = pdx["document_type_id"].apply(to_int_safe)

    commentary_rows = pdx[pdx["document_type_id_int"].isin([int(i) for i in commentary_ids])]
    print(f"Total iptp_policydocument rows in shard: {len(pdx):,}")
    print(f"Commentary-type iptp_policydocument rows in shard: {len(commentary_rows):,}\n")

    show_cols = [
        "id",
        "policy_id",
        "date",
        "display_title",
        "link_title",
        "link",
        "description",
        "document_file_id",
        "document_type_id",
        "action_actor_id",
        "post_trump_action_id",
        "slug",
    ]
    show_cols = [c for c in show_cols if c in commentary_rows.columns]

    if len(commentary_rows) == 0:
        print("No commentary policydocument rows found in this shard.")
        print("NOTE: This can happen if shard sampling is unlucky; rerun with --all-shards for this table if needed.")
        return

    # sample a few rows
    sample = commentary_rows.sample(min(sample_n, len(commentary_rows)), random_state=0)
    print("Sample commentary rows (key columns):")
    print(sample[show_cols].to_string(index=False))


def parse_related_policies_cell(cell) -> list[int]:
    """
    related_policies appears to be a list of dicts like:
      [{"type":"related_policy","value":25,...}, ...]
    Return list of referenced policy IDs (ints) if we can parse them.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []

    # Sometimes it is already a list (parquet can keep nested types) OR a JSON string
    try:
        obj = cell
        if isinstance(cell, str):
            s = cell.strip()
            if not s:
                return []
            obj = json.loads(s)

        if not isinstance(obj, list):
            return []

        out = []
        for item in obj:
            if isinstance(item, dict) and "value" in item:
                v = item["value"]
                try:
                    out.append(int(v))
                except Exception:
                    pass
        return out
    except Exception:
        # as a fallback, try to find `"value": <num>` patterns
        try:
            s = str(cell)
            return [int(m.group(1)) for m in re.finditer(r'"value"\s*:\s*(\d+)', s)]
        except Exception:
            return []


def probe_related_policies(backup_prefix: str, out_dir: Path, sample_n: int = 10) -> None:
    """
    Check whether associated/derivative policies are represented in iptp_policy.related_policies.
    """
    print("\n==============================")
    print("PROBE 2: RELATED / ASSOCIATED POLICIES")
    print("==============================\n")

    policy = load_table_shard(backup_prefix, "public.iptp_policy", out_dir)

    if "related_policies" not in policy.columns:
        print("iptp_policy.related_policies column not present in this shard.")
        print("Columns:", list(policy.columns))
        return

    px = policy.copy()
    px["related_ids"] = px["related_policies"].apply(parse_related_policies_cell)
    px["n_related"] = px["related_ids"].apply(len)

    with_related = px[px["n_related"] > 0]
    print(f"Total iptp_policy rows in shard: {len(px):,}")
    print(f"Rows with >=1 related policy: {len(with_related):,}")
    if len(with_related):
        print(f"Max number of related policies on a single row (in shard): {with_related['n_related'].max()}")

    if len(with_related) == 0:
        print("\nNo rows with related policies found in this shard.")
        print("NOTE: could be shard sampling; if needed, scan more shards for iptp_policy.")
        return

    cols = ["page_ptr_id", "announced_date", "effective_date", "current_status_id", "related_policies", "related_ids", "n_related"]
    cols = [c for c in cols if c in with_related.columns]

    sample = with_related.sample(min(sample_n, len(with_related)), random_state=0)
    print("\nSample rows with related policies:")
    print(sample[cols].to_string(index=False))


def probe_tags(backup_prefix: str, out_dir: Path, sample_n: int = 10) -> None:
    """
    Optional: check taggit tagging, which could also support "associated/derivative" style groupings.
    This will run only if both iptp_taggedpolicy and taggit_tag exist under the backup prefix.
    """
    print("\n==============================")
    print("PROBE 3 (OPTIONAL): TAG-BASED RELATIONSHIPS")
    print("==============================\n")

    # Try to load; if missing, skip gracefully
    try:
        tagged = load_table_shard(backup_prefix, "public.iptp_taggedpolicy", out_dir)
    except Exception as e:
        print("Skipping: could not load public.iptp_taggedpolicy:", e)
        return

    try:
        tags = load_table_shard(backup_prefix, "public.taggit_tag", out_dir)
    except Exception as e:
        print("Skipping: could not load public.taggit_tag:", e)
        return

    # Basic join
    if not {"tag_id", "content_object_id"}.issubset(set(tagged.columns)):
        print("iptp_taggedpolicy missing expected columns; got:", list(tagged.columns))
        return
    if "id" not in tags.columns:
        print("taggit_tag missing expected 'id'; got:", list(tags.columns))
        return

    merged = tagged.merge(tags, left_on="tag_id", right_on="id", how="left", suffixes=("", "_tag"))
    # Try to show best-effort columns
    show_cols = []
    for c in ["content_object_id", "tag_id", "name", "slug"]:
        if c in merged.columns:
            show_cols.append(c)

    print(f"iptp_taggedpolicy rows in shard: {len(tagged):,}")
    print(f"taggit_tag rows in shard: {len(tags):,}")
    print("\nSample tagged rows:")
    sample = merged.sample(min(sample_n, len(merged)), random_state=0)
    print(sample[show_cols].to_string(index=False))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup-prefix", required=True, help="e.g. s3://<backup-bucket>/<snapshot>/iptp_db")
    ap.add_argument("--out-dir", default="data/db_peek_probe", help="local cache dir")
    ap.add_argument("--sample-n", type=int, default=10, help="rows to sample per probe")
    ap.add_argument("--skip-tags", action="store_true", help="skip optional tag probe")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    probe_commentary(args.backup_prefix, out_dir, sample_n=args.sample_n)
    probe_related_policies(args.backup_prefix, out_dir, sample_n=args.sample_n)
    if not args.skip_tags:
        probe_tags(args.backup_prefix, out_dir, sample_n=args.sample_n)


if __name__ == "__main__":
    main()
