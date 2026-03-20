#!/usr/bin/env python3

"""
How to run
------------
python3 scripts/inspect_iptp_tables.py --backup-prefix "$BACKUP_PREFIX" --focus --max-tables 50

"""

import argparse
import os
import re
import subprocess
from pathlib import Path

import pandas as pd


def split_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Split s3://bucket/prefix... into (bucket, prefix_without_leading_slash).
    """
    assert s3_uri.startswith("s3://"), f"Expected s3://..., got {s3_uri}"
    rest = s3_uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key  # key may be "" if no prefix


def sh(cmd: list[str], check: bool = True) -> str:
    """Run a shell command and return stdout as text."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDERR:\n{p.stderr.strip()}\nSTDOUT:\n{p.stdout.strip()}"
        )
    return p.stdout


def list_table_prefixes(backup_prefix: str) -> list[str]:
    """
    Return a list of "public.<table>/" prefixes under BACKUP_PREFIX.
    Example prefix returned: 'public.iptp_policy/'
    """
    out = sh(["aws", "s3", "ls", backup_prefix.rstrip("/") + "/"])
    prefixes = []
    for line in out.splitlines():
        line = line.strip()
        # Lines look like: "PRE public.iptp_policy/"
        m = re.match(r"^PRE\s+(.+?/)$", line)
        if m:
            prefixes.append(m.group(1))
    return prefixes


def find_first_parquet_key(backup_prefix: str, table_prefix: str) -> str | None:
    """
    Find the first parquet object for a given table prefix.
    Typical layout: <BACKUP_PREFIX>/<table_prefix>/1/part-....parquet
    """
    full = backup_prefix.rstrip("/") + "/" + table_prefix.strip("/") + "/"
    # Recursively list and pick the first parquet file.
    out = sh(["aws", "s3", "ls", full, "--recursive"], check=False)
    parquet_lines = [ln for ln in out.splitlines() if ln.strip().endswith(".parquet")]
    if not parquet_lines:
        return None
    # Each line: "2026-01-30 09:08:07  1234 prod-.../iptp_db/public.xxx/1/part-....parquet"
    first = parquet_lines[0].split()
    key = first[-1]
    return key


def download_parquet(backup_prefix: str, parquet_key: str, out_dir: Path) -> Path:
    """
    Download the parquet object to out_dir mirroring table name.
    parquet_key is a bucket-relative key like:
      prod-1-30-2026/iptp_db/public.iptp_doc/1/part-....parquet
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket, prefix_key = split_s3_uri(backup_prefix.rstrip("/"))

    fname = parquet_key.split("/")[-1]

    # put in a folder by table name (public.xxx)
    m = re.search(r"(public\.[^/]+)/", parquet_key)
    table = m.group(1) if m else "unknown_table"
    table_dir = out_dir / table
    table_dir.mkdir(parents=True, exist_ok=True)

    local_path = table_dir / fname
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    # parquet_key is already bucket-relative; do NOT prepend backup_prefix again
    s3_uri = f"s3://{bucket}/{parquet_key}"

    sh(["aws", "s3", "cp", s3_uri, str(local_path)])
    return local_path


def safe_preview(df: pd.DataFrame, n: int = 2) -> str:
    cols = list(df.columns)
    shown_cols = cols[: min(len(cols), 12)]
    head = df[shown_cols].head(n)
    return head.to_string(index=False)


def guess_mapping(table_name: str, cols: list[str]) -> list[str]:
    """
    Heuristic: suggest which filter/concept this table likely supports.
    This is not authoritative, but it speeds up triage.
    """
    t = table_name.lower()
    cset = set([c.lower() for c in cols])
    guesses = []

    # Known filter structures you already found
    if "iptp_currentstatus" in t:
        guesses.append("FILTER: Policies by Current Status (iptp_policy.current_status_id -> iptp_currentstatus)")
    if t.endswith("iptp_status"):
        guesses.append("FILTER: Status tags incl. 'In Litigation' (via iptp_policy_status)")
    if "iptp_policy_status" in t:
        guesses.append("JOIN: policy -> status tags (litigation/enjoined/etc.)")
    if "iptp_typeofaction" in t:
        guesses.append("FILTER: Administrative Action (via iptp_policy_types_of_action)")
    if "iptp_policy_types_of_action" in t:
        guesses.append("JOIN: policy -> typeofaction")
    if "iptp_subjectmattercategory" in t:
        guesses.append("FILTER: Subject Matter category (top-level)")
    if t.endswith("iptp_subjectmatter"):
        guesses.append("FILTER: Subject Matter (leaf tags; FK to subjectmattercategory)")
    if "iptp_policysubjectmatter" in t:
        guesses.append("JOIN: policy -> subject matter")
    if "iptp_agencydepartment" in t:
        guesses.append("FILTER: Agency department")
    if t.endswith("iptp_agency"):
        guesses.append("FILTER: Agencies (FK to agencydepartment)")
    if "iptp_policyagency" in t:
        guesses.append("JOIN: policy -> agencies")

    # High-priority missing ones
    if "iptp_policyfeedback" in t or ("publisher" in cset and "title" in cset):
        guesses.append("LIKELY: Commentary / analysis / notes attached to policy")
    if "iptp_externalpage" in t or ("link" in cset and "title" in cset and "policy" in "".join(cset)):
        guesses.append("LIKELY: External links / referenced pages (possible commentary)")
    if "posttrumpaction" in t:
        guesses.append("FILTER: Post-Trump action category (revoked/replaced/enjoined/etc.)")
    if "policyactionactor" in t:
        guesses.append("FILTER: Action actor (Biden Admin / Courts / Other)")
    if "taggedpolicy" in t or ("related" in cset and "policy_id" in cset):
        guesses.append("LIKELY: Associated/Derivative policy relationships (or tagging)")
    if "wagtaildocs_document" in t:
        guesses.append("FILES: Wagtail documents; file path -> S3 key 'media/<file>'")
    if "iptp_policydocument" in t:
        guesses.append("ATTACHMENTS: Policy -> documents (pdfs); contains document_file_id, document_type_id, etc.")
    if "wagtailcore_page" in t:
        guesses.append("CORE: Wagtail pages (title/slug/url_path/content_type_id)")

    # Generic relational hints
    if any(k in cset for k in ["policy_id", "subject_matter_id", "agency_id", "typeofaction_id", "status_id"]):
        guesses.append("RELATION: join table/dimension likely supports filters")

    return guesses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup-prefix", required=True,
                    help="e.g. s3://<backup-bucket>/<snapshot>/iptp_db")
    ap.add_argument("--out-dir", default="data/db_peek_auto", help="local cache dir")
    ap.add_argument("--include-regex", default=None,
                    help="only inspect table prefixes matching this regex (e.g. 'iptp_(policyfeedback|externalpage|taggedpolicy)')")
    ap.add_argument("--focus", action="store_true",
                    help="focus on likely commentary + associated/derivative + action-actor/posttrump tables")
    ap.add_argument("--max-tables", type=int, default=9999)
    args = ap.parse_args()

    backup_prefix = args.backup_prefix.rstrip("/")
    out_dir = Path(args.out_dir)

    table_prefixes = list_table_prefixes(backup_prefix)

    # Focus shortlist: the ones you said you want + the most likely missing filter tables
    focus_patterns = [
        r"public\.iptp_policyfeedback/",
        r"public\.iptp_externalpage/",
        r"public\.iptp_taggedpolicy/",
        r"public\.iptp_policyactionactor/",
        r"public\.iptp_posttrumpaction/",
        r"public\.iptp_policydocument/",
        r"public\.wagtaildocs_document/",
        r"public\.wagtailcore_page/",
        r"public\.iptp_policy/",
        r"public\.postgres_search_indexentry/",
        r"public\.iptp_documenttype/",
    ]

    if args.focus:
        keep = []
        for p in table_prefixes:
            if any(re.search(pat, p) for pat in focus_patterns):
                keep.append(p)
        # Also include any unknown-but-promising names
        for p in table_prefixes:
            if re.search(r"(related|derivative|commentary|analysis|subsequent|litigat|enjoin)", p, re.I):
                if p not in keep:
                    keep.append(p)
        table_prefixes = keep

    if args.include_regex:
        rx = re.compile(args.include_regex)
        table_prefixes = [p for p in table_prefixes if rx.search(p)]

    table_prefixes = table_prefixes[: args.max_tables]

    print(f"\nFound {len(table_prefixes)} table prefixes to inspect under {backup_prefix}\n")

    for table_prefix in table_prefixes:
        table_name = table_prefix.strip("/")

        parquet_key = find_first_parquet_key(backup_prefix, table_prefix)
        if parquet_key is None:
            print(f"=== {table_name} ===")
            print("No parquet files found (may be empty table or different export format).\n")
            continue

        local_path = download_parquet(backup_prefix, parquet_key, out_dir)

        # Read parquet
        try:
            df = pd.read_parquet(local_path)
        except Exception as e:
            print(f"=== {table_name} ===")
            print(f"Downloaded: {local_path}")
            print(f"Failed to read parquet: {e}\n")
            continue

        cols = list(df.columns)

        print(f"=== {table_name} ===")
        print(f"Parquet: {parquet_key}")
        print(f"Local : {local_path}")
        print(f"Rows  : {len(df):,}")
        print("Columns:")
        for c in cols:
            print(f" - {c}")

        # Preview
        if len(df) > 0:
            print("\nPreview (first 2 rows, first ~12 cols):")
            print(safe_preview(df, n=2))
        else:
            print("\nPreview: <empty table shard>")

        # Helpful uniques if small
        if "title" in df.columns and len(df) > 0:
            uniq = df["title"].dropna().astype(str).unique().tolist()
            if len(uniq) <= 50:
                print("\nUnique values in 'title' (<=50):")
                for v in uniq:
                    print(f" - {v}")

        # Heuristic mapping
        guesses = guess_mapping(table_name, cols)
        if guesses:
            print("\nLikely mapping:")
            for g in guesses:
                print(f" * {g}")

        print("\n")


if __name__ == "__main__":
    main()
