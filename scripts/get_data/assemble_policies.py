#!/usr/bin/env python3
"""
assemble_policies.py

Goal
----
Assemble a local, self-contained dataset of ~N diverse IPTP policies (default 100),
including joined metadata + a manifest of ONLY the needed Wagtail document files
(PDFs) to download later.

This script:
- Reads IPTP DB parquet exports from an S3 backup prefix (e.g. s3://<backup-bucket>/<snapshot>/iptp_db)
- Loads the *relevant tables* (all parquet parts, not just one shard)
- Samples N policies (diverse/stratified-ish)
- Joins in:
    * current status
    * agencies (+ departments)
    * subject matter (+ categories)
    * administrative action types
    * status tags (litigation/enjoined/etc.)
    * policy documents (attachments) split by document_type (source docs vs commentary vs subsequent action, etc.)
    * related policies list (resolve titles/dates)
- Writes per-policy folder:
    out_dir/policies/<policy_folder>/
        policy.json                  # all metadata + commentary text items
        documents/                   # empty for now; downloader fills this
- Writes global manifest:
    out_dir/manifest.jsonl           # one line per needed S3 file (Wagtail doc)
- Writes a small index:
    out_dir/policies_index.jsonl     # one line per sampled policy (quick skim)

Resumability
------------
- Safe to rerun.
- It will NOT overwrite existing policy.json unless --overwrite.
- manifest.jsonl is rebuilt each run (deterministic for a fixed sample/seed).

Assumptions / Key mappings discovered
-------------------------------------
- iptp_policy.page_ptr_id corresponds to wagtailcore_page.id for the policy page.
- iptp_policydocument.document_file_id refers to wagtaildocs_document.id.
- wagtaildocs_document.file is a relative path like 'documents/foo.pdf'
  => S3 key is 'media/' + that file path (e.g. media/documents/foo.pdf)

How to run
------------
python -m scripts.get_data.assemble_policies \
  --config config.json \
  --backup-prefix "$BACKUP_PREFIX" \
  --documents-bucket "$DOCUMENTS_BUCKET"

"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.common.json_sanitize import json_dumps_strict

from scripts.common.config import (
    load_config,
    get_cfg_value,
    get_path,
    get_int,
    get_str,
)

import numpy as np
import pandas as pd


# --------------------------
# Shell / AWS helpers
# --------------------------

def sh(cmd: List[str], check: bool = True) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDERR:\n{p.stderr.strip()}\nSTDOUT:\n{p.stdout.strip()}"
        )
    return p.stdout


def split_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Split s3://bucket/prefix... into (bucket, key_prefix_without_leading_slash)."""
    assert s3_uri.startswith("s3://"), f"Expected s3://..., got {s3_uri}"
    rest = s3_uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def list_parquet_keys(backup_prefix: str, table_prefix: str) -> List[str]:
    """
    Return all parquet object keys under:
      <backup_prefix>/<table_prefix>/**.parquet

    backup_prefix: s3://bucket/prod-.../iptp_db
    table_prefix:  public.iptp_policy  (no trailing slash required)
    Returns bucket-relative keys (no s3://bucket/ prefix).
    """
    backup_prefix = backup_prefix.rstrip("/")
    table_prefix = table_prefix.strip("/")

    # Build full listing prefix:
    # e.g. s3://<backup-bucket>/<snapshot>/iptp_db/public.iptp_policy/
    full = f"{backup_prefix}/{table_prefix}/"
    out = sh(["aws", "s3", "ls", full, "--recursive"], check=False)
    keys: List[str] = []
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln.endswith(".parquet"):
            continue
        parts = ln.split()
        if len(parts) < 4:
            continue
        keys.append(parts[-1])
    return keys


def download_s3_key(bucket: str, key: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return
    s3_uri = f"s3://{bucket}/{key}"
    sh(["aws", "s3", "cp", s3_uri, str(dest_path)])


# --------------------------
# Parquet loading (cache)
# --------------------------

def cache_table_parts(
    backup_prefix: str,
    table_prefix: str,
    cache_dir: Path,
    max_parts: Optional[int] = None,
) -> List[Path]:
    """
    Download ALL parquet parts for a table into cache_dir/<table_prefix>/.

    Returns local file paths.
    """
    bucket, _ = split_s3_uri(backup_prefix)
    keys = list_parquet_keys(backup_prefix, table_prefix)
    if not keys:
        return []

    if max_parts is not None:
        keys = keys[:max_parts]

    local_paths: List[Path] = []
    out_table_dir = cache_dir / table_prefix
    out_table_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        fname = key.split("/")[-1]
        local_path = out_table_dir / fname
        download_s3_key(bucket, key, local_path)
        local_paths.append(local_path)

    return local_paths


def read_table(
    backup_prefix: str,
    table_prefix: str,
    cache_dir: Path,
    columns: Optional[List[str]] = None,
    max_parts: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read a table as a concatenation of all parquet parts.
    Uses local cache to avoid re-downloading.
    """
    parts = cache_table_parts(backup_prefix, table_prefix, cache_dir, max_parts=max_parts)
    if not parts:
        return pd.DataFrame()

    dfs: List[pd.DataFrame] = []
    for p in parts:
        dfp = pd.read_parquet(p, columns=columns) if columns else pd.read_parquet(p)
        dfs.append(dfp)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# --------------------------
# Text / filename helpers
# --------------------------

def safe_slug(s: str, max_len: int = 100) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-_]+", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s:
        s = "untitled"
    return s[:max_len]


def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def json_load_maybe(x: Any) -> Any:
    """
    iptp_policy.related_policies is sometimes already parsed or stored as JSON-ish.
    Try to parse if it's a string; otherwise return as-is.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list) or isinstance(x, dict):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def html_to_text(s: Any) -> str:
    """
    Conservative HTML-ish -> plain text for IPTP policy summaries.
    - Decodes HTML entities
    - Preserves basic structure: paragraphs, line breaks, list items
    - Strips remaining tags
    - Normalizes whitespace
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    if not isinstance(s, str):
        s = str(s)

    s = s.strip()
    if not s:
        return ""

    # Decode entities first (e.g., &quot; -> ")
    s = html.unescape(s)

    # Normalize common block/line tags into newlines.
    s = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*p\s*>", "\n\n", s)
    s = re.sub(r"(?i)<\s*p(\s+[^>]*)?\s*>", "", s)
    s = re.sub(r"(?i)</\s*div\s*>", "\n\n", s)
    s = re.sub(r"(?i)<\s*div(\s+[^>]*)?\s*>", "", s)

    # Lists: convert <li>...</li> to "- ...\n"
    s = re.sub(r"(?i)<\s*li(\s+[^>]*)?\s*>", "- ", s)
    s = re.sub(r"(?i)</\s*li\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*ul\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*ol\s*>", "\n", s)
    s = re.sub(r"(?i)<\s*(ul|ol)(\s+[^>]*)?\s*>", "", s)

    # Strip any remaining tags.
    s = re.sub(r"<[^>]+>", "", s)

    # Whitespace normalization
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)   # trailing spaces before newline
    s = re.sub(r"\n[ \t]+", "\n", s)   # leading spaces after newline
    s = re.sub(r"[ \t]{2,}", " ", s)   # collapse runs of spaces/tabs
    s = re.sub(r"\n{3,}", "\n\n", s)   # cap blank lines
    return s.strip()


# --------------------------
# Data assembly
# --------------------------

@dataclass
class ManifestItem:
    page_ptr_id: int
    wagtaildoc_id: int
    s3_bucket: str
    s3_key: str
    dest_relpath: str
    file_title: str
    document_type_title: str
    policydocument_id: int

    def to_json(self) -> Dict[str, Any]:
        return {
            "page_ptr_id": self.page_ptr_id,
            "wagtaildoc_id": self.wagtaildoc_id,
            "policydocument_id": self.policydocument_id,
            "document_type": self.document_type_title,
            "file_title": self.file_title,
            "s3_bucket": self.s3_bucket,
            "s3_key": self.s3_key,
            "dest_relpath": self.dest_relpath,
        }


def build_policy_folder_name(page_ptr_id: int, title: str, slug: str) -> str:
    # Prefer Wagtail slug; fall back to title.
    base = slug or safe_slug(title)
    base = safe_slug(base, max_len=80)
    return f"{page_ptr_id}__{base}"


def choose_diverse_sample(
    policies: pd.DataFrame,
    policy_subject_cat: pd.DataFrame,
    policy_action: pd.DataFrame,
    policy_dept: pd.DataFrame,
    n: int,
    seed: int,
) -> List[int]:
    """
    Practical "diverse" sampling:
    - Create a bucket key per policy based on (subject_category, department, action_type)
      using the first (sorted) value in each dimension if multiple.
    - Sample approximately evenly across buckets.
    """
    rng = np.random.RandomState(seed)

    # Build first-category per policy
    sub_first = (
        policy_subject_cat.sort_values(["policy_id", "category_sort", "subject_sort"])
        .groupby("policy_id", as_index=False)
        .first()[["policy_id", "category_title"]]
        .rename(columns={"policy_id": "page_ptr_id", "category_title": "subject_category"})
    )

    dept_first = (
        policy_dept.sort_values(["policy_id", "department_sort", "agency_sort"])
        .groupby("policy_id", as_index=False)
        .first()[["policy_id", "department_title"]]
        .rename(columns={"policy_id": "page_ptr_id", "department_title": "department"})
    )

    act_first = (
        policy_action.sort_values(["policy_id", "action_sort", "typeofaction_title"])
        .groupby("policy_id", as_index=False)
        .first()[["policy_id", "typeofaction_title"]]
        .rename(columns={"policy_id": "page_ptr_id", "typeofaction_title": "action_type"})
    )

    base = policies[["page_ptr_id"]].copy()
    base = base.merge(sub_first, on="page_ptr_id", how="left")
    base = base.merge(dept_first, on="page_ptr_id", how="left")
    base = base.merge(act_first, on="page_ptr_id", how="left")

    base["subject_category"] = base["subject_category"].fillna("unknown_subject")
    base["department"] = base["department"].fillna("unknown_dept")
    base["action_type"] = base["action_type"].fillna("unknown_action")

    base["bucket"] = (
        base["subject_category"].astype(str)
        + " | "
        + base["department"].astype(str)
        + " | "
        + base["action_type"].astype(str)
    )

    # Bucketed sampling
    buckets = base["bucket"].unique().tolist()
    rng.shuffle(buckets)

    # Round-robin across buckets
    bucket_to_ids: Dict[str, List[int]] = {}
    for b, grp in base.groupby("bucket"):
        ids = grp["page_ptr_id"].tolist()
        rng.shuffle(ids)
        bucket_to_ids[b] = ids

    chosen: List[int] = []
    while len(chosen) < n and bucket_to_ids:
        progressed = False
        for b in list(buckets):
            ids = bucket_to_ids.get(b, [])
            if not ids:
                bucket_to_ids.pop(b, None)
                continue
            chosen.append(ids.pop())
            progressed = True
            if len(chosen) >= n:
                break
        if not progressed:
            break

    # Fallback if still short: random from remaining policies not already picked
    if len(chosen) < n:
        remaining = list(set(policies["page_ptr_id"].tolist()) - set(chosen))
        rng.shuffle(remaining)
        chosen.extend(remaining[: (n - len(chosen))])

    return chosen[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config.json (optional)")
    ap.add_argument(
        "--backup-prefix",
        default=None,
        help="e.g. s3://<backup-bucket>/<snapshot>/iptp_db (or set aws.backup_prefix in config)",
    )
    ap.add_argument(
        "--documents-bucket",
        default=None,
        help="S3 bucket containing Wagtail files referenced in manifest (e.g. from config aws.documents_bucket)",
    )
    ap.add_argument("--out-dir", default="data/sample_100", help="Output dataset dir")
    ap.add_argument("--cache-dir", default="data/parquet_cache", help="Local parquet cache dir")
    ap.add_argument("--n", type=int, default=100, help="How many policies to sample")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing policy.json files")
    ap.add_argument("--include-nonpdf", action="store_true", help="Include non-PDF wagtail documents in manifest")
    ap.add_argument(
        "--include-curation-notes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include internal curation_notes_html in policy.json output",
    )
    ap.add_argument("--max-parts", type=int, default=None, help="Debug: limit parquet parts per table")
    args = ap.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(Path(args.config))

    # CLI > config > defaults

    # out-dir
    if args.out_dir == ap.get_default("out_dir"):
        args.out_dir = str(
            get_path(cfg, "assemble.out_dir", default=args.out_dir)
        )

    # cache-dir
    if args.cache_dir == ap.get_default("cache_dir"):
        args.cache_dir = str(
            get_path(cfg, "assemble.cache_dir", default=args.cache_dir)
        )

    # n
    if args.n == ap.get_default("n"):
        args.n = get_int(cfg, "assemble.n", default=args.n)

    # seed
    if args.seed == ap.get_default("seed"):
        args.seed = get_int(cfg, "assemble.seed", default=args.seed)

    # include_nonpdf (bool flag)
    # only pull from config if user did NOT explicitly pass the flag
    if not args.include_nonpdf:
        v = get_cfg_value(cfg, "assemble.include_nonpdf")
        if v is not None:
            args.include_nonpdf = bool(v)

    # include-curation-notes (tri-state: None means defer to config/default false)
    if args.include_curation_notes is None:
        v = get_cfg_value(cfg, "assemble.include_curation_notes")
        args.include_curation_notes = bool(v) if v is not None else False

    # backup prefix
    if not args.backup_prefix:
        args.backup_prefix = get_str(cfg, "aws.backup_prefix", default=None)
    if not args.backup_prefix:
        raise SystemExit("Need --backup-prefix or config aws.backup_prefix")

    # documents bucket
    if not args.documents_bucket:
        args.documents_bucket = get_str(cfg, "aws.documents_bucket", default=None)
    if not args.documents_bucket:
        raise SystemExit("Need --documents-bucket or config aws.documents_bucket")


    backup_prefix = args.backup_prefix.rstrip("/")
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_policies_dir = out_dir / "policies"
    out_policies_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _bucket, _backup_key_prefix = split_s3_uri(backup_prefix)

    # --------------------------
    # Load tables (all parts)
    # --------------------------
    # Keep columns minimal to reduce memory.
    print("\nLoading tables from:", backup_prefix)

    # Core policy tables
    iptp_policy = read_table(
        backup_prefix, "public.iptp_policy", cache_dir,
        columns=["page_ptr_id", "curation_notes", "announced_date", "effective_date", "expired_date",
                 "description", "related_policies", "current_status_id", "priority_id", "search_document"],
        max_parts=args.max_parts
    )

    wagtailcore_page = read_table(
        backup_prefix, "public.wagtailcore_page", cache_dir,
        columns=["id", "title", "slug", "url_path", "content_type_id", "first_published_at", "last_published_at", "live"],
        max_parts=args.max_parts
    )

    # Current status
    iptp_currentstatus = read_table(
        backup_prefix, "public.iptp_currentstatus", cache_dir,
        columns=["id", "title", "slug", "description", "sort_order"],
        max_parts=args.max_parts
    )

    # Status tags (litigation/enjoined/etc.)
    iptp_status = read_table(
        backup_prefix, "public.iptp_status", cache_dir,
        columns=["id", "title", "slug", "description", "sort_order"],
        max_parts=args.max_parts
    )
    iptp_policy_status = read_table(
        backup_prefix, "public.iptp_policy_status", cache_dir,
        columns=["policy_id", "status_id"],
        max_parts=args.max_parts
    )

    # Administrative action types
    iptp_typeofaction = read_table(
        backup_prefix, "public.iptp_typeofaction", cache_dir,
        columns=["id", "title", "slug", "description", "sort_order"],
        max_parts=args.max_parts
    )
    iptp_policy_types_of_action = read_table(
        backup_prefix, "public.iptp_policy_types_of_action", cache_dir,
        columns=["policy_id", "typeofaction_id"],
        max_parts=args.max_parts
    )

    # Subject matter
    iptp_subjectmattercategory = read_table(
        backup_prefix, "public.iptp_subjectmattercategory", cache_dir,
        columns=["id", "title", "slug", "sort_order", "description"],
        max_parts=args.max_parts
    )
    iptp_subjectmatter = read_table(
        backup_prefix, "public.iptp_subjectmatter", cache_dir,
        columns=["id", "title", "slug", "sort_order", "description", "category_id"],
        max_parts=args.max_parts
    )
    iptp_policysubjectmatter = read_table(
        backup_prefix, "public.iptp_policysubjectmatter", cache_dir,
        columns=["policy_id", "subject_matter_id", "sort_order"],
        max_parts=args.max_parts
    )

    # Agencies
    iptp_agencydepartment = read_table(
        backup_prefix, "public.iptp_agencydepartment", cache_dir,
        columns=["id", "title", "acronym", "display_title", "slug", "sort_order", "description"],
        max_parts=args.max_parts
    )
    iptp_agency = read_table(
        backup_prefix, "public.iptp_agency", cache_dir,
        columns=["id", "acronym", "title", "display_title", "slug", "sort_order", "description", "department_id"],
        max_parts=args.max_parts
    )
    iptp_policyagency = read_table(
        backup_prefix, "public.iptp_policyagency", cache_dir,
        columns=["policy_id", "agency_id", "sort_order"],
        max_parts=args.max_parts
    )

    # Policy documents / attachments + document type dimension
    iptp_documenttype = read_table(
        backup_prefix, "public.iptp_documenttype", cache_dir,
        columns=["id", "title", "slug", "description", "sort_order"],
        max_parts=args.max_parts
    )
    iptp_policydocument = read_table(
        backup_prefix, "public.iptp_policydocument", cache_dir,
        columns=["id", "policy_id", "date", "display_title", "description", "link", "link_title",
                 "document_file_id", "document_type_id", "sort_order", "slug",
                 "action_actor_id", "post_trump_action_id"],
        max_parts=args.max_parts
    )

    # Wagtail document files (to turn doc ids into S3 keys)
    wagtaildocs_document = read_table(
        backup_prefix, "public.wagtaildocs_document", cache_dir,
        columns=["id", "title", "file", "file_size", "file_hash", "created_at", "collection_id"],
        max_parts=args.max_parts
    )

    # Related policy references are stored inside iptp_policy.related_policies streamfield JSON.
    # We resolve "value" IDs to policy pages via wagtailcore_page (title/slug), and iptp_policy for dates/status.
    # (Both share page_ptr_id <-> wagtailcore_page.id)
    print("Loaded:")
    print(f"  iptp_policy: {len(iptp_policy):,}")
    print(f"  wagtailcore_page: {len(wagtailcore_page):,}")
    print(f"  iptp_policydocument: {len(iptp_policydocument):,}")
    print(f"  wagtaildocs_document: {len(wagtaildocs_document):,}")

    # --------------------------
    # Normalize / prep joins
    # --------------------------
    # Ensure integer-ish ids where possible
    for df, col in [
        (iptp_policy, "page_ptr_id"),
        (wagtailcore_page, "id"),
        (wagtaildocs_document, "id"),
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Join: policy -> wagtail page metadata
    page_meta = wagtailcore_page.rename(columns={"id": "page_ptr_id"})[
        ["page_ptr_id", "title", "slug", "url_path", "content_type_id", "live", "first_published_at", "last_published_at"]
    ]
    iptp_policy = iptp_policy.merge(page_meta, on="page_ptr_id", how="left")

    # Join: current status
    cur_status_map = iptp_currentstatus.rename(columns={"id": "current_status_id", "title": "current_status_title", "slug": "current_status_slug"})
    iptp_policy = iptp_policy.merge(cur_status_map[["current_status_id", "current_status_title", "current_status_slug"]], on="current_status_id", how="left")

    # Status tags per policy
    if len(iptp_policy_status) > 0 and len(iptp_status) > 0:
        status_dim = iptp_status.rename(columns={"id": "status_id", "title": "status_title", "slug": "status_slug"})
        policy_status = iptp_policy_status.merge(status_dim[["status_id", "status_title", "status_slug"]], on="status_id", how="left")
    else:
        policy_status = pd.DataFrame(columns=["policy_id", "status_id", "status_title", "status_slug"])

    # Actions per policy
    if len(iptp_policy_types_of_action) > 0 and len(iptp_typeofaction) > 0:
        act_dim = iptp_typeofaction.rename(columns={"id": "typeofaction_id", "title": "typeofaction_title", "slug": "typeofaction_slug"})
        policy_action = iptp_policy_types_of_action.merge(act_dim[["typeofaction_id", "typeofaction_title", "typeofaction_slug"]], on="typeofaction_id", how="left")
        policy_action["action_sort"] = 0
        if "sort_order" in iptp_typeofaction.columns:
            # best-effort: bring in sort_order if present
            policy_action = policy_action.merge(
                iptp_typeofaction.rename(columns={"id": "typeofaction_id"})[["typeofaction_id", "sort_order"]],
                on="typeofaction_id",
                how="left"
            )
            policy_action["action_sort"] = pd.to_numeric(policy_action["sort_order"], errors="coerce").fillna(0)
    else:
        policy_action = pd.DataFrame(columns=["policy_id", "typeofaction_title", "typeofaction_slug", "action_sort"])

    # Subject matter per policy with category
    subj_dim = iptp_subjectmatter.rename(columns={"id": "subject_matter_id", "title": "subject_title", "slug": "subject_slug"})
    cat_dim = iptp_subjectmattercategory.rename(columns={"id": "category_id", "title": "category_title", "slug": "category_slug", "sort_order": "category_sort"})
    subj_dim = subj_dim.merge(cat_dim[["category_id", "category_title", "category_slug", "category_sort"]], on="category_id", how="left")
    subj_dim["subject_sort"] = pd.to_numeric(subj_dim.get("sort_order", 0), errors="coerce").fillna(0)

    policy_subject = iptp_policysubjectmatter.merge(
        subj_dim[["subject_matter_id", "subject_title", "subject_slug", "category_title", "category_slug", "category_sort", "subject_sort"]],
        on="subject_matter_id",
        how="left"
    )
    policy_subject_cat = policy_subject.rename(columns={"policy_id": "policy_id"})  # keep name for sampler

    # Agency per policy with department
    dept_dim = iptp_agencydepartment.rename(columns={"id": "department_id", "title": "department_title", "slug": "department_slug", "sort_order": "department_sort"})
    ag_dim = iptp_agency.rename(columns={"id": "agency_id", "title": "agency_title", "slug": "agency_slug"})
    ag_dim["agency_sort"] = pd.to_numeric(ag_dim.get("sort_order", 0), errors="coerce").fillna(0)
    ag_dim = ag_dim.merge(dept_dim[["department_id", "department_title", "department_slug", "department_sort"]], on="department_id", how="left")

    policy_agency = iptp_policyagency.merge(
        ag_dim[["agency_id", "agency_title", "agency_slug", "department_title", "department_slug", "department_sort", "agency_sort"]],
        on="agency_id",
        how="left"
    )
    policy_dept = policy_agency.rename(columns={"policy_id": "policy_id"})  # for sampler

    # Policy documents with document type
    doc_type_dim = iptp_documenttype.rename(columns={"id": "document_type_id", "title": "document_type_title", "slug": "document_type_slug"})
    iptp_policydocument = iptp_policydocument.merge(
        doc_type_dim[["document_type_id", "document_type_title", "document_type_slug"]],
        on="document_type_id",
        how="left"
    )

    # Attach actual file path from wagtaildocs_document (document_file_id -> wagtaildocs_document.id)
    wagtaildocs_document = wagtaildocs_document.rename(columns={"id": "wagtaildoc_id", "title": "wagtaildoc_title", "file": "wagtaildoc_file"})
    iptp_policydocument["document_file_id"] = pd.to_numeric(iptp_policydocument["document_file_id"], errors="coerce").astype("Int64")
    iptp_policydocument = iptp_policydocument.merge(
        wagtaildocs_document[["wagtaildoc_id", "wagtaildoc_title", "wagtaildoc_file", "file_size", "file_hash"]],
        left_on="document_file_id",
        right_on="wagtaildoc_id",
        how="left"
    )

    # --------------------------
    # Choose a diverse sample of policies
    # --------------------------
    # Only sample policies that have a page_ptr_id
    iptp_policy_valid = iptp_policy.dropna(subset=["page_ptr_id"]).copy()
    iptp_policy_valid["page_ptr_id"] = iptp_policy_valid["page_ptr_id"].astype(int)

    sampled_ids = choose_diverse_sample(
        policies=iptp_policy_valid[["page_ptr_id"]],
        policy_subject_cat=policy_subject,  # has policy_id + category fields
        policy_action=policy_action,
        policy_dept=policy_dept,
        n=args.n,
        seed=args.seed,
    )

    sampled_set = set(sampled_ids)
    sample_policies = iptp_policy_valid[iptp_policy_valid["page_ptr_id"].isin(sampled_set)].copy()

    # --------------------------
    # Build manifest + write per-policy JSON
    # --------------------------
    manifest_items: List[ManifestItem] = []

    # Create maps for quick resolution
    policy_by_id = sample_policies.set_index("page_ptr_id", drop=False)

    # For resolving related policies
    all_policy_meta = iptp_policy_valid.set_index("page_ptr_id", drop=False)[
        ["page_ptr_id", "title", "slug", "announced_date", "effective_date", "expired_date", "current_status_title", "current_status_slug"]
    ]

    def is_pdf_path(p: Any) -> bool:
        if not isinstance(p, str):
            return False
        return p.lower().endswith(".pdf")

    # Write per-policy outputs
    policies_index_path = out_dir / "policies_index.jsonl"
    manifest_path = out_dir / "manifest.jsonl"

    # rebuild these each run (cheap + consistent)
    if policies_index_path.exists():
        policies_index_path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()

    with policies_index_path.open("w", encoding="utf-8") as idx_f:
        for pid in sampled_ids:
            if pid not in policy_by_id.index:
                continue

            prow = policy_by_id.loc[pid]

            title = str(prow.get("title") or "").strip()
            slug = str(prow.get("slug") or "").strip()
            folder = build_policy_folder_name(pid, title=title, slug=slug)
            policy_dir = out_policies_dir / folder
            docs_dir = policy_dir / "documents"
            policy_json_path = policy_dir / "policy.json"

            # If not overwriting and already exists, skip writing; but still include in index + manifest rebuild.
            should_write = args.overwrite or (not policy_json_path.exists())

            policy_dir.mkdir(parents=True, exist_ok=True)
            docs_dir.mkdir(parents=True, exist_ok=True)

            # statuses
            st_rows = policy_status[policy_status["policy_id"] == pid] if len(policy_status) else pd.DataFrame()
            statuses_list = []
            if len(st_rows):
                for _, r in st_rows.dropna(subset=["status_title"]).iterrows():
                    statuses_list.append({"title": r["status_title"], "slug": r["status_slug"]})
                seen = set()
                uniq = []
                for s in statuses_list:
                    key = (s.get("slug") or s.get("title") or "")
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(s)
                statuses_list = uniq

            # actions
            act_rows = policy_action[policy_action["policy_id"] == pid] if len(policy_action) else pd.DataFrame()
            actions_list = []
            if len(act_rows):
                act_rows = act_rows.sort_values(["action_sort", "typeofaction_title"])
                for _, r in act_rows.iterrows():
                    if pd.isna(r.get("typeofaction_title")):
                        continue
                    actions_list.append({"title": r["typeofaction_title"], "slug": r["typeofaction_slug"]})
                seen = set()
                uniq = []
                for a in actions_list:
                    key = (a.get("slug") or a.get("title") or "")
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(a)
                actions_list = uniq

            # subject matters
            subj_rows = policy_subject[policy_subject["policy_id"] == pid] if len(policy_subject) else pd.DataFrame()
            subject_list = []
            if len(subj_rows):
                subj_rows = subj_rows.sort_values(["category_sort", "subject_sort", "subject_title"])
                for _, r in subj_rows.iterrows():
                    subject_list.append({
                        "category": r.get("category_title"),
                        "category_slug": r.get("category_slug"),
                        "title": r.get("subject_title"),
                        "slug": r.get("subject_slug"),
                    })
                seen = set()
                uniq = []
                for s in subject_list:
                    key = (s.get("category_slug"), s.get("slug"))
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(s)
                subject_list = uniq

            # agencies
            ag_rows = policy_agency[policy_agency["policy_id"] == pid] if len(policy_agency) else pd.DataFrame()
            agencies_list = []
            if len(ag_rows):
                ag_rows = ag_rows.sort_values(["department_sort", "agency_sort", "agency_title"])
                for _, r in ag_rows.iterrows():
                    agencies_list.append({
                        "department": r.get("department_title"),
                        "department_slug": r.get("department_slug"),
                        "agency": r.get("agency_title"),
                        "agency_slug": r.get("agency_slug"),
                    })
                seen = set()
                uniq = []
                for a in agencies_list:
                    key = (a.get("department_slug"), a.get("agency_slug"))
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(a)
                agencies_list = uniq

            # policy documents
            pd_rows = iptp_policydocument[iptp_policydocument["policy_id"] == pid] if len(iptp_policydocument) else pd.DataFrame()
            attachments = []
            commentary = []

            if len(pd_rows):
                pd_rows = pd_rows.copy()
                pd_rows["sort_order"] = pd.to_numeric(pd_rows.get("sort_order", 0), errors="coerce").fillna(0)
                pd_rows = pd_rows.sort_values(["document_type_title", "date", "sort_order", "id"])

                for _, r in pd_rows.iterrows():
                    doc_type_title = r.get("document_type_title")
                    doc_type_slug = r.get("document_type_slug")
                    policydoc_id = int(r.get("id")) if not pd.isna(r.get("id")) else None

                    item = {
                        "policydocument_id": policydoc_id,
                        "document_type": doc_type_title,
                        "document_type_slug": doc_type_slug,
                        "date": str(r.get("date")) if not pd.isna(r.get("date")) else None,
                        "display_title": r.get("display_title"),
                        "link_title": r.get("link_title"),
                        "link": r.get("link"),
                        "description_html": r.get("description"),
                        "wagtaildoc_id": int(r.get("wagtaildoc_id")) if not pd.isna(r.get("wagtaildoc_id")) else None,
                        "wagtaildoc_file": r.get("wagtaildoc_file"),
                        "wagtaildoc_title": r.get("wagtaildoc_title"),
                        "file_size": int(r.get("file_size")) if not pd.isna(r.get("file_size")) else None,
                        "file_hash": r.get("file_hash"),
                        "slug": r.get("slug"),
                    }

                    # Commentary: keep TEXT (description/link) even if no pdf.
                    if (doc_type_title or "").strip().lower() == "commentary":
                        commentary.append(item)
                        continue

                    # Otherwise: treat as attachment if it has a wagtail file reference
                    has_file = isinstance(item.get("wagtaildoc_file"), str) and bool(item.get("wagtaildoc_file").strip())
                    if not has_file:
                        attachments.append(item)
                        continue

                    attachments.append(item)

                    # Add to manifest if the file is a PDF (or include-nonpdf)
                    file_rel = item["wagtaildoc_file"].strip()
                    is_pdf = is_pdf_path(file_rel)
                    if (not is_pdf) and (not args.include_nonpdf):
                        continue

                    # S3 key resolution discovered: media/<file>
                    s3_key = f"media/{file_rel}"

                    ext = os.path.splitext(file_rel)[1] or (".pdf" if is_pdf else "")
                    base_title = item.get("display_title") or item.get("wagtaildoc_title") or item.get("link_title") or f"doc-{item.get('wagtaildoc_id')}"
                    fname = (
                        f"{safe_slug(doc_type_slug or doc_type_title or 'document', 40)}"
                        f"__{policydoc_id or 'unknown'}"
                        f"__{safe_slug(str(base_title), 80)}"
                        f"__{short_hash(s3_key, 10)}"
                        f"{ext}"
                    )

                    dest_relpath = f"policies/{folder}/documents/{fname}"

                    manifest_items.append(
                        ManifestItem(
                            page_ptr_id=pid,
                            wagtaildoc_id=int(item["wagtaildoc_id"]) if item["wagtaildoc_id"] is not None else -1,
                            policydocument_id=policydoc_id if policydoc_id is not None else -1,
                            s3_bucket=args.documents_bucket,
                            s3_key=s3_key,
                            dest_relpath=dest_relpath,
                            file_title=str(base_title),
                            document_type_title=str(doc_type_title),
                        )
                    )

            # related policies
            related_raw = json_load_maybe(prow.get("related_policies"))
            related_ids: List[int] = []
            if isinstance(related_raw, list):
                for obj in related_raw:
                    if isinstance(obj, dict) and obj.get("type") == "related_policy":
                        val = obj.get("value")
                        try:
                            related_ids.append(int(val))
                        except Exception:
                            pass

            related_resolved = []
            if related_ids:
                for rid in related_ids:
                    if rid in all_policy_meta.index:
                        rr = all_policy_meta.loc[rid]
                        related_resolved.append({
                            "page_ptr_id": int(rr["page_ptr_id"]),
                            "title": rr.get("title"),
                            "slug": rr.get("slug"),
                            "announced_date": str(rr.get("announced_date")) if rr.get("announced_date") is not None else None,
                            "effective_date": str(rr.get("effective_date")) if rr.get("effective_date") is not None else None,
                            "expired_date": str(rr.get("expired_date")) if rr.get("expired_date") is not None else None,
                            "current_status": rr.get("current_status_title"),
                        })
                    else:
                        related_resolved.append({"page_ptr_id": rid})

            # NEW: summary derivation from description_html
            description_html = prow.get("description")
            description_text = html_to_text(description_html)

            policy_record = {
                "page_ptr_id": int(pid),
                "title": title or None,
                "slug": slug or None,
                "url_path": prow.get("url_path"),
                "live": bool(prow.get("live")) if prow.get("live") is not None else None,
                "announced_date": str(prow.get("announced_date")) if prow.get("announced_date") is not None else None,
                "effective_date": str(prow.get("effective_date")) if prow.get("effective_date") is not None else None,
                "expired_date": str(prow.get("expired_date")) if prow.get("expired_date") is not None else None,
                "current_status": {
                    "title": prow.get("current_status_title"),
                    "slug": prow.get("current_status_slug"),
                } if prow.get("current_status_title") is not None else None,
                "description_html": description_html,
                "description_text": description_text,
                "search_document": prow.get("search_document"),
                "filters": {
                    "administrative_actions": actions_list,
                    "subject_matter": subject_list,
                    "agencies": agencies_list,
                    "status_tags": statuses_list,
                },
                "documents": {
                    "attachments": attachments,
                    "commentary": commentary,
                },
                "related_policies": related_resolved,
                "dataset": {
                    "assembled_at": datetime.utcnow().isoformat() + "Z",
                    "backup_prefix": backup_prefix,
                    "seed": args.seed,
                },
            }
            if args.include_curation_notes:
                policy_record["curation_notes_html"] = prow.get("curation_notes")

            # Write policy.json atomically
            if should_write:
                tmp = policy_json_path.with_suffix(".json.tmp")
                tmp.write_text(json_dumps_strict(policy_record, indent=2), encoding="utf-8")
                tmp.replace(policy_json_path)

            # Write index line (+ NEW description_chars)
            idx_f.write(json_dumps_strict({
                "page_ptr_id": int(pid),
                "folder": folder,
                "title": title,
                "announced_date": policy_record["announced_date"],
                "effective_date": policy_record["effective_date"],
                "current_status": (policy_record["current_status"] or {}).get("title"),
                "n_attachments": len(attachments),
                "n_commentary_items": len(commentary),
                "n_related": len(related_resolved),
                "description_chars": len(description_text),
            }) + "\n")

    # De-dupe manifest by (s3_key, dest_relpath)
    seen = set()
    uniq_manifest: List[ManifestItem] = []
    for mi in manifest_items:
        key = (mi.s3_key, mi.dest_relpath)
        if key in seen:
            continue
        seen.add(key)
        uniq_manifest.append(mi)

    # Write manifest.jsonl
    with manifest_path.open("w", encoding="utf-8") as mf:
        for mi in uniq_manifest:
            mf.write(json_dumps_strict(mi.to_json()) + "\n")

    print("\nDone.")
    print(f"Sampled policies: {len(sampled_ids)}")
    print(f"Wrote policies to: {out_policies_dir}")
    print(f"Manifest lines   : {len(uniq_manifest)} -> {manifest_path}")
    print(f"Policy index     : {policies_index_path}")
    print("\nNext step: run download_documents.py using manifest.jsonl\n")


if __name__ == "__main__":
    main()
