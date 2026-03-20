#!/usr/bin/env python3
"""

How to run
------------
python -m scripts.get_data.download_documents --config config.json

--------- manual way (old)
python3 scripts/download_documents.py \
  --manifest data/sample_100/manifest.jsonl \
  --out-dir data/sample_100 \
  --jobs 12 \
  --skip-existing \
  --verify-size \
  --quiet

"""


import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from scripts.common.config import (
    load_config,
    get_cfg_value,
)


def cfg_default(cfg: Dict[str, Any], key: str, fallback: Any) -> Any:
    v = get_cfg_value(cfg, key)
    return fallback if v is None else v


def sh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDERR:\n{p.stderr.strip()}\nSTDOUT:\n{p.stdout.strip()}"
        )
    return p


def parse_manifest_line(obj: Dict[str, Any]) -> Tuple[str, str, str, str, Optional[int]]:
    dest_rel = (
        obj.get("dest_relpath")
        or obj.get("dest_rel")
        or obj.get("local_relpath")
        or obj.get("local_path")
        or obj.get("dest_path")
    )
    if not dest_rel:
        raise ValueError(f"Manifest line missing dest_relpath (keys={list(obj.keys())})")

    expected_size = obj.get("size") or obj.get("content_length") or obj.get("ContentLength")
    if isinstance(expected_size, str) and expected_size.isdigit():
        expected_size = int(expected_size)
    if not isinstance(expected_size, int):
        expected_size = None

    s3_uri = obj.get("s3_uri")
    bucket = obj.get("bucket") or obj.get("s3_bucket")
    key = obj.get("key") or obj.get("s3_key")

    if s3_uri:
        if s3_uri.startswith("s3://") and (bucket is None or key is None):
            rest = s3_uri[len("s3://"):]
            b, _, k = rest.partition("/")
            bucket = bucket or b
            key = key or k
    else:
        if not bucket or not key:
            raise ValueError(f"Manifest line missing s3_uri or s3_bucket+s3_key (keys={list(obj.keys())})")
        s3_uri = f"s3://{bucket}/{key}"

    return s3_uri, bucket or "", key or "", dest_rel, expected_size


def head_object_size(bucket: str, key: str, aws_profile: Optional[str], aws_region: Optional[str]) -> Optional[int]:
    cmd = ["aws"]
    if aws_profile:
        cmd += ["--profile", aws_profile]
    if aws_region:
        cmd += ["--region", aws_region]
    cmd += ["s3api", "head-object", "--bucket", bucket, "--key", key]
    p = sh(cmd, check=False)
    if p.returncode != 0:
        return None
    try:
        data = json.loads(p.stdout)
        sz = data.get("ContentLength")
        return int(sz) if isinstance(sz, int) else None
    except Exception:
        return None


def should_skip(dest_path: Path, expected_size: Optional[int], skip_existing: bool) -> bool:
    if not dest_path.exists():
        return False
    if not skip_existing:
        return False
    if expected_size is None:
        return dest_path.stat().st_size > 0
    return dest_path.stat().st_size == expected_size


def download_one(
    obj: Dict[str, Any],
    s3_uri: str,
    bucket: str,
    key: str,
    dest_path: Path,
    expected_size: Optional[int],
    skip_existing: bool,
    verify_size: bool,
    aws_profile: Optional[str],
    aws_region: Optional[str],
    quiet: bool,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns (ok, msg, meta). meta includes useful identifying info for failure logs.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # If requested, fill expected_size from S3 only when we might skip or validate
    if verify_size and expected_size is None and dest_path.exists():
        expected_size = head_object_size(bucket, key, aws_profile, aws_region)

    if should_skip(dest_path, expected_size, skip_existing):
        return True, "skipped", {}

    # If verify_size and file exists but wrong size, delete it and re-download
    if verify_size and dest_path.exists() and expected_size is not None:
        if dest_path.stat().st_size != expected_size:
            try:
                dest_path.unlink()
            except Exception:
                pass

    cmd = ["aws"]
    if aws_profile:
        cmd += ["--profile", aws_profile]
    if aws_region:
        cmd += ["--region", aws_region]
    cmd += ["s3", "cp", s3_uri, str(dest_path)]
    if quiet:
        cmd += ["--no-progress"]

    p = sh(cmd, check=False)
    if p.returncode != 0:
        meta = {
            "page_ptr_id": obj.get("page_ptr_id"),
            "policydocument_id": obj.get("policydocument_id"),
            "wagtaildoc_id": obj.get("wagtaildoc_id"),
            "s3_bucket": bucket,
            "s3_key": key,
            "dest_relpath": obj.get("dest_relpath"),
        }
        return False, (p.stderr.strip() or p.stdout.strip() or "unknown aws error"), meta

    if verify_size:
        # ensure size matches (fetch from S3 if still missing)
        if expected_size is None:
            expected_size = head_object_size(bucket, key, aws_profile, aws_region)
        if expected_size is not None:
            got = dest_path.stat().st_size
            if got != expected_size:
                meta = {
                    "page_ptr_id": obj.get("page_ptr_id"),
                    "policydocument_id": obj.get("policydocument_id"),
                    "wagtaildoc_id": obj.get("wagtaildoc_id"),
                    "s3_bucket": bucket,
                    "s3_key": key,
                    "dest_relpath": obj.get("dest_relpath"),
                    "expected_size": expected_size,
                    "got_size": got,
                }
                return False, f"size mismatch after download: expected {expected_size}, got {got}", meta

    return True, "downloaded", {}


def main():
    # parse only --config so we can load cfg before setting defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="Path to config.json (optional)")
    pre_args, _ = pre.parse_known_args()

    cfg: Dict[str, Any] = {}
    if pre_args.config:
        cfg = load_config(Path(pre_args.config))

    # full parser with cfg-driven defaults (CLI still overrides)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=pre_args.config, help="Path to config.json (optional)")

    ap.add_argument("--manifest", default=None, help="Path to manifest.jsonl (optional if --config)")
    ap.add_argument("--out-dir", default=None, help="Root directory to download into (optional if --config)")

    ap.add_argument("--jobs", type=int, default=int(cfg_default(cfg, "download.jobs", 8)), help="Parallel downloads")

    # Allow config defaults + CLI override both ways
    ap.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg_default(cfg, "download.skip_existing", False)),
        help="Skip if dest exists",
    )
    ap.add_argument(
        "--verify-size",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg_default(cfg, "download.verify_size", False)),
        help="Use head-object to verify size for resume; re-download truncated files",
    )
    ap.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg_default(cfg, "download.quiet", False)),
        help="Disable aws progress output",
    )
    ap.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg_default(cfg, "download.dry_run", False)),
        help="Print planned downloads and exit",
    )

    ap.add_argument(
        "--failures-out",
        default=cfg_default(cfg, "download.failures_out", None),
        help="Write failures to this jsonl (default: <out-dir>/download_failures.jsonl)",
    )

    ap.add_argument("--aws-profile", default=cfg_default(cfg, "aws.profile", None))
    ap.add_argument("--aws-region", default=cfg_default(cfg, "aws.region", None))

    args = ap.parse_args()


    # Resolve manifest path
    if args.manifest is None:
        v = get_cfg_value(cfg, "paths.manifest_jsonl")
        if v is not None:
            args.manifest = str(v)
        else:
            # Derive from existing config (no new keys required)
            idx = get_cfg_value(cfg, "paths.policies_index_jsonl")
            if idx is None:
                raise SystemExit("Need --manifest or config paths.manifest_jsonl or paths.policies_index_jsonl")
            args.manifest = str(Path(str(idx)).parent / "manifest.jsonl")

    # Resolve out-dir
    if args.out_dir is None:
        v = get_cfg_value(cfg, "assemble.out_dir") or get_cfg_value(cfg, "paths.dataset_dir")
        if v is not None:
            args.out_dir = str(v)
        else:
            # safest default: dataset root = manifest's parent directory
            args.out_dir = str(Path(args.manifest).parent)




    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_out = Path(args.failures_out) if args.failures_out else (out_dir / "download_failures.jsonl")

    items = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[manifest] line {i}: JSON error: {e}", file=sys.stderr)
                continue
            try:
                s3_uri, bucket, key, dest_rel, expected_size = parse_manifest_line(obj)
            except Exception as e:
                print(f"[manifest] line {i}: parse error: {e}", file=sys.stderr)
                continue

            dest_path = out_dir / dest_rel
            items.append((obj, s3_uri, bucket, key, dest_path, expected_size))

    print(f"Manifest: {manifest_path} ({len(items)} items)")
    print(f"Out dir : {out_dir}")
    print(f"Jobs    : {args.jobs}")
    print(f"Resume  : {'on' if args.skip_existing else 'off'}")
    print(f"Verify  : {'on' if args.verify_size else 'off'}")

    if args.dry_run:
        shown = 0
        for obj, s3_uri, bucket, key, dest_path, expected_size in items[:25]:
            print(f"GET {s3_uri} -> {dest_path}")
            shown += 1
        print(f"... ({len(items)} total)")
        return

    n_ok = 0
    n_skip = 0
    n_fail = 0
    failures_out.parent.mkdir(parents=True, exist_ok=True)

    with failures_out.open("w", encoding="utf-8") as fail_f, ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {}
        for obj, s3_uri, bucket, key, dest_path, expected_size in items:
            fut = ex.submit(
                download_one,
                obj, s3_uri, bucket, key, dest_path, expected_size,
                args.skip_existing, args.verify_size,
                args.aws_profile, args.aws_region,
                args.quiet
            )
            futs[fut] = (obj, s3_uri, dest_path)

        for idx, fut in enumerate(as_completed(futs), start=1):
            ok, msg, meta = fut.result()
            if ok:
                if msg == "skipped":
                    n_skip += 1
                else:
                    n_ok += 1
            else:
                n_fail += 1
                obj, s3_uri, dest_path = futs[fut]
                record = {
                    "error": msg,
                    "s3_uri": s3_uri,
                    "dest_path": str(dest_path),
                    **meta,
                }
                fail_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if idx % 25 == 0 or idx == len(futs):
                print(f"Progress: {idx}/{len(futs)} | downloaded={n_ok} skipped={n_skip} failed={n_fail}")

    print("\nDone.")
    print(f"Downloaded: {n_ok}")
    print(f"Skipped   : {n_skip}")
    print(f"Failed    : {n_fail}")
    print(f"Failures  : {failures_out}")


if __name__ == "__main__":
    main()
