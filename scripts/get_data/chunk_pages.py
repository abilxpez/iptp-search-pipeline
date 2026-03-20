# Convert extracted_pages.jsonl into chunk-level records for indexing.
# Reads pages line-by-line (JSONL) and writes chunks.jsonl (one line per chunk).
#
# Updated to:
# - Read paths from a config value (JSON for now; we'll write the config file next)
# - Use the real extracted_pages schema: page_ptr_id + policydocument_id
# - Option A metadata join: duplicate policy + attachment metadata into each chunk
# - Use policies_index.jsonl to locate each policy folder, then load policy.json for rich metadata
# - Handle missing/partial metadata gracefully and log all anomalies


"""
How to use:

python -m scripts.get_data.chunk_pages --config config.json 

"""

from __future__ import annotations

import argparse
import html
import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from scripts.common.policy_meta import flatten_policy_meta  # type: ignore

RE_PUNCT_SPLIT = re.compile(r"(?<=[;:,\)])\s+")  # gentle split points before hard windowing

# Compiled regexes for speed and consistency
RE_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
RE_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])(?:[ \t]+)")
RE_FR_HEADER = re.compile(
    r"\bFederal Register\s*/\s*Vol\.\s*\d+.*?Rules and Regulations\b",
    flags=re.IGNORECASE,
)
RE_FR_FOOTER_TOKENS = re.compile(
    r"\bVerDate\b.*?\bJkt\b|\bE:\\FR\\FM\\b",
    flags=re.IGNORECASE,
)
RE_NL_3PLUS = re.compile(r"\n{3,}")
RE_NL = re.compile(r"\n")
RE_SPACES_TABS = re.compile(r"[ \t]+")

# Strip C0/C1 control chars (keeps \n handling elsewhere; this runs after newline normalization anyway)
RE_CONTROL_CHARS = re.compile(r"[\x00-\x1F\x7F-\x9F]")


# -------------------------
# Data records
# -------------------------

@dataclass(frozen=True)
class PageRecord:
    # Real schema keys (kept as strings for stable IDs / consistent joins)
    page_ptr_id: str
    policydocument_id: str
    page_num: int
    text: str
    source_path: str

    # Extra fields present in extracted_pages.jsonl that are useful for metadata join
    extra: Dict[str, Any]


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_type: str
    page_ptr_id: str
    policydocument_id: Optional[str]
    page_start: int
    page_end: int
    text: str
    source_path: Optional[str]
    meta: Dict[str, Any]
    meta_flat: Dict[str, Any]


# -------------------------
# Small utilities
# -------------------------

def ensure_parent_dir(path: Path) -> None:
    """Ensure the parent directory exists for a given output file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


class JsonlLogger:
    def __init__(self, log_path: Path):
        ensure_parent_dir(log_path)
        self.log_path = log_path
        self.f = log_path.open("a", encoding="utf-8")

    def log(self, event: Dict[str, Any]) -> None:
        self.f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.f.flush()
        self.f.close()


def load_json(path: Path) -> Any:
    """Load a JSON file with a clear error."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path, logger: Optional[JsonlLogger] = None) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list. (Used for small index files.)"""
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                if logger is not None:
                    logger.log(
                        {
                            "type": "BAD_JSON_LINE",
                            "source": str(path),
                            "line_num": line_num,
                            "error": str(e),
                        }
                    )
    return out


# -------------------------
# Config handling
# -------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load a config file (JSON for now) that contains path values.
    Reader is intentionally strict:
    - file must exist
    - must be a dict
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_json(config_path)

    if not isinstance(cfg, dict):
        raise ValueError("Config must be a JSON object at top-level")

    # Optional local override (ignored by git): config.local.json next to base config.
    local_path = config_path.with_name("config.local.json")
    if local_path.exists():
        local_cfg = load_json(local_path)
        if not isinstance(local_cfg, dict):
            raise ValueError("Local config must be a JSON object at top-level")

        def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = dict(base)
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = _deep_merge(out[k], v)
                else:
                    out[k] = v
            return out

        cfg = _deep_merge(cfg, local_cfg)

    return cfg


def get_cfg_value(cfg: Dict[str, Any], key: str) -> Any:
    """
    Read a possibly-nested config key.
    Supports dot notation like "paths.extracted_pages_jsonl".
    """
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def get_path(
    cfg: Dict[str, Any],
    key: str,
    *,
    default: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    val = get_cfg_value(cfg, key)
    if val is None:
        val = default
    if val is None:
        raise KeyError(f"Missing required config key: {key}")

    p = Path(str(val))
    if base_dir is not None and not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def get_int(cfg: Dict[str, Any], key: str, *, default: Optional[int] = None) -> int:
    val = get_cfg_value(cfg, key)
    if val is None:
        val = default
    if val is None:
        raise KeyError(f"Missing required config key: {key}")
    try:
        return int(val)
    except (TypeError, ValueError):
        raise ValueError(f"Config key {key} must be an int, got: {val!r}")


# -------------------------
# Metadata: policies_index + policy.json join (Option A)
# -------------------------

def build_policy_folder_index(policies_index_path: Path, logger: JsonlLogger) -> Dict[str, Dict[str, Any]]:
    """
    Build a lookup:
      page_ptr_id (string) -> {folder, quick_fields...}
    """
    idx_rows = load_jsonl(policies_index_path, logger=logger)
    out: Dict[str, Dict[str, Any]] = {}

    for row in idx_rows:
        page_ptr_id = row.get("page_ptr_id")
        folder = row.get("folder")
        if page_ptr_id is None or not folder:
            logger.log(
                {
                    "type": "WARN_BAD_POLICIES_INDEX_ROW",
                    "source": str(policies_index_path),
                    "row": row,
                }
            )
            continue

        key = str(page_ptr_id)
        out[key] = {
            "folder": folder,
            "title": row.get("title"),
            "announced_date": row.get("announced_date"),
            "effective_date": row.get("effective_date"),
            "current_status": row.get("current_status"),
            "n_attachments": row.get("n_attachments"),
            "n_commentary_items": row.get("n_commentary_items"),
            "n_related": row.get("n_related"),
        }

    return out


def normalize_status(status_obj: Any) -> Dict[str, Any]:
    """
    Normalize current_status, which can be:
    - string in policies_index.jsonl
    - dict in policy.json (often {title, slug, ...})
    """
    if status_obj is None:
        return {"title": None, "slug": None}

    if isinstance(status_obj, str):
        return {"title": status_obj, "slug": None}

    if isinstance(status_obj, dict):
        return {"title": status_obj.get("title"), "slug": status_obj.get("slug")}

    return {"title": str(status_obj), "slug": None}


def extract_policy_meta(policy_json: Dict[str, Any]) -> Dict[str, Any]:
    """Extract policy-level metadata from policy.json (kept relatively flat)."""
    current_status = normalize_status(policy_json.get("current_status"))

    return {
        "page_ptr_id": policy_json.get("page_ptr_id"),
        "title": policy_json.get("title"),
        "slug": policy_json.get("slug"),
        "url_path": policy_json.get("url_path"),
        "announced_date": policy_json.get("announced_date") or policy_json.get("date_announced"),
        "effective_date": policy_json.get("effective_date"),
        "current_status": current_status,
        "filters": policy_json.get("filters"),
    }


def build_attachment_meta_map(policy_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build:
      policydocument_id (string) -> attachment_meta dict

    policy.json commonly nests attachments under policy_json["documents"]["attachments"].
    We handle missing structure safely.
    """
    out: Dict[str, Dict[str, Any]] = {}

    docs = policy_json.get("documents") or {}
    attachments = docs.get("attachments") or []

    if not isinstance(attachments, list):
        return out

    for a in attachments:
        if not isinstance(a, dict):
            continue
        pdid = a.get("policydocument_id")
        if pdid is None:
            continue

        out[str(pdid)] = {
            "policydocument_id": pdid,
            "document_type": a.get("document_type"),
            "date": a.get("date"),
            "display_title": a.get("display_title") or a.get("title"),
            "link": a.get("link"),
            "wagtaildoc_id": a.get("wagtaildoc_id"),
            "wagtaildoc_title": a.get("wagtaildoc_title"),
        }

    return out


class PolicyMetaCache:
    """Small cache so we don't repeatedly load policy.json for the same policy."""

    def __init__(self, policies_dir: Path, folder_index: Dict[str, Dict[str, Any]], logger: JsonlLogger):
        self.policies_dir = policies_dir
        self.folder_index = folder_index
        self.logger = logger
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, page_ptr_id: str) -> Dict[str, Any]:
        """
        Return:
          {"policy_meta": {...}, "attachment_meta_map": {...}, "index_fallback": {...}, "summary_text": ...}
        """
        if page_ptr_id in self._cache:
            return self._cache[page_ptr_id]

        idx = self.folder_index.get(page_ptr_id)

        index_fallback = {
            "folder": None,
            "title": None,
            "announced_date": None,
            "effective_date": None,
            "current_status": None,
            "n_attachments": None,
            "n_commentary_items": None,
            "n_related": None,
        }

        if idx is None:
            self.logger.log({"type": "WARN_POLICY_NOT_IN_INDEX", "page_ptr_id": page_ptr_id})
        else:
            index_fallback.update(idx)

        policy_meta: Dict[str, Any] = {
            "page_ptr_id": int(page_ptr_id) if page_ptr_id.isdigit() else page_ptr_id,
            "title": index_fallback.get("title"),
            "announced_date": index_fallback.get("announced_date"),
            "effective_date": index_fallback.get("effective_date"),
            "current_status": normalize_status(index_fallback.get("current_status")),
        }
        attachment_map: Dict[str, Dict[str, Any]] = {}
        summary_text: Optional[str] = None

        folder = index_fallback.get("folder")
        if folder:
            policy_path = self.policies_dir / folder / "policy.json"
            if policy_path.exists():
                try:
                    policy_json = load_json(policy_path)
                    if isinstance(policy_json, dict):
                        policy_meta = extract_policy_meta(policy_json)
                        attachment_map = build_attachment_meta_map(policy_json)
                        dt = policy_json.get("description_text")
                        if isinstance(dt, str):
                            summary_text = dt.strip() or None
                        if summary_text is None:
                            dh = policy_json.get("description_html")
                            if isinstance(dh, str):
                                summary_text = html_to_text(dh).strip() or None
                    else:
                        self.logger.log(
                            {
                                "type": "WARN_POLICY_JSON_NOT_OBJECT",
                                "page_ptr_id": page_ptr_id,
                                "policy_path": str(policy_path),
                            }
                        )
                except Exception as e:
                    self.logger.log(
                        {
                            "type": "WARN_POLICY_JSON_LOAD_FAILED",
                            "page_ptr_id": page_ptr_id,
                            "policy_path": str(policy_path),
                            "error": str(e),
                        }
                    )
            else:
                self.logger.log(
                    {
                        "type": "WARN_MISSING_POLICY_JSON",
                        "page_ptr_id": page_ptr_id,
                        "policy_path": str(policy_path),
                    }
                )

        value = {
            "policy_meta": policy_meta,
            "attachment_meta_map": attachment_map,
            "index_fallback": index_fallback,
            "summary_text": summary_text,
        }
        self._cache[page_ptr_id] = value
        return value


# -------------------------
# Input: stream extracted_pages.jsonl
# -------------------------

def iter_pages(pages_jsonl_path: Path, logger: JsonlLogger) -> Iterator[PageRecord]:
    """
    Stream pages from extracted_pages.jsonl.

    Expected keys:
      - page_ptr_id
      - policydocument_id
      - page_num
      - text
      - source_path
      - (optional extras: dest_relpath, document_type, wagtaildoc_id, s3_key, etc.)
    """
    with pages_jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.log(
                    {"type": "BAD_JSON_LINE", "source": str(pages_jsonl_path), "line_num": line_num, "error": str(e)}
                )
                continue

            page_ptr_id = obj.get("page_ptr_id")
            policydocument_id = obj.get("policydocument_id")
            page_num_raw = obj.get("page_num")

            if page_ptr_id is None or policydocument_id is None or page_num_raw is None:
                logger.log(
                    {
                        "type": "SKIP_BAD_RECORD",
                        "source": str(pages_jsonl_path),
                        "line_num": line_num,
                        "missing_or_invalid": {
                            "page_ptr_id": page_ptr_id is None,
                            "policydocument_id": policydocument_id is None,
                            "page_num": page_num_raw is None,
                        },
                    }
                )
                continue

            try:
                page_num = int(page_num_raw)
            except (TypeError, ValueError):
                logger.log(
                    {
                        "type": "SKIP_BAD_PAGE_NUM",
                        "source": str(pages_jsonl_path),
                        "line_num": line_num,
                        "page_ptr_id": str(page_ptr_id),
                        "policydocument_id": str(policydocument_id),
                        "page_num": page_num_raw,
                    }
                )
                continue

            if page_num <= 0:
                logger.log(
                    {
                        "type": "SKIP_NONPOSITIVE_PAGE_NUM",
                        "source": str(pages_jsonl_path),
                        "line_num": line_num,
                        "page_ptr_id": str(page_ptr_id),
                        "policydocument_id": str(policydocument_id),
                        "page_num": page_num,
                    }
                )
                continue

            extra = dict(obj)
            extra.pop("text", None)

            yield PageRecord(
                page_ptr_id=str(page_ptr_id),
                policydocument_id=str(policydocument_id),
                page_num=page_num,
                text=(obj.get("text") if isinstance(obj.get("text"), str) else ""),
                source_path=str(obj.get("source_path") or ""),
                extra=extra,
            )


# -------------------------
# Text cleaning / splitting
# -------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace and remove common PDF artifacts."""
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = RE_HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = text.replace("\t", " ")
    text = RE_SPACES_TABS.sub(" ", text)
    text = RE_NL_3PLUS.sub("\n\n", text)

    text = text.replace("\n\n", "\u0000")
    text = RE_NL.sub(" ", text)
    text = text.replace("\u0000", "\n\n")

    text = RE_FR_HEADER.sub("", text)
    text = RE_FR_FOOTER_TOKENS.sub("", text)

    text = RE_SPACES_TABS.sub(" ", text)
    text = RE_NL_3PLUS.sub("\n\n", text)

    # Remove problematic control characters (keep newline + tab)
    text = "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or (0x20 <= ord(ch) <= 0x7E) or ord(ch) >= 0xA0
    )

    # Remove non-printable control chars that can break logs/indexing (e.g., \x9d showing as "A%")
    text = RE_CONTROL_CHARS.sub("", text)

    return text.strip()


def is_boilerplate(text: str) -> bool:
    """Return True if the text is mostly boilerplate we don't want indexed."""
    t = text.lower()
    patterns = [
        "this section of the federal register",
        "table of contents",
        "table of abbreviations",
        "list of subjects",
        "public participation",
        "for further information contact",
        "addresses:",
        "comments must be submitted",
        "http:// www.regulations.gov",
        "verdate sep",
        "jkt ",
        " frm ",
        " fmt ",
        " sfmt ",
        "e:\\fr\\fm",
    ]
    return any(p in t for p in patterns)


def split_oversized_unit(unit: str, *, max_chars: int, hard_window: int) -> List[str]:
    """Split one oversized unit into smaller pieces (paragraph -> punctuation -> hard windows)."""
    if len(unit) <= max_chars:
        return [unit]

    if "\n\n" in unit:
        parts = [p.strip() for p in unit.split("\n\n") if p.strip()]
    else:
        parts = [unit]

    refined: List[str] = []
    for p in parts:
        if len(p) <= max_chars:
            refined.append(p)
            continue
        subparts = [sp.strip() for sp in RE_PUNCT_SPLIT.split(p) if sp.strip()]
        refined.extend(subparts if subparts else [p])

    final: List[str] = []
    for p in refined:
        if len(p) <= max_chars:
            final.append(p)
            continue
        for i in range(0, len(p), hard_window):
            window = p[i : i + hard_window].strip()
            if window:
                final.append(window)

    return final


def split_into_units(text: str) -> List[str]:
    """Split cleaned text into sentence-ish units, preserving paragraph structure."""
    if not text:
        return []

    units: List[str] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in paragraphs:
        sentences = RE_SENT_SPLIT.split(para)
        sentences = [s.strip() for s in sentences if s.strip()]
        units.extend(sentences)
    return units


def split_summary_paragraphs(text: str) -> List[str]:
    """Split summary text into paragraphs (blank-line delimited)."""
    if not text:
        return []
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def html_to_text(s: Any) -> str:
    """
    Conservative HTML-ish -> plain text for policy summaries.
    - Decodes HTML entities
    - Preserves basic structure: paragraphs, line breaks, list items
    - Strips remaining tags
    - Normalizes whitespace
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    s = s.strip()
    if not s:
        return ""

    s = html.unescape(s)

    s = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*p\s*>", "\n\n", s)
    s = re.sub(r"(?i)<\s*p(\s+[^>]*)?\s*>", "", s)
    s = re.sub(r"(?i)</\s*div\s*>", "\n\n", s)
    s = re.sub(r"(?i)<\s*div(\s+[^>]*)?\s*>", "", s)

    s = re.sub(r"(?i)<\s*li(\s+[^>]*)?\s*>", "- ", s)
    s = re.sub(r"(?i)</\s*li\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*ul\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*ol\s*>", "\n", s)
    s = re.sub(r"(?i)<\s*(ul|ol)(\s+[^>]*)?\s*>", "", s)

    s = re.sub(r"<[^>]+>", "", s)

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# -------------------------
# Grouping / chunk IDs
# -------------------------

def iter_page_groups(pages: Iterator[PageRecord]) -> Iterator[Tuple[str, str, List[PageRecord]]]:
    """Group pages by (page_ptr_id, policydocument_id). Assumes input is ordered."""
    current_key: Optional[Tuple[str, str]] = None
    current_group: List[PageRecord] = []

    for page in pages:
        key = (page.page_ptr_id, page.policydocument_id)

        if current_key is None:
            current_key = key

        if key != current_key:
            yield current_key[0], current_key[1], current_group
            current_key = key
            current_group = []

        current_group.append(page)

    if current_key is not None and current_group:
        yield current_key[0], current_key[1], current_group


def stable_chunk_hash(
    source_kind: str,
    page_ptr_id: str,
    policydocument_id: Optional[str],
    page_start: int,
    page_end: int,
    chunk_text: str,
) -> str:
    """Compute a stable short hash for chunk identity (helps caching + dedupe)."""
    normalized_text = RE_SPACES_TABS.sub(" ", chunk_text).strip()
    doc_id = policydocument_id or ""
    payload = f"{source_kind}|{page_ptr_id}|{doc_id}|{page_start}|{page_end}|{normalized_text}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def stable_summary_hash(
    source_kind: str,
    page_ptr_id: str,
    paragraph_index: int,
    summary_text: str,
) -> str:
    """Stable hash for summary chunks (based on text + paragraph index)."""
    normalized_text = RE_SPACES_TABS.sub(" ", summary_text).strip()
    payload = f"{source_kind}|{page_ptr_id}|{paragraph_index}|{normalized_text}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


# -------------------------
# Chunk creation helpers
# -------------------------

def make_chunk_from_buffer(
    page_ptr_id: str,
    policydocument_id: str,
    source_path: str,
    meta: Dict[str, Any],
    flat_base: Dict[str, Any],
    units: List[Tuple[str, int]],  # (unit_text, page_num)
    min_chars: int,
) -> Optional[ChunkRecord]:
    """Convert buffered units into a ChunkRecord if it passes size/boilerplate checks."""
    if not units:
        return None

    chunk_text = " ".join(u for (u, _) in units).strip()
    if len(chunk_text) < min_chars:
        return None
    if is_boilerplate(chunk_text):
        return None

    page_start = min(p for (_, p) in units)
    page_end = max(p for (_, p) in units)

    h = stable_chunk_hash("pdf", page_ptr_id, policydocument_id, page_start, page_end, chunk_text)
    chunk_id = f"{page_ptr_id}_pdf_{policydocument_id}_p{page_start:04d}-p{page_end:04d}_h{h}"

    flat_meta = dict(flat_base)
    flat_meta["source_type"] = "pdf_page"
    if policy_date := meta.get("policy", {}).get("announced_date"):
        flat_meta["announced_date"] = str(policy_date)
    flat_meta["source_type"] = "pdf_page"
    if policy_date := meta.get("policy", {}).get("announced_date"):
        flat_meta["announced_date"] = str(policy_date)

    return ChunkRecord(
        chunk_id=chunk_id,
        source_type="pdf_page",
        page_ptr_id=page_ptr_id,
        policydocument_id=policydocument_id,
        page_start=page_start,
        page_end=page_end,
        text=chunk_text,
        source_path=source_path,
        meta=meta,
        meta_flat=flat_meta,
    )


def build_summary_chunks(
    page_ptr_id: str,
    cached: Dict[str, Any],
) -> List[ChunkRecord]:
    summary_text = cached.get("summary_text")
    if not isinstance(summary_text, str) or not summary_text.strip():
        return []

    policy_meta = cached.get("policy_meta", {}) or {}
    paragraphs = split_summary_paragraphs(summary_text)
    if not paragraphs:
        return []

    split_by_paragraph = any(len(p) > 1500 for p in paragraphs)
    if not split_by_paragraph:
        paragraphs = ["\n\n".join(paragraphs)]

    chunks: List[ChunkRecord] = []
    num_paragraphs = len(paragraphs)
    split_method = "paragraph" if split_by_paragraph else "single"

    for idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        h = stable_summary_hash(source_kind="sum", page_ptr_id=page_ptr_id, paragraph_index=idx, summary_text=para)
        if split_by_paragraph:
            chunk_id = f"{page_ptr_id}_sum_par{idx:04d}_h{h}"
        else:
            chunk_id = f"{page_ptr_id}_sum_h{h}"

        meta = {
            "policy": policy_meta,
            "attachment": None,
            "summary": {
                "kind": "policy_level",
                "split_method": split_method,
                "paragraph_index": idx,
                "num_paragraphs": num_paragraphs,
            },
            "extracted": None,
        }
        flat_meta = flatten_policy_meta(meta)
        flat_meta["source_type"] = "policy_summary"
        if policy_date := policy_meta.get("announced_date"):
            flat_meta["announced_date"] = str(policy_date)

        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                source_type="policy_summary",
                page_ptr_id=page_ptr_id,
                policydocument_id=None,
                page_start=0,
                page_end=0,
                text=para,
                source_path=None,
                meta=meta,
                meta_flat=flat_meta,
            )
        )

    return chunks


def build_title_chunks(
    page_ptr_id: str,
    cached: Dict[str, Any],
) -> List[ChunkRecord]:
    policy_meta = cached.get("policy_meta", {}) or {}
    title = policy_meta.get("title")
    if not isinstance(title, str):
        return []
    title = title.strip()
    if not title:
        return []

    h = stable_summary_hash(source_kind="ttl", page_ptr_id=page_ptr_id, paragraph_index=0, summary_text=title)
    chunk_id = f"{page_ptr_id}_ttl_h{h}"

    meta = {
        "policy": policy_meta,
        "attachment": None,
        "title": {
            "kind": "policy_title",
        },
        "extracted": None,
    }
    flat_meta = flatten_policy_meta(meta)
    flat_meta["source_type"] = "policy_title"
    if policy_date := policy_meta.get("announced_date"):
        flat_meta["announced_date"] = str(policy_date)

    return [
        ChunkRecord(
            chunk_id=chunk_id,
            source_type="policy_title",
            page_ptr_id=page_ptr_id,
            policydocument_id=None,
            page_start=0,
            page_end=0,
            text=title,
            source_path=None,
            meta=meta,
            meta_flat=flat_meta,
        )
    ]


def reset_chunk_buffer() -> Tuple[List[Tuple[str, int]], int]:
    """Reset chunk buffer state."""
    return [], 0


def add_unit_to_buffer(
    buffer_units: List[Tuple[str, int]],
    buffer_text_len: int,
    page_num: int,
    unit: str,
) -> Tuple[List[Tuple[str, int]], int]:
    """Append a unit into the chunk buffer and update tracking variables."""
    buffer_units.append((unit, page_num))
    buffer_text_len += len(unit) + 1
    return buffer_units, buffer_text_len


# -------------------------
# Per-PDF chunker
# -------------------------

def log_group_warnings(
    logger: JsonlLogger,
    page_ptr_id: str,
    policydocument_id: str,
    pages: List[PageRecord],
    large_pdf_threshold: int,
) -> None:
    """Log warnings about per-PDF conditions (large PDFs)."""
    if large_pdf_threshold and len(pages) > large_pdf_threshold:
        logger.log(
            {
                "type": "WARN_LARGE_PDF",
                "page_ptr_id": page_ptr_id,
                "policydocument_id": policydocument_id,
                "pages": len(pages),
                "threshold": large_pdf_threshold,
            }
        )



def build_chunk_meta(
    page_ptr_id: str,
    policydocument_id: str,
    first_page_extra: Dict[str, Any],
    logger: JsonlLogger,
    cached: Dict[str, Any],
) -> Dict[str, Any]:
    """Option A: duplicate policy + attachment + extracted metadata into each chunk."""
    policy_meta = cached.get("policy_meta", {}) or {}
    attachment_map = cached.get("attachment_meta_map", {}) or {}

    attachment_meta = attachment_map.get(policydocument_id)
    if attachment_meta is None:
        logger.log(
            {
                "type": "WARN_MISSING_ATTACHMENT_META",
                "page_ptr_id": page_ptr_id,
                "policydocument_id": policydocument_id,
            }
        )
        attachment_meta = {"policydocument_id": policydocument_id, "missing_attachment_meta": True}

    extracted_meta = dict(first_page_extra) if isinstance(first_page_extra, dict) else {}
    extracted_meta.pop("page_ptr_id", None)
    extracted_meta.pop("policydocument_id", None)
    extracted_meta.pop("page_num", None)

    return {"policy": policy_meta, "attachment": attachment_meta, "extracted": extracted_meta}


def chunk_pages_for_pdf(
    page_ptr_id: str,
    policydocument_id: str,
    pages: List[PageRecord],
    cached: Dict[str, Any],
    target_chars: int,
    min_chars: int,
    max_unit_chars: int,
    hard_split_chars: int,
    overlap_enabled: bool,
    overlap_units: int,
    overlap_min_units: int,
    overlap_max_frac: float,
    logger: JsonlLogger,
) -> Iterator[ChunkRecord]:

    """Yield chunks for one (page_ptr_id, policydocument_id) group."""
    pages_sorted = sorted(pages, key=lambda p: p.page_num)
    source_path = pages_sorted[0].source_path if pages_sorted else ""

    source_paths = {p.source_path for p in pages_sorted if p.source_path}
    if len(source_paths) > 1:
        logger.log(
            {
                "type": "ERROR_MIXED_SOURCE_PATH",
                "page_ptr_id": page_ptr_id,
                "policydocument_id": policydocument_id,
                "source_paths": sorted(source_paths),
                "message": "Multiple source_path values in same (policy, document) group; skipping group.",
            }
        )
        return

    first_page_extra = pages_sorted[0].extra if pages_sorted else {}
    meta = build_chunk_meta(
        page_ptr_id=page_ptr_id,
        policydocument_id=policydocument_id,
        first_page_extra=first_page_extra,
        logger=logger,
        cached=cached,
    )
    flat_base = flatten_policy_meta(meta)
    if policy_date := meta.get("policy", {}).get("announced_date"):
        flat_base["announced_date"] = str(policy_date)

    buffer_units: List[Tuple[str, int]] = []
    buffer_text_len = 0
    oversized_units = 0

    for page in pages_sorted:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue

        units = split_into_units(cleaned)

        for u in units:
            if not u:
                continue

            if len(u) > max_unit_chars:
                oversized_units += 1
                sub_units = split_oversized_unit(u, max_chars=max_unit_chars, hard_window=hard_split_chars)
            else:
                sub_units = [u]

            for su in sub_units:
                if not su:
                    continue

                buffer_units, buffer_text_len = add_unit_to_buffer(
                    buffer_units=buffer_units,
                    buffer_text_len=buffer_text_len,
                    page_num=page.page_num,
                    unit=su,
                )


                if buffer_text_len >= target_chars:
                    # Freeze the current buffer as the candidate boundary
                    units_for_chunk = list(buffer_units)

                    candidate = make_chunk_from_buffer(
                        page_ptr_id=page_ptr_id,
                        policydocument_id=policydocument_id,
                        source_path=source_path,
                        meta=meta,
                        flat_base=flat_base,
                        units=units_for_chunk,
                        min_chars=min_chars,
                    )

                    if candidate is None:
                        # Important: do NOT clear or overlap-reset; keep accumulating
                        continue

                    yield candidate

                    # Only reset the buffer AFTER a chunk is emitted
                    if overlap_enabled and units_for_chunk:
                        overlap_count = max(overlap_units, overlap_min_units)
                        overlap_char_budget = int(target_chars * overlap_max_frac)

                        overlap_buffer: List[Tuple[str, int]] = []
                        char_count = 0

                        # take from the end, keep order
                        for (unit_text, unit_page) in reversed(units_for_chunk):
                            if len(overlap_buffer) >= overlap_count:
                                break

                            if len(overlap_buffer) == 0:
                                # ALWAYS include at least one unit
                                overlap_buffer.insert(0, (unit_text, unit_page))
                                char_count += len(unit_text)
                                continue

                            # For additional units, enforce char budget
                            if char_count + len(unit_text) > overlap_char_budget:
                                break

                            overlap_buffer.insert(0, (unit_text, unit_page))
                            char_count += len(unit_text)

                        buffer_units = overlap_buffer
                        buffer_text_len = sum(len(unit_text) + 1 for (unit_text, _) in buffer_units)
                    else:
                        buffer_units, buffer_text_len = reset_chunk_buffer()



    if buffer_units:
        candidate = make_chunk_from_buffer(
            page_ptr_id=page_ptr_id,
            policydocument_id=policydocument_id,
            source_path=source_path,
            meta=meta,
            flat_base=flat_base,
            units=buffer_units,
            min_chars=min_chars,
        )

        if candidate is not None:
            yield candidate

    if oversized_units > 0:
        logger.log(
            {
                "type": "STATS_OVERSIZED_UNITS",
                "page_ptr_id": page_ptr_id,
                "policydocument_id": policydocument_id,
                "count": oversized_units,
            }
        )


# -------------------------
# Output writers
# -------------------------

def write_chunk_jsonl(out_f, chunk: ChunkRecord) -> None:
    """Write one chunk record as a single JSONL line."""
    out_f.write(
        json.dumps(
            {
                "chunk_id": chunk.chunk_id,
                "source_type": chunk.source_type,
                "page_ptr_id": chunk.page_ptr_id,
                "policydocument_id": chunk.policydocument_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "text": chunk.text,
                "source_path": chunk.source_path,
                "meta": chunk.meta,
                "meta_flat": chunk.meta_flat,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def stream_chunks_to_file(
    out_f,
    page_ptr_id: str,
    policydocument_id: str,
    pages: List[PageRecord],
    cached: Dict[str, Any],
    target_chars: int,
    min_chars: int,
    max_unit_chars: int,
    hard_split_chars: int,
    overlap_enabled: bool,
    overlap_units: int,
    overlap_min_units: int,
    overlap_max_frac: float,
    logger: JsonlLogger,
) -> int:
    """Stream chunks for one PDF group into the output file."""
    n = 0
    for chunk in chunk_pages_for_pdf(
        page_ptr_id=page_ptr_id,
        policydocument_id=policydocument_id,
        pages=pages,
        cached=cached,
        target_chars=target_chars,
        min_chars=min_chars,
        max_unit_chars=max_unit_chars,
        hard_split_chars=hard_split_chars,
        overlap_enabled=overlap_enabled,
        overlap_units=overlap_units,
        overlap_min_units=overlap_min_units,
        overlap_max_frac=overlap_max_frac,
        logger=logger,
    ):
        write_chunk_jsonl(out_f, chunk)
        n += 1
    return n



# -------------------------
# Pipeline orchestrator
# -------------------------

def chunk_pages_pipeline(
    config_path: Path,
    pages_path: Optional[Path],
    out_chunks_path: Optional[Path],
    target_chars: Optional[int],
    min_chars: Optional[int],
    log_path: Optional[Path],
) -> None:
    """
    Orchestrator:
    - Load config (paths + chunking defaults)
    - Build policies_index folder lookup
    - Stream extracted_pages.jsonl
    - Chunk per (page_ptr_id, policydocument_id) group
    - Write chunks.jsonl
    """
    cfg = load_config(config_path)

    # Resolve relative paths in config relative to the config file location
    base_dir = config_path.parent.resolve()

    # Paths from config (with CLI override if provided)
    pages_jsonl_path = pages_path or get_path(cfg, "paths.extracted_pages_jsonl", base_dir=base_dir)
    policies_index_path = get_path(cfg, "paths.policies_index_jsonl", base_dir=base_dir)
    policies_dir = get_path(cfg, "paths.policies_dir", base_dir=base_dir)
    out_path = out_chunks_path or get_path(cfg, "paths.chunks_jsonl", base_dir=base_dir)
    log_path_final = log_path or get_path(
        cfg,
        "paths.chunk_log_jsonl",
        default="data/chunks/chunk_log.jsonl",
        base_dir=base_dir,
    )

    # Chunking params: CLI overrides config; config overrides defaults
    target_chars_final = target_chars if target_chars is not None else get_int(cfg, "chunking.target_chars", default=1500)
    min_chars_final = min_chars if min_chars is not None else get_int(cfg, "chunking.min_chars", default=300)

    max_unit_chars = get_int(cfg, "chunking.max_unit_chars", default=2000)
    hard_split_chars = get_int(cfg, "chunking.hard_split_chars", default=800)
    large_pdf_threshold = get_int(cfg, "chunking.large_pdf_page_threshold", default=300)

    overlap_enabled = bool(get_cfg_value(cfg, "chunking.overlap.enabled") or False)
    overlap_units = get_int(cfg, "chunking.overlap.units", default=2)
    overlap_min_units = get_int(cfg, "chunking.overlap.min_units", default=1)
    overlap_max_frac = float(get_cfg_value(cfg, "chunking.overlap.max_frac_of_target") or 0.2)

    # Ensure output directories exist
    ensure_parent_dir(out_path)
    ensure_parent_dir(log_path_final)
    logger = JsonlLogger(log_path_final)

    folder_index = build_policy_folder_index(policies_index_path, logger)
    policy_cache = PolicyMetaCache(policies_dir=policies_dir, folder_index=folder_index, logger=logger)
    pages_iter = iter_pages(pages_jsonl_path, logger)

    closed_keys: set[Tuple[str, str]] = set()
    bad_keys: set[Tuple[str, str]] = set()
    summary_written: set[str] = set()
    title_written: set[str] = set()

    bad_keys_path = out_path.parent / "bad_keys.jsonl"
    ensure_parent_dir(bad_keys_path)

    n_written = 0
    next_progress = 500
    try:
        with out_path.open("w", encoding="utf-8") as out_f:
            for page_ptr_id, policydocument_id, pages in iter_page_groups(pages_iter):
                key = (page_ptr_id, policydocument_id)

                if key in closed_keys:
                    logger.log(
                        {
                            "type": "ERROR_UNSORTED_INPUT",
                            "page_ptr_id": page_ptr_id,
                            "policydocument_id": policydocument_id,
                            "message": "Key reappeared after being processed; skipping key to avoid inconsistent chunks.",
                        }
                    )

                    if key not in bad_keys:
                        with bad_keys_path.open("a", encoding="utf-8") as bf:
                            bf.write(
                                json.dumps(
                                    {"page_ptr_id": page_ptr_id, "policydocument_id": policydocument_id},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        bad_keys.add(key)
                    continue

                cached = policy_cache.get(page_ptr_id)

                log_group_warnings(logger, page_ptr_id, policydocument_id, pages, large_pdf_threshold)

                if page_ptr_id not in title_written:
                    title_chunks = build_title_chunks(page_ptr_id=page_ptr_id, cached=cached)
                    for tc in title_chunks:
                        write_chunk_jsonl(out_f, tc)
                    n_written += len(title_chunks)
                    while n_written >= next_progress:
                        print(f"Wrote {next_progress} chunks...")
                        next_progress += 500
                    title_written.add(page_ptr_id)

                if page_ptr_id not in summary_written:
                    summary_chunks = build_summary_chunks(page_ptr_id=page_ptr_id, cached=cached)
                    for sc in summary_chunks:
                        write_chunk_jsonl(out_f, sc)
                    n_written += len(summary_chunks)
                    while n_written >= next_progress:
                        print(f"Wrote {next_progress} chunks...")
                        next_progress += 500
                    summary_written.add(page_ptr_id)

                written = stream_chunks_to_file(
                    out_f=out_f,
                    page_ptr_id=page_ptr_id,
                    policydocument_id=policydocument_id,
                    pages=pages,
                    cached=cached,
                    target_chars=target_chars_final,
                    min_chars=min_chars_final,
                    max_unit_chars=max_unit_chars,
                    hard_split_chars=hard_split_chars,
                    overlap_enabled=overlap_enabled,
                    overlap_units=overlap_units,
                    overlap_min_units=overlap_min_units,
                    overlap_max_frac=overlap_max_frac,
                    logger=logger,
                )

                fallback_info: Optional[Dict[str, Any]] = None
                if pages and written == 0:
                    relaxed_min = min(min_chars_final, 150)
                    fallback_info = {
                        "min_chars_original": min_chars_final,
                        "min_chars_relaxed": relaxed_min,
                    }

                    written += stream_chunks_to_file(
                        out_f=out_f,
                        page_ptr_id=page_ptr_id,
                        policydocument_id=policydocument_id,
                        pages=pages,
                        cached=cached,
                        target_chars=target_chars_final,
                        min_chars=relaxed_min,
                        max_unit_chars=max_unit_chars,
                        hard_split_chars=hard_split_chars,
                        overlap_enabled=overlap_enabled,
                        overlap_units=overlap_units,
                        overlap_min_units=overlap_min_units,
                        overlap_max_frac=overlap_max_frac,
                        logger=logger,
                    )

                if pages and written == 0:
                    group_source_path = pages[0].source_path if pages else ""
                    event = {
                        "type": "WARN_ZERO_CHUNKS",
                        "page_ptr_id": page_ptr_id,
                        "policydocument_id": policydocument_id,
                        "pages": len(pages),
                        "source_path": group_source_path,
                    }
                    if fallback_info is not None:
                        event.update(fallback_info)
                    logger.log(event)

                n_written += written
                while n_written >= next_progress:
                    print(f"Wrote {next_progress} chunks...")
                    next_progress += 500
                closed_keys.add(key)

            # Add title/summary chunks for policies that never appeared in extracted pages
            for page_ptr_id in sorted(folder_index.keys()):
                if page_ptr_id in title_written and page_ptr_id in summary_written:
                    continue
                cached = policy_cache.get(page_ptr_id)
                if page_ptr_id not in title_written:
                    title_chunks = build_title_chunks(page_ptr_id=page_ptr_id, cached=cached)
                    for tc in title_chunks:
                        write_chunk_jsonl(out_f, tc)
                    n_written += len(title_chunks)
                    while n_written >= next_progress:
                        print(f"Wrote {next_progress} chunks...")
                        next_progress += 500
                    title_written.add(page_ptr_id)
                if page_ptr_id not in summary_written:
                    summary_chunks = build_summary_chunks(page_ptr_id=page_ptr_id, cached=cached)
                    for sc in summary_chunks:
                        write_chunk_jsonl(out_f, sc)
                    n_written += len(summary_chunks)
                    while n_written >= next_progress:
                        print(f"Wrote {next_progress} chunks...")
                        next_progress += 500
                    summary_written.add(page_ptr_id)
    finally:
        logger.close()

    print("Chunking complete.")
    print(f"Config:         {config_path}")
    print(f"Input pages:    {pages_jsonl_path}")
    print(f"Output chunks:  {out_path}")
    print(f"Chunks written: {n_written}")
    print(f"Bad keys:       {bad_keys_path}")
    print(f"Log:            {log_path_final}")


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk extracted PDF pages into chunk-level JSONL.")
    p.add_argument("--config", default="config.json", help="Path to config JSON.")
    p.add_argument("--pages", default=None, help="Override: extracted_pages.jsonl (otherwise config).")
    p.add_argument("--out", default=None, help="Override: chunks.jsonl (otherwise config).")
    p.add_argument("--target_chars", type=int, default=None, help="Override: target chunk size (otherwise config).")
    p.add_argument("--min_chars", type=int, default=None, help="Override: min chunk size (otherwise config).")
    p.add_argument("--log", default=None, help="Override: chunk log path (otherwise config).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    chunk_pages_pipeline(
        config_path=Path(args.config),
        pages_path=Path(args.pages) if args.pages else None,
        out_chunks_path=Path(args.out) if args.out else None,
        target_chars=args.target_chars,
        min_chars=args.min_chars,
        log_path=Path(args.log) if args.log else None,
    )


if __name__ == "__main__":
    main()
