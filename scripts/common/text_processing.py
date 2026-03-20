# Shared text processing utilities for IPTP search.
# Used by BM25 now, and can be reused later for hybrid search / analytics.
#
# Updated to use scripts/common/config.py (single source of truth):
# - paths.stopwords_en_txt
# - text_processing.use_stopwords
# - text_processing.min_token_len
#
# Design note:
# This module must remain safe to import even if config.json is missing.
# Call init_text_processing_from_config() once at program startup to apply config.

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from scripts.common.config import load_config, get_path, get_int, get_bool  # type: ignore

# -------------------------
# Defaults
# -------------------------

# Default stopwords location (human-editable). Used if config does not specify a path.
DEFAULT_STOPWORDS_PATH = Path("data/text/stopwords_en.txt")

# Default behavior if config is missing or init() is never called.
DEFAULT_USE_STOPWORDS = True
DEFAULT_MIN_TOKEN_LEN = 2

# Simple, dependency-free tokenization for BM25-style indexing.
RE_TOKEN = re.compile(r"[a-z0-9]+")

# Common whitespace runs (tabs/newlines/multiple spaces).
RE_WS_RUN = re.compile(r"\s+")

# Fix hyphenation across line breaks: "immi-\ngration" -> "immigration"
RE_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")

_DEFAULT_CONFIG_PATH: Path = Path("config.json")

# -------------------------
# Config
# -------------------------

@dataclass(frozen=True)
class TextProcessingConfig:
    stopwords_path: Path = DEFAULT_STOPWORDS_PATH
    use_stopwords: bool = DEFAULT_USE_STOPWORDS
    min_token_len: int = DEFAULT_MIN_TOKEN_LEN


def load_text_processing_config(config_path: Path = _DEFAULT_CONFIG_PATH) -> TextProcessingConfig:
    """
    Reads config.json using scripts/common/config.py.

    Expected keys (per your config.json):
      - paths.stopwords_en_txt
      - text_processing.use_stopwords
      - text_processing.min_token_len

    Relative paths are resolved relative to the config file location.

    If config.json is missing, returns defaults (MVP-friendly).
    """
    if not config_path.exists():
        return TextProcessingConfig()

    cfg = load_config(config_path)
    base_dir = config_path.parent.resolve()

    stopwords_path = get_path(
        cfg,
        "paths.stopwords_en_txt",
        default=str(DEFAULT_STOPWORDS_PATH),
        base_dir=base_dir,
    )

    use_stopwords = get_bool(cfg, "text_processing.use_stopwords", default=DEFAULT_USE_STOPWORDS)

    min_token_len = get_int(cfg, "text_processing.min_token_len", default=DEFAULT_MIN_TOKEN_LEN)

    return TextProcessingConfig(
        stopwords_path=stopwords_path,
        use_stopwords=use_stopwords,
        min_token_len=min_token_len,
    )


# -------------------------
# Stopwords
# -------------------------

def load_stopwords(path: Path = DEFAULT_STOPWORDS_PATH) -> Set[str]:
    # If the file doesn't exist, return an empty set (MVP-friendly).
    if not path.exists():
        return set()

    words: Set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        words.add(line.lower())
    return words


# Minimal-change: keep STOPWORDS global, but allow BM25/others to initialize it from config.
STOPWORDS: Set[str] = load_stopwords(DEFAULT_STOPWORDS_PATH)

# Also keep configurable defaults as module globals so tokenize() can default to config behavior
# after init_text_processing_from_config() is called.
USE_STOPWORDS_DEFAULT: bool = DEFAULT_USE_STOPWORDS
MIN_TOKEN_LEN_DEFAULT: int = DEFAULT_MIN_TOKEN_LEN

_INITIALIZED: bool = False


def init_text_processing_from_config(config_path: Path = _DEFAULT_CONFIG_PATH) -> TextProcessingConfig:
    """
    Call this once at program startup (BM25 build/search) to make behavior config-driven.

    Side effects:
    - sets STOPWORDS from cfg.stopwords_path
    - sets USE_STOPWORDS_DEFAULT and MIN_TOKEN_LEN_DEFAULT from cfg

    Returns the TextProcessingConfig used (useful for debugging).
    """
    cfg_obj = load_text_processing_config(config_path)

    global STOPWORDS, USE_STOPWORDS_DEFAULT, MIN_TOKEN_LEN_DEFAULT, _INITIALIZED
    STOPWORDS = load_stopwords(cfg_obj.stopwords_path)
    USE_STOPWORDS_DEFAULT = bool(cfg_obj.use_stopwords)
    MIN_TOKEN_LEN_DEFAULT = int(cfg_obj.min_token_len)
    _INITIALIZED = True

    return cfg_obj


# -------------------------
# Text normalization / tokenization
# -------------------------

def normalize_text(text: str) -> str:
    # Guard against empty or missing text.
    if not text:
        return ""

    # Fix hyphenation across line breaks.
    text = RE_HYPHEN_LINEBREAK.sub(r"\1\2", text)

    # Normalize newlines/tabs.
    text = text.replace("\t", " ")
    text = text.replace("\r", "\n")

    # Collapse whitespace.
    text = RE_WS_RUN.sub(" ", text)

    return text.strip()


def _ensure_initialized() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    init_text_processing_from_config(_DEFAULT_CONFIG_PATH)


def tokenize(
    text: str,
    use_stopwords: Optional[bool] = None,
    *,
    min_token_len: Optional[int] = None,
) -> List[str]:
    """
    Normalize, tokenize, and optionally remove stopwords.

    Backward-compatible behavior:
    - Existing callers that pass use_stopwords=True/False still work.
    - If use_stopwords is None, we use USE_STOPWORDS_DEFAULT (config-driven after init()).
    - If min_token_len is None, we use MIN_TOKEN_LEN_DEFAULT (config-driven after init()).
    """
    _ensure_initialized()
    text = normalize_text(text).lower()
    raw = RE_TOKEN.findall(text)

    if use_stopwords is None:
        use_sw = USE_STOPWORDS_DEFAULT
    elif isinstance(use_stopwords, bool):
        use_sw = use_stopwords
    else:
        raise TypeError("use_stopwords must be None or bool")

    min_len = MIN_TOKEN_LEN_DEFAULT if min_token_len is None else int(min_token_len)

    sw = STOPWORDS if use_sw else set()
    return filter_tokens(raw, stopwords=sw, min_token_len=min_len)


def filter_tokens(tokens: List[str], stopwords: Optional[Set[str]] = None, *, min_token_len: int = DEFAULT_MIN_TOKEN_LEN) -> List[str]:
    sw = stopwords or set()

    filtered: List[str] = []
    for tok in tokens:
        if len(tok) < min_token_len:
            continue
        if tok in sw:
            continue
        filtered.append(tok)

    return filtered

