#!/usr/bin/env python3
"""
scripts/common/json_sanitize.py

Purpose
-------
Utilities to sanitize Python / pandas / numpy objects so they can be safely serialized
to strict JSON (no NaN/Infinity tokens). Intended for JSONL pipelines.

Policy
------
- Convert pandas/numpy missing values (NaN, pd.NA, NaT, etc.) -> None
- Convert string "nan" (case-insensitive, stripped) -> None
- Do NOT convert empty strings "" -> None
- Recurse through dict/list/tuple/set
- Convert numpy scalar types to native Python types
- Convert datetime/date/Timestamp to ISO strings
- Decimal supported:
    * Decimal("NaN") -> None
    * finite Decimal -> float
- Sets supported:
    * set/frozenset -> sorted list (deterministic), after sanitizing elements
- Dict keys preserved exactly (sanitize values only)
- json_dumps_strict uses allow_nan=False to guarantee strict JSON output
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

# Optional deps: pandas/numpy. We handle absence gracefully.
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _is_missing(x: Any) -> bool:
    """
    True for pandas/numpy missing values:
      - np.nan, float('nan')
      - pd.NA, pd.NaT
      - pandas Timestamp NaT
    """
    # Fast path
    if x is None:
        return False

    # pandas-aware missing check (robust for scalars)
    if pd is not None:
        try:
            # pd.isna works on scalars; for non-scalars it returns array-like.
            v = pd.isna(x)
            if isinstance(v, (bool,)):
                return bool(v)
        except Exception:
            pass

    # numpy-aware missing check
    if np is not None:
        try:
            # np.isnan works for float-like scalars; will raise for others.
            if isinstance(x, (float, np.floating)):
                return bool(np.isnan(x))
        except Exception:
            pass

    # Python float NaN check without numpy
    if isinstance(x, float):
        # NaN is the only value not equal to itself.
        return x != x

    return False


def sanitize_for_json(x: Any) -> Any:
    """
    Recursively sanitize an object for strict JSON serialization.
    """
    # Missing values -> None
    if _is_missing(x):
        return None

    # Strings: harden "nan" -> None (case-insensitive)
    if isinstance(x, str):
        if x.strip().lower() == "nan":
            return None
        return x

    # Decimal: Decimal("NaN") -> None; finite Decimal -> float
    if isinstance(x, Decimal):
        try:
            if x.is_nan():
                return None
        except Exception:
            # If Decimal is in a weird state, fall back to string
            return str(x)
        return float(x)

    # Native primitives
    if isinstance(x, (bool, int)):
        return x

    if isinstance(x, float):
        # _is_missing already handled NaN
        return x

    # datetime-like -> ISO string
    if isinstance(x, (datetime, date)):
        return x.isoformat()

    if pd is not None:
        # pandas Timestamp -> ISO
        try:
            if isinstance(x, pd.Timestamp):
                # Missing handled above via _is_missing(pd.NaT) / pd.isna
                return x.isoformat()
        except Exception:
            pass

    # numpy scalar types -> native
    if np is not None:
        try:
            if isinstance(x, np.generic):
                # Missing handled above.
                if isinstance(x, np.integer):
                    return int(x)
                if isinstance(x, np.floating):
                    return float(x)
                if isinstance(x, np.bool_):
                    return bool(x)
                # fallback: best-effort
                return x.item()
        except Exception:
            pass

    # dict -> preserve keys exactly; sanitize values only
    # (json_dumps will enforce JSON key rules at serialization time)
    if isinstance(x, dict):
        return {k: sanitize_for_json(v) for k, v in x.items()}

    # set/frozenset -> deterministic sorted list (sanitize elements)
    if isinstance(x, (set, frozenset)):
        items = [sanitize_for_json(v) for v in x]
        try:
            return sorted(items)
        except TypeError:
            return sorted(items, key=lambda v: repr(v))

    # list/tuple -> list
    if isinstance(x, (list, tuple)):
        return [sanitize_for_json(v) for v in x]

    # Fallback: try to preserve if already JSON-serializable, else stringify.
    try:
        json.dumps(x, ensure_ascii=False, allow_nan=False)
        return x
    except Exception:
        return str(x)


def json_dumps_strict(x: Any, *, indent: int | None = None) -> str:
    """
    Strict JSON dump:
      - sanitizes input recursively
      - disallows NaN/Infinity in output (allow_nan=False)
      - fails fast on non-string dict keys (e.g. tuple keys)
    """
    clean = sanitize_for_json(x)

    def _check_keys(obj, path="root"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if not isinstance(k, str):
                    raise TypeError(f"Non-string dict key at {path}: {k!r}")
                _check_keys(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _check_keys(v, f"{path}[{i}]")

    _check_keys(clean)

    return json.dumps(clean, ensure_ascii=False, allow_nan=False, indent=indent)
