# tests/test_json_sanitize.py
#
# Pytest tests for scripts/common/json_sanitize.py
#
# Notes:
# - These tests assume you expose:
#     - sanitize_for_json(obj)  -> returns JSON-safe Python objects (no NaN/NaT/NA)
#     - json_dumps_strict(obj, **kwargs) -> json.dumps wrapper that never emits NaN
#
# If your module uses different function names, either:
#   (a) add thin aliases in json_sanitize.py, or
#   (b) update the imports below.
#
# Run:
#   pytest -q

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from scripts.common.json_sanitize import sanitize_for_json, json_dumps_strict


# -------------------------
# Scalar sanitization
# -------------------------

@pytest.mark.parametrize(
    "val",
    [
        np.nan,
        float("nan"),
        pd.NA,
        pd.NaT,
        pd.Timestamp("NaT"),
    ],
)
def test_nan_like_scalars_become_none(val):
    assert sanitize_for_json(val) is None


@pytest.mark.parametrize(
    "val",
    [
        "nan",
        "NaN",
        " NAN ",
        "\nNaN\t",
        "nAn",
    ],
)
def test_string_nan_variants_become_none(val):
    assert sanitize_for_json(val) is None


@pytest.mark.parametrize("val", ["", " ", "  \n", "banana", "none", "null", "NaNd"])
def test_strings_other_than_nan_are_preserved(val):
    # We deliberately do NOT collapse empty strings.
    out = sanitize_for_json(val)
    assert out == val


def test_basic_types_unchanged():
    assert sanitize_for_json(0) == 0
    assert sanitize_for_json(1.25) == 1.25
    assert sanitize_for_json(True) is True
    assert sanitize_for_json(False) is False
    assert sanitize_for_json(None) is None


# -------------------------
# Recursive container behavior
# -------------------------

def test_nested_containers_recursive_sanitization():
    obj = {
        "a": np.nan,
        "b": [1, 2, float("nan"), {"c": pd.NA, "d": "NaN"}],
        "e": {"f": pd.NaT, "g": "ok"},
        "h": ("nan", 3, pd.NA),
    }
    out = sanitize_for_json(obj)

    assert out["a"] is None
    assert out["b"][2] is None
    assert out["b"][3]["c"] is None
    assert out["b"][3]["d"] is None
    assert out["e"]["f"] is None
    assert out["e"]["g"] == "ok"

    # tuple may remain tuple or become list depending on implementation;
    # either is fine as long as elements are sanitized.
    assert list(out["h"]) == [None, 3, None]


def test_dict_keys_are_not_corrupted():
    # json.dumps will stringify non-string keys; sanitizer should not break this.
    obj = {1: np.nan, "2": "nan", (3, 4): "ok"}
    out = sanitize_for_json(obj)

    # Keys are implementation-dependent; we only require values are sanitized.
    # The "1" key may remain int or become str later in json.dumps.
    assert out[1] is None
    assert out["2"] is None
    assert out[(3, 4)] == "ok"


# -------------------------
# Datetime / Timestamp handling
# -------------------------

def test_datetime_and_date_isoformat():
    dt = datetime(2026, 2, 10, 12, 34, 56, tzinfo=timezone.utc)
    d = date(2026, 2, 10)
    ts = pd.Timestamp("2026-02-10T12:34:56Z")

    out = sanitize_for_json({"dt": dt, "d": d, "ts": ts})

    assert isinstance(out["dt"], str)
    assert isinstance(out["d"], str)
    assert isinstance(out["ts"], str)

    # avoid over-constraining exact formatting, but should be ISO-ish
    assert out["dt"].startswith("2026-02-10T12:34:56")
    assert out["d"] == "2026-02-10"
    assert out["ts"].startswith("2026-02-10")


def test_nat_timestamp_becomes_none():
    out = sanitize_for_json({"ts": pd.Timestamp("NaT")})
    assert out["ts"] is None


# -------------------------
# json_dumps_strict behavior
# -------------------------

def test_json_dumps_strict_never_emits_nan_token():
    # This is the core guarantee: output must be strict JSON (no NaN/Infinity tokens).
    obj = {"x": np.nan, "y": [1, float("nan"), "NaN"], "z": pd.NA}
    s = json_dumps_strict(obj)

    assert "NaN" not in s  # no bare NaN token, no string "NaN" either after sanitization
    assert "Infinity" not in s
    assert "-Infinity" not in s

    # And the result must parse with the standard JSON parser.
    parsed = json.loads(s)
    assert parsed["x"] is None
    assert parsed["y"][1] is None
    assert parsed["y"][2] is None
    assert parsed["z"] is None


def test_json_dumps_strict_round_trip_complex():
    obj = {
        "page_ptr_id": 93,
        "expired_date": "nan",
        "effective_date": pd.NaT,
        "attachments": [{"display_title": np.nan, "file_size": 123}],
        "flags": [True, False, None],
    }
    s = json_dumps_strict(obj, indent=2)
    back = json.loads(s)

    assert back["expired_date"] is None
    assert back["effective_date"] is None
    assert back["attachments"][0]["display_title"] is None
    assert back["attachments"][0]["file_size"] == 123
    assert back["flags"] == [True, False, None]


def test_decimal_handling_policy():
    # Some pipelines may produce Decimal (e.g., from DB adapters).
    # Decide whether to coerce to float, string, or reject.
    obj = {"a": Decimal("1.25"), "b": Decimal("NaN")}
    s = json_dumps_strict(obj)
    back = json.loads(s)
    assert back["a"] == pytest.approx(1.25)
    assert back["b"] is None

def test_set_coercion_policy():
    obj = {"tags": {"b", "a"}, "x": np.nan}
    s = json_dumps_strict(obj)
    back = json.loads(s)
    assert back["tags"] == ["a", "b"]
    assert back["x"] is None
