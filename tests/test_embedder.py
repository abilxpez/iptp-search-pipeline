# tests/test_embedder.py
# Run with:
#   pytest -q tests/test_embedder.py -m "not slow"
#   pytest -q tests/test_embedder.py

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from typing import List

import numpy as np
import pytest

from scripts.bge_embedder import BGEConfig, BGEEmbedder


# -----------------------------
# pytest markers
# -----------------------------
SLOW_MARK = pytest.mark.slow


# -----------------------------
# helpers
# -----------------------------
def _cos_sim_matrix(X: np.ndarray) -> np.ndarray:
    return X @ X.T


def _make_texts_small() -> List[str]:
    return [
        "USCIS eliminated the one-year foreign residency requirement for R-1 religious workers.",
        "A USCIS interim final rule removes a one-year foreign residence requirement for certain R-1 readmissions.",
        "I like pizza and surfing on weekends.",
    ]


def _stable_unit_vectors(texts: List[str], dim: int, normalize: bool) -> np.ndarray:
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.blake2b((t or "").encode("utf-8"), digest_size=16).digest()
        seed = int.from_bytes(h[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(dim,)).astype(np.float32)
        if normalize:
            n = float(np.linalg.norm(v))
            v = v / max(n, 1e-12)
        X[i] = v
    return X.astype(np.float32)


# -----------------------------
# fixtures
# -----------------------------
@pytest.fixture(scope="session")
def real_cfg() -> BGEConfig:
    return BGEConfig(
        device=None,          # auto: will pick "mps" on your M2
        pooling="mean",
        normalize=True,
        batch_size=8,         # safer for MPS
        max_length=256,       # smaller => faster tests while still meaningful
        empty_policy="replace",
        empty_replacement=".",
        use_inference_mode=True,
        oom_retries=2,
    )


@pytest.fixture(scope="session")
def real_embedder(real_cfg: BGEConfig) -> BGEEmbedder:
    return BGEEmbedder(real_cfg)


@pytest.fixture()
def dummy_embedder(monkeypatch: pytest.MonkeyPatch) -> BGEEmbedder:
    # build a BGEEmbedder instance without loading HF model/tokenizer
    cfg = BGEConfig(
        device="cpu",
        pooling="mean",
        normalize=True,
        batch_size=4,
        max_length=32,
        empty_policy="replace",
        empty_replacement=".",
        use_inference_mode=True,
        oom_retries=2,
    )

    emb = BGEEmbedder.__new__(BGEEmbedder)
    emb.cfg = cfg
    emb.device = "cpu"
    emb.dim = 768
    emb._amp_dtype = None
    emb.tokenizer = None
    emb.model = None

    def _encode_batch(batch: List[str]) -> np.ndarray:
        return _stable_unit_vectors(batch, dim=emb.dim, normalize=emb.cfg.normalize)

    monkeypatch.setattr(emb, "_encode_batch", _encode_batch)
    return emb


# ============================================================
# 1) embed returns float32 and correct shape
# ============================================================
@SLOW_MARK
def test_embed_returns_float32_and_correct_shape(real_embedder: BGEEmbedder):
    texts = _make_texts_small()[:2]
    X = real_embedder.embed_texts(texts)
    assert isinstance(X, np.ndarray)
    assert X.dtype == np.float32
    assert X.shape[0] == len(texts)
    assert X.shape[1] == real_embedder.dim
    assert X.shape[1] > 0


# ============================================================
# 2) empty input returns (0, d)
# ============================================================
def test_empty_input_returns_empty_0_by_d(dummy_embedder: BGEEmbedder):
    X = dummy_embedder.embed_texts([])
    assert X.dtype == np.float32
    assert X.shape == (0, dummy_embedder.dim)


# ============================================================
# 3) normalize=True produces ~unit norms
# ============================================================
@SLOW_MARK
def test_normalize_true_produces_unit_norms(real_embedder: BGEEmbedder):
    texts = _make_texts_small()[:2]
    X = real_embedder.embed_texts(texts)
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


# ============================================================
# 4) normalize=False not all unit norms
# ============================================================
def test_normalize_false_not_all_unit_norms(dummy_embedder: BGEEmbedder):
    dummy_embedder.cfg = replace(dummy_embedder.cfg, normalize=False)
    texts = _make_texts_small()[:2]
    X = dummy_embedder.embed_texts(texts)
    norms = np.linalg.norm(X, axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-3)


# ============================================================
# 5) pooling mean vs cls changes vectors
# ============================================================
@SLOW_MARK
def test_pooling_mode_mean_vs_cls_changes_vectors(real_cfg: BGEConfig):
    texts = _make_texts_small()[:2]

    emb_mean = BGEEmbedder(replace(real_cfg, pooling="mean", normalize=True))
    emb_cls = BGEEmbedder(replace(real_cfg, pooling="cls", normalize=True))

    X_mean = emb_mean.embed_texts(texts)
    X_cls = emb_cls.embed_texts(texts)

    assert X_mean.shape == X_cls.shape
    assert not np.allclose(X_mean, X_cls, atol=1e-6)



# ============================================================
# 6) empty_policy=replace preserves alignment
# ============================================================
def test_empty_policy_replace_preserves_alignment(dummy_embedder: BGEEmbedder):
    texts = ["", "   ", None, "hello world"]
    X = dummy_embedder.embed_texts(texts)  # type: ignore[arg-type]
    assert X.shape[0] == 4
    assert X.shape[1] == dummy_embedder.dim
    assert np.isfinite(X).all()


# ============================================================
# 7) empty_policy=error raises
# ============================================================
def test_empty_policy_error_raises(monkeypatch: pytest.MonkeyPatch, dummy_embedder: BGEEmbedder):
    dummy_embedder.cfg = replace(dummy_embedder.cfg, empty_policy="error")
    with pytest.raises(ValueError):
        _ = dummy_embedder.embed_texts(["   "])


# ============================================================
# 8) batching equivalence bs1 vs bs32
# ============================================================
def test_batching_equivalence_bs1_vs_bs32(dummy_embedder: BGEEmbedder):
    texts = _make_texts_small()

    dummy_embedder.cfg = replace(dummy_embedder.cfg, batch_size=1, normalize=True, pooling="mean")
    X1 = dummy_embedder.embed_texts(texts)

    dummy_embedder.cfg = replace(dummy_embedder.cfg, batch_size=8, normalize=True, pooling="mean")
    X2 = dummy_embedder.embed_texts(texts)

    assert X1.shape == X2.shape
    assert np.allclose(X1, X2, atol=1e-6)


# ============================================================
# 9) long text truncation runs and returns embedding
# ============================================================
@SLOW_MARK
def test_long_text_truncation_runs_and_returns_embedding(real_cfg: BGEConfig):
    long_text = "immigration policy " * 5000
    emb = BGEEmbedder(replace(real_cfg, max_length=64, normalize=True))
    X = emb.embed_texts([long_text])
    assert X.shape == (1, emb.dim)
    assert np.isfinite(X).all()


# ============================================================
# 10) output finite: no NaNs / infs
# ============================================================
@SLOW_MARK
def test_output_is_finite_no_nans_or_infs(real_embedder: BGEEmbedder):
    texts = _make_texts_small()[:2]
    X = real_embedder.embed_texts(texts)
    assert np.isfinite(X).all()


# ============================================================
# 11) OOM retry logic halves batch and succeeds (mocked)
# ============================================================
def test_oom_retry_logic_halves_batch_and_succeeds(monkeypatch: pytest.MonkeyPatch):
    cfg = BGEConfig(
        device="cpu",
        batch_size=8,
        oom_retries=3,
        normalize=True,
        empty_policy="replace",
    )
    emb = BGEEmbedder.__new__(BGEEmbedder)
    emb.cfg = cfg
    emb.device = "cpu"
    emb.dim = 32
    emb._amp_dtype = None
    emb.tokenizer = None
    emb.model = None

    def _encode_batch(batch: List[str]) -> np.ndarray:
        if len(batch) > 1:
            raise RuntimeError("MPS out of memory")
        return _stable_unit_vectors(batch, dim=emb.dim, normalize=emb.cfg.normalize)

    monkeypatch.setattr(emb, "_encode_batch", _encode_batch)

    texts = ["a", "b", "c", "d"]
    X = emb.embed_texts(texts)
    assert X.shape == (4, emb.dim)
    assert X.dtype == np.float32


# ============================================================
# 12) slow debug test: print vectors, norms, similarity matrix
# ============================================================
@SLOW_MARK
def test_debug_print_vectors_norms_similarity_matrix(real_embedder: BGEEmbedder):
    texts = _make_texts_small()
    X = real_embedder.embed_texts(texts)

    # show first few dims of each vector (so you can “see” it’s not all zeros)
    for i, t in enumerate(texts):
        print("\n---")
        print("text:", t[:80] + ("..." if len(t) > 80 else ""))
        print("first 10 dims:", np.array2string(X[i, :10], precision=5, suppress_small=False))

    # sanity: norms should be ~1.0 if normalize=True
    norms = np.linalg.norm(X, axis=1)
    print("\nL2 norms:", np.array2string(norms, precision=6))
    assert np.allclose(norms, 1.0, atol=1e-3)

    # sanity: similarity matrix (should show higher sim for related immigration texts)
    S = _cos_sim_matrix(X)  # because normalized => cosine similarity
    print("\nCosine similarity matrix:")
    print(np.array2string(S, precision=4))

    # related immigration texts are texts[0] and texts[1]; unrelated is texts[2]
    assert S[0, 1] > S[0, 2]


# -----------------------------
# CLI entrypoint
# -----------------------------
def _run_pytest(mode: str) -> int:
    import pytest as _pytest

    if mode == "fast":
        # run all tests NOT marked slow
        args = ["-q", __file__, "-m", "not slow"]
    else:
        # run everything (includes slow model tests)
        args = ["-q", __file__]
    return int(_pytest.main(args))


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fast", action="store_true", help="run fast tests only (no HF model load)")
    g.add_argument("--slow", action="store_true", help="run all tests (includes HF model + debug prints)")
    args = p.parse_args()

    mode = "slow" if args.slow or not args.fast else "fast"
    rc = _run_pytest(mode)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
