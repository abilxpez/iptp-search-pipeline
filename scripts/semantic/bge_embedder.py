from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


PoolingMode = Literal["mean", "cls"]
EmptyPolicy = Literal["replace", "error"]


@dataclass(frozen=True)
class BGEConfig:
    model_name: str = "BAAI/bge-base-en-v1.5"

    # device control
    device: Optional[str] = None  # "cuda", "mps", "cpu" (auto if None)

    # batching / truncation
    batch_size: int = 8
    max_length: int = 512

    # embedding behavior
    pooling: PoolingMode = "mean"  # "mean" is a strong default; "cls" is easy to benchmark
    normalize: bool = True  # L2 normalize so dot-product == cosine similarity

    # input hygiene
    empty_policy: EmptyPolicy = "replace"  # keep 1:1 alignment with chunk_id list
    empty_replacement: str = "."  # stable placeholder if a chunk becomes empty after strip

    # reproducibility (optional)
    seed: Optional[int] = None  # set to an int if you want stable behavior across runs

    # performance knobs (safe defaults)
    use_inference_mode: bool = True  # slightly faster than no_grad
    use_amp: bool = True  # only applies on CUDA; keeps output float32
    amp_dtype: Literal["fp16", "bf16"] = "bf16"  # bf16 is often safer than fp16 on modern GPUs

    # robustness for memory
    oom_retries: int = 4  # how many times to halve batch size on OOM

    # HF loading options (useful for deployment)
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    trust_remote_code: bool = False  # explicit for safety


class BGEEmbedder:
    def __init__(self, cfg: BGEConfig):
        self.cfg = cfg

        # pick device if not specified
        self.device = cfg.device or (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # seed is optional; embeddings are typically stable, but this helps if you benchmark
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(cfg.seed)

        # load tokenizer/model
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            trust_remote_code=cfg.trust_remote_code,
        )

        self.model = AutoModel.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            trust_remote_code=cfg.trust_remote_code,
        )

        self.model.eval()
        self.model.to(self.device)

        # hidden size lets us return a correct (0, d) array for empty input
        self.dim = int(getattr(self.model.config, "hidden_size", 0))

        # choose an AMP dtype if requested
        self._amp_dtype = None
        if self.device == "cuda" and cfg.use_amp:
            self._amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16

    def _sanitize_texts(self, texts: Sequence[str]) -> List[str]:
        out: List[str] = []
        for t in texts:
            s = (t or "").strip()
            if not s:
                if self.cfg.empty_policy == "error":
                    raise ValueError("Encountered empty/whitespace-only text after stripping.")
                s = self.cfg.empty_replacement
            out.append(s)
        return out

    def _pool(self, last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden: (B, T, H)
        # attn_mask: (B, T)
        if self.cfg.pooling == "cls":
            # CLS pooling uses the first token embedding
            # this is easy to benchmark vs mean pooling for your corpus/queries
            return last_hidden[:, 0, :]  # (B, H)

        # mean pooling over non-padding tokens using the attention mask
        # masking is required because we pad sequences within a batch
        mask = attn_mask.unsqueeze(-1).type_as(last_hidden)  # (B, T, 1)
        masked = last_hidden * mask
        summed = masked.sum(dim=1)  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        return summed / counts  # (B, H)

    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # stable L2 normalization; keeps comparisons consistent for FAISS
        return torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)

    def _iter_batches(self, texts: Sequence[str], batch_size: int) -> Iterable[Tuple[int, List[str]]]:
        for i in range(0, len(texts), batch_size):
            yield i, list(texts[i : i + batch_size])

    def _encode_batch(self, batch: List[str]) -> np.ndarray:
        # tokenize
        tok = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )

        tok = {k: v.to(self.device) for k, v in tok.items()}
        attn_mask = tok.get("attention_mask")
        if attn_mask is None:
            raise RuntimeError("Tokenizer did not return attention_mask; cannot pool safely.")

        # forward pass
        # inference_mode is a bit faster and uses less overhead than no_grad
        if self.cfg.use_inference_mode:
            ctx = torch.inference_mode()
        else:
            ctx = torch.no_grad()

        with ctx:
            # AMP for speed on CUDA; output will be converted to float32
            if self.device == "cuda" and self._amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                    out = self.model(**tok)
            else:
                out = self.model(**tok)

            last_hidden = out.last_hidden_state  # (B, T, H)
            emb = self._pool(last_hidden, attn_mask)  # (B, H)

            if self.cfg.normalize:
                emb = self._l2_normalize(emb)

        # always return float32 on CPU for FAISS compatibility
        return emb.detach().to("cpu", dtype=torch.float32).numpy()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # preserve 1:1 alignment with your chunk_id list
        clean_texts = self._sanitize_texts(texts)

        # handle empty input robustly
        if len(clean_texts) == 0:
            d = self.dim if self.dim > 0 else 0
            return np.empty((0, d), dtype=np.float32)

        all_vecs: List[np.ndarray] = []

        # dynamic batch sizing for OOM robustness
        bs = max(1, int(self.cfg.batch_size))
        retries_left = max(0, int(self.cfg.oom_retries))

        i = 0
        while i < len(clean_texts):
            batch = clean_texts[i : i + bs]
            try:
                vecs = self._encode_batch(batch)
                all_vecs.append(vecs)
                i += bs
            except RuntimeError as e:
                msg = str(e).lower()
                is_oom = ("out of memory" in msg) or ("memory" in msg and ("cuda" in msg or "mps" in msg))
                is_mps_oom = "mps" in msg and "memory" in msg
                if (is_oom or is_mps_oom) and retries_left > 0 and bs > 1:
                    # free cache and retry with smaller batch
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    retries_left -= 1
                    continue
                raise

        return np.vstack(all_vecs).astype(np.float32)


# Example usage:
# cfg = BGEConfig(pooling="mean", normalize=True, batch_size=32)
# embedder = BGEEmbedder(cfg)
# X = embedder.embed_texts(["hello world", "immigration policy memo ..."])
# X.shape == (2, d)
