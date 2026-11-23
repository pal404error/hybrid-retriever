"""Embedding providers. Local only — no API keys, no network services.

Two providers:
- ``SentenceTransformerEmbedder``: all-MiniLM-L6-v2 (384-dim), the default.
  Requires ``sentence-transformers`` (+ torch). Small, CPU-friendly.
- ``HashedEmbedder``: deterministic token-hash placeholder. NOT semantic —
  it exists so the store / retrieval / eval code runs without torch.
  Swap point: replace it with ``SentenceTransformerEmbedder``; the
  interface (``embed`` returning L2-normalised float32 rows) is identical.

``get_embedder`` tries MiniLM first and falls back to the hashed provider
with a warning, so the library never crashes on a machine without torch.
"""

from __future__ import annotations

import hashlib
import re
import warnings

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder:
    """Protocol: embed a list of texts -> (n, dim) float32, L2-normalised."""

    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class HashedEmbedder(Embedder):
    """Deterministic bag-of-hashed-tokens vectors.

    Each token hashes (SHA-256) to one of ``dim`` buckets; the vector is
    the count vector, L2-normalised. Lexical overlap -> high cosine.
    This is a *placeholder*, not an embedding model: it cannot do
    paraphrase matching, which is exactly what the vector leg is for.
    Every eval report must say which embedder was used.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashed-{dim}"

    def _token_bucket(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                out[i, self._token_bucket(token)] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (out / norms).astype(np.float32)


class SentenceTransformerEmbedder(Embedder):
    """all-MiniLM-L6-v2 via sentence-transformers (local, CPU)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st-{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class OnnxEmbedder(Embedder):
    """all-MiniLM-L6-v2 via ONNX Runtime (no torch needed).

    Loads ``model.onnx`` + ``tokenizer.json`` from ``model_dir``
    (Xenova/all-MiniLM-L6-v2 layout). Mean-pools the last hidden state
    with the attention mask, then L2-normalises — the same pooling
    sentence-transformers uses for this model.
    """

    def __init__(self, model_dir: str) -> None:
        import os

        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(
            os.path.join(model_dir, "tokenizer.json")
        )
        self._tokenizer.enable_truncation(max_length=256)
        self._tokenizer.enable_padding(length=256)
        self._session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.dim = 384
        self.name = "onnx-all-MiniLM-L6-v2"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            enc = self._tokenizer.encode_batch(batch)
            input_ids = np.array([e.ids for e in enc], dtype=np.int64)
            attn = np.array([e.attention_mask for e in enc], dtype=np.int64)
            token_type = np.array(
                [e.type_ids for e in enc], dtype=np.int64
            )
            last_hidden = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attn,
                    "token_type_ids": token_type,
                },
            )[0]
            mask = attn[..., None].astype(np.float32)
            summed = (last_hidden * mask).sum(axis=1)
            counts = mask.sum(axis=1)
            counts[counts == 0] = 1.0
            mean_pooled = summed / counts
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            out.append((mean_pooled / norms).astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)


def get_embedder(prefer: str = "minilm") -> Embedder:
    """Return the best available embedder.

    Order: sentence-transformers (MiniLM) -> ONNX Runtime (same model,
    no torch) -> HashedEmbedder placeholder. Each fallback warns loudly.
    ``prefer="hashed"`` skips straight to the placeholder.
    """
    import os

    if prefer == "minilm":
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # noqa: BLE001 - fall through to ONNX
            warnings.warn(
                "sentence-transformers unavailable (%s); trying ONNX." % exc,
                stacklevel=2,
            )
        onnx_dir = os.path.join(
            os.path.dirname(__file__), "..", "models", "all-minilm-l6-v2-onnx"
        )
        if os.path.exists(os.path.join(onnx_dir, "model.onnx")):
            try:
                return OnnxEmbedder(onnx_dir)
            except Exception as exc:  # noqa: BLE001 - fall through to hashed
                warnings.warn(
                    "ONNX embedder failed (%s); using HashedEmbedder." % exc,
                    stacklevel=2,
                )
    return HashedEmbedder()
