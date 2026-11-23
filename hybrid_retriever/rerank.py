"""Optional cross-encoder rerank. Lazy import, never a hard dependency.

Two implementations, same interface (``rerank(query, ids, texts)``):
- ``OnnxReranker``: ms-marco-MiniLM-L-6-v2 as ONNX (Xenova layout).
  Preferred — no torch needed. Loads ``model.onnx`` + ``tokenizer.json``
  from the model dir.
- ``CrossEncoderReranker``: the same model via sentence-transformers.
  Needs torch; kept as the fallback.

If neither is available, constructing either raises a clear error.
The retriever works fine without reranking.
"""

from __future__ import annotations

import os

_DEFAULT_ONNX_DIR = os.path.join(
    os.path.dirname(__file__), "..", "models", "ms-marco-minilm-l6-v2-onnx"
)


class OnnxReranker:
    """Cross-encoder rerank via ONNX Runtime."""

    def __init__(self, model_dir: str = _DEFAULT_ONNX_DIR) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(
            os.path.join(model_dir, "tokenizer.json")
        )
        self._tokenizer.enable_truncation(max_length=512)
        self._tokenizer.enable_padding()  # pad to longest in batch
        self._session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.name = "onnx-cross-encoder-ms-marco-MiniLM-L-6-v2"

    def rerank(
        self, query: str, ids: list[int], texts: list[str]
    ) -> list[int]:
        import numpy as np

        enc = self._tokenizer.encode_batch([(query, t) for t in texts])
        input_ids = np.array([e.ids for e in enc], dtype=np.int64)
        attn = np.array([e.attention_mask for e in enc], dtype=np.int64)
        token_type = np.array([e.type_ids for e in enc], dtype=np.int64)
        logits = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attn,
                "token_type_ids": token_type,
            },
        )[0]
        scores = np.asarray(logits, dtype=np.float64).reshape(len(ids))
        order = sorted(range(len(ids)), key=lambda i: -float(scores[i]))
        return [ids[i] for i in order]


def get_reranker():
    """Best available reranker: ONNX first, then sentence-transformers."""
    import os

    onnx_model = os.path.join(_DEFAULT_ONNX_DIR, "model.onnx")
    if os.path.exists(onnx_model):
        try:
            return OnnxReranker()
        except Exception:
            pass
    return CrossEncoderReranker()


class CrossEncoderReranker:
    """Cross-encoder rerank over a candidate list."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "cross-encoder rerank needs sentence-transformers: "
                "pip install sentence-transformers"
            ) from exc
        try:
            self._model = CrossEncoder(model)
        except Exception as exc:  # noqa: BLE001 - surface the real cause
            raise RuntimeError(
                "could not load cross-encoder model %r (%s). "
                "Check network access for the first download, or run "
                "without reranking." % (model, exc)
            ) from exc
        self.name = f"cross-encoder-{model}"

    def rerank(
        self, query: str, ids: list[int], texts: list[str]
    ) -> list[int]:
        """Return ``ids`` re-ordered by cross-encoder score, best first."""
        pairs = [(query, t) for t in texts]
        scores = self._model.predict(pairs)
        order = sorted(range(len(ids)), key=lambda i: -float(scores[i]))
        return [ids[i] for i in order]
