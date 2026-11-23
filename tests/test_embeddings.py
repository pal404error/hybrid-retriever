"""Embedder tests: determinism, normalisation, and the semantic gap."""

import os

import numpy as np
import pytest

from hybrid_retriever.embeddings import HashedEmbedder, OnnxEmbedder

ONNX_DIR = os.path.join(
    os.path.dirname(__file__), "..", "models", "all-minilm-l6-v2-onnx"
)
HAS_ONNX = os.path.exists(os.path.join(ONNX_DIR, "model.onnx"))


def test_hashed_is_deterministic_and_normalised():
    e = HashedEmbedder()
    texts = ["Maya feeds her sourdough starter.", "Dev rides his bike."]
    a = e.embed(texts)
    b = e.embed(texts)
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    # Lexical overlap beats no overlap, but a paraphrase with no shared
    # content words scores ~0: this is the placeholder's known limit.
    q = e.embed(["sourdough starter"])[0]
    assert float(a[0] @ q) > float(a[1] @ q)


@pytest.mark.skipif(not HAS_ONNX, reason="ONNX model not downloaded")
def test_onnx_paraphrase_beats_unrelated():
    e = OnnxEmbedder(ONNX_DIR)
    assert e.dim == 384
    texts = [
        "Maya feeds her sourdough starter every morning.",
        "How does Maya maintain her bread culture?",
        "Dev rides his bike along the river path.",
    ]
    v = e.embed(texts)
    assert v.shape == (3, 384)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)
    para = float(v[0] @ v[1])
    unrelated = float(v[0] @ v[2])
    assert para > unrelated + 0.3  # real semantic signal, not lexical
