"""RRF fusion + MMR unit tests (pure functions, no store needed)."""

import numpy as np

from hybrid_retriever.retrieval import mmr, rrf_fuse


def test_rrf_order_and_weights():
    # Two legs disagree; the doc ranked #1 by one leg and #2 by the other wins.
    fused = rrf_fuse([[1, 2, 3], [2, 3, 1]], weights=(0.5, 0.5), k_rrf=60)
    assert [rid for rid, _ in fused] == [2, 1, 3]
    # Scores: 2 -> 0.5/62 + 0.5/61, 1 -> 0.5/61 + 0.5/63
    assert abs(fused[0][1] - (0.5 / 62 + 0.5 / 61)) < 1e-9


def test_rrf_weight_scales_contribution():
    # Same ranks, different weights: the heavier leg decides.
    heavy_first = rrf_fuse([[1, 2], [2, 1]], weights=(0.9, 0.1), k_rrf=60)
    assert heavy_first[0][0] == 1
    heavy_second = rrf_fuse([[1, 2], [2, 1]], weights=(0.1, 0.9), k_rrf=60)
    assert heavy_second[0][0] == 2


def test_rrf_empty_leg_and_zero_weight():
    assert rrf_fuse([[1, 2], []], weights=(0.5, 0.5)) == rrf_fuse(
        [[1, 2]], weights=(0.5,)
    )
    assert [r for r, _ in rrf_fuse([[1, 2], [2, 1]], weights=(1.0, 0.0))] == [1, 2]


def test_rrf_top_n():
    fused = rrf_fuse([[1, 2, 3, 4]], weights=(1.0,), top_n=2)
    assert [r for r, _ in fused] == [1, 2]


def test_mmr_keeps_relevance_order_by_default():
    q = np.array([1.0, 0.0], dtype=np.float32)
    vecs = {
        1: np.array([0.99, 0.14], dtype=np.float32),
        2: np.array([0.80, 0.60], dtype=np.float32),
        3: np.array([0.10, 0.99], dtype=np.float32),
    }
    vecs = {k: v / np.linalg.norm(v) for k, v in vecs.items()}
    q = q / np.linalg.norm(q)
    out = mmr([(1, 1.0), (2, 0.9), (3, 0.8)], q, vecs, lambda_=0.65, top_n=3)
    assert out[0] == 1  # most query-similar first


def test_mmr_demotes_near_duplicate_with_diversity_lambda():
    q = np.array([1.0, 0.0], dtype=np.float32)
    vecs = {
        1: np.array([0.99, 0.14], dtype=np.float32),
        2: np.array([0.98, 0.20], dtype=np.float32),  # near-duplicate of 1
        3: np.array([0.10, 0.99], dtype=np.float32),  # different direction
    }
    vecs = {k: v / np.linalg.norm(v) for k, v in vecs.items()}
    out = mmr([(1, 1.0), (2, 0.95), (3, 0.5)], q, vecs, lambda_=0.3, top_n=3)
    assert out == [1, 3, 2]  # diverse doc 3 beats the duplicate 2
    assert mmr([], q, vecs) == []
