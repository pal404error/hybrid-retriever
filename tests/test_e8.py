"""P9 — E8 lattice key quantisation (k2v2) tests.

Covers: the codebook (size, E8 membership, even-sum check,
half-split reconstruction), the lattice decoder (exactness
vs brute force, identity on codewords, parity cases),
quantise/dequantise round-trips and error bounds (E8 vs the
scalar baselines, monotone in bits, zero handling, prefix
slicing, the counted minimal-vector fallback), the
table-based scoring equivalence with the dequantise path,
the stored-size accounting, and per-setting benchmark
determinism (the eval's done condition is pinned as
measured — False on this synthetic data — not as hoped).
"""

import numpy as np
import pytest

from hybrid_retriever.e8 import (
    MAX_NORM_SQ,
    NORM_MULTS,
    Codebook,
    build_codebook,
    dequantise_keys_e8,
    dequantise_scalar,
    fidelity_eval,
    get_codebook,
    kv_bits_per_dim,
    kv_ratio,
    nearest_e8,
    quantise_keys_e8,
    quantise_scalar,
    run_e8_eval,
    attention_table,
)
from hybrid_retriever.kvcache import (
    attention_float,
    generate_synthetic_kv,
)


def _small_kv(n_tokens: int = 256):
    kv = generate_synthetic_kv("small-model-like")
    return {k: v[:n_tokens] for k, v in kv.items()}


def _k_mse(orig: np.ndarray, recon: np.ndarray) -> float:
    return float(
        np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    )


# ---------------------------------------------------------------------------
# Codebook
# ---------------------------------------------------------------------------

def test_codebook_size_and_lattice_membership():
    cb = get_codebook()
    assert len(cb) == 56881
    pts = cb.points.astype(np.float64)
    # Every codeword is integer or half-integer in every coord.
    doubled = pts * 2
    assert np.all(np.abs(doubled - np.round(doubled)) < 1e-6)
    is_int = np.all(np.abs(pts - np.round(pts)) < 1e-6, axis=1)
    is_half = ~is_int
    # Half-integer codewords have all coords half-integer.
    assert np.all(
        np.abs(pts[is_half] * 2 - np.round(pts[is_half] * 2)) < 1e-6
    )
    assert np.all(
        np.abs(pts[is_half] - np.round(pts[is_half])) > 1e-6
    )
    # Even coordinate sum (the E8 parity condition).
    sums = np.round(pts.sum(axis=1))
    assert np.all(np.abs(pts.sum(axis=1) - sums) < 1e-6)
    assert np.all(sums % 2 == 0)
    # Radius and the zero + minimal vectors.
    norms_sq = (pts ** 2).sum(axis=1)
    assert norms_sq.max() <= MAX_NORM_SQ + 1e-6
    assert int((norms_sq == 0).sum()) == 1
    assert int((np.abs(norms_sq - 2.0) < 1e-6).sum()) == 240
    assert int((np.abs(norms_sq - 4.0) < 1e-6).sum()) == 2160


def test_codebook_deterministic_and_halves_reconstruct():
    a = build_codebook()
    b = build_codebook()
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.half1_ids, b.half1_ids)
    assert np.array_equal(a.half2_ids, b.half2_ids)
    recon = np.concatenate(
        [a.halves1[a.half1_ids], a.halves2[a.half2_ids]], axis=1
    )
    assert np.array_equal(recon, a.points)
    assert a.halves1.shape[1] == 4 and a.halves2.shape[1] == 4
    assert a.half1_ids.max() < len(a.halves1)
    assert a.half2_ids.max() < len(a.halves2)


def test_codebook_lookup_roundtrip():
    cb = get_codebook()
    rng = np.random.default_rng(0)
    sample = rng.choice(len(cb), size=500, replace=False)
    doubled = np.round(cb.points[sample].astype(np.float64) * 2).astype(np.int64)
    assert np.array_equal(cb.lookup(doubled), sample)
    # A point outside the codebook (norm^2 12) maps to -1.
    outside = np.array([[2, 2, 2, 0, 0, 0, 0, 0]], dtype=np.int64)
    assert cb.lookup(outside)[0] == -1


# ---------------------------------------------------------------------------
# Lattice decoder
# ---------------------------------------------------------------------------

def test_nearest_e8_matches_brute_force():
    cb = get_codebook()
    rng = np.random.default_rng(3)
    # Small-norm vectors, so the true nearest point is inside
    # the finite codebook and brute force over it is exact:
    # every vector is rescaled to norm <= 2.0, and the E8
    # covering radius is 1, so the nearest lattice point has
    # norm <= 3.0 < sqrt(10) and is in the codebook.
    y = rng.normal(scale=0.9, size=(200, 8))
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    y = y * np.minimum(1.0, 2.0 / norms)
    got = nearest_e8(y)
    pts = cb.points.astype(np.float64)
    d2 = ((y[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
    brute = pts[np.argmin(d2, axis=1)]
    got_d2 = ((y - got) ** 2).sum(axis=1)
    brute_d2 = ((y - brute) ** 2).sum(axis=1)
    assert np.allclose(got_d2, brute_d2, atol=1e-9)
    # Point equality is NOT asserted: the nearest lattice point
    # need not be unique (ties), so the decoder may return a
    # different equidistant codeword than argmin's first hit.


def test_nearest_e8_identity_on_codewords():
    cb = get_codebook()
    rng = np.random.default_rng(1)
    sample = cb.points[rng.choice(len(cb), size=2000, replace=False)]
    assert np.allclose(nearest_e8(sample), sample)


def test_nearest_e8_parity_cases():
    # [0.6, 0, ...]: integer rounding gives sum 1 (odd); the
    # cheapest fix drops it to the zero vector, which beats the
    # half-integer candidate ([0.5]*8, dist^2 1.76).
    assert np.allclose(
        nearest_e8(np.array([[0.6, 0, 0, 0, 0, 0, 0, 0]])),
        np.zeros((1, 8)),
    )
    # The all-halves vector is on the lattice already.
    assert np.allclose(
        nearest_e8(np.full((1, 8), 0.5)), np.full((1, 8), 0.5)
    )


# ---------------------------------------------------------------------------
# Quantise / dequantise
# ---------------------------------------------------------------------------

def test_e8_roundtrip_beats_scalar_k2_on_sqnr():
    kv = _small_kv()
    q, _ = quantise_keys_e8(kv["K"])
    e8_mse = _k_mse(kv["K"], dequantise_keys_e8(q))
    k2_mse = _k_mse(kv["K"], dequantise_scalar(quantise_scalar(kv["K"], 2)))
    k3_mse = _k_mse(kv["K"], dequantise_scalar(quantise_scalar(kv["K"], 3)))
    assert e8_mse < 0.02
    assert e8_mse < k2_mse  # lattice wins the same-rate geometry comparison
    assert k3_mse < k2_mse  # scalar error is monotone in bits
    assert k3_mse < e8_mse  # ...and the extra bit still beats the lattice here


def test_scalar_zero_bound_and_e8_zero_exact():
    x = np.zeros((4, 2, 64), dtype=np.float32)
    sq = quantise_scalar(x, 2)
    recon = dequantise_scalar(sq)
    # Even level count: no zero level, so zero reconstructs to
    # +-step/2 with step = 1.0 * first multiplier.
    assert np.abs(recon).max() <= 0.5 * float(NORM_MULTS[0]) + 1e-6
    q, _ = quantise_keys_e8(x)
    assert np.array_equal(dequantise_keys_e8(q), x)
    assert np.all(q.indices == 0) or np.all(
        get_codebook().points[q.indices.reshape(-1)] == 0
    )


def test_norm_codes_scales_and_bad_dim():
    kv = _small_kv(32)
    q, _ = quantise_keys_e8(kv["K"])
    assert q.norm_codes.min() >= 0 and q.norm_codes.max() <= 3
    assert np.all(np.isfinite(q.scales.astype(np.float64)))
    assert q.scales.shape == kv["K"].shape[:2]
    sq = quantise_scalar(kv["K"], 3)
    assert sq.codes.shape == kv["K"].shape
    assert sq.bits == 3
    with pytest.raises(ValueError):
        quantise_keys_e8(np.zeros((2, 2, 100), dtype=np.float32))
    with pytest.raises(ValueError):
        quantise_scalar(np.zeros((2, 2, 100), dtype=np.float32), 2)


def test_e8_prefix_slicing_consistent():
    kv = _small_kv()
    q, _ = quantise_keys_e8(kv["K"])
    q32, _ = quantise_keys_e8(kv["K"][:32])
    assert np.array_equal(
        dequantise_keys_e8(q.prefix(32)), dequantise_keys_e8(q32)
    )


def test_fallback_minimal_blocks_counted_and_small():
    kv = _small_kv()
    _, info = quantise_keys_e8(kv["K"])
    assert 0 <= info["fallback_minimal_blocks"] < info["total_blocks"]
    assert info["fallback_minimal_blocks"] / info["total_blocks"] < 0.05
    assert info["codebook_size"] == 56881


# ---------------------------------------------------------------------------
# Table-based scoring
# ---------------------------------------------------------------------------

def test_table_scoring_matches_dequantised_path():
    kv = _small_kv(64)
    q, _ = quantise_keys_e8(kv["K"])
    vq = quantise_scalar(kv["V"], 2)
    vr = dequantise_scalar(vq)
    out_table = attention_table(kv["K"][:16], q, vr)
    out_deq = attention_float(kv["K"][:16], dequantise_keys_e8(q), vr)
    diff = float(
        np.abs(out_table.astype(np.float64) - out_deq.astype(np.float64)).max()
    )
    assert diff < 1e-5


# ---------------------------------------------------------------------------
# Size accounting
# ---------------------------------------------------------------------------

def test_ratio_accounting():
    # E8 k2v2 and scalar k2v2 store the same number of bytes
    # (same index/code width, same metadata); scalar k3v2 stores
    # more. d=128: 76 stored bytes per (token, head) vs 512.
    assert kv_ratio(128, 2, True) == kv_ratio(128, 2, False)
    assert kv_ratio(128, 2, True) > kv_ratio(128, 3, False)
    assert kv_ratio(128, 2, True) == pytest.approx(512 / 76, rel=1e-9)
    assert kv_ratio(128, 3, False) == pytest.approx(512 / 92, rel=1e-9)
    assert kv_ratio(64, 2, True) == pytest.approx(256 / 40, rel=1e-9)
    assert kv_ratio(64, 3, False) == pytest.approx(256 / 48, rel=1e-9)
    assert kv_bits_per_dim(128, 2, True) == pytest.approx(2.375, rel=1e-9)
    assert kv_bits_per_dim(128, 3, False) == pytest.approx(2.875, rel=1e-9)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def test_fidelity_eval_context_actually_matters():
    kv = _small_kv(512)
    r64 = fidelity_eval(
        {"K": kv["K"][:64], "V": kv["V"][:64]},
        {"K": kv["K"][:64], "V": kv["V"][:64]},
    )
    r512 = fidelity_eval(kv, kv)
    assert r64["attended_position_top1"] == 1.0
    assert r512["attended_position_top1"] == 1.0
    assert r64["n_tokens"] == 64 and r512["n_tokens"] == 512


def test_eval_determinism_and_structure():
    ctx = {"small-model-like": [64, 128]}
    a = run_e8_eval(contexts=ctx, verbose=False)
    b = run_e8_eval(contexts=ctx, verbose=False)
    assert a == b
    assert a["n_settings"] == 2
    prof = a["profiles"]["small-model-like"]
    for ctx_key, s in prof["settings"].items():
        for label in ("e8-k2v2", "scalar-k2v2", "scalar-k3v2"):
            assert s[label]["ratio"] > 1.0
            assert 0.0 <= s[label]["attended_position_top1"] <= 1.0
        assert s["e8_table_equivalent"] is True
        assert s["e8_table_max_abs_diff"] < 1e-5
    wins = sum(
        1
        for s in prof["settings"].values()
        if s["e8-k2v2"]["attended_position_top1"]
        > s["scalar-k3v2"]["attended_position_top1"]
    )
    assert a["e8_wins_vs_scalar_k3v2"] == wins


def test_eval_done_condition_reported_as_measured():
    # On this synthetic data E8 does NOT beat scalar k3v2 in
    # most settings — the eval must say so, not claim the win.
    result = run_e8_eval(
        contexts={"small-model-like": [64, 128]}, verbose=False
    )
    assert result["done_beats_k3_in_most"] is False
    assert result["e8_wins_vs_scalar_k3v2"] == 0
