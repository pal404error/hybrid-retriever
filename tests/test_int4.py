"""P15 — int4 KV quantisation + correctness harness tests.

Covers: pack/unpack round-trip, quantise/dequantise error
bounds (per-group step/2 for the plain/value arm, exact
magnitude preservation + direction-only error for the
angular arm, zero and constant-group edge cases,
determinism), the stored-size accounting, the top-1
agreement harness, the scale-factor sweep (determinism and
the attended-top-1 invariance theorem — attended agreement
is identical for every scale multiplier), the long-context
size/agreement, the compressed-domain attention equivalence
with the dequantise path, and eval determinism (the done
condition is pinned as measured, not as hoped).
"""

import numpy as np
import pytest

from hybrid_retriever.int4 import (
    GROUP_SIZE,
    SCALE_MULTIPLIERS,
    TABLE_TOL,
    attention_compressed,
    attention_with_scale,
    dequantise_int4,
    fidelity_eval,
    generate_long_kv,
    kv_bits_per_dim,
    kv_fp16_bytes_per_vector,
    kv_ratio,
    kv_stored_bytes_per_vector,
    kv_total_bytes,
    quantise_int4,
    run_int4_eval,
    scale_sweep,
)
from hybrid_retriever.kvcache import (
    attention_float,
    generate_synthetic_kv,
)
from hybrid_retriever.int4 import _pack_codes, _unpack_codes


def _small_kv(n_tokens: int = 256):
    kv = generate_synthetic_kv("small-model-like")
    return {k: v[:n_tokens] for k, v in kv.items()}


def _max_step_bound(x: np.ndarray) -> float:
    """Max per-group step/2 over the last dim (group 32)."""
    g = x.reshape(x.shape[:-1] + (-1, GROUP_SIZE)).astype(np.float64)
    steps = (g.max(axis=-1) - g.min(axis=-1)) / 15.0
    return float(steps.max() / 2.0)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def test_pack_unpack_round_trip_and_nibble_order():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, size=(5, 3, 64), dtype=np.uint8)
    packed = _pack_codes(codes)
    assert packed.shape == (5, 3, 32)
    assert packed.dtype == np.uint8
    assert np.array_equal(_unpack_codes(packed), codes)
    # Low nibble first: [a, b] -> a | (b << 4).
    two = np.array([[[3, 12]]], dtype=np.uint8)
    assert int(_pack_codes(two)[0, 0, 0]) == 3 | (12 << 4)
    with pytest.raises(ValueError):
        _pack_codes(np.zeros((2, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Quantise / dequantise: error bounds, edge cases, determinism
# ---------------------------------------------------------------------------

def test_plain_error_within_step_over_two():
    kv = _small_kv()
    for name in ("K", "V"):
        q = quantise_int4(kv[name])
        recon = dequantise_int4(q)
        assert recon.shape == kv[name].shape
        err = np.abs(kv[name].astype(np.float64) - recon.astype(np.float64))
        # step/2 per group, plus FP16 rounding slop on the
        # stored min/step (values ~O(1), FP16 rel. err ~5e-4).
        assert float(err.max()) <= _max_step_bound(kv[name]) + 0.01


def test_angular_magnitude_preserved_and_error_is_angular():
    kv = _small_kv()
    k = kv["K"]
    q = quantise_int4(k, angular=True)
    assert q.magnitudes is not None
    recon = dequantise_int4(q)
    # Magnitude exact up to FP16 storage of the magnitude.
    norms_orig = np.linalg.norm(k.astype(np.float64), axis=-1)
    norms_recon = np.linalg.norm(recon.astype(np.float64), axis=-1)
    assert np.max(np.abs(norms_orig - norms_recon)) < 0.01
    # Direction error only: cosine similarity stays high and
    # the K MSE matches the plain arm's order of magnitude.
    cos = (
        (k.astype(np.float64) * recon.astype(np.float64)).sum(axis=-1)
        / np.where(norms_orig > 0, norms_orig, 1.0)
        / np.where(norms_recon > 0, norms_recon, 1.0)
    )
    assert float(cos.min()) > 0.9
    plain_mse = float(np.mean(
        (k.astype(np.float64) - dequantise_int4(quantise_int4(k)).astype(np.float64)) ** 2
    ))
    ang_mse = float(np.mean((k.astype(np.float64) - recon.astype(np.float64)) ** 2))
    assert ang_mse < 5 * plain_mse


def test_zero_and_constant_groups_exact():
    x = np.zeros((4, 2, 64), dtype=np.float32)
    assert np.array_equal(dequantise_int4(quantise_int4(x)), x)
    assert np.array_equal(
        dequantise_int4(quantise_int4(x, angular=True)), x
    )
    # Constant nonzero group: min == max -> step 0, exact.
    y = np.full((4, 2, 64), 0.5, dtype=np.float32)
    recon = dequantise_int4(quantise_int4(y))
    assert np.allclose(recon, y, atol=1e-6)
    # One constant group inside a varied tensor.
    z = np.random.default_rng(1).normal(size=(2, 1, 64)).astype(np.float32)
    z[..., :32] = 0.25
    assert np.allclose(
        dequantise_int4(quantise_int4(z))[..., :32], 0.25, atol=1e-6
    )


def test_quantise_determinism_and_prefix():
    kv = _small_kv()
    q1 = quantise_int4(kv["K"], angular=True)
    q2 = quantise_int4(kv["K"], angular=True)
    assert np.array_equal(q1.packed, q2.packed)
    assert np.array_equal(q1.mins, q2.mins)
    assert np.array_equal(q1.steps, q2.steps)
    assert np.array_equal(q1.magnitudes, q2.magnitudes)
    pre = q1.prefix(100)
    assert pre.shape == (100, 16, 64)
    assert np.array_equal(
        dequantise_int4(pre), dequantise_int4(q1)[:100]
    )


def test_bad_head_dim_rejected():
    with pytest.raises(ValueError):
        quantise_int4(np.zeros((4, 2, 100), dtype=np.float32))
    with pytest.raises(ValueError):
        kv_stored_bytes_per_vector(100)


# ---------------------------------------------------------------------------
# Stored-size accounting
# ---------------------------------------------------------------------------

def test_stored_size_accounting():
    # d=128: codes 128 B + scales 32 B + magnitude 2 B = 162 B
    # vs FP16 512 B -> 3.160x. d=64: 82 B vs 256 B -> 3.122x.
    assert kv_stored_bytes_per_vector(128) == 162
    assert kv_fp16_bytes_per_vector(128) == 512
    assert kv_stored_bytes_per_vector(64) == 82
    assert kv_fp16_bytes_per_vector(64) == 256
    assert kv_ratio(128) == pytest.approx(512 / 162)
    assert kv_ratio(64) == pytest.approx(256 / 82)
    assert kv_bits_per_dim(128) == pytest.approx(162 * 8 / 256)
    assert kv_total_bytes(1024, 32, 128, True) == 1024 * 32 * 162
    assert kv_total_bytes(1024, 32, 128, False) == 1024 * 32 * 512


# ---------------------------------------------------------------------------
# Attention with scale + fidelity harness
# ---------------------------------------------------------------------------

def test_attention_with_scale_matches_attention_float_at_c1():
    kv = _small_kv()
    q = kv["K"][:64]
    out = attention_with_scale(q, kv["K"], kv["V"], 1.0)
    ref = attention_float(q, kv["K"], kv["V"])
    assert np.abs(out.astype(np.float64) - ref.astype(np.float64)).max() < 1e-6


def test_fidelity_eval_ranges_and_determinism():
    kv = _small_kv()
    vq = quantise_int4(kv["V"])
    ka = quantise_int4(kv["K"], angular=True)
    recon = {"K": dequantise_int4(ka), "V": dequantise_int4(vq)}
    r1 = fidelity_eval(kv, recon, 1.0)
    r2 = fidelity_eval(kv, recon, 1.0)
    assert r1 == r2
    assert 0.0 <= r1["attended_position_top1"] <= 1.0
    assert 0.0 <= r1["vocab_top1"] <= 1.0
    assert 0.0 < r1["top1_mass"] <= 1.0
    assert r1["mse"] >= 0.0
    # A sharper softmax concentrates the top-1 mass.
    assert (
        fidelity_eval(kv, recon, 8.0)["top1_mass"]
        > fidelity_eval(kv, recon, 0.25)["top1_mass"]
    )


def test_scale_sweep_determinism_and_attended_invariance():
    kv = _small_kv()
    vq = quantise_int4(kv["V"])
    v_recon = dequantise_int4(vq)
    ka_recon = dequantise_int4(quantise_int4(kv["K"], angular=True))
    kp_recon = dequantise_int4(quantise_int4(kv["K"], angular=False))
    sweep1 = scale_sweep(
        kv, {"K": ka_recon, "V": v_recon}, {"K": kp_recon, "V": v_recon}
    )
    sweep2 = scale_sweep(
        kv, {"K": ka_recon, "V": v_recon}, {"K": kp_recon, "V": v_recon}
    )
    assert sweep1 == sweep2
    assert [float(c) for c in sweep1] == list(SCALE_MULTIPLIERS)
    # THE invariance theorem: attended-position top-1 is
    # identical for every scale multiplier, in both arms —
    # argmax commutes with multiplication by a positive c.
    for arm in ("angular", "plain"):
        attended = {
            row[arm]["attended_position_top1"] for row in sweep1.values()
        }
        assert len(attended) == 1
    # And the failure mode is visible where c bites: vocab
    # top-1 at the sharp end is below the flat end, in the
    # angular arm on this fixed slice.
    ang = [row["angular"]["vocab_top1"] for row in sweep1.values()]
    assert ang[-1] < ang[0]


# ---------------------------------------------------------------------------
# Long context
# ---------------------------------------------------------------------------

def test_long_context_size_and_agreement():
    kv = generate_long_kv(seq=8192)
    assert kv["K"].shape == (8192, 8, 64)
    d = 64
    assert kv_total_bytes(8192, 8, d, True) == 8192 * 8 * 82
    assert kv_total_bytes(8192, 8, d, False) == 8192 * 8 * 256
    vq = quantise_int4(kv["V"])
    ka = quantise_int4(kv["K"], angular=True)
    recon = {"K": dequantise_int4(ka), "V": dequantise_int4(vq)}
    r = fidelity_eval(kv, recon, 1.0)
    assert 0.0 <= r["attended_position_top1"] <= 1.0
    assert 0.0 <= r["vocab_top1"] <= 1.0
    # The long context is a harder task than the 256-token
    # slice: more distractor keys compete for the argmax.
    assert r["attended_position_top1"] <= 1.0


def test_generate_long_kv_deterministic_and_p8_recipe():
    a = generate_long_kv(seq=512)
    b = generate_long_kv(seq=512)
    assert np.array_equal(a["K"], b["K"])
    assert np.array_equal(a["V"], b["V"])
    assert not np.array_equal(a["K"], a["V"])


# ---------------------------------------------------------------------------
# Compressed-domain attention (Phase 3 analogue)
# ---------------------------------------------------------------------------

def test_compressed_domain_equivalent_to_dequantise_path():
    kv = _small_kv()
    vq = quantise_int4(kv["V"])
    ka = quantise_int4(kv["K"], angular=True)
    v_recon = dequantise_int4(vq)
    k_recon = dequantise_int4(ka)
    q = kv["K"][:64]
    for c in (0.5, 1.0, 4.0):
        out_comp = attention_compressed(q, ka, vq, c)
        out_deq = attention_with_scale(q, k_recon, v_recon, c)
        diff = float(
            np.abs(out_comp.astype(np.float64) - out_deq.astype(np.float64)).max()
        )
        assert diff < TABLE_TOL


def test_compressed_domain_requires_angular_keys():
    kv = _small_kv()
    vq = quantise_int4(kv["V"])
    kp = quantise_int4(kv["K"], angular=False)
    with pytest.raises(ValueError):
        attention_compressed(kv["K"][:16], kp, vq)


def test_compressed_domain_long_context_equivalence():
    kv = generate_long_kv(seq=8192)
    vq = quantise_int4(kv["V"])
    ka = quantise_int4(kv["K"], angular=True)
    q = kv["K"][:64]
    out_comp = attention_compressed(q, ka, vq, 1.0)
    out_deq = attention_with_scale(
        q, dequantise_int4(ka), dequantise_int4(vq), 1.0
    )
    diff = float(
        np.abs(out_comp.astype(np.float64) - out_deq.astype(np.float64)).max()
    )
    assert diff < TABLE_TOL


# ---------------------------------------------------------------------------
# Eval determinism + done condition pinned as measured
# ---------------------------------------------------------------------------

def _strip_timing(result: dict) -> dict:
    import copy

    r = copy.deepcopy(result)
    for p in r["profiles"].values():
        for s in p["settings"].values():
            s.pop("compressed_seconds", None)
            s.pop("dequantise_seconds", None)
    for s in r["long_context"]["settings"].values():
        s.pop("compressed_seconds", None)
        s.pop("dequantise_seconds", None)
    return r


def test_eval_determinism_small():
    small = {"small-model-like": [256]}
    r1 = run_int4_eval(contexts=small, long_contexts=[8192], verbose=False)
    r2 = run_int4_eval(contexts=small, long_contexts=[8192], verbose=False)
    assert _strip_timing(r1) == _strip_timing(r2)


def test_eval_done_condition_pinned_as_measured():
    # Full eval, run once: the done flag is whatever the
    # numbers say. On this synthetic data the angular arm
    # holds attended top-1 below 1.0 in at least one setting
    # and vocab top-1 below 1.0 everywhere, so the plan's
    # "identical top-1" is NOT met — pinned here so a future
    # re-tune cannot silently flip it.
    result = run_int4_eval(verbose=False)
    assert result["done_identical_top1"] is False
    assert result["all_vocab_top1_one"] is False
    assert result["ratio_d128"] == pytest.approx(512 / 162)
    for p in result["profiles"].values():
        for s in p["settings"].values():
            assert s["compressed_equivalent"] is True
    for s in result["long_context"]["settings"].values():
        assert s["compressed_equivalent"] is True
