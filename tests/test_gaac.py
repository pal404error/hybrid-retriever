"""P7 — GAAC lossless KV codec tests.

Covers: bit-exact round-trips on both synthetic profiles (K and V,
plus the serialised form), the measured ratio, bit-identical
attention from original vs decompressed KV, on-the-fly attention
equivalence, row-decode correctness, and edge cases (constant
exponents, single-value groups, empty / degenerate tensors,
wide exponent ranges, channels not divisible by the group size).
"""

import numpy as np
import pytest

from hybrid_retriever.gaac import (
    PROFILES,
    CompressedKV,
    attention,
    attention_onthefly,
    compress,
    decode_rows,
    decompress,
    generate_synthetic_kv,
    run_gaac_eval,
)


def _bits_with_exponents(exponents, mantissa=0, sign=0):
    """Build a (rows, channels) BF16 bit array from exponent values."""
    exponents = np.asarray(exponents, dtype=np.uint16)
    return ((np.asarray(sign, dtype=np.uint16) << 15)
            | (exponents << 7)
            | np.asarray(mantissa, dtype=np.uint16))


def test_roundtrip_both_profiles_bit_exact():
    for name in PROFILES:
        kv = generate_synthetic_kv(name)
        for kind in ("K", "V"):
            blob = compress(kv[kind])
            assert np.array_equal(decompress(blob), kv[kind])
            assert np.array_equal(
                decompress(CompressedKV.from_bytes(blob.to_bytes())), kv[kind]
            )


def test_ratio_measured_and_above_target():
    # Profiles were fixed before measuring; the ratios below are the
    # measured numbers, pinned so a regression is visible. Profile A
    # (width 3) -> 2 / (1 + 3/8) = 1.4545, profile B (width 2) ->
    # 2 / (1 + 2/8) = 1.6, up to negligible header overhead.
    ratios = {}
    for name in PROFILES:
        kv = generate_synthetic_kv(name)
        total_orig = total_comp = 0
        for kind in ("K", "V"):
            blob = compress(kv[kind])
            total_orig += blob.original_n_bytes
            total_comp += blob.n_bytes
        ratios[name] = total_orig / total_comp
    assert ratios["llama-7b-like"] >= 1.4
    assert ratios["small-model-like"] >= 1.4
    assert ratios["llama-7b-like"] == pytest.approx(1.4545, rel=1e-3)
    assert ratios["small-model-like"] == pytest.approx(1.6, rel=1e-3)


def test_group_widths_match_profile_spread():
    kv = generate_synthetic_kv("llama-7b-like")
    blob = compress(kv["K"])
    assert set(np.unique(blob.widths)) == {3}  # deltas 0..6 need 3 bits
    kv2 = generate_synthetic_kv("small-model-like")
    blob2 = compress(kv2["K"])
    assert set(np.unique(blob2.widths)) == {2}  # deltas 0..2 need 2 bits


def test_attention_bit_identical_original_vs_decompressed():
    for name in PROFILES:
        kv = generate_synthetic_kv(name)
        n_tok = 64
        q = kv["K"][:n_tok]
        out_orig = attention(q, kv["K"][:n_tok], kv["V"][:n_tok])
        out_dec = attention(
            q,
            decompress(compress(kv["K"]))[:n_tok],
            decompress(compress(kv["V"]))[:n_tok],
        )
        assert np.array_equal(out_orig, out_dec)


def test_onthefly_attention_equivalent_and_cheaper():
    kv = generate_synthetic_kv("small-model-like")
    n_tok = 64
    q = kv["K"][:n_tok]
    k_blob = compress(kv["K"][:n_tok])
    v_blob = compress(kv["V"][:n_tok])
    out_full = attention(q, kv["K"][:n_tok], kv["V"][:n_tok])
    out_otf, stats = attention_onthefly(k_blob, v_blob, q)
    assert np.allclose(out_otf, out_full, rtol=1e-5, atol=1e-6)
    # One float32 tile vs the full decompressed + float32 caches.
    n_elem = q.size
    full_extra = 2 * (n_elem * 2 + n_elem * 4)
    assert stats["tile_peak_extra_bytes"] < full_extra


def test_decode_rows_matches_full_decompress():
    kv = generate_synthetic_kv("llama-7b-like")
    blob = compress(kv["K"])
    full = decompress(blob).reshape(-1, 128)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(full.shape[0], size=100, replace=False))
    assert np.array_equal(decode_rows(blob, rows), full[rows])
    assert decode_rows(blob, np.array([], dtype=np.int64)).shape == (0, 128)


def test_constant_exponents_width_zero():
    bits = _bits_with_exponents(np.full((1024, 32), 130, dtype=np.uint16))
    blob = compress(bits)
    assert set(np.unique(blob.widths)) == {0}
    assert len(blob.packed) == 0
    assert np.array_equal(decompress(blob), bits)
    # Only the verbatim sign+mantissa bytes remain (plus a small
    # header / per-channel base table, negligible at this size).
    assert blob.ratio == pytest.approx(2.0, rel=1e-2)


def test_single_value_group_and_single_channel():
    bits = _bits_with_exponents(np.full((4, 1), 120, dtype=np.uint16))
    assert np.array_equal(decompress(compress(bits)), bits)
    bits = _bits_with_exponents(
        np.array([[120], [121], [120], [127]], dtype=np.uint16)
    )
    blob = compress(bits)
    assert np.array_equal(decompress(blob), bits)


def test_channels_not_divisible_by_group_size():
    rng = np.random.default_rng(3)
    exponents = rng.integers(120, 130, size=(16, 37)).astype(np.uint16)
    bits = _bits_with_exponents(exponents, mantissa=rng.integers(0, 128, size=(16, 37)))
    blob = compress(bits, group_size=16)
    assert len(blob.widths) == 3  # 16 + 16 + 5 channels
    assert np.array_equal(decompress(blob), bits)
    assert np.array_equal(
        decompress(CompressedKV.from_bytes(blob.to_bytes())), bits
    )


def test_wide_exponent_range_uses_width_8():
    # One channel spanning the full exponent range: offsets 0..255
    # need the full 8 bits. (Per-channel bases mean a wide range
    # across *different* channels does not widen any one group.)
    exponents = np.array([[0], [255]], dtype=np.uint16)
    bits = _bits_with_exponents(exponents)
    blob = compress(bits, group_size=2)
    assert int(blob.widths[0]) == 8
    assert np.array_equal(decompress(blob), bits)


def test_empty_and_degenerate_tensors():
    empty = np.zeros((0, 16), dtype=np.uint16)
    blob = compress(empty)
    assert decompress(blob).shape == (0, 16)
    assert np.array_equal(
        decompress(CompressedKV.from_bytes(blob.to_bytes())), empty
    )
    single = _bits_with_exponents(np.array([[133]], dtype=np.uint16))
    assert np.array_equal(decompress(compress(single)), single)


def test_serialised_blob_rejects_bad_magic():
    with pytest.raises(ValueError):
        CompressedKV.from_bytes(b"XXXX" + b"\x00" * 32)


def test_eval_smoke():
    result = run_gaac_eval(verbose=False)
    assert result["done_1p4x_on_both"] is True
    for name in PROFILES:
        p = result["profiles"][name]
        assert p["attention_bit_identical"] is True
        assert p["onthefly_equivalent"] is True
        assert all(
            t["roundtrip_bit_exact"] and t["serialised_roundtrip_bit_exact"]
            for t in p["tensors"].values()
        )
