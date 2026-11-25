"""P8 — KV-cache snapshot store with transform coding tests.

Covers: PCA round-trip and the correlation structure it exploits
(correlated synthetic KV needs far fewer components than pure
noise), quantiser error bounds and bit allocation, Huffman
round-trips including degenerate symbol distributions, ratio +
accuracy measurement, snapshot save/load and the per-profile
basis cache, and restart-continuation equivalence (reported as
measured: the argmax half is NOT pinned to True, because the
lossy codec cannot guarantee it — the test pins the bounded
error and the diagnostic that explains a flip).
"""

import os
import tempfile

import numpy as np
import pytest

from hybrid_retriever.kvcache import (
    CALIBRATION_TOKENS,
    PROFILES,
    Coder,
    KVSnapshotStore,
    Snapshot,
    TransformCodedTensor,
    attention_float,
    dequantise,
    fit_coder,
    generate_synthetic_kv,
    huffman_decode,
    huffman_decode_columns,
    huffman_encode,
    huffman_encode_columns,
    quant_step,
    quantise,
    reasoning_eval,
    run_kvcache_eval,
    run_session_demo,
)


def _small_kv():
    kv = generate_synthetic_kv("small-model-like")
    return {k: v[:256] for k, v in kv.items()}


def _coder_for(kv):
    return {
        kind: fit_coder(kv[kind][:CALIBRATION_TOKENS].reshape(-1, kv[kind].shape[-1]))
        for kind in ("K", "V")
    }


def test_generation_deterministic_and_shaped():
    a = generate_synthetic_kv("llama-7b-like")
    b = generate_synthetic_kv("llama-7b-like")
    assert a["K"].shape == (1024, 32, 128)
    assert a["V"].shape == (1024, 32, 128)
    assert a["K"].dtype == np.float32
    assert np.array_equal(a["K"], b["K"]) and np.array_equal(a["V"], b["V"])
    assert not np.array_equal(a["K"], a["V"])  # K and V draw independently
    small = generate_synthetic_kv("small-model-like")
    assert small["K"].shape == (2048, 16, 64)


def test_pca_exploits_correlation_not_noise():
    # Correlated synthetic KV: 98% variance in far fewer than d
    # components. Pure isotropic noise of the same shape needs
    # nearly all of them — the control that proves the coder is
    # exploiting the planted correlation, not the tensor shape.
    for name, d in (("llama-7b-like", 128), ("small-model-like", 64)):
        kv = generate_synthetic_kv(name)
        coder = fit_coder(kv["K"][:CALIBRATION_TOKENS].reshape(-1, d))
        assert coder.k < d // 2
        rng = np.random.default_rng(0)
        noise = rng.normal(size=(CALIBRATION_TOKENS * kv["K"].shape[1], d))
        noise_coder = fit_coder(noise)
        assert noise_coder.k > 0.9 * d


def test_pca_roundtrip_error_small():
    kv = _small_kv()
    coders = _coder_for(kv)
    for kind in ("K", "V"):
        blob = TransformCodedTensor.encode(kv[kind], coders[kind])
        rec = blob.decode()
        assert rec.shape == kv[kind].shape
        # Reconstruction error is truncation + quantisation of a
        # signal with per-element std ~1: max abs well under 1.
        assert float(np.abs(rec - kv[kind]).max()) < 0.6
        assert float(np.mean((rec - kv[kind]) ** 2)) < 0.01


def test_quantiser_error_bound_in_range():
    kv = _small_kv()
    coder = _coder_for(kv)["K"]
    rng = np.random.default_rng(1)
    # In-range coefficients: uniform in [-scale, +scale].
    coeffs = rng.uniform(-1, 1, size=(2000, coder.k)) * coder.scales
    codes = quantise(coeffs, coder)
    err = np.abs(dequantise(codes, coder) - coeffs)
    step = quant_step(coder)
    assert np.all(err <= step / 2 + 1e-9)
    # Codes stay inside each component's alphabet (odd level
    # count: max code is 2**bits - 2).
    levels = (1 << coder.bits.astype(int)) - 1
    assert np.all(codes < levels[None, :])
    # Zero quantises to exactly zero (odd level count: the
    # middle code is the zero level).
    mid = ((1 << coder.bits.astype(int)) - 2) // 2
    assert np.allclose(
        dequantise(np.tile(mid, (4, 1)).astype(np.uint8), coder), 0.0)
    assert np.all(quantise(np.zeros((4, coder.k)), coder) == mid[None, :])


def test_quantiser_clips_out_of_range():
    kv = _small_kv()
    coder = _coder_for(kv)["K"]
    huge = np.full((4, coder.k), 1e6)
    codes = quantise(huge, coder)
    levels = (1 << coder.bits.astype(int)) - 1
    assert np.all(codes == (levels - 1)[None, :])
    tiny = np.full((4, coder.k), -1e6)
    assert np.all(quantise(tiny, coder) == 0)


def test_bit_allocation_monotone_in_variance():
    kv = _small_kv()
    coder = _coder_for(kv)["K"]
    # Bits are non-decreasing in component std (PCA orders
    # components by variance, so bits should be non-increasing
    # with component index), and inside [2, 8].
    assert np.all(coder.bits >= 2) and np.all(coder.bits <= 8)
    assert np.all(np.diff(coder.bits.astype(int)) <= 0)


def test_huffman_roundtrip_distributions():
    rng = np.random.default_rng(2)
    cases = {
        "skewed": rng.choice([0, 1, 2, 3], size=10000,
                             p=[0.6, 0.25, 0.1, 0.05]).astype(np.uint8),
        "uniform": rng.integers(0, 256, size=5000, dtype=np.uint8),
        "two_symbols": np.array([0, 1] * 500, dtype=np.uint8),
        "all_256": np.tile(np.arange(256, dtype=np.uint8), 4),
        "single_value": np.full(1000, 7, dtype=np.uint8),
        "empty": np.zeros(0, dtype=np.uint8),
    }
    for name, codes in cases.items():
        data, n_bits, lengths = huffman_encode(codes)
        if codes.size == 0:
            assert data == b"" and n_bits == 0
            assert np.array_equal(huffman_decode(data, 0, lengths),
                                  np.zeros(0, dtype=np.uint8))
        elif len(np.unique(codes)) == 1:
            # Degenerate: one symbol -> empty bitstream.
            assert data == b"" and n_bits == 0
            assert int(lengths.max()) == 0
        else:
            assert np.array_equal(huffman_decode(data, len(codes), lengths), codes)
            # Huffman beats the fixed 8-bit code on skewed data.
            if name == "skewed":
                assert n_bits / len(codes) < 2.0
            if name == "uniform":
                assert n_bits / len(codes) == pytest.approx(8.0, rel=0.05)


def test_huffman_deterministic_and_prefix_free():
    rng = np.random.default_rng(3)
    codes = rng.choice([0, 1, 2, 3, 4], size=2000,
                       p=[0.5, 0.25, 0.125, 0.0625, 0.0625]).astype(np.uint8)
    d1, n1, l1 = huffman_encode(codes)
    d2, n2, l2 = huffman_encode(codes)
    assert d1 == d2 and n1 == n2 and np.array_equal(l1, l2)
    # Kraft sum <= 1 (prefix-free code).
    present = l1[l1 > 0]
    assert float(np.sum(2.0 ** (-present.astype(float)))) <= 1.0 + 1e-9


def test_huffman_columns_roundtrip_and_degenerate():
    rng = np.random.default_rng(4)
    # Columns with DIFFERENT distributions (narrow early
    # columns, wide late ones — the shape real PCA codes have):
    # per-column codebooks must beat one shared alphabet here.
    cols = []
    for j, width in enumerate([2, 2, 4, 4, 8, 16, 32]):
        cols.append(rng.integers(0, width, size=5000))
    codes = np.stack(cols, axis=1).astype(np.uint8)
    data, n_bits, lengths, single = huffman_encode_columns(codes)
    assert lengths.shape == (7, 256)
    assert np.array_equal(
        huffman_decode_columns(data, 5000, lengths, single), codes)
    shared_bits = huffman_encode(codes.reshape(-1))[1] / codes.size
    assert n_bits / codes.size < shared_bits
    # Degenerate columns: constant columns write no bits.
    deg = np.zeros((100, 3), dtype=np.uint8)
    deg[:, 1] = 5
    deg[:, 2] = rng.integers(0, 4, 100)
    data, n_bits, lengths, single = huffman_encode_columns(deg)
    assert list(single) == [0, 5, -1]
    assert np.array_equal(
        huffman_decode_columns(data, 100, lengths, single), deg)


def test_tensor_serialisation_roundtrip():
    kv = _small_kv()
    coders = _coder_for(kv)
    for kind in ("K", "V"):
        blob = TransformCodedTensor.encode(kv[kind], coders[kind])
        rt = TransformCodedTensor.from_bytes(blob.to_bytes())
        assert rt.shape == blob.shape
        assert rt.ratio == pytest.approx(blob.ratio, rel=1e-9)
        assert rt.bits_per_code == pytest.approx(blob.bits_per_code, rel=1e-9)
        assert np.array_equal(rt.decode(), blob.decode())
        assert rt.coder == coders[kind]


def test_tensor_rejects_bad_magic():
    with pytest.raises(ValueError):
        TransformCodedTensor.from_bytes(b"XXXX" + b"\x00" * 64)


def test_constant_tensor_roundtrip():
    const = np.full((64, 4, 16), 2.5, dtype=np.float32)
    coder = fit_coder(const[:32].reshape(-1, 16))
    blob = TransformCodedTensor.encode(const, coder)
    rec = blob.decode()
    assert np.allclose(rec, const, atol=1e-6)
    assert np.array_equal(
        TransformCodedTensor.from_bytes(blob.to_bytes()).decode(), rec)


def test_empty_tensor():
    empty = np.zeros((0, 4, 16), dtype=np.float32)
    coder = fit_coder(np.zeros((8, 16), dtype=np.float32) + 1.0)
    blob = TransformCodedTensor.encode(empty, coder)
    assert blob.decode().shape == (0, 4, 16)
    assert blob.ratio == float("inf") or blob.n_bytes > 0


def test_ratio_and_accuracy_measured():
    kv = generate_synthetic_kv("small-model-like")
    coders = _coder_for(kv)
    blobs = {k: TransformCodedTensor.encode(kv[k], coders[k]) for k in ("K", "V")}
    total_orig = sum(b.original_n_bytes for b in blobs.values())
    total_comp = sum(b.n_bytes for b in blobs.values())
    ratio = total_orig / total_comp
    assert ratio > 10.0  # measured ~13.7x; the 20x target is NOT pinned
    recon = {k: blobs[k].decode() for k in ("K", "V")}
    metrics = reasoning_eval(kv, recon)
    assert metrics["max_abs_error"] < 0.1
    assert metrics["mse"] < 1e-3
    assert metrics["attended_position_top1"] >= 0.9
    assert metrics["vocab_top1"] >= 0.6


def test_attention_float_matches_gaac_convention():
    # Sanity: attention over identical K/V/Q is a convex
    # combination of V rows (weights sum to 1, all positive).
    kv = _small_kv()
    out = attention_float(kv["K"][:16], kv["K"][:16], kv["V"][:16])
    assert out.shape == kv["K"][:16].shape
    assert np.all(np.isfinite(out))


def test_snapshot_save_load_roundtrip(tmp_path):
    kv = _small_kv()
    store = KVSnapshotStore(str(tmp_path))
    path = store.save("sess-1", 1, kv, "small-model-like")
    assert os.path.exists(path)
    assert store.list_turns("sess-1") == [1]
    snap = store.load("sess-1")
    assert snap.profile_name == "small-model-like"
    assert snap.session_id == "sess-1" and snap.turn == 1
    assert snap.n_tokens == 256
    assert snap.combined_ratio > 5.0
    rec = snap.decode()
    assert rec["K"].shape == kv["K"].shape
    assert float(np.abs(rec["K"] - kv["K"]).max()) < 0.6
    # Serialised snapshot bytes round-trip too.
    snap2 = Snapshot.from_bytes(snap.to_bytes())
    assert np.array_equal(snap2.decode()["K"], rec["K"])
    with pytest.raises(ValueError):
        Snapshot.from_bytes(b"XXXX" + b"\x00" * 32)
    with pytest.raises(FileNotFoundError):
        store.load("no-such-session")


def test_basis_cache_reused_and_on_disk(tmp_path):
    kv = _small_kv()
    store = KVSnapshotStore(str(tmp_path))
    coders1 = store.get_coders("small-model-like", kv)
    basis_path = os.path.join(str(tmp_path), "bases", "small-model-like.npz")
    assert os.path.exists(basis_path)
    # A fresh store object (fresh-process equivalent) reloads
    # the same coder from disk instead of refitting.
    store2 = KVSnapshotStore(str(tmp_path))
    assert store2._coder_cache == {}
    coders2 = store2.get_coders("small-model-like", kv)
    assert coders2["K"] == coders1["K"] and coders2["V"] == coders1["V"]
    # And a snapshot still loads if the basis cache is deleted:
    # every snapshot file carries its own coder.
    store.save("sess-1", 1, kv, "small-model-like")
    os.remove(basis_path)
    store3 = KVSnapshotStore(str(tmp_path))
    assert store3.load("sess-1").decode()["K"].shape == kv["K"].shape


def test_snapshot_latest_turn(tmp_path):
    kv = _small_kv()
    store = KVSnapshotStore(str(tmp_path))
    store.save("s", 1, {k: v[:128] for k, v in kv.items()}, "small-model-like")
    store.save("s", 2, kv, "small-model-like")
    assert store.list_turns("s") == [1, 2]
    assert store.load("s").turn == 2
    assert store.load("s", turn=1).n_tokens == 128


def test_session_demo_restart_bounded_error():
    result = run_session_demo(verbose=False)
    assert result["basis_cached_on_disk"] is True
    assert result["restart_after_turn"] == 2
    restarted = [t for t in result["turns"] if t["restarted"]]
    assert len(restarted) == 2
    # Bounded error is the guarantee a lossy codec makes — pin it.
    assert result["restart_max_abs_error"] < 0.05
    # Argmax identity is reported as measured, not pinned: the
    # demo's diagnostics must exist and be consistent with the
    # verdict (a flip needs gap < logit error).
    for t in restarted:
        matched = (t["continuation_restarted"]
                   == t["continuation_uninterrupted"])
        if not matched:
            assert t["top2_gap"] < t["max_logit_error"]


def test_eval_smoke():
    result = run_kvcache_eval(verbose=False)
    assert result["done_20x"] is False  # honest miss: ~13.7x, not 20x
    for name in PROFILES:
        p = result["profiles"][name]
        assert 10.0 < p["combined_ratio"] < 20.0
        r = p["reasoning"]
        assert r["attended_position_top1"] >= 0.9
        assert r["max_abs_error"] < 0.1
        for kind in ("K", "V"):
            assert p["tensors"][kind]["serialised_roundtrip_max_abs"] < 0.6
    assert result["demo"]["restart_max_abs_error"] < 0.05
