"""P14 tests: Q4 quantisation, the offload runner under a RAM
cap, prefetch correctness, the LRU cache, and benchmark
determinism of the non-timing columns."""

import os

import numpy as np
import pytest

from hybrid_retriever.offload import (
    LAYER_FP32_BYTES,
    LAYER_Q4_BYTES,
    AllInRamModel,
    OffloadRunner,
    build_toy_model,
    build_weights_in_memory,
    dequantise_q4,
    q4_error_stats,
    q4_size_bytes,
    quantise_q4,
    run_offload_eval,
)


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "model"
    build_toy_model(str(d), n_layers=8)
    return str(d)


@pytest.fixture
def model32_dir(tmp_path):
    d = tmp_path / "model32"
    build_toy_model(str(d), n_layers=32)
    return str(d)


# ---------------------------------------------------------------------------
# Q4 quantisation
# ---------------------------------------------------------------------------

def test_q4_round_trip_error_within_rounding_bound():
    rng = np.random.default_rng(7)
    w = rng.normal(0.0, 0.1, (128, 128)).astype(np.float32)
    packed, scales, shape = quantise_q4(w)
    dq = dequantise_q4(packed, scales, shape)
    assert dq.shape == w.shape
    assert dq.dtype == np.float32
    err = np.abs(dq - w)
    # Per-element error is at most scale/2 (rounding), so the
    # global max error is at most max(scale)/2.
    assert float(err.max()) <= float(scales.max()) / 2.0 + 1e-9
    assert float(err.mean()) < float(scales.max()) / 4.0


def test_q4_zero_stays_zero_and_deterministic():
    w = np.zeros((64, 64), dtype=np.float32)
    w[3, 5] = 0.5
    packed, scales, shape = quantise_q4(w)
    dq = dequantise_q4(packed, scales, shape)
    assert float(dq[0, 0]) == 0.0
    assert float(dq[10, 10]) == 0.0
    packed2, scales2, _ = quantise_q4(w)
    assert np.array_equal(packed, packed2)
    assert np.array_equal(scales, scales2)


def test_q4_size_reduction_about_8x():
    n = 128 * 128
    assert q4_size_bytes(n) == n // 2 + (n // 32) * 4
    assert q4_size_bytes(n) * 6 < n * 4  # > 6x smaller than FP32
    stats = q4_error_stats(n_layers=32)
    assert stats["bound_holds"] is True
    assert stats["size_ratio_fp32_over_q4"] > 6.0
    assert stats["max_abs_weight_error"] > 0.0  # lossy, not exact


def test_q4_odd_length_and_non_multiple_of_group():
    w = np.arange(100, dtype=np.float32) / 100.0
    packed, scales, shape = quantise_q4(w)
    dq = dequantise_q4(packed, scales, shape)
    assert dq.shape == (100,)
    assert float(np.abs(dq - w).max()) <= float(scales.max()) / 2 + 1e-9


# ---------------------------------------------------------------------------
# Offload runner: output match + RAM cap
# ---------------------------------------------------------------------------

def test_offload_matches_all_in_ram(model_dir):
    ref = AllInRamModel(model_dir)
    tokens_ref, h_ref, _ = ref.generate(n_tokens=8)
    for use_cache in (False, True):
        for prefetch in (False, True):
            runner = OffloadRunner(
                model_dir, LAYER_FP32_BYTES,
                use_cache=use_cache, prefetch=prefetch)
            tokens, h, stats = runner.generate(n_tokens=8)
            assert tokens == tokens_ref
            assert float(np.max(np.abs(h - h_ref))) <= 1e-5
            assert stats["cap_respected"] is True
            assert stats["peak_held_bytes"] <= LAYER_FP32_BYTES


def test_model_bigger_than_cap_runs_correctly(model32_dir):
    manifest_params = 32 * (128 * 128 + 128) + 2 * 64 * 128
    assert manifest_params == 544768
    total_q4 = 32 * LAYER_Q4_BYTES
    cap = LAYER_FP32_BYTES  # one layer only
    assert total_q4 > cap  # the model does NOT fit in the cap
    ref = AllInRamModel(model32_dir)
    tokens_ref, h_ref, _ = ref.generate(n_tokens=16)
    runner = OffloadRunner(model32_dir, cap, use_cache=True,
                           prefetch=True)
    tokens, h, stats = runner.generate(n_tokens=16)
    assert tokens == tokens_ref
    assert float(np.max(np.abs(h - h_ref))) <= 1e-5
    assert stats["peak_held_bytes"] <= cap
    assert stats["peak_held_bytes"] == cap  # cache fills the cap
    assert stats["evictions"] > 0  # layers really were freed


def test_cap_smaller_than_one_layer_raises(model_dir):
    with pytest.raises(ValueError):
        OffloadRunner(model_dir, LAYER_FP32_BYTES - 1)


def test_no_cache_loads_every_layer_every_token(model_dir):
    runner = OffloadRunner(model_dir, LAYER_FP32_BYTES,
                           use_cache=False)
    _, _, stats = runner.generate(n_tokens=4)
    # 8 layers x 4 tokens, no caching: one disk load each.
    assert stats["loads"] == 8 * 4
    assert stats["cache_hits"] == 0
    assert stats["peak_held_bytes"] == 0


def test_all_in_ram_holds_full_stack(model32_dir):
    ref = AllInRamModel(model32_dir)
    assert ref.held_bytes == 32 * LAYER_FP32_BYTES
    assert ref.held_bytes > LAYER_FP32_BYTES


# ---------------------------------------------------------------------------
# Prefetch: same outputs, real overlap accounting
# ---------------------------------------------------------------------------

def test_prefetch_same_outputs_as_no_prefetch(model32_dir):
    base = OffloadRunner(model32_dir, LAYER_FP32_BYTES,
                         prefetch=False)
    tokens_base, _, _ = base.generate(n_tokens=16)
    pre = OffloadRunner(model32_dir, LAYER_FP32_BYTES,
                        prefetch=True)
    tokens_pre, _, stats = pre.generate(n_tokens=16)
    assert tokens_pre == tokens_base
    # Prefetch ran (some layers arrived via the worker thread).
    assert stats["prefetch_hits"] + stats["prefetch_misses"] > 0


# ---------------------------------------------------------------------------
# LRU cache eviction behaviour
# ---------------------------------------------------------------------------

def test_lru_cache_eviction_order(model_dir):
    # 2-layer cap over an 8-layer model: walking the stack
    # forward evicts the oldest layer each time.
    cap = 2 * LAYER_FP32_BYTES
    runner = OffloadRunner(model_dir, cap, use_cache=True)
    runner.generate(n_tokens=1)
    # After one forward pass the cache holds the last 2 layers.
    assert set(runner.cache.keys()) == {6, 7}
    assert runner.cache_bytes == cap
    assert runner.evictions == 6
    # A second token re-walks the stack: every layer is a miss
    # again (nothing survives a full pass at this cap) except
    # none — the first layer of the new pass was evicted last.
    hits_before = runner.cache_hits
    runner.generate(n_tokens=1)
    assert runner.cache_hits == hits_before
    assert runner.peak_held_bytes <= cap


def test_lru_cache_hit_when_model_fits_cap(model_dir):
    # 8-layer model, 8-layer cap: second token is all hits.
    cap = 8 * LAYER_FP32_BYTES
    runner = OffloadRunner(model_dir, cap, use_cache=True)
    runner.generate(n_tokens=1)
    assert runner.loads == 8
    assert runner.cache_hits == 0
    runner.generate(n_tokens=1)
    assert runner.loads == 8  # no further disk loads
    assert runner.cache_hits == 8
    assert runner.evictions == 0


def test_lru_reuse_within_a_token_skips_reload(model_dir):
    # 4-layer cap: layers 0-3 load, then 4 evicts 0, 5 evicts
    # 1... — a second forward pass reloads everything, but a
    # direct re-request of the most recent layer hits.
    cap = 4 * LAYER_FP32_BYTES
    runner = OffloadRunner(model_dir, cap, use_cache=True)
    runner.generate(n_tokens=1)
    assert set(runner.cache.keys()) == {4, 5, 6, 7}
    loads = runner.loads
    runner._get_layer(7)
    assert runner.loads == loads  # most recent layer: hit
    runner._get_layer(0)
    assert runner.loads == loads + 1  # evicted layer: reload


# ---------------------------------------------------------------------------
# Benchmark table determinism (non-timing columns)
# ---------------------------------------------------------------------------

def test_benchmark_table_deterministic_non_timing_columns():
    first = run_offload_eval(n_tokens=4, repeats=1)
    second = run_offload_eval(n_tokens=4, repeats=1)
    assert len(first["benchmark_table"]) == len(second["benchmark_table"])
    for a, b in zip(first["benchmark_table"],
                    second["benchmark_table"]):
        for key in ("layers", "n_params", "ram_cap_bytes",
                    "ram_cap_layers", "disk", "mode", "tokens",
                    "tokens_generated", "evictions",
                    "peak_held_bytes"):
            assert a[key] == b[key], (key, a, b)
        # Loads are deterministic except under prefetch, where
        # the worker/main race decides who loads a layer.
        if a["mode"] != "offload+prefetch":
            assert a["loads"] == b["loads"]
            assert a["cache_hits"] == b["cache_hits"]
    assert first["done_condition"]["tokens_offload"] == \
        second["done_condition"]["tokens_offload"]
    assert first["q4"]["max_abs_weight_error"] == \
        second["q4"]["max_abs_weight_error"]


def test_eval_done_condition_shape():
    result = run_offload_eval(n_tokens=4, repeats=1)
    dc = result["done_condition"]
    assert dc["model_bigger_than_cap"] is True
    assert dc["outputs_match_all_in_ram"] is True
    assert dc["cap_respected"] is True
    assert result["done_model_bigger_than_cap_runs_correctly"] is True
    assert result["q4"]["bound_holds"] is True
    assert dc["peak_rss_offload_kb"] > 0
    assert dc["peak_rss_all_in_ram_kb"] > 0
    # Prefetch arm generates the same tokens as plain offload.
    assert result["prefetch"]["offload"]["tokens_generated"] == \
        result["prefetch"]["offload+prefetch"]["tokens_generated"]
    # The table covers layers x caps x disks x modes.
    assert len(result["benchmark_table"]) == 2 * 3 * 2 * 3
