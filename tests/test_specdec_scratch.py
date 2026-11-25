"""P11 — from-scratch lossless speculative decoding tests.

Covers: the hand-rolled chi^2 survival function (known
values), chi^2 homogeneity on a known-identical vs a
known-different pair, the adjusted-distribution resample
(support + empirical distribution), rejection-sampling
correctness (empirical frequencies match the target, and a
naive retry is NOT the target — the reason tree width gets
only one trial per position), the chain loop (length,
determinism, K=6 accounting, distribution match on random
contexts), the sampled tree (shape, contexts, lossless
walk), wall-clock accounting, determinism of the eval, and
the from-scratch pin (no speculative-loop import from
specdec).
"""

import numpy as np
import pytest

import hybrid_retriever.specdec_scratch as scratch
from hybrid_retriever.specdec import (
    PROMPT_IDS,
    TOK,
    V,
    VOCAB,
    build_corpus,
)
from hybrid_retriever.specdec_scratch import (
    ALPHA,
    CHI2_N,
    CHI2_SEED,
    K,
    CostModel,
    build_sampled_tree,
    chi2_homogeneity,
    chi2_sf,
    adjusted_resample,
    generate_lossless,
    rejection_sample,
    run_benchmark,
    run_chi2_eval,
    run_longgen_comparison,
    run_scratch_eval,
    sample_target_only,
    speculative_round,
    speculative_tree_round,
    verify_tree_lossless,
)


def _models():
    from hybrid_retriever.specdec import DraftModel, TargetModel
    sents = build_corpus()
    return TargetModel(sents), DraftModel(sents)


# ---------------------------------------------------------------------------
# chi^2 machinery
# ---------------------------------------------------------------------------

def test_chi2_sf_known_values():
    # 95th percentiles: chi2_1 = 3.841459, chi2_2 = 5.991465,
    # chi2_63 ~ 82.53 (p = 0.05).
    assert chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-9)
    assert chi2_sf(5.991464547107979, 2) == pytest.approx(0.05, abs=1e-9)
    assert chi2_sf(82.5, 63) == pytest.approx(0.05, abs=1e-3)
    assert chi2_sf(0.0, 5) == 1.0
    assert chi2_sf(1e6, 5) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        chi2_sf(1.0, 0)


def test_chi2_homogeneity_identical_vs_different():
    rng = np.random.default_rng(42)
    p = np.array([0.4, 0.3, 0.2, 0.1])
    a = rng.multinomial(5000, p)
    b = rng.multinomial(5000, p)
    stat, df, p_value = chi2_homogeneity(a, b)
    assert df == 3
    assert p_value >= 0.05  # same distribution: must not reject
    q = np.array([0.1, 0.2, 0.3, 0.4])
    c = rng.multinomial(5000, q)
    stat2, df2, p2 = chi2_homogeneity(a, c)
    assert df2 == 3
    assert p2 < 1e-6  # different distribution: must reject
    assert stat2 > stat
    with pytest.raises(ValueError):
        chi2_homogeneity([1, 2], [1, 2, 3])


def test_chi2_homogeneity_drops_empty_columns():
    stat, df, p_value = chi2_homogeneity(
        [100, 50, 0, 0], [90, 60, 0, 0])
    assert df == 1
    assert 0.0 <= p_value <= 1.0


# ---------------------------------------------------------------------------
# Adjusted resample + rejection sampling
# ---------------------------------------------------------------------------

def test_adjusted_resample_support_only_where_target_exceeds_draft():
    rng = np.random.default_rng(0)
    p = np.array([0.5, 0.3, 0.2, 0.0])
    q = np.array([0.2, 0.3, 0.5, 0.0])
    # norm(max(0, p-q)) puts mass only on token 0.
    for _ in range(50):
        assert adjusted_resample(p, q, rng) == 0
    # Balanced case: mass on tokens where p > q.
    p2 = np.array([0.6, 0.1, 0.2, 0.1])
    q2 = np.array([0.1, 0.4, 0.2, 0.3])
    seen = {adjusted_resample(p2, q2, rng) for _ in range(200)}
    assert seen <= {0}  # only token 0 has p > q
    # Degenerate p == q falls back to argmax, never crashes.
    assert adjusted_resample(p, p, rng) == int(np.argmax(p))


def test_adjusted_resample_empirical_distribution():
    rng = np.random.default_rng(1)
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    # norm(max(0, p-q)) = [0.3, 0, 0] / 0.3 = [1, 0, 0].
    counts = np.zeros(3)
    for _ in range(500):
        counts[adjusted_resample(p, q, rng)] += 1
    assert counts[0] == 500
    p2 = np.array([0.7, 0.2, 0.1])
    q2 = np.array([0.1, 0.4, 0.5])
    # norm(max(0, p2-q2)) = [0.6, 0, 0]/0.6 again degenerate;
    # use a non-degenerate pair instead:
    p3 = np.array([0.5, 0.4, 0.1])
    q3 = np.array([0.2, 0.3, 0.5])
    # norm([0.3, 0.1, 0]) = [0.75, 0.25, 0].
    counts = np.zeros(3)
    for _ in range(4000):
        counts[adjusted_resample(p3, q3, rng)] += 1
    freq = counts / counts.sum()
    assert freq[0] == pytest.approx(0.75, abs=0.03)
    assert freq[1] == pytest.approx(0.25, abs=0.03)
    assert freq[2] == 0.0


def test_rejection_sample_matches_target_empirically():
    rng = np.random.default_rng(0)
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    counts = np.zeros(3)
    for _ in range(20000):
        tok, _ = rejection_sample(p, q, rng)
        counts[tok] += 1
    freq = counts / counts.sum()
    for i in range(3):
        assert freq[i] == pytest.approx(p[i], abs=0.02)
    # Acceptance happens sometimes and rejection sometimes.
    rng = np.random.default_rng(0)
    oks = [rejection_sample(p, q, rng)[1] for _ in range(2000)]
    assert 0.3 < float(np.mean(oks)) < 0.9


def test_rejection_sample_degenerate_draft_equals_target():
    rng = np.random.default_rng(3)
    p = np.array([0.25, 0.25, 0.5])
    counts = np.zeros(3)
    for _ in range(6000):
        tok, ok = rejection_sample(p, p, rng)
        assert ok
        counts[tok] += 1
    freq = counts / counts.sum()
    for i in range(3):
        assert freq[i] == pytest.approx(p[i], abs=0.02)


def test_retry_after_rejection_is_not_the_target():
    """The reason a tree gets one trial per position: keep
    proposing from q until one is accepted and the accepted
    token is distributed as min(p, q)/sum(min(p, q)), NOT p.
    With p=[0.5,0.3,0.2], q=[0.2,0.3,0.5] that is
    [0.2,0.3,0.2]/0.7 = [0.286,0.429,0.286] — far from p.
    """
    rng = np.random.default_rng(0)
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    counts = np.zeros(3)
    for _ in range(20000):
        while True:
            x = int(rng.choice(3, p=q))
            if rng.random() < min(1.0, p[x] / q[x]):
                counts[x] += 1
                break
    freq = counts / counts.sum()
    expect_biased = np.array([0.2, 0.3, 0.2]) / 0.7
    for i in range(3):
        assert freq[i] == pytest.approx(expect_biased[i], abs=0.02)
        assert abs(freq[i] - p[i]) > 0.05


# ---------------------------------------------------------------------------
# Chain loop
# ---------------------------------------------------------------------------

def test_speculative_round_length_and_acceptance_bounds():
    target, draft = _models()
    cost = CostModel(work_rounds=0)
    rng = np.random.default_rng(0)
    out, n_acc = speculative_round(
        target, draft, PROMPT_IDS[0], rng, cost, k=K)
    assert 1 <= len(out) <= K + 1
    assert 0 <= n_acc <= K
    assert n_acc <= len(out)
    assert all(0 <= t < V for t in out)
    # Exactly one target forward per round (batched).
    assert cost.forwards == 1


def test_generate_lossless_length_and_determinism():
    target, draft = _models()
    cost = CostModel(work_rounds=0)
    gen1, rounds1, _, fwd1 = generate_lossless(
        target, draft, PROMPT_IDS[0], np.random.default_rng(5),
        cost, k=K, max_new=24)
    assert len(gen1) == 24
    assert 1 <= rounds1 <= 24
    assert fwd1 == rounds1
    cost2 = CostModel(work_rounds=0)
    gen2, _, _, _ = generate_lossless(
        target, draft, PROMPT_IDS[0], np.random.default_rng(5),
        cost2, k=K, max_new=24)
    assert gen1 == gen2
    # Different seed may differ (sampling), same seed never does.
    cost3 = CostModel(work_rounds=0)
    gen3, _, _, _ = generate_lossless(
        target, draft, PROMPT_IDS[0], np.random.default_rng(6),
        cost3, k=K, max_new=24)
    assert gen3 != gen1 or True  # no assertion on inequality


def test_speculative_first_token_matches_target_on_contexts():
    """Mini chi^2: 2,000 independent contexts, first token of
    a fresh speculative round vs a target-only sample."""
    target, draft = _models()
    sents = build_corpus()
    from hybrid_retriever.specdec_scratch import _random_contexts
    ctx_t = _random_contexts(sents, 2000, 777)
    ctx_s = _random_contexts(sents, 2000, 888)
    cost = CostModel(work_rounds=0)
    rt = np.random.default_rng(11)
    rs = np.random.default_rng(12)
    ca = np.zeros(V, dtype=np.int64)
    cb = np.zeros(V, dtype=np.int64)
    for ctx in ctx_t:
        ca[int(rt.choice(V, p=target.dist(ctx)))] += 1
    for ctx in ctx_s:
        out, _ = speculative_round(target, draft, ctx, rs, cost, k=K)
        cb[out[0]] += 1
    _, _, p_value = chi2_homogeneity(ca, cb)
    assert p_value >= ALPHA


def test_sample_target_only_costs_one_forward_per_token():
    target, _ = _models()
    cost = CostModel(work_rounds=0)
    out, fwd = sample_target_only(
        target, PROMPT_IDS[0], np.random.default_rng(0), cost,
        max_new=24)
    assert len(out) == 24
    assert fwd == 24
    assert cost.forwards == 24


def test_tokens_per_forward_above_one():
    target, draft = _models()
    cost = CostModel(work_rounds=0)
    _, rounds, _, fwd = generate_lossless(
        target, draft, PROMPT_IDS[1], np.random.default_rng(0),
        cost, k=K, max_new=24)
    assert fwd == rounds
    assert 24 / fwd > 1.0


# ---------------------------------------------------------------------------
# Sampled tree
# ---------------------------------------------------------------------------

def test_tree_shape_and_contexts():
    _, draft = _models()
    cost = CostModel(work_rounds=0)
    nodes = build_sampled_tree(
        draft, [TOK["the"]], np.random.default_rng(0), cost,
        depth=3, width=2, max_nodes=127)
    assert len(nodes) == 1 + 2 + 4 + 8
    for parent, tok, ctx, _qt, _q in nodes[1:]:
        assert ctx == nodes[parent][2] + (tok,)
        assert ctx[-1] == tok
    # Siblings never duplicate.
    from collections import defaultdict
    kids = defaultdict(list)
    for parent, tok, _c, _qt, _q in nodes[1:]:
        kids[parent].append(tok)
    for toks in kids.values():
        assert len(set(toks)) == len(toks)


def test_tree_cap_and_width_one_is_a_chain():
    _, draft = _models()
    cost = CostModel(work_rounds=0)
    nodes = build_sampled_tree(
        draft, [TOK["the"]], np.random.default_rng(0), cost,
        depth=5, width=3, max_nodes=10)
    assert len(nodes) == 10
    nodes = build_sampled_tree(
        draft, [TOK["the"]], np.random.default_rng(0), cost,
        depth=3, width=1, max_nodes=127)
    assert len(nodes) == 4


def test_tree_round_length_and_determinism():
    target, draft = _models()
    cost = CostModel(work_rounds=0)
    out1, acc1 = speculative_tree_round(
        target, draft, PROMPT_IDS[0], np.random.default_rng(4),
        cost, depth=K, width=2, max_nodes=127)
    assert 1 <= len(out1) <= K + 1
    assert 0 <= acc1 <= K
    assert cost.forwards == 1  # one batched forward, all nodes
    cost2 = CostModel(work_rounds=0)
    out2, acc2 = speculative_tree_round(
        target, draft, PROMPT_IDS[0], np.random.default_rng(4),
        cost2, depth=K, width=2, max_nodes=127)
    assert out1 == out2 and acc1 == acc2


def test_tree_first_token_matches_target_on_contexts():
    """Same mini chi^2 as the chain, for the tree walk."""
    target, draft = _models()
    sents = build_corpus()
    from hybrid_retriever.specdec_scratch import _random_contexts
    ctx_t = _random_contexts(sents, 2000, 777)
    ctx_s = _random_contexts(sents, 2000, 888)
    cost = CostModel(work_rounds=0)
    rt = np.random.default_rng(11)
    rs = np.random.default_rng(12)
    ca = np.zeros(V, dtype=np.int64)
    cb = np.zeros(V, dtype=np.int64)
    for ctx in ctx_t:
        ca[int(rt.choice(V, p=target.dist(ctx)))] += 1
    for ctx in ctx_s:
        out, _ = speculative_tree_round(
            target, draft, ctx, rs, cost, depth=K, width=2,
            max_nodes=127)
        cb[out[0]] += 1
    _, _, p_value = chi2_homogeneity(ca, cb)
    assert p_value >= ALPHA


def test_verify_tree_walk_follows_first_child_only():
    target, draft = _models()
    cost = CostModel(work_rounds=0)
    rng = np.random.default_rng(2)
    nodes = build_sampled_tree(
        draft, PROMPT_IDS[0], rng, cost, depth=2, width=2,
        max_nodes=127)
    dists = [target.dist(n[2]) for n in nodes]
    out, n_acc, n_ex = verify_tree_lossless(nodes, dists, rng)
    assert 1 <= len(out) <= 3
    assert n_ex <= 2
    assert n_acc <= n_ex


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def test_cost_model_counts_and_checksum():
    cost = CostModel(work_rounds=8, draft_cost_rounds=2)
    target, draft = _models()
    cost.target_batch(target, [PROMPT_IDS[0], PROMPT_IDS[1]])
    assert cost.forwards == 1
    assert cost.work_checksum != 0.0
    before = cost.work_checksum
    cost.draft_work()
    assert cost.draft_steps == 1
    assert cost.work_checksum != before
    raw = CostModel(work_rounds=0, draft_cost_rounds=0)
    raw.target_batch(target, [PROMPT_IDS[0]])
    assert raw.forwards == 1
    assert raw.work_checksum == 0.0


# ---------------------------------------------------------------------------
# From-scratch pin + eval determinism
# ---------------------------------------------------------------------------

def test_no_specdec_loop_imported():
    for name in ("generate_speculative", "verify_tree",
                 "build_draft_tree", "margin_accept",
                 "TreePolicy", "FixedPolicy"):
        assert not hasattr(scratch, name), name


def test_chi2_eval_small_is_deterministic():
    r1 = run_chi2_eval(n_tokens=500, seed=123)
    r2 = run_chi2_eval(n_tokens=500, seed=123)
    assert r1["statistic"] == r2["statistic"]
    assert r1["p_value"] == r2["p_value"]
    assert r1["df"] == r2["df"]
    assert (r1["counts_target"] == r2["counts_target"]).all()
    assert (r1["counts_spec"] == r2["counts_spec"]).all()
    assert r1["n_tokens_per_arm"] == 500
    assert isinstance(r1["pass"], bool)


def test_longgen_comparison_deterministic_and_close():
    r1 = run_longgen_comparison(n_per_prompt=50, seed=321)
    r2 = run_longgen_comparison(n_per_prompt=50, seed=321)
    assert r1["total_variation_distance"] == r2["total_variation_distance"]
    assert r1["n_tokens_per_arm"] == 40 * 50
    # Same target, same draft: long-gen unigram distance small.
    assert r1["total_variation_distance"] < 0.15


def test_benchmark_small_reports_all_configs_and_accounting():
    res = run_benchmark(repeats=1, seed=99)
    assert len(res) == 7
    for row in res:
        assert row["spec"]["tokens"] == 40 * 24
        assert row["target_only"]["tokens"] == 40 * 24
        assert row["target_only"]["tokens_per_target_forward"] == 1.0
        assert row["spec"]["tokens_per_target_forward"] > 1.0
        assert row["speedup"] > 0.0
        assert 0.0 <= row["spec"]["acceptance_rate"] <= 1.0
    names = [r["config"] for r in res]
    assert any("RAW" in n for n in names)
    assert any("tree" in n for n in names)
