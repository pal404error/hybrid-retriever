"""P10 — AdaptiveSpec toy decoder tests.

Covers: vocab/corpus construction and determinism, the
margin rule (including the boundary), the arithmetic head,
draft-tree construction, the adaptive policy's reshaping
rules, verification correctness (margin 0 == target-only),
throughput accounting, the agreement / arithmetic-answer
metrics, and end-to-end eval determinism.
"""

import numpy as np
import pytest

from hybrid_retriever.specdec import (
    CORPUS_SEED,
    FIXED_DEPTH,
    FIXED_WIDTH,
    MARGIN,
    PROMPTS,
    PROMPT_IDS,
    TOK,
    V,
    VOCAB,
    DraftModel,
    FixedPolicy,
    TargetModel,
    TreePolicy,
    build_corpus,
    build_draft_tree,
    build_models,
    extract_answer,
    generate_speculative,
    generate_target_only,
    margin_accept,
    run_spec_eval,
    token_agreement,
    verify_tree,
)


def _models():
    return build_models()


# ---------------------------------------------------------------------------
# Vocab / corpus
# ---------------------------------------------------------------------------

def test_vocab_size_and_coverage():
    assert len(VOCAB) == 64
    assert len(set(VOCAB)) == 64
    sents = build_corpus()
    seen = {t for s in sents for t in s}
    assert set(VOCAB) <= seen


def test_corpus_deterministic():
    a = build_corpus(seed=CORPUS_SEED)
    b = build_corpus(seed=CORPUS_SEED)
    assert a == b
    assert len(a) == 6000


def test_distributions_sum_to_one_and_full_support():
    target, draft = _models()
    for ctx in ([TOK["the"]], [TOK["what"], TOK["is"]],
                [TOK["3"], TOK["+"], TOK["4"], TOK["="]]):
        for p in (target.dist(ctx), draft.dist(ctx)):
            assert abs(float(p.sum()) - 1.0) < 1e-9
            assert float(p.min()) > 0.0


# ---------------------------------------------------------------------------
# Margin rule
# ---------------------------------------------------------------------------

def test_margin_rule_accept_reject_and_boundary():
    p = np.array([0.60, 0.35, 0.05])
    # gap 0.25 > margin 0.05 -> reject; gap 0.0 -> accept.
    assert margin_accept(p, 0, MARGIN)
    assert not margin_accept(p, 1, MARGIN)
    # Boundary inclusive: gap exactly == margin -> accept.
    assert margin_accept(p, 1, 0.25)
    assert not margin_accept(p, 1, 0.249999)
    # A token 0.049 below top-1 is accepted at margin 0.05.
    q = np.array([0.50, 0.451, 0.049])
    assert margin_accept(q, 1, MARGIN)
    assert not margin_accept(q, 2, MARGIN)


def test_margin_zero_accepts_only_top1():
    p = np.array([0.20, 0.50, 0.30])
    assert margin_accept(p, 1, 0.0)
    assert not margin_accept(p, 0, 0.0)
    assert not margin_accept(p, 2, 0.0)


# ---------------------------------------------------------------------------
# Arithmetic head
# ---------------------------------------------------------------------------

def test_head_puts_mass_on_correct_answer_digit():
    target, _ = _models()
    cases = [
        (["what", "is", "3", "+", "4", "?", "the", "answer", "is"], "7"),
        (["3", "+", "4", "="], "7"),
        (["9", "*", "9", "="], "8"),
        (["compute", "4", "times", "6", ".", "the", "total", "is"], "2"),
        (["find", "the", "product", "of", "3", "and", "8", ".",
          "the", "answer", "is"], "2"),
        (["compute", "8", "minus", "2", ".", "the", "total", "is"], "6"),
    ]
    for words, expect in cases:
        ctx = [TOK[t] for t in words]
        p = target.dist(ctx)
        assert VOCAB[int(np.argmax(p))] == expect
        assert float(p[TOK[expect]]) >= 0.90


def test_head_second_digit_and_switch_off():
    target, _ = _models()
    # 9*9 = 81: after "8" the head wants "1", then switches off.
    ctx = [TOK[t] for t in ["9", "*", "9", "=", "8"]]
    assert VOCAB[int(np.argmax(target.dist(ctx)))] == "1"
    ctx = [TOK[t] for t in ["9", "*", "9", "=", "8", "1"]]
    p = target.dist(ctx)
    # Answer complete: no 0.90-mass digit any more.
    assert float(p.max()) < 0.90


def test_target_only_answers_all_arithmetic_prompts():
    target, _ = _models()
    for i in range(24):
        gen = generate_target_only(target, PROMPT_IDS[i])
        assert extract_answer(PROMPT_IDS[i], gen) is not None


def test_draft_has_no_arithmetic_head():
    _, draft = _models()
    # The bigram draft cannot compute 9*9: its top-1 after
    # "=" is the globally most frequent first answer digit,
    # not the correct "8".
    ctx = [TOK[t] for t in ["9", "*", "9", "="]]
    assert VOCAB[int(np.argmax(draft.dist(ctx)))] != "8"


# ---------------------------------------------------------------------------
# Draft tree
# ---------------------------------------------------------------------------

def test_tree_shape_depth_width_and_cap():
    _, draft = _models()
    ctx = [TOK["the"]]
    nodes = build_draft_tree(draft, ctx, depth=3, width=2)
    assert len(nodes) == 1 + 2 + 4 + 8
    depths = {0: 0}
    for idx, (parent, _tok, _c) in enumerate(nodes):
        if parent >= 0:
            depths[idx] = depths[parent] + 1
    assert max(depths.values()) == 3
    nodes = build_draft_tree(draft, ctx, depth=5, width=3,
                             max_nodes=10)
    assert len(nodes) == 10
    nodes = build_draft_tree(draft, ctx, depth=2, width=1)
    assert len(nodes) == 3  # root + 1 + 1


def test_tree_contexts_extend_parents():
    _, draft = _models()
    nodes = build_draft_tree(draft, [TOK["the"]], depth=2, width=2)
    for parent, tok, ctx in nodes[1:]:
        assert ctx == nodes[parent][2] + (tok,)
        assert ctx[-1] == tok


# ---------------------------------------------------------------------------
# Adaptive policy
# ---------------------------------------------------------------------------

def test_policy_reshaping_rules():
    pol = TreePolicy()
    # init ema 0.5, high confidence -> (4, 2).
    assert pol.choose(0.9) == (4, 2)
    # Raise ema above 0.7 with perfect rounds, high conf -> (5, 3).
    for _ in range(6):
        pol.update(3, 3)
    assert pol.ema_accept >= 0.7
    assert pol.choose(0.6) == (5, 3)
    # High ema but low confidence still widens? No: needs c >= 0.5.
    assert pol.choose(0.4) == (4, 2)
    # Crash the ema -> narrow (2, 1) whatever the confidence.
    for _ in range(10):
        pol.update(0, 3)
    assert pol.ema_accept < 0.3
    assert pol.choose(0.9) == (2, 1)
    # Low confidence alone narrows.
    pol.ema_accept = 0.5
    assert pol.choose(0.1) == (2, 1)
    # Mid-range -> base (3, 2).
    pol.ema_accept = 0.45
    assert pol.choose(0.3) == (3, 2)


def test_policy_ema_update():
    pol = TreePolicy(alpha=0.25, init_accept=0.5)
    assert pol.update(1, 2) == pytest.approx(0.5)
    assert pol.ema_accept == pytest.approx(0.5)
    pol.update(2, 2)
    assert pol.ema_accept == pytest.approx(0.625)
    pol.update(0, 2)
    assert pol.ema_accept == pytest.approx(0.46875)


def test_fixed_policy_is_constant():
    pol = FixedPolicy()
    for conf in (0.0, 0.5, 1.0):
        assert pol.choose(conf) == (FIXED_DEPTH, FIXED_WIDTH)


# ---------------------------------------------------------------------------
# Verification correctness
# ---------------------------------------------------------------------------

def test_margin_zero_speculative_equals_target_only():
    target, draft = _models()
    for i in list(range(6)) + [18, 19, 24, 30]:
        ref = generate_target_only(target, PROMPT_IDS[i])
        for policy in (FixedPolicy(), TreePolicy()):
            gen, _ = generate_speculative(
                target, draft, PROMPT_IDS[i], policy, margin=0.0)
            assert gen == ref, (PROMPTS[i], gen, ref)


def test_verification_accepts_at_least_bonus_and_counts_forwards():
    target, draft = _models()
    target.forwards = 0
    gen, rounds = generate_speculative(
        target, draft, PROMPT_IDS[0], FixedPolicy(), margin=MARGIN)
    assert len(gen) == 24
    # One target forward per verification round, no more.
    assert target.forwards == rounds
    assert 1 <= rounds <= 24


def test_target_only_costs_one_forward_per_token():
    target, _ = _models()
    target.forwards = 0
    gen = generate_target_only(target, PROMPT_IDS[0])
    assert len(gen) == 24
    assert target.forwards == 24


def test_speculative_beats_one_token_per_forward():
    target, draft = _models()
    target.forwards = 0
    gen, rounds = generate_speculative(
        target, draft, PROMPT_IDS[1], FixedPolicy(), margin=MARGIN)
    assert len(gen) / target.forwards > 1.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_token_agreement_metric():
    assert token_agreement([1, 2, 3], [1, 2, 3]) == 1.0
    assert token_agreement([1, 2, 3], [1, 9, 3]) == pytest.approx(2 / 3)
    assert token_agreement([1, 2], [9, 9]) == 0.0


def test_extract_answer_forms():
    def ids(words):
        return [TOK[t] for t in words]
    assert extract_answer(ids(["3", "+", "4", "="]),
                          ids(["7", "."])) == "7"
    assert extract_answer(ids(["9", "*", "9", "="]),
                          ids(["8", "1", "."])) == "81"
    assert extract_answer(ids(["what", "is", "2", "*", "6", "?"]),
                          ids(["the", "answer", "is", "1", "2",
                               "."])) == "12"
    assert extract_answer(ids(["the", "cat", "sits"]),
                          ids(["on", "the", "book"])) is None


# ---------------------------------------------------------------------------
# End-to-end eval
# ---------------------------------------------------------------------------

def test_spec_eval_small_is_deterministic():
    r1 = run_spec_eval(repeats=1, n_prompts=8, max_new=12)
    r2 = run_spec_eval(repeats=1, n_prompts=8, max_new=12)
    for arm in ("target-only", "fixed", "adaptive"):
        a, b = r1["arms"][arm], r2["arms"][arm]
        assert a["tokens_per_target_forward"] == b["tokens_per_target_forward"]
        assert a["token_agreement"] == b["token_agreement"]
        assert a["generations"] == b["generations"]
    # The done flag depends on wall-clock tok/sec, which is
    # noisy by nature — only the deterministic columns above
    # are pinned. The flag itself is reported as measured.
    assert isinstance(r1["done_adaptive_beats_fixed_with_agreement"], bool)


def test_spec_eval_reports_all_arms_and_metrics():
    res = run_spec_eval(repeats=1, n_prompts=8, max_new=12)
    assert res["margin"] == MARGIN
    assert res["n_prompts"] == 8
    for arm in ("target-only", "fixed", "adaptive"):
        a = res["arms"][arm]
        assert a["tokens"] == 8 * 12
        assert a["target_forwards"] >= 1
        assert 0.0 <= a["token_agreement"] <= 1.0
        assert 0.0 <= a["sequence_exact_match"] <= 1.0
        assert a["arith_total"] >= 1
    assert res["arms"]["target-only"]["tokens_per_target_forward"] == 1.0
    assert res["arms"]["fixed"]["tokens_per_target_forward"] > 1.0
    assert res["arms"]["adaptive"]["tokens_per_target_forward"] > 1.0
    assert res["arms"]["adaptive"]["arith_match_rate"] >= 0.94
