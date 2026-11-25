"""P12 — retrieval-based drafter for small local models tests.

Covers: the prompt-lookup matching (longest suffix wins,
history before corpus at the same length, corpus hit,
no-match fallback to uniform), retrieval drafter
determinism, overhead accounting (retrieval pays zero
simulated draft work, the model drafter pays
DRAFT_COST_ROUNDS per step), acceptance-rate measurement
(accepted / examined, reported separately from
wall-clock), tokens-per-target-forward > 1 for both
speculative arms, the wall-clock comparison structure at
both cost settings (the done flag is the comparison, not
a baked-in outcome), and the reuse pin (verification is
P11's speculative_round, not P10's speculative loop).
"""

import numpy as np
import pytest

import hybrid_retriever.retrieval_drafter as rd
from hybrid_retriever.retrieval_drafter import (
    DRAFT_COST_ROUNDS,
    MAX_NGRAM,
    PromptLookupDrafter,
    build_lookup_index,
    generate_with_drafter,
    run_retrieval_eval,
)
from hybrid_retriever.specdec import (
    PROMPT_IDS,
    TOK,
    V,
    DraftModel,
    TargetModel,
    build_corpus,
)
from hybrid_retriever.specdec_scratch import (
    CostModel,
    speculative_round,
)


def _models():
    sents = build_corpus()
    return TargetModel(sents), DraftModel(sents), sents


# ---------------------------------------------------------------------------
# Prompt-lookup matching
# ---------------------------------------------------------------------------

def test_longest_suffix_wins_over_shorter():
    # Corpus: "the cat sits" and "cat sits on" — the suffix
    # ("cat", "sits") has continuation "on"; the suffix
    # ("sits",) alone has continuations "on" (from "cat sits
    # on") and nothing else... use a corpus where the
    # length-1 and length-2 continuations differ.
    sents = [["the", "cat", "runs", "."],
             ["cat", "sits", "on", "the", "book", "."],
             ["the", "dog", "sits", "near", "the", "tree", "."]]
    d = PromptLookupDrafter(sents)
    # Context ends in "cat sits": the length-3 suffix
    # ("dog","cat","sits") matches nothing, so the length-2
    # corpus match ("cat","sits") -> "on" wins.
    tok, n, src = d.lookup([TOK["dog"], TOK["cat"], TOK["sits"]])
    assert tok == TOK["on"] and n == 2 and src == "corpus"
    # Context ends in "dog sits": the length-3 corpus match
    # ("the","dog","sits") -> "near" wins over length 2.
    tok, n, src = d.lookup([TOK["the"], TOK["dog"], TOK["sits"]])
    assert tok == TOK["near"] and n == 3 and src == "corpus"


def test_history_match_beats_corpus_at_same_length():
    # Corpus says "cat sits" -> "on", but the running context
    # itself already contains "cat sits under": the history
    # match at the same length wins (prompt lookup first).
    sents = [["cat", "sits", "on", "the", "book", "."]]
    d = PromptLookupDrafter(sents)
    ctx = [TOK["cat"], TOK["sits"], TOK["under"], TOK["the"],
           TOK["cat"], TOK["sits"]]
    tok, n, src = d.lookup(ctx)
    assert tok == TOK["under"] and n == 2 and src == "history"


def test_longest_match_wins_across_sources():
    # History has only a length-1 match ("sits" -> "under"
    # from the earlier "sits under"), the corpus has a
    # length-2 match ("cat sits" -> "on"): longest wins,
    # even though it comes from the corpus.
    sents = [["cat", "sits", "on", "the", "book", "."]]
    d = PromptLookupDrafter(sents)
    ctx = [TOK["sits"], TOK["under"], TOK["the"], TOK["cat"],
           TOK["sits"]]
    tok, n, src = d.lookup(ctx)
    assert tok == TOK["on"] and n == 2 and src == "corpus"


def test_no_match_fallback_is_uniform():
    # A drafter built from sentences that never mention
    # "dog": a context ending in "dog" matches nothing.
    d = PromptLookupDrafter([["cat", "runs", "."]])
    tok, n, src = d.lookup([TOK["dog"]])
    assert (tok, n, src) == (None, 0, None)
    p = d.dist([TOK["dog"]])
    assert p.shape == (V,)
    assert np.allclose(p, 1.0 / V)
    assert p.sum() == pytest.approx(1.0)
    # Empty context also falls back.
    tok, n, src = d.lookup([])
    assert (tok, n, src) == (None, 0, None)


def test_lookup_respects_max_ngram():
    sents = build_corpus()
    d = PromptLookupDrafter(sents, max_ngram=3)
    ctx = PROMPT_IDS[0] + [TOK["the"], TOK["answer"], TOK["is"]]
    _, n, _ = d.lookup(ctx)
    assert n <= 3


def test_index_deterministic_tie_break():
    # "the cat" is followed equally often by "sits" and
    # "runs": the lowest token id must win, every build.
    sents = [["the", "cat", "sits", "."],
             ["the", "cat", "runs", "."]]
    idx1 = build_lookup_index(sents)
    idx2 = build_lookup_index(sents)
    assert idx1 == idx2
    assert idx1[(TOK["the"], TOK["cat"])] == min(TOK["sits"],
                                                 TOK["runs"])


def test_drafter_dist_is_one_hot_on_lookup():
    sents = build_corpus()
    d = PromptLookupDrafter(sents)
    ctx = PROMPT_IDS[0]
    tok, _, _ = d.lookup(ctx)
    p = d.dist(ctx)
    assert p.sum() == pytest.approx(1.0)
    if tok is not None:
        assert int(np.argmax(p)) == tok
        assert p[tok] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_retrieval_generation_deterministic():
    target, _, sents = _models()
    d = PromptLookupDrafter(sents)
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    cost1 = CostModel(work_rounds=1, draft_cost_rounds=0)
    cost2 = CostModel(work_rounds=1, draft_cost_rounds=0)
    out1 = generate_with_drafter(target, d, PROMPT_IDS[0], rng1,
                                 cost1)
    out2 = generate_with_drafter(target, d, PROMPT_IDS[0], rng2,
                                 cost2)
    assert out1 == out2


def test_lookup_stats_add_up():
    _, _, sents = _models()
    d = PromptLookupDrafter(sents)
    target, _, _ = _models()
    rng = np.random.default_rng(3)
    cost = CostModel(work_rounds=1, draft_cost_rounds=0)
    generate_with_drafter(target, d, PROMPT_IDS[1], rng, cost)
    assert d.lookups > 0
    assert (d.history_hits + d.corpus_hits + d.misses
            == d.lookups)


# ---------------------------------------------------------------------------
# Overhead accounting
# ---------------------------------------------------------------------------

def test_zero_draft_cost_rounds_charges_no_simulated_work():
    cost = CostModel(work_rounds=1, draft_cost_rounds=0)
    before = cost.work_checksum
    for _ in range(10):
        cost.draft_work()
    assert cost.draft_steps == 10
    assert cost.work_checksum == before


def test_model_draft_cost_rounds_charges_simulated_work():
    cost = CostModel(work_rounds=1,
                     draft_cost_rounds=DRAFT_COST_ROUNDS)
    before = cost.work_checksum
    for _ in range(10):
        cost.draft_work()
    assert cost.draft_steps == 10
    assert cost.work_checksum != before


def test_eval_records_draft_cost_per_arm():
    result = run_retrieval_eval(repeats=1, n_prompts=4)
    for setting in result["settings"]:
        assert setting["arms"]["retrieval"]["draft_cost_rounds"] == 0
        assert setting["arms"]["model-based"]["draft_cost_rounds"] \
            == DRAFT_COST_ROUNDS
        # The retrieval drafter does real dict lookups, so its
        # microbenchmark time is finite and smaller than the
        # bigram draft model's distribution computation.
        assert result["overhead"]["retrieval_us_per_draft_token"] > 0
        assert result["overhead"]["model_us_per_draft_token"] > 0


# ---------------------------------------------------------------------------
# Acceptance-rate measurement (separate from wall-clock)
# ---------------------------------------------------------------------------

def test_acceptance_rate_is_accepted_over_examined():
    target, draft_model, sents = _models()
    retrieval = PromptLookupDrafter(sents)
    prompts = PROMPT_IDS[:4]
    for draft, draft_cost in ((draft_model, DRAFT_COST_ROUNDS),
                              (retrieval, 0)):
        rng = np.random.default_rng(11)
        cost = CostModel(work_rounds=1, draft_cost_rounds=draft_cost)
        total_acc = 0
        total_ex = 0
        for prompt in prompts:
            _, _, acc, ex, _ = generate_with_drafter(
                target, draft, prompt, rng, cost)
            total_acc += acc
            total_ex += ex
        assert 0 <= total_acc <= total_ex
        assert total_ex > 0
        rate = total_acc / total_ex
        assert 0.0 <= rate <= 1.0


def test_tokens_per_target_forward_above_one_both_arms():
    target, draft_model, sents = _models()
    retrieval = PromptLookupDrafter(sents)
    prompts = PROMPT_IDS[:4]
    for draft, draft_cost in ((draft_model, DRAFT_COST_ROUNDS),
                              (retrieval, 0)):
        rng = np.random.default_rng(11)
        cost = CostModel(work_rounds=1, draft_cost_rounds=draft_cost)
        tokens = 0
        forwards = 0
        for prompt in prompts:
            gen, _, _, _, fwd = generate_with_drafter(
                target, draft, prompt, rng, cost)
            tokens += len(gen)
            forwards += fwd
        assert tokens / forwards > 1.0


# ---------------------------------------------------------------------------
# Eval structure: both cost settings, done flag = comparison
# ---------------------------------------------------------------------------

def test_eval_has_both_cost_settings_and_done_flag():
    result = run_retrieval_eval(repeats=1, n_prompts=8)
    assert len(result["settings"]) == 2
    assert result["settings"][0]["name"].startswith("small-target")
    assert result["settings"][1]["name"].startswith("large-target")
    assert result["settings"][0]["work_rounds"] == 1
    assert result["settings"][1]["work_rounds"] == 32
    for setting in result["settings"]:
        arms = setting["arms"]
        assert set(arms) == {"target-only", "model-based",
                             "retrieval"}
        for label in ("model-based", "retrieval"):
            assert "acceptance_rate" in arms[label]
            assert "tokens_per_target_forward" in arms[label]
            assert "tokens_per_sec" in arms[label]
            assert 0.0 <= arms[label]["acceptance_rate"] <= 1.0
        # The done flag is exactly the wall-clock comparison
        # at the small-target setting — whatever it measures.
        assert setting["retrieval_speedup_vs_model"] == pytest.approx(
            arms["retrieval"]["tokens_per_sec"]
            / arms["model-based"]["tokens_per_sec"])
    small = result["settings"][0]
    assert result["done_retrieval_beats_model_small_target"] == bool(
        small["retrieval_speedup_vs_model"] > 1.0)


def test_eval_deterministic_columns_stable():
    r1 = run_retrieval_eval(repeats=1, n_prompts=8)
    r2 = run_retrieval_eval(repeats=1, n_prompts=8)
    for s1, s2 in zip(r1["settings"], r2["settings"]):
        for label in ("target-only", "model-based", "retrieval"):
            assert s1["arms"][label]["tokens"] \
                == s2["arms"][label]["tokens"]
            assert s1["arms"][label]["target_forwards"] \
                == s2["arms"][label]["target_forwards"]


# ---------------------------------------------------------------------------
# Reuse pin: verification is P11's loop, not P10's
# ---------------------------------------------------------------------------

def test_verification_is_p11_lossless_round():
    assert rd.speculative_round is speculative_round
    # P10's lossy speculative loop is NOT imported.
    assert not hasattr(rd, "generate_speculative")
    assert not hasattr(rd, "verify_tree")
    assert not hasattr(rd, "margin_accept")
    assert not hasattr(rd, "TreePolicy")
