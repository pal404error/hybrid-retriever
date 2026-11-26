"""P16 — context-engineering lab tests.

Covers: policy config parsing/validation, each primitive's effect
on a fixed context, trigger firing rules, the cost model, grid
determinism, the harness end-to-end on a tiny task subset, and
the context-bound read path (an evicted/cleared episode is
genuinely forgotten by the agent).
"""

import os
import tempfile

import pytest

from hybrid_retriever.compress import verify_no_invention
from hybrid_retriever.context_lab import (
    POLICY_GRID,
    Context,
    PolicyConfig,
    apply_summarise,
    apply_trc,
    apply_truncate,
    cost_usd,
    maybe_compact,
    run_context_lab_eval,
    run_task_with_policy,
)
from hybrid_retriever.eviction import estimate_tokens
from hybrid_retriever.tobench import build_tobench_tasks
from hybrid_retriever.verify import build_vbd_store, build_vbd_tasks


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _fixed_context() -> Context:
    """A small fixed context: user + plan + tool call + a long
    JSON tool result + an extract reasoning episode."""
    ctx = Context()
    ctx.append("user", "goal", "Total the expenses in expenses.json.")
    ctx.append("reasoning", "plan", "I will read the file and total it.")
    ctx.append("tool_call", "read", "files.read_file(expenses.json)")
    big_json = (
        '{"status": "ok", "count": 13, "results": ['
        + ", ".join(
            '{"id": %d, "note": "entry %d about topic %d", '
            '"score": 0.5, "detail": "%s"}' % (i, i, i % 5, "x" * 50)
            for i in range(12)
        )
        + "]}"
    )
    ctx.append("tool_result", "read", big_json)
    ctx.append(
        "reasoning", "extract",
        "The answer is 33.0.",
    )
    return ctx


@pytest.fixture(scope="module")
def vbd_store():
    from hybrid_retriever.embeddings import get_embedder

    emb = get_embedder()
    store, _ = build_vbd_store(
        os.path.join("eval_data", "synthetic_corpus.jsonl"),
        embedder=emb,
    )
    yield store, emb
    store.close()


# ----------------------------------------------------------------------
# Phase 1 — policy config
# ----------------------------------------------------------------------


def test_policy_defaults_are_valid():
    cfg = PolicyConfig()
    assert cfg.primitive == "none"
    assert cfg.trigger == "never"
    assert cfg.depth == 0.50
    assert cfg.budget == 120
    assert cfg.every_n == 2


def test_policy_from_dict_round_trip():
    cfg = PolicyConfig.from_dict(
        {"primitive": "trc", "trigger": "budget", "depth": 0.25,
         "budget": 120, "every_n": 2}
    )
    assert cfg.as_dict() == {
        "primitive": "trc", "trigger": "budget", "depth": 0.25,
        "budget": 120, "every_n": 2,
    }


def test_policy_rejects_unknown_primitive_and_trigger():
    with pytest.raises(ValueError):
        PolicyConfig(primitive="shrink")
    with pytest.raises(ValueError):
        PolicyConfig(trigger="sometimes")


def test_policy_rejects_bad_depth_budget_every_n():
    with pytest.raises(ValueError):
        PolicyConfig(depth=0.0)
    with pytest.raises(ValueError):
        PolicyConfig(depth=1.5)
    with pytest.raises(ValueError):
        PolicyConfig(budget=0)
    with pytest.raises(ValueError):
        PolicyConfig(every_n=0)


def test_policy_rejects_unknown_dict_keys():
    with pytest.raises(ValueError):
        PolicyConfig.from_dict({"primitive": "trc", "strengh": 3})


def test_policy_none_needs_never_or_budget_trigger():
    with pytest.raises(ValueError):
        PolicyConfig(primitive="none", trigger="every_n")
    # ... but the no-op control (none + budget) is legal.
    PolicyConfig(primitive="none", trigger="budget")


def test_policy_grid_shape():
    names = [n for n, _ in POLICY_GRID]
    assert names == [
        "baseline", "noop", "trc-50", "trc-25",
        "truncate-50", "truncate-25",
        "summarise-50", "summarise-25", "trc-every2",
    ]
    assert all(isinstance(cfg, PolicyConfig) for _, cfg in POLICY_GRID)


# ----------------------------------------------------------------------
# Primitives on a fixed context
# ----------------------------------------------------------------------


def test_trc_clears_tool_results_only_and_keeps_the_record():
    ctx = _fixed_context()
    before = ctx.live_tokens()
    report = apply_trc(ctx, target_tokens=0)
    assert report["n_affected"] == 1
    assert report["tokens_saved"] > 0
    assert ctx.live_tokens() < before
    # The episode survives (the record that a result existed),
    # but its payload is gone and the original is untouched.
    result = ctx.visible("read")
    assert result is not None
    assert "cleared" in result
    assert "expenses" not in result
    ep = [
        e for e in ctx.live()
        if e.kind == "read" and e.type == "tool_result"
    ][0]
    assert ep.original_text.startswith('{"status"')
    assert ep.cleared
    # Nothing else was touched.
    assert len(ctx.live()) == 5
    assert ctx.visible_reasoning() == "The answer is 33.0."


def test_truncate_drops_oldest_and_protects_user_and_active_window():
    ctx = _fixed_context()
    report = apply_truncate(ctx, target_tokens=0)
    assert report["reached_target"] is False  # protections hold
    live = ctx.live()
    kinds = [(e.type, e.kind) for e in live]
    # User turn and the last episode (the extract reasoning) survive.
    assert ("user", "goal") in kinds
    assert ("reasoning", "extract") in kinds
    # The plan, the tool call and the tool result are gone.
    assert ("reasoning", "plan") not in kinds
    assert ("tool_call", "read") not in kinds
    assert ("tool_result", "read") not in kinds
    assert report["n_affected"] == 3


def test_truncate_reaches_a_loose_target_by_dropping_one_episode():
    ctx = _fixed_context()
    # Dropping just the big tool result (oldest evictable is the
    # plan, but the loop drops oldest-first; either way it stops
    # once under target).
    target = ctx.live_tokens() - estimate_tokens(
        "I will read the file and total it."
    )
    report = apply_truncate(ctx, target_tokens=target)
    assert report["reached_target"] is True
    assert ctx.live_tokens() <= target


def test_summarise_compresses_long_episodes_and_never_rewrites_user():
    ctx = _fixed_context()
    user_before = [e for e in ctx.live() if e.type == "user"][0].text
    before = ctx.live_tokens()
    report = apply_summarise(ctx, target_tokens=0)
    assert report["n_affected"] >= 1
    assert report["tokens_saved"] > 0
    assert ctx.live_tokens() < before
    user_after = [e for e in ctx.live() if e.type == "user"][0].text
    assert user_after == user_before
    # Extractive only: nothing invented in any compressed episode.
    for ep in ctx.live():
        if ep.compressed:
            assert verify_no_invention(ep.original_text, ep.text) == []


def test_summarise_charges_tokens_and_a_call():
    ctx = _fixed_context()
    processed_before = ctx.tokens_processed
    calls_before = ctx.n_summarise_calls
    apply_summarise(ctx, target_tokens=0)
    assert ctx.tokens_processed > processed_before
    assert ctx.n_summarise_calls > calls_before


# ----------------------------------------------------------------------
# Triggers
# ----------------------------------------------------------------------


def test_budget_trigger_fires_only_above_budget():
    small = PolicyConfig("trc", "budget", 0.50, budget=10**9)
    ctx = _fixed_context()
    assert maybe_compact(ctx, small) is None
    assert ctx.trigger_checks == 1
    tight = PolicyConfig("trc", "budget", 0.50, budget=10)
    assert maybe_compact(ctx, tight) is not None
    assert len(ctx.compactions) == 1


def test_every_n_trigger_fires_once_per_turn_count():
    policy = PolicyConfig("trc", "every_n", 0.50, every_n=2)
    ctx = Context()
    ctx.append("user", "goal", "do a thing")
    ctx.append("reasoning", "plan", "step one")
    assert maybe_compact(ctx, policy) is None  # turn 1: no fire
    ctx.append("reasoning", "extract", "step two")
    assert maybe_compact(ctx, policy) is not None  # turn 2: fire
    # Same turn count, later check (e.g. after a final action):
    # must NOT fire again for turn 2.
    assert maybe_compact(ctx, policy) is None
    ctx.append("reasoning", "extract", "step three")
    assert maybe_compact(ctx, policy) is None  # turn 3: no fire
    ctx.append("reasoning", "extract", "step four")
    assert maybe_compact(ctx, policy) is not None  # turn 4: fire


def test_end_of_task_trigger_fires_only_at_the_end():
    policy = PolicyConfig("trc", "end_of_task", 0.50)
    ctx = _fixed_context()
    assert maybe_compact(ctx, policy) is None
    assert maybe_compact(ctx, policy, end_of_task=True) is not None


def test_never_trigger_and_none_primitive_never_compact():
    ctx = _fixed_context()
    before = ctx.live_tokens()
    assert maybe_compact(ctx, PolicyConfig("trc", "never")) is None
    assert maybe_compact(
        ctx, PolicyConfig("none", "budget", budget=1)
    ) is None
    assert ctx.live_tokens() == before
    assert ctx.compactions == []


# ----------------------------------------------------------------------
# Cost model
# ----------------------------------------------------------------------


def test_cost_model_rates():
    assert cost_usd(1_000_000, 0) == pytest.approx(3.00)
    assert cost_usd(0, 0) == 0.0
    assert cost_usd(0, 1) == pytest.approx(0.002)
    assert cost_usd(500_000, 2) == pytest.approx(1.50 + 0.004)


# ----------------------------------------------------------------------
# The context-bound read path (causality pins)
# ----------------------------------------------------------------------


def test_scan_is_deterministic_and_bills_live_tokens_once_per_call():
    ctx = _fixed_context()
    first = ctx.scan({"expenses", "file"})
    billed = ctx.tokens_processed
    assert billed == ctx.live_tokens()
    second = ctx.scan({"expenses", "file"})
    assert [e.seq for e, _ in second] == [e.seq for e, _ in first]
    assert ctx.tokens_processed == 2 * billed


def test_cleared_tool_result_is_forgotten_before_consumption():
    # The counterfactual the consumption rule exists for: if
    # compaction fired on the raw tool result, the agent would
    # have nothing to extract from.
    ctx = Context()
    ctx.append("user", "goal", "What is the sum?")
    ctx.append("tool_result", "recall",
               "record #1: deduplicate by id first")
    apply_trc(ctx, target_tokens=0)
    assert "deduplicate" not in (ctx.visible("recall") or "")


def test_evicted_reasoning_is_forgotten_by_the_final_step():
    ctx = _fixed_context()
    assert ctx.visible_reasoning() == "The answer is 33.0."
    ep = [e for e in ctx.live() if e.kind == "extract"][0]
    ep.evicted = True
    # The final step reads the latest live reasoning episode: the
    # extract (with the answer) is gone, so only the plan remains
    # and the answer is no longer visible anywhere in context.
    assert ctx.visible_reasoning() == "I will read the file and total it."
    assert all(
        "33.0" not in e.text for e in ctx.live()
    )


# ----------------------------------------------------------------------
# Harness end-to-end on a tiny subset + determinism
# ----------------------------------------------------------------------


def _tiny_policies():
    return [
        ("baseline", PolicyConfig(primitive="none", trigger="never")),
        ("noop", PolicyConfig(primitive="none", trigger="budget")),
        ("trc-50", PolicyConfig("trc", "budget", 0.50)),
    ]


def test_harness_tiny_end_to_end_and_noop_equals_baseline():
    result = run_context_lab_eval(
        n_tobench=2, n_vbd=2, policies=_tiny_policies(), verbose=False
    )
    assert result["n_tasks"] == 4
    assert result["cost_model"]["modelled"] is True
    base = result["rows"]["baseline"]
    noop = result["rows"]["noop"]
    # S1, S2 pass; json-00 fails (distractor rule), log-01 passes.
    assert base["passed"] == 3
    assert base["total"] == 4
    # The no-op control reproduces the baseline exactly on every
    # deterministic metric.
    assert noop["passed"] == base["passed"]
    assert noop["tokens"] == base["tokens"]
    assert noop["cost_usd"] == base["cost_usd"]
    assert noop["n_compactions"] == 0
    # TRC holds success on the tiny set (consumption precedes
    # clearing) while cutting tokens.
    trc = result["rows"]["trc-50"]
    assert trc["passed"] == base["passed"]
    assert trc["tokens"] < base["tokens"]
    assert trc["n_compactions"] > 0
    assert set(result["rows"]) == {"baseline", "noop", "trc-50"}


def test_task_runs_are_deterministic(vbd_store):
    store, emb = vbd_store
    run_dir = tempfile.mkdtemp(prefix="hr-ctxlab-test-")
    tasks = build_tobench_tasks()[:2] + build_vbd_tasks(2)
    policy = PolicyConfig("trc", "budget", 0.50)
    first = [
        run_task_with_policy(t, policy, vbd_store=store,
                             embedder=emb, run_dir=run_dir)
        for t in tasks
    ]
    second = [
        run_task_with_policy(t, policy, vbd_store=store,
                             embedder=emb, run_dir=run_dir)
        for t in tasks
    ]
    for a, b in zip(first, second):
        assert a["passed"] == b["passed"]
        assert a["tokens"] == b["tokens"]
        assert a["cost_usd"] == b["cost_usd"]
        assert a["n_compactions"] == b["n_compactions"]
        assert a["peak_live_tokens"] == b["peak_live_tokens"]
