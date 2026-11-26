"""P17 — SUMONE-style learned summariser tests.

Covers: config validation, masking scheme (numbers / quoted /
goal words, seeded determinism), recovery-rate computation
(perfect / empty), extractive-only output, the accept/reject
boundary (reject -> looser budget -> uncompressed fallback),
budget and per-segment minimum, the reward, parameter-search
determinism and improvement over the default config, the
checkpoint round-trip, the summarising agent context, and
held-out eval determinism plus cost/success accounting on
both harnesses.
"""

import os

import pytest

from hybrid_retriever.eviction import estimate_tokens
from hybrid_retriever.sumone import (
    DEFAULT_CONFIG,
    RECOVERY_THRESHOLD,
    SummariserConfig,
    SummarisingContext,
    _candidate_spans,
    _content_words,
    _split_units,
    build_grid,
    coverage_rate,
    mean_reward,
    parameter_search,
    recovery_rate,
    reward_fn,
    run_sumone_eval,
    sample_masks,
    summarise_text,
    summarise_trajectory,
    summarise_units,
    verify_extractive,
    _units_from_episodes,
)
from hybrid_retriever.tobench import build_tobench_tasks
from hybrid_retriever.verify import build_vbd_store, build_vbd_tasks


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _fact_trajectory() -> list[dict]:
    """A small trajectory: user goal + recall tool result with
    numbered, quoted facts + a reasoning distillation."""
    return [
        {
            "type": "user",
            "kind": "goal",
            "text": "What is Maya's favourite tea and how much "
            "does she spend on rye flour?",
        },
        {
            "type": "tool_result",
            "kind": "recall",
            "text": "\n".join(
                [
                    "record #4: Maya's favourite tea is 'earl grey'.",
                    "record #7: Maya spends 12.50 on rye flour weekly.",
                    "record #9: Maya bakes sourdough every Sunday morning.",
                    "record #11: Tom delivers flour for 8.25 per bag.",
                    "record #13: Lena counts heads three times before trips.",
                    "record #15: Sam writes quotes on carbon paper daily.",
                ]
            ),
        },
        {
            "type": "reasoning",
            "kind": "extract",
            "text": "Relevant facts: Maya's favourite tea is earl grey. "
            "Maya spends 12.50 on rye flour weekly.",
        },
    ]


def _number_heavy_trajectory(n_lines: int = 10) -> list[dict]:
    """Each line carries one distinct number and otherwise
    disjoint context words — dropping a line loses its number
    for good, which is what the accept/reject boundary needs."""
    lines = [
        "entry %d: zebras %d kilograms alpha%d"
        % (i, 100 + i, i)
        for i in range(n_lines)
    ]
    return [
        {
            "type": "user",
            "kind": "goal",
            "text": "Count the zebras in the ledger.",
        },
        {
            "type": "tool_result",
            "kind": "read",
            "text": "\n".join(lines),
        },
    ]


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
# Config
# ----------------------------------------------------------------------


def test_config_defaults_are_valid():
    cfg = SummariserConfig()
    assert cfg.budget_frac == 0.25
    assert cfg.min_per_segment == 1
    assert cfg.as_dict()["w_number"] == 2.0


def test_config_validation():
    with pytest.raises(ValueError):
        SummariserConfig(budget_frac=0.0)
    with pytest.raises(ValueError):
        SummariserConfig(budget_frac=1.5)
    with pytest.raises(ValueError):
        SummariserConfig(w_number=-1.0)
    with pytest.raises(ValueError):
        SummariserConfig(min_per_segment=-1)
    with pytest.raises(ValueError):
        SummariserConfig.from_dict({"budget_frac": 0.25, "nope": 1})


def test_checkpoint_round_trip():
    cfg = SummariserConfig(
        budget_frac=0.4, w_number=3.0, w_goal=1.5, w_tool=0.0,
        min_per_segment=2,
    )
    again = SummariserConfig.from_dict(cfg.as_dict())
    assert again.as_dict() == cfg.as_dict()


def test_grid_is_48_configs_in_fixed_order():
    grid = build_grid()
    assert len(grid) == 48
    assert build_grid()[0].as_dict() == grid[0].as_dict()
    assert len({repr(c.as_dict()) for c in grid}) == 48


# ----------------------------------------------------------------------
# Units and masking
# ----------------------------------------------------------------------


def test_split_units_keeps_json_lines_whole():
    text = '{"id": 1, "score": 0.5}\n{"id": 2, "score": 0.75}'
    assert _split_units(text) == [
        '{"id": 1, "score": 0.5}',
        '{"id": 2, "score": 0.75}',
    ]


def test_split_units_splits_long_prose():
    text = "First sentence here. Second sentence here. " * 12
    assert len(_split_units(text)) > 5


def test_masking_scheme_covers_numbers_quoted_and_goal_words():
    units, _ = _units_from_episodes(_fact_trajectory())
    goal_words = _content_words(
        "What is Maya's favourite tea and how much "
        "does she spend on rye flour?"
    )
    spans = {(m.span, m.unit_index) for m in _candidate_spans(units, goal_words)}
    span_texts = {s for s, _ in spans}
    assert "4" in span_texts  # record id (number)
    assert "12.50" in span_texts  # number
    assert "earl grey" in span_texts  # quoted fact value
    assert "maya" in span_texts  # goal key token
    assert "rye" in span_texts


def test_mask_sampling_is_seeded_and_deterministic():
    units, _ = _units_from_episodes(_fact_trajectory())
    goal_words = _content_words("Maya favourite tea rye flour")
    first = sample_masks(units, goal_words, 8)
    second = sample_masks(units, goal_words, 8)
    assert [(m.span, m.unit_index) for m in first] == [
        (m.span, m.unit_index) for m in second
    ]
    assert len(first) <= 8


def test_recovery_perfect_and_empty():
    units, _ = _units_from_episodes(_fact_trajectory())
    goal_words = _content_words("Maya favourite tea rye flour")
    masks = sample_masks(units, goal_words, 8)
    assert masks, "fixture must yield masks"
    assert recovery_rate(masks, units) == 1.0
    assert recovery_rate(masks, []) == 0.0
    assert recovery_rate([], units) == 1.0  # nothing to recover


def test_coverage_is_diagnostic_not_gate():
    units, _ = _units_from_episodes(_fact_trajectory())
    goal_words = _content_words("Maya favourite tea rye flour")
    assert coverage_rate(units, goal_words, units) == 1.0
    assert coverage_rate(units, goal_words, units[:1]) < 1.0


# ----------------------------------------------------------------------
# Summarisation: extractive, budget, gate
# ----------------------------------------------------------------------


def test_summary_is_extractive():
    result = summarise_trajectory(
        _fact_trajectory(),
        "Maya favourite tea rye flour",
        DEFAULT_CONFIG,
    )
    original = "\n".join(e["text"] for e in _fact_trajectory())
    assert verify_extractive(original, result.text) == []


def test_budget_is_respected_when_accepted():
    # Redundant lines carrying the SAME number and goal words:
    # any kept line recovers every sampled mask, so the gate
    # accepts the tight budget instead of falling back.
    episodes = [
        {
            "type": "user",
            "kind": "goal",
            "text": "Summarise the sourdough notes.",
        },
        {
            "type": "tool_result",
            "kind": "recall",
            "text": "\n".join(
                "Maya feeds her sourdough starter with rye "
                "flour daily and keeps it warm overnight. "
                "Batch 3."
                for _ in range(12)
            ),
        },
    ]
    cfg = SummariserConfig(budget_frac=0.25, min_per_segment=1)
    result = summarise_trajectory(episodes, "sourdough notes", cfg)
    assert result.tokens_before > 0
    assert result.token_ratio <= 0.60  # budget + min-per-segment slack


def test_min_per_segment_is_enforced():
    episodes = [
        {"type": "tool_result", "kind": "a", "text": "Alpha one.\nAlpha two.\nAlpha three."},
        {"type": "tool_result", "kind": "b", "text": "Beta one.\nBeta two.\nBeta three."},
        {"type": "tool_result", "kind": "c", "text": "Gamma one.\nGamma two.\nGamma three."},
    ]
    cfg = SummariserConfig(
        budget_frac=0.15, min_per_segment=2, w_number=0.0,
        w_goal=0.0, w_tool=0.0, w_first=0.0,
    )
    result = summarise_trajectory(episodes, "nothing relevant", cfg)
    kept = result.text.splitlines()
    assert len(kept) >= 6  # >= 2 per segment
    for word in ("Alpha", "Beta", "Gamma"):
        assert any(word in line for line in kept)


def test_reject_falls_back_uncompressed():
    # Ten disjoint number lines, tiny budget: the sampled
    # masks cannot be recovered from one kept line, so every
    # retry fails and the summary falls back uncompressed.
    traj = _number_heavy_trajectory(10)
    cfg = SummariserConfig(
        budget_frac=0.05, w_number=0.0, w_goal=0.0, w_tool=0.0,
        w_first=0.0, min_per_segment=1,
    )
    result = summarise_trajectory(traj, "Count the zebras", cfg)
    assert result.accepted is False
    assert result.fell_back is True
    assert result.text == "\n".join(e["text"] for e in traj)
    assert result.token_ratio == 1.0
    assert len(result.attempts) >= 2  # initial + at least one retry
    assert all(
        a["recovery_rate"] < RECOVERY_THRESHOLD
        for a in result.attempts
    )


def test_accept_boundary_full_summary_passes():
    traj = _number_heavy_trajectory(10)
    cfg = SummariserConfig(budget_frac=1.0, min_per_segment=1)
    result = summarise_trajectory(traj, "Count the zebras", cfg)
    assert result.accepted is True
    assert result.fell_back is False
    assert result.recovery_rate >= RECOVERY_THRESHOLD


def test_reward_fn_is_recovery_minus_length_penalty():
    assert reward_fn(0.9, 0.25) == pytest.approx(0.9 - 0.5 * 0.25)
    assert reward_fn(1.0, 1.0) == pytest.approx(0.5)
    assert reward_fn(0.0, 0.0) == pytest.approx(0.0)


def test_summarise_text_single_episode():
    text = "\n".join(
        "record #%d: fact about topic %d with value %d."
        % (i, i, i * 3)
        for i in range(6)
    )
    result = summarise_text(text, "facts about topic", DEFAULT_CONFIG)
    assert result.tokens_before == estimate_tokens(text)
    assert result.n_masks > 0
    assert verify_extractive(text, result.text) == [] or result.fell_back


# ----------------------------------------------------------------------
# Parameter search
# ----------------------------------------------------------------------


def test_search_is_deterministic_and_improves_on_default():
    trajs = [
        {"goal": "Maya favourite tea rye flour", "episodes": _fact_trajectory()},
        {"goal": "Count the zebras", "episodes": _number_heavy_trajectory(10)},
        {"goal": "Maya favourite tea rye flour", "episodes": _fact_trajectory()},
    ]
    first = parameter_search(trajs)
    second = parameter_search(trajs)
    assert first["best"]["config"] == second["best"]["config"]
    assert first["best"]["mean_reward"] == second["best"]["mean_reward"]
    assert first["best"]["mean_reward"] >= first["default"]["mean_reward"]
    assert first["n_configs"] == 48


def test_mean_reward_reports_recovery_ratio_and_fallbacks():
    trajs = [
        {"goal": "Count the zebras", "episodes": _number_heavy_trajectory(10)},
    ]
    scored = mean_reward(trajs, DEFAULT_CONFIG)
    assert 0.0 <= scored["mean_recovery"] <= 1.0
    assert 0.0 <= scored["mean_token_ratio"] <= 1.0
    assert scored["n_trajectories"] == 1
    assert "n_fell_back" in scored


# ----------------------------------------------------------------------
# SummarisingContext (the agent arm)
# ----------------------------------------------------------------------


def test_summarising_context_summarises_tool_results_only():
    ctx = SummarisingContext(DEFAULT_CONFIG)
    ctx.append("user", "goal", "Maya favourite tea rye flour")
    ctx.append("reasoning", "plan", "I will recall the facts.")
    big = "\n".join(
        "record #%d: Maya fact number %d about tea." % (i, i)
        for i in range(8)
    )
    ctx.append("tool_result", "recall", big)
    assert ctx.n_summarised == 1
    assert ctx.summariser_tokens_read == estimate_tokens(big)
    # The live episode is the summary, not the full result.
    visible = ctx.visible("recall")
    assert visible is not None
    assert estimate_tokens(visible) <= estimate_tokens(big)
    # User and reasoning episodes are untouched.
    assert ctx.episodes[0].text.startswith("Maya favourite")
    assert ctx.episodes[1].text == "I will recall the facts."
    assert ctx.recovery_rates, "gate ran on the episode"


def test_summarising_context_fallback_flag_is_recorded():
    cfg = SummariserConfig(
        budget_frac=0.05, w_number=0.0, w_goal=0.0, w_tool=0.0,
        w_first=0.0, min_per_segment=1,
    )
    ctx = SummarisingContext(cfg)
    ctx.append("user", "goal", "Count the zebras")
    text = "\n".join(
        "entry %d: zebras %d kilograms alpha%d" % (i, 100 + i, i)
        for i in range(10)
    )
    ctx.append("tool_result", "read", text)
    assert ctx.n_fell_back == 1
    assert ctx.visible("read") == text  # uncompressed, flagged


# ----------------------------------------------------------------------
# Full eval: determinism + accounting on both harnesses (tiny sets)
# ----------------------------------------------------------------------


def test_sumone_eval_small_is_deterministic(vbd_store):
    store, emb = vbd_store
    first = run_sumone_eval(n_tobench=2, n_vbd=2, verbose=False)
    second = run_sumone_eval(n_tobench=2, n_vbd=2, verbose=False)
    assert first["checkpoint"] == second["checkpoint"]
    assert first["search"]["best"]["mean_reward"] == pytest.approx(
        second["search"]["best"]["mean_reward"]
    )
    for name in ("tobench", "vbd"):
        assert first["harnesses"][name]["full_passed"] == (
            second["harnesses"][name]["full_passed"]
        )
        assert first["harnesses"][name]["sum_passed"] == (
            second["harnesses"][name]["sum_passed"]
        )
        assert first["harnesses"][name]["full_tokens"] == (
            second["harnesses"][name]["full_tokens"]
        )
        assert first["harnesses"][name]["sum_tokens"] == (
            second["harnesses"][name]["sum_tokens"]
        )
    assert isinstance(first["done"], bool)


def test_sumone_eval_accounting(vbd_store):
    store, emb = vbd_store
    result = run_sumone_eval(n_tobench=2, n_vbd=2, verbose=False)
    assert result["n_tobench"] == 2
    assert result["n_vbd"] == 2
    assert result["n_train_trajectories"] >= 2  # agent + P3
    assert result["n_heldout_trajectories"] == 2
    assert result["heldout_best"]["mean_reward"] is not None
    assert result["heldout_default"]["mean_reward"] is not None
    # Held-out improvement is measured and reported, not
    # asserted — the search optimises the train split.
    for name, row in result["harnesses"].items():
        assert row["n_tasks"] == (2 if name == "tobench" else 2)
        assert row["tasks_lost"] == (
            row["full_passed"] - row["sum_passed"]
        )
        assert 0.0 <= row["cost_reduction"] <= 1.0
        assert row["token_ratio"] == pytest.approx(
            row["sum_tokens"] / row["full_tokens"]
        )
        assert row["summariser_tokens_read"] >= 0
        assert row["n_tool_results_summarised"] > 0
        assert 0.0 <= row["mean_episode_recovery"] <= 1.0
        assert row["done"] == (
            row["cost_reduction"] >= 0.75
            and row["tasks_lost"] <= 1
        )
    assert result["done"] == all(
        h["done"] for h in result["harnesses"].values()
    )
