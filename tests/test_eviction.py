"""P18 tests: annotation, tiers, protections, budget, explainability."""

import os

import pytest

from hybrid_retriever.embeddings import HashedEmbedder
from hybrid_retriever.eviction import (
    EPISODE_TYPES,
    annotate_episode,
    dependency_violations,
    estimate_tokens,
    evict_to_budget,
    eviction_tier,
    explain_episode,
    ingest_session,
    mark_session_persisted_by_merges,
)
from hybrid_retriever.eval import run_eviction_eval
from hybrid_retriever.lifecycle import draft_merge_text, propose_merge
from hybrid_retriever.segments import turn_text
from hybrid_retriever.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "t.db"), embedder=HashedEmbedder())
    yield s
    s.close()


def _record(store, text):
    return store.add_record(text, type_="episode")


def _session_with_tiers(store):
    """One session exercising every tier (see eviction_tier).

    seq: 0 user (protected) / 1 reasoning / 2 tool_call /
    3 tool_result persisted (tier 0) / 4 chatter (tier 1) /
    5 tool_call persisted (tier 2) / 6 fact persisted (tier 3) /
    7 decision (tier 5) / 8 reasoning (tier 4) /
    9 tool_result unpersisted (tier 6).
    Episode 9 depends on episode 5 (its tool_call), so 5 is
    protected while 9 lives — the 3rd eviction is 6, not 5.
    """
    persist = _record(store, "the persisted outcome")
    turns = [
        {"role": "user", "text": "question one"},
        {"role": "assistant", "text": "thinking about it"},
        {"role": "assistant", "text": "lookup one", "episode_type": "tool_call"},
        {"role": "tool", "text": "result text here", "persisted_by": persist},
        {"role": "assistant", "text": "noted", "episode_type": "chatter"},
        {"role": "assistant", "text": "lookup two", "episode_type": "tool_call", "persisted_by": persist},
        {"role": "assistant", "text": "a fact here", "episode_type": "fact", "persisted_by": persist},
        {"role": "assistant", "text": "we decided to go with red", "episode_type": "decision"},
        {"role": "assistant", "text": "more thinking here"},
        {"role": "tool", "text": "result two here"},
    ]
    return ingest_session(store, "s", turns)


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("x" * 100) == 25
    # Deterministic: same text, same number, every time.
    assert estimate_tokens("hello world") == estimate_tokens("hello world")


def test_annotate_validates(store):
    rid = _record(store, "some text")
    with pytest.raises(ValueError, match="unknown episode type"):
        annotate_episode(store, "s", rid, "nonsense")
    with pytest.raises(ValueError, match="unknown record"):
        annotate_episode(store, "s", 999, "user")
    eid = annotate_episode(store, "s", rid, "user")
    with pytest.raises(ValueError, match="unknown dependency"):
        annotate_episode(store, "s", rid, "reasoning", depends_on=[999])
    other = annotate_episode(store, "other", rid, "user")
    with pytest.raises(ValueError, match="belongs to session"):
        annotate_episode(store, "s", rid, "reasoning", depends_on=[other])
    with pytest.raises(ValueError, match="unknown persisted_by"):
        annotate_episode(store, "s", rid, "fact", persisted_by=999)
    assert eid is not None


def test_ingest_session_defaults(store):
    turns = [
        {"role": "user", "text": "what time is it"},
        {"role": "assistant", "text": "let me check the clock"},
        {"role": "assistant", "text": "read clock", "episode_type": "tool_call"},
        {"role": "tool", "text": "the clock says noon"},
        {"role": "assistant", "text": "we decided on noon", "episode_type": "decision"},
    ]
    ids = ingest_session(store, "s", turns)
    eps = store.list_episodes("s")
    types = [e["type"] for e in eps]
    assert types == ["user", "reasoning", "tool_call", "tool_result", "decision"]
    # The tool_result auto-links to the preceding tool_call.
    assert eps[3]["depends_on"] == [ids[2]]
    # Explicit episode_type overrides the role default.
    assert eps[4]["type"] == "decision"
    # Token totals are the deterministic estimates over the
    # stored text (turn_text prefixes the role).
    assert store.session_token_total("s") == sum(
        estimate_tokens(turn_text(t)) for t in turns
    )


def test_ingest_depends_on_indices(store):
    turns = [
        {"role": "user", "text": "first question"},
        {"role": "assistant", "text": "thinking", "depends_on": [0]},
        {"role": "assistant", "text": "more", "depends_on": [5]},
    ]
    with pytest.raises(ValueError, match="out of range"):
        ingest_session(store, "s", turns)


def test_tier_order_and_dependency_protection(store):
    ids = _session_with_tiers(store)
    eps = store.list_episodes("s")
    # Tiers in seq order: user is tier 6 by the function but never
    # a candidate (protected separately); the rest are exact.
    assert [eviction_tier(e) for e in eps] == [6, 4, 6, 0, 1, 2, 3, 5, 4, 6]
    total = store.session_token_total("s")
    # Evict exactly three: tier 0, tier 1, then tier 3 — tier 2 is
    # protected while episode 9 (which depends on it) lives.
    want = [ids[3], ids[4], ids[6]]
    budget = total - sum(eps[i]["tokens"] for i in (3, 4, 6))
    summary = evict_to_budget(store, "s", budget, protect_last=0)
    assert summary["under_budget"] is True
    assert [e["episode_id"] for e in summary["evicted"]] == want
    assert summary["evicted_by_tier"] == {0: 1, 1: 1, 3: 1}
    assert store.get_episode(ids[5])["status"] == "live"
    assert store.get_episode(ids[0])["status"] == "live"


def test_user_never_evicted_and_overage_reported(store):
    turns = [
        {"role": "user", "text": "question %d" % i} for i in range(4)
    ] + [{"role": "assistant", "text": "noted", "episode_type": "chatter"}]
    ingest_session(store, "s", turns)
    summary = evict_to_budget(store, "s", 0, protect_last=0)
    assert summary["n_evicted"] == 1  # only the chatter could go
    assert summary["under_budget"] is False
    assert summary["overage"] > 0
    assert summary["user_turns_live"] == 4
    assert all(
        e["status"] == "live"
        for e in store.list_episodes("s")
        if e["type"] == "user"
    )


def test_dependency_protection_dynamic(store):
    rid_a = _record(store, "the fact itself")
    rid_b = _record(store, "reasoning that uses the fact")
    b = annotate_episode(store, "s", rid_a, "fact")
    a = annotate_episode(store, "s", rid_b, "reasoning", depends_on=[b])
    total = store.session_token_total("s")
    # Budget forces one eviction: B is protected (A depends on it),
    # so A goes first even though B is the lower tier.
    summary = evict_to_budget(
        store, "s", total - store.get_episode(a)["tokens"], protect_last=0
    )
    assert [e["episode_id"] for e in summary["evicted"]] == [a]
    # Now nothing depends on B: a tighter budget evicts it.
    summary = evict_to_budget(store, "s", 0, protect_last=0)
    assert [e["episode_id"] for e in summary["evicted"]] == [b]
    assert dependency_violations(store, "s") == []


def test_active_window_protected(store):
    turns = [
        {"role": "assistant", "text": "filler %d" % i, "episode_type": "chatter"}
        for i in range(6)
    ]
    ids = ingest_session(store, "s", turns)
    summary = evict_to_budget(store, "s", 0, protect_last=3)
    assert [e["episode_id"] for e in summary["evicted"]] == ids[:3]
    assert [e["id"] for e in store.list_episodes("s", status="live")] == ids[3:]


def test_dry_run_writes_nothing(store):
    _session_with_tiers(store)
    n_before = len(store.list_evictions())
    summary = evict_to_budget(store, "s", 0, protect_last=0, dry_run=True)
    assert summary["would_evict"] > 0
    assert summary["dry_run"] is True
    assert len(store.list_episodes("s", status="live")) == 10
    assert len(store.list_evictions()) == n_before


def test_explain_episode(store):
    ids = _session_with_tiers(store)
    # Evict one, then explain it, a protected user turn, and a
    # live evictable episode.
    total = store.session_token_total("s")
    eps = store.list_episodes("s")
    evict_to_budget(store, "s", total - eps[3]["tokens"], protect_last=0)
    gone = explain_episode(store, ids[3])
    assert "evicted" in gone and "persisted by record" in gone
    assert "never" in explain_episode(store, ids[0])
    live = explain_episode(store, ids[9], protect_last=0)
    assert "tier 6" in live and "unpersisted" in live


def test_mark_persisted_by_merges(store):
    turns = [
        {"role": "user", "text": "what does Maya use"},
        {"role": "assistant", "text": "Maya uses coarse flour for bread"},
        {"role": "assistant", "text": "the crumb opened up nicely"},
    ]
    ids = ingest_session(store, "s", turns)
    recs = [store.get_episode(e)["record_id"] for e in ids]
    texts = [store.get_record(r)["text"] for r in recs]
    res = propose_merge(
        store,
        HashedEmbedder(),
        recs,
        merged_text=draft_merge_text(texts),
        max_length_ratio=1.0,
    )
    assert res.status == "committed"
    marked = mark_session_persisted_by_merges(store, "s")
    assert marked == 3
    assert all(
        e["persisted_by"] == res.merged_record_id
        for e in store.list_episodes("s")
    )


def test_naive_evicts_user_turns(store):
    turns = []
    for i in range(5):
        turns.append({"role": "user", "text": "question %d here" % i})
        turns.append(
            {"role": "assistant", "text": "noted %d" % i, "episode_type": "chatter"}
        )
    ingest_session(store, "s", turns)
    total = store.session_token_total("s")
    eps = store.list_episodes("s")
    # Evict three episodes' worth of tokens (oldest first).
    budget = total - sum(e["tokens"] for e in eps[:3])
    naive = evict_to_budget(store, "s", budget, protect_last=0, policy="naive")
    assert naive["user_turns_live"] < naive["user_turns_total"]
    # Same session shape, CWL policy: user turns all survive, and
    # exactly three chatter episodes go (budget = 3 chatter tokens).
    store2 = Store(str(store.path) + "2", embedder=HashedEmbedder())
    try:
        ingest_session(store2, "s", turns)
        total2 = store2.session_token_total("s")
        eps2 = store2.list_episodes("s")
        chat_tok = next(
            e["tokens"] for e in eps2 if e["type"] == "chatter"
        )
        budget2 = total2 - 3 * chat_tok
        cwl = evict_to_budget(store2, "s", budget2, protect_last=0)
        assert cwl["user_turns_live"] == cwl["user_turns_total"]
        assert cwl["n_evicted"] == 3
    finally:
        store2.close()


def test_eviction_demo_small(tmp_path):
    root = os.path.join(os.path.dirname(__file__), "..")
    result = run_eviction_eval(
        os.path.join(root, "eval_data", "synthetic_corpus.jsonl"),
        os.path.join(root, "eval_data", "synthetic_queries.json"),
        embedder=HashedEmbedder(),
        n_tasks=4,
        verbose=False,
    )
    assert result["n_tasks"] == 4
    cwl = result["arms"]["cwl"]
    naive = result["arms"]["naive"]
    iso = result["arms"]["isolated"]
    # The policy's value is in the conversation, not the facts:
    # CWL keeps every user turn in context, naive truncation
    # drops them, and CWL never strands a dependent.
    assert cwl["user_turn_recall"] == 1.0
    assert naive["user_turn_recall"] < cwl["user_turn_recall"]
    assert cwl["dependency_violations"] == 0
    assert naive["dependency_violations"] >= 0
    assert cwl["under_budget"] is True
    assert naive["under_budget"] is True
    # Facts survive consolidation in both long arms.
    assert cwl["task_any_hit5"] >= naive["task_any_hit5"]
    assert iso["task_any_hit5"] >= 0.0  # isolated is the reference arm
