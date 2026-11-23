"""P2 lifecycle tests: TRACE-UCB credit, judge-gated merge, sleep job.

Uses HashedEmbedder (no torch) so the tests run anywhere; the
certificate's embedding checks then run on lexical vectors, which is
the honest fallback the code documents.
"""

import sqlite3

import pytest

from hybrid_retriever.embeddings import HashedEmbedder
from hybrid_retriever.lifecycle import (
    attribute_credit,
    draft_merge_text,
    find_merge_candidates,
    propose_merge,
    record_use,
    rollback_merge,
    sleep,
)
from hybrid_retriever.retrieval import retrieve
from hybrid_retriever.store import Store

EMB = HashedEmbedder()

S1 = "Maya feeds her sourdough starter with rye flour every morning at six."
S2 = "The starter Maya keeps in the pantry bubbled over after four days without feeding."
S3 = "Maya's starter, begun in 2019, now leavens twelve loaves a week."


@pytest.fixture()
def store(tmp_path):
    s = Store(str(tmp_path / "l.db"), embedder=EMB)
    yield s
    s.close()


def _retrieve_all(store, query):
    return retrieve(store, EMB, query, k=10, use_mmr=False)


# ----------------------------------------------------------------------
# TRACE-UCB
# ----------------------------------------------------------------------
def test_credit_rewards_used_records_only(store):
    store.add_texts(
        [
            "Maya feeds her sourdough starter every morning at six.",
            "Dev rides his bike along the river path.",
        ]
    )
    res = _retrieve_all(store, "Maya sourdough starter")
    assert [r.id for r in res] == [1]
    credits = record_use(
        store, res, "Maya feeds her sourdough starter every morning."
    )
    assert credits[1] > 0.3
    rec1 = store.get_record(1)
    assert rec1["uses"] == 1 and rec1["credit"] > 0
    assert rec1["importance"] > 0.5
    # Never retrieved: untouched, neutral prior.
    rec2 = store.get_record(2)
    assert rec2["uses"] == 0 and rec2["credit"] == 0.0
    assert rec2["importance"] == 0.5
    # Retrieved but unused: earns no credit. Its importance is still
    # above the prior after one use (the UCB exploration bonus
    # dominates a single trial) and only sinks below it after many
    # fruitless uses.
    res2 = _retrieve_all(store, "Dev river path bike")
    assert res2[0].id == 2
    credits2 = record_use(
        store, res2, "Maya feeds her sourdough starter every morning."
    )
    assert credits2[2] == 0.0
    assert store.get_record(2)["uses"] == 1
    assert store.get_record(2)["importance"] > 0.5
    for _ in range(20):
        record_use(
            store, res2, "Maya feeds her sourdough starter every morning."
        )
    assert store.get_record(2)["importance"] < 0.5
    assert store.get_record(1)["importance"] > store.get_record(2)["importance"]


def test_empty_answer_records_nothing(store):
    store.add_texts(["Maya feeds her sourdough starter every morning."])
    res = _retrieve_all(store, "Maya sourdough starter")
    assert record_use(store, res, "   ") == {}
    assert store.get_record(1)["uses"] == 0


def test_importance_boost_reorders_retrieval(store):
    store.add_texts(
        [
            "Maya feeds her sourdough starter every morning at six.",
            "Maya's starter began in 2019 and leavens twelve loaves a week.",
        ]
    )
    # Baseline: record 1 (exact query words) ranks first.
    base = _retrieve_all(store, "Maya sourdough starter")
    assert base[0].id == 1
    # Warm-up: an answer grounded in record 2 only.
    credits = record_use(store, base, store.get_record(2)["text"])
    assert credits[2] > credits[1]
    boosted = retrieve(
        store, EMB, "Maya sourdough starter", k=10, use_mmr=False,
        importance_weight=2.0,
    )
    assert boosted[0].id == 2
    # w = 0 is exactly P1 behaviour.
    plain = retrieve(
        store, EMB, "Maya sourdough starter", k=10, use_mmr=False,
        importance_weight=0.0,
    )
    assert [r.id for r in plain] == [r.id for r in base]


def test_co_recall_logged_only_for_credited_pairs(store):
    store.add_texts([S1, S2, "Dev rides his bike along the river path."])
    res = _retrieve_all(store, "Maya sourdough starter pantry")
    record_use(store, res, S1 + " " + S2)
    pairs = store.co_recall_pairs(min_count=1)
    assert (1, 2, 1) in pairs
    # The bike record earned no credit, so it pairs with nothing.
    assert all(3 not in (a, b) for a, b, _ in pairs)


def test_attribute_credit_uniqueness_is_leave_one_out():
    from hybrid_retriever.retrieval import Retrieved

    retrieved = [
        Retrieved(id=1, text=S1, score=1.0, legs={"bm25", "vector"}),
        Retrieved(id=2, text=S2, score=0.9, legs={"bm25"}),
    ]
    # Answer uses only record 1's distinctive words.
    credits = attribute_credit(retrieved, S1)
    assert credits[1] > credits[2]
    # Answer uses both records' words: both earn, the shared name
    # ("Maya", "starter") does not double-count.
    credits = attribute_credit(retrieved, S1 + " " + S2)
    assert credits[1] > 0 and credits[2] > 0


# ----------------------------------------------------------------------
# Judge-gated merge
# ----------------------------------------------------------------------
def test_merge_commits_for_overlapping_records(store):
    a = store.add_record(S1 + " " + S2)
    b = store.add_record(S2 + " " + S3)
    res = propose_merge(store, EMB, [a, b])
    assert res.status == "committed"
    assert res.merged_record_id is not None
    cert = res.certificate
    assert cert["passed"] and cert["checks"]["faithfulness"]["passed"]
    assert cert["checks"]["length_ratio"]["value"] <= 0.90
    assert cert["checks"]["coverage"]["value"] >= 0.45
    # Sources superseded, merged record live and retrievable.
    assert store.get_record(a)["status"] == "superseded"
    assert store.get_record(b)["status"] == "superseded"
    assert store.get_record(res.merged_record_id)["status"] == "live"
    assert store.get_record(res.merged_record_id)["text"] == S1 + " " + S2 + " " + S3
    hits = store.bm25_search("sourdough starter pantry", k=10)
    assert res.merged_record_id in [h[0] for h in hits]
    assert a not in [h[0] for h in hits]
    assert b not in [h[0] for h in hits]
    # Audit row recorded.
    row = store.get_merge(res.merge_id)
    assert row["status"] == "committed"
    assert row["merged_ids"] == [a, b]


def test_merge_rejects_invented_content(store):
    a = store.add_record(S1)
    b = store.add_record(S2)
    res = propose_merge(
        store, EMB, [a, b],
        merged_text=S1 + " And the moon is made of green cheese.",
    )
    assert res.status == "rejected"
    assert "faithfulness" in res.certificate["reasons"]
    assert res.merged_record_id is None
    assert store.get_record(a)["status"] == "live"
    assert store.get_record(b)["status"] == "live"
    assert store.get_merge(res.merge_id)["status"] == "rejected"


def test_merge_rejects_distinct_facts(store):
    # Two single-sentence records about different things: the union
    # compresses nothing, so the length check must reject it.
    a = store.add_record(S1)
    b = store.add_record("Dev rides his bike along the river path.")
    res = propose_merge(store, EMB, [a, b])
    assert res.status == "rejected"
    assert "length_ratio" in res.certificate["reasons"]


def test_merge_rejects_mixed_types_and_single_source(store):
    a = store.add_record(S1, type_="fact")
    b = store.add_record(S2, type_="episode")
    res = propose_merge(store, EMB, [a, b])
    assert res.status == "rejected"
    assert "mixed record types" in res.certificate["reasons"]
    res = propose_merge(store, EMB, [a])
    assert res.status == "rejected"


def test_rollback_restores_sources(store):
    a = store.add_record(S1 + " " + S2)
    b = store.add_record(S2 + " " + S3)
    res = propose_merge(store, EMB, [a, b])
    assert res.status == "committed"
    assert rollback_merge(store, res.merge_id) is True
    assert store.get_record(a)["status"] == "live"
    assert store.get_record(b)["status"] == "live"
    assert store.get_record(res.merged_record_id)["status"] == "archived"
    assert store.get_merge(res.merge_id)["status"] == "rolled-back"
    with pytest.raises(ValueError):
        rollback_merge(store, res.merge_id)  # already rolled back
    with pytest.raises(ValueError):
        rollback_merge(store, 9999)


def test_draft_merge_text_is_sentence_union():
    assert draft_merge_text([S1 + " " + S2, S2 + " " + S3]) == S1 + " " + S2 + " " + S3
    assert draft_merge_text([S1, S1]) == S1


# ----------------------------------------------------------------------
# Sleep job
# ----------------------------------------------------------------------
def test_sleep_merges_prunes_and_promotes(store):
    # Near-duplicate pair (Jaccard 1.0 on content tokens).
    store.add_record("Maya feeds her sourdough starter every morning.")
    store.add_record("Maya feeds her sourdough starter every morning!")
    # A dead record: never used, aged 100 days via direct SQL.
    dead = store.add_record("An old note nobody ever retrieves.")
    conn = sqlite3.connect(str(store.path))
    conn.execute(
        "UPDATE records SET created_at = created_at - ? WHERE id = ?",
        (100 * 86400, dead),
    )
    conn.commit()
    conn.close()
    # Co-recall: use two records together once, twice (min count 2).
    store.add_texts([S1, S2])
    res = _retrieve_all(store, "Maya sourdough starter pantry")
    record_use(store, res, S1 + " " + S2)
    record_use(store, res, S1 + " " + S2)

    summary = sleep(store, EMB, prune_days=30, min_co_recall=2)
    assert summary["merges_proposed"] == 1
    assert summary["merges_committed"] == 1
    assert summary["merges_rejected"] == 0
    assert summary["pruned"] == 1
    assert summary["co_recall_pairs_promoted"] == 1
    assert store.get_record(dead)["status"] == "pruned"
    # Live at the end: the merged record + S1 + S2. The pair's
    # originals are superseded, the dead record pruned.
    assert store.count_live() == 3
    assert store.non_live_ids() == {1, 2, dead}


def test_sleep_dry_run_writes_nothing(store):
    store.add_record("Maya feeds her sourdough starter every morning.")
    store.add_record("Maya feeds her sourdough starter every morning!")
    before = store.count_live()
    summary = sleep(store, EMB, prune_days=0, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["merges_proposed"] == 1
    assert summary["merges_would_commit"] == 1
    assert summary["merges_committed"] == 0
    assert store.count_live() == before
    assert store.list_merges() == []


def test_find_merge_candidates_ignores_distinct_records(store):
    store.add_texts([S1, S2, S3])
    assert find_merge_candidates(store, EMB) == []


# ----------------------------------------------------------------------
# Store migration
# ----------------------------------------------------------------------
def test_v1_database_is_migrated_in_place(tmp_path):
    path = str(tmp_path / "v1.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE records(id INTEGER PRIMARY KEY, text TEXT NOT NULL, "
        "type TEXT NOT NULL DEFAULT 'fact', tags TEXT NOT NULL DEFAULT '[]', "
        "created_at REAL NOT NULL, meta TEXT NOT NULL DEFAULT '{}')"
    )
    conn.execute(
        "INSERT INTO records(text, type, created_at) VALUES (?, ?, ?)",
        ("Maya feeds her sourdough starter every morning.", "fact", 1000.0),
    )
    conn.commit()
    conn.close()
    s = Store(path, embedder=EMB)
    rec = s.get_record(1)
    assert rec["status"] == "live"
    assert rec["importance"] == 0.5
    assert rec["uses"] == 0
    assert rec["credit"] == 0.0
    s.close()
